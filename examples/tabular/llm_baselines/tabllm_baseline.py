#!/usr/bin/env python
"""
TabLLM baseline evaluation module.
Contains functions for evaluating the TabLLM baseline on tabular datasets.
"""

import os
import sys
import numpy as np
import torch
import json
import glob
import time
import logging
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List, Tuple
from marvis.utils.model_loader import model_loader, GenerationConfig


def create_regression_bins(y_train: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, List[str], float, float]:
    """
    Create bins for converting regression to classification.
    
    Args:
        y_train: Training target values
        n_bins: Number of bins to create
        
    Returns:
        Tuple of (bin_edges, bin_labels, min_val, max_val)
    """
    min_val, max_val = y_train.min(), y_train.max()
    if min_val == max_val:
        # Handle constant values by adding small perturbation
        min_val -= 0.1
        max_val += 0.1
    
    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Create bin labels with range descriptions
    bin_labels = []
    for i in range(n_bins):
        bin_labels.append(f"Range_{i}_{bin_edges[i]:.3g}_to_{bin_edges[i+1]:.3g}")
    
    return bin_edges, bin_labels, min_val, max_val


def convert_targets_to_bins(y: np.ndarray, bin_edges: np.ndarray, bin_labels: List[str]) -> List[str]:
    """
    Convert continuous target values to bin labels.
    
    Args:
        y: Target values to convert
        bin_edges: Bin edge values
        bin_labels: Bin label strings
        
    Returns:
        List of bin labels
    """
    # Use digitize to assign values to bins (1-indexed)
    bin_indices = np.digitize(y, bin_edges, right=False)
    
    # Handle edge cases
    bin_indices = np.clip(bin_indices - 1, 0, len(bin_labels) - 1)
    
    return [bin_labels[i] for i in bin_indices]


def convert_bin_predictions_to_values(predictions: List[str], bin_edges: np.ndarray, bin_labels: List[str]) -> np.ndarray:
    """
    Convert bin predictions back to continuous values using bin centers.
    
    Args:
        predictions: List of predicted bin labels
        bin_edges: Bin edge values
        bin_labels: Bin label strings
        
    Returns:
        Array of continuous values
    """
    values = []
    
    # Create mapping from bin labels to bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    label_to_center = {label: center for label, center in zip(bin_labels, bin_centers)}
    
    for pred in predictions:
        if pred in label_to_center:
            values.append(label_to_center[pred])
        else:
            # Fallback: try to extract range from label name
            try:
                # Look for "Range_X_min_to_max" pattern
                parts = pred.split('_')
                if len(parts) >= 4:
                    min_val = float(parts[2])
                    max_val = float(parts[4])
                    center = (min_val + max_val) / 2
                    values.append(center)
                else:
                    # Default to middle of overall range
                    values.append((bin_edges[0] + bin_edges[-1]) / 2)
            except (ValueError, IndexError):
                # Default to middle of overall range
                values.append((bin_edges[0] + bin_edges[-1]) / 2)
    
    return np.array(values)


def load_tabllm_config_by_openml_id(openml_task_id, original_feature_count=None):
    """Load TabLLM configuration by OpenML task ID with feature count validation.
    
    Args:
        openml_task_id: OpenML task ID to look up
        original_feature_count: Number of features in dataset before preprocessing
        
    Returns:
        Tuple of (template_data, feature_mapping) or (None, None)
    """
    logger = logging.getLogger(__name__)
    
    # Use resource manager for path resolution
    try:
        from marvis.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        tabllm_dir = resource_manager.path_resolver.get_config_path('tabllm_like', '')
        
        if tabllm_dir and tabllm_dir.exists():
            # Try direct task ID lookup first (new approach)
            template_path = tabllm_dir / f"templates_task_{openml_task_id}.yaml"
        else:
            template_path = None
            
    except Exception as e:
        logger.warning(f"Could not use resource manager for TabLLM template lookup: {e}")
        # Fallback to relative path from current script location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tabllm_dir = os.path.join(current_dir, "tabllm_like")
        template_path = os.path.join(tabllm_dir, f"templates_task_{openml_task_id}.yaml")
    
    template_data = None
    try:
        if template_path and template_path.exists():
            import yaml
            
            # Define constructors for custom YAML tags
            def template_constructor(loader, node):
                return loader.construct_mapping(node)
            def template_metadata_constructor(loader, node):
                return loader.construct_mapping(node)
            
            # Create a custom loader
            CustomLoader = yaml.SafeLoader
            CustomLoader.add_constructor('!Template', template_constructor)
            CustomLoader.add_constructor('!TemplateMetadata', template_metadata_constructor)
            
            with open(template_path, 'r') as f:
                template_data = yaml.load(f, Loader=CustomLoader)
            
            dataset_name = template_data.get('dataset_name', f'task_{openml_task_id}')
            logger.info(f"Found TabLLM config for OpenML task {openml_task_id} (dataset: {dataset_name})")
            
            # Load semantic information for feature count validation
            semantic_file = None
            try:
                # Try resource manager first with new general search
                if 'rm' in locals():
                    semantic_file = rm.find_semantic_metadata(openml_task_id)
                    if semantic_file is None:
                        # Fallback to old cc18_semantic method for backward compatibility
                        semantic_file = rm.path_resolver.get_config_path('cc18_semantic', str(openml_task_id))
            except:
                pass
            
            # Fallback to metadata loader
            if semantic_file is None or (hasattr(semantic_file, 'exists') and not semantic_file.exists()):
                try:
                    from marvis.utils.metadata_loader import get_metadata_loader
                    loader = get_metadata_loader()
                    semantic_file = loader.detect_metadata_file(openml_task_id)
                except Exception as e:
                    logger.debug(f"Could not use metadata loader: {e}")
                    semantic_file = None
            
            semantic_file_exists = (hasattr(semantic_file, 'exists') and semantic_file.exists()) or \
                                   (isinstance(semantic_file, str) and os.path.exists(semantic_file))
            
            if semantic_file_exists and original_feature_count is not None:
                try:
                    with open(semantic_file, 'r') as f:
                        semantic_info = json.load(f)
                    
                    # Skip early feature count validation since we'll expand semantics online
                    # to match the actual processed feature count after reduction
                    logger.info(f"Semantic info loaded for online note generation (will expand to match processed features)")
                    
                    # Create feature mapping to preserve semantic descriptions
                    feature_mapping = {
                        'semantic_info': semantic_info,
                        'task_id': openml_task_id,
                        'dataset_name': dataset_name
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not validate feature count from semantic info: {e}")
                    feature_mapping = None
            else:
                feature_mapping = None
            
            return template_data, feature_mapping
        
        # Fallback: try old dataset name-based approach for backward compatibility
        return _load_tabllm_config_by_dataset_name_fallback(openml_task_id, original_feature_count)
        
    except ValueError:
        # Re-raise validation errors (like feature count mismatch)
        raise
    except Exception as e:
        logger.error(f"Error loading TabLLM config for OpenML task {openml_task_id}: {e}")
        return None, None


def _load_tabllm_config_by_dataset_name_fallback(openml_task_id, original_feature_count=None):
    """Fallback method using the old dataset name mapping approach."""
    logger = logging.getLogger(__name__)
    
    # Try to use the new resource manager first
    task_mapping = None
    try:
        from marvis.utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        mapping_path = rm.path_resolver.get_configs_dir() / 'tabllm' / 'openml_task_mapping.json'
        
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                task_mapping = json.load(f)
            logger.debug(f"Loaded OpenML task mapping from managed config")
        else:
            logger.debug(f"OpenML task mapping not found in managed config")
    except Exception as e:
        logger.debug(f"Could not load from managed config: {e}")
    
    # Fallback to legacy method
    if task_mapping is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tabllm_dir = os.path.join(current_dir, "tabllm_like")
        
        # Load OpenML task mapping
        mapping_path = os.path.join(tabllm_dir, "openml_task_mapping.json")
        if not os.path.exists(mapping_path):
            logger.debug(f"OpenML task mapping not found at {mapping_path}")
            return None, None
        
        try:
            with open(mapping_path, 'r') as f:
                task_mapping = json.load(f)
        except Exception as e:
            logger.debug(f"Error loading OpenML task mapping: {e}")
            return None, None
    
    # Find dataset name that maps to this OpenML task ID
    dataset_name = None
    for name, task_id in task_mapping.items():
        if task_id == openml_task_id:
            dataset_name = name
            break
    
    if dataset_name is None:
        logger.debug(f"No TabLLM config found for OpenML task ID {openml_task_id}")
        return None, None
    
    # Load the corresponding template file using legacy approach
    template_data = load_tabllm_template(dataset_name)
    if template_data is None:
        logger.debug(f"TabLLM template file not found for dataset: {dataset_name}")
        return None, None
    
    logger.info(f"Found TabLLM config for OpenML task {openml_task_id} (dataset: {dataset_name}) via fallback")
    
    # Feature validation (simplified for fallback)
    feature_mapping = None
    if original_feature_count is not None:
        # Basic feature mapping without extensive validation
        feature_mapping = {
            'task_id': openml_task_id,
            'dataset_name': dataset_name
        }
    
    return template_data, feature_mapping


def load_tabllm_template(dataset_name):
    """Load TabLLM template if available for the dataset (legacy function)."""
    # Use relative path from current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "tabllm_like", f"templates_{dataset_name}.yaml")
    
    try:
        if os.path.exists(template_path):
            import yaml
            
            # Define constructors for custom YAML tags
            def template_constructor(loader, node):
                return loader.construct_mapping(node)
            def template_metadata_constructor(loader, node):
                return loader.construct_mapping(node)
            
            # Create a custom loader
            CustomLoader = yaml.SafeLoader
            CustomLoader.add_constructor('!Template', template_constructor)
            CustomLoader.add_constructor('!TemplateMetadata', template_metadata_constructor)
            
            with open(template_path, 'r') as f:
                template_data = yaml.load(f, Loader=CustomLoader)
            return template_data
        return None
    except Exception as e:
        logging.getLogger(__name__).debug(f"Could not load template for {dataset_name}: {e}")
        return None

def create_feature_mapping_after_preprocessing(original_feature_names, processed_feature_names, feature_mapping):
    """Create mapping from processed feature names to original semantic descriptions.
    
    Args:
        original_feature_names: List of original feature names from dataset
        processed_feature_names: List of feature names after preprocessing
        feature_mapping: Original feature mapping from TabLLM config
        
    Returns:
        Dictionary mapping processed feature names to semantic info
    """
    if feature_mapping is None or 'semantic_info' not in feature_mapping:
        return {}
    
    semantic_info = feature_mapping['semantic_info']
    processed_mapping = {
        'semantic_info': {},
        'task_id': feature_mapping.get('task_id'),
        'dataset_name': feature_mapping.get('dataset_name')
    }
    
    # Extract original descriptions based on semantic info structure
    original_descriptions = {}
    if 'columns' in semantic_info:
        for col in semantic_info['columns']:
            if col.get('name') != 'target':
                original_descriptions[col['name']] = col.get('semantic_description', col['name'])
    elif 'feature_descriptions' in semantic_info:
        original_descriptions = semantic_info['feature_descriptions']
    elif 'feature_description' in semantic_info:
        original_descriptions = semantic_info['feature_description']
    
    # Create mapping from original to processed names
    for i, processed_name in enumerate(processed_feature_names):
        if i < len(original_feature_names):
            original_name = original_feature_names[i]
            if original_name in original_descriptions:
                processed_mapping['semantic_info'][processed_name] = original_descriptions[original_name]
            # Also try to match by name similarity for robustness
            else:
                # Look for partial matches in original descriptions
                for orig_key, description in original_descriptions.items():
                    if orig_key.lower() in processed_name.lower() or processed_name.lower() in orig_key.lower():
                        processed_mapping['semantic_info'][processed_name] = description
                        break
    
    return processed_mapping

# NOTE: Old note loading functions removed - TabLLM now generates notes online during evaluation


def expand_semantic_features(semantic_info: Dict[str, Any], target_feature_count: int, 
                           actual_feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Expand semantic features by duplicating them to match actual dataset features.
    
    Args:
        semantic_info: Original semantic information dictionary
        target_feature_count: Number of features the dataset actually has
        actual_feature_names: List of actual feature names from the processed dataset
        
    Returns:
        Updated semantic_info with expanded features
    """
    # Make a copy to avoid modifying the original
    expanded_info = semantic_info.copy()
    
    # Handle different semantic info structures
    if 'columns' in semantic_info:
        original_columns = semantic_info['columns']
        current_count = len(original_columns)
        
        # Check if we need to expand or realign features
        needs_expansion = current_count < target_feature_count
        needs_realignment = (actual_feature_names and 
                           len(actual_feature_names) == target_feature_count and 
                           current_count <= target_feature_count)
        
        if needs_expansion or needs_realignment:
            logger = logging.getLogger(__name__)
            if needs_expansion:
                logger.info(f"Expanding {current_count} semantic features to {target_feature_count} by duplicating with actual feature names...")
            else:
                logger.info(f"Realigning {current_count} semantic features to match actual feature names...")
            expanded_columns = []
            
            # If we have actual feature names, use them; otherwise create suffixed names
            if actual_feature_names and len(actual_feature_names) == target_feature_count:
                for i, actual_name in enumerate(actual_feature_names):
                    # Get the base feature to duplicate (cycle through original features)
                    base_feature = original_columns[i % current_count]
                    
                    # Create new feature with actual name
                    new_feature = base_feature.copy()
                    new_feature['name'] = actual_name
                    if 'semantic_description' in new_feature:
                        variant_num = (i // current_count) + 1 if i >= current_count else ""
                        variant_suffix = f" (variant {variant_num})" if variant_num else ""
                        new_feature['semantic_description'] = f"{base_feature['semantic_description']}{variant_suffix}"
                    
                    expanded_columns.append(new_feature)
            else:
                # Fallback to suffixed names
                for i in range(target_feature_count):
                    # Get the base feature to duplicate (cycle through original features)
                    base_feature = original_columns[i % current_count]
                    suffix_num = (i // current_count) + 1
                    
                    # Create new feature with suffix
                    new_feature = base_feature.copy()
                    new_feature['name'] = f"{base_feature['name']}_{suffix_num}" if i >= current_count else base_feature['name']
                    if 'semantic_description' in new_feature:
                        variant_suffix = f" (variant {suffix_num})" if i >= current_count else ""
                        new_feature['semantic_description'] = f"{base_feature['semantic_description']}{variant_suffix}"
                    
                    expanded_columns.append(new_feature)
            
            expanded_info['columns'] = expanded_columns
    
    elif 'feature_descriptions' in semantic_info:
        original_features = semantic_info['feature_descriptions']
        current_count = len(original_features)
        
        # Check if we need to expand or realign features
        needs_expansion = current_count < target_feature_count
        needs_realignment = (actual_feature_names and 
                           len(actual_feature_names) == target_feature_count and 
                           current_count <= target_feature_count)
        
        if needs_expansion or needs_realignment:
            logger = logging.getLogger(__name__)
            if needs_expansion:
                logger.info(f"Expanding {current_count} semantic features to {target_feature_count} by duplicating with actual feature names...")
            else:
                logger.info(f"Realigning {current_count} semantic features to match actual feature names...")
            expanded_features = {}
            
            # If we have actual feature names, use them; otherwise create suffixed names
            if actual_feature_names and len(actual_feature_names) == target_feature_count:
                original_names = list(original_features.keys())
                for i, actual_name in enumerate(actual_feature_names):
                    # Get the base feature to duplicate (cycle through original features)
                    base_name = original_names[i % current_count]
                    base_desc = original_features[base_name]
                    
                    variant_num = (i // current_count) + 1 if i >= current_count else ""
                    variant_suffix = f" (variant {variant_num})" if variant_num else ""
                    expanded_features[actual_name] = f"{base_desc}{variant_suffix}"
            else:
                # Fallback to suffixed names
                original_names = list(original_features.keys())
                for i in range(target_feature_count):
                    base_name = original_names[i % current_count]
                    base_desc = original_features[base_name]
                    suffix_num = (i // current_count) + 1
                    
                    new_name = f"{base_name}_{suffix_num}" if i >= current_count else base_name
                    variant_suffix = f" (variant {suffix_num})" if i >= current_count else ""
                    expanded_features[new_name] = f"{base_desc}{variant_suffix}"
            
            expanded_info['feature_descriptions'] = expanded_features
    
    elif 'feature_description' in semantic_info:
        original_features = semantic_info['feature_description']
        current_count = len(original_features)
        
        # Check if we need to expand or realign features
        needs_expansion = current_count < target_feature_count
        needs_realignment = (actual_feature_names and 
                           len(actual_feature_names) == target_feature_count and 
                           current_count <= target_feature_count)
        
        if needs_expansion or needs_realignment:
            logger = logging.getLogger(__name__)
            if needs_expansion:
                logger.info(f"Expanding {current_count} semantic features to {target_feature_count} by duplicating with actual feature names...")
            else:
                logger.info(f"Realigning {current_count} semantic features to match actual feature names...")
            expanded_features = {}
            
            # If we have actual feature names, use them; otherwise create suffixed names
            if actual_feature_names and len(actual_feature_names) == target_feature_count:
                original_names = list(original_features.keys())
                for i, actual_name in enumerate(actual_feature_names):
                    # Get the base feature to duplicate (cycle through original features)
                    base_name = original_names[i % current_count]
                    base_desc = original_features[base_name]
                    
                    variant_num = (i // current_count) + 1 if i >= current_count else ""
                    variant_suffix = f" (variant {variant_num})" if variant_num else ""
                    expanded_features[actual_name] = f"{base_desc}{variant_suffix}"
            else:
                # Fallback to suffixed names
                original_names = list(original_features.keys())
                for i in range(target_feature_count):
                    base_name = original_names[i % current_count]
                    base_desc = original_features[base_name]
                    suffix_num = (i // current_count) + 1
                    
                    new_name = f"{base_name}_{suffix_num}" if i >= current_count else base_name
                    variant_suffix = f" (variant {suffix_num})" if i >= current_count else ""
                    expanded_features[new_name] = f"{base_desc}{variant_suffix}"
            
            expanded_info['feature_description'] = expanded_features
    
    return expanded_info


def generate_note_from_row(row, semantic_info: Dict[str, Any], attribute_names: List[str], exclude_target: bool = True) -> str:
    """Generate a TabLLM-style note from a data row using semantic info."""
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    note_parts = []
    
    # Convert row to pandas Series if it isn't already
    if not isinstance(row, pd.Series):
        if hasattr(row, '__len__') and len(attribute_names) == len(row):
            row = pd.Series(row, index=attribute_names)
        else:
            # Fallback: create basic series
            row = pd.Series(row) if hasattr(row, '__iter__') else pd.Series([row])
    
    # Determine the number of features we need to describe
    feature_count = len(row)
    
    # Extract semantic feature names and descriptions based on structure
    semantic_features = []
    if 'columns' in semantic_info:
        semantic_features = [(col['name'], col.get('semantic_description', col['name'])) 
                           for col in semantic_info['columns'] 
                           if col.get('name') != 'target']
    elif 'feature_descriptions' in semantic_info:
        semantic_features = [(name, desc) for name, desc in semantic_info['feature_descriptions'].items()]
    elif 'feature_description' in semantic_info:
        semantic_features = [(name, desc) for name, desc in semantic_info['feature_description'].items()]
    elif 'feature_names' in semantic_info:
        semantic_features = [(name, meaning) for name, meaning in semantic_info['feature_names'].items()]
    
    # Check if semantic features match the feature count
    if len(semantic_features) == feature_count:
        # Use semantic features by position
        logger.debug(f"Using {len(semantic_features)} semantic features by position")
        for i, (value, (sem_name, sem_desc)) in enumerate(zip(row, semantic_features)):
            if pd.isna(value):
                note_parts.append(f"The {sem_desc} is missing.")
            elif isinstance(value, (int, float)):
                if value == int(value):
                    note_parts.append(f"The {sem_desc} is {int(value)}.")
                else:
                    note_parts.append(f"The {sem_desc} is {value:.2f}.")
            else:
                note_parts.append(f"The {sem_desc} is {value}.")
    
    elif len(attribute_names) == feature_count:
        # Use attribute names by position, ensuring they're strings and unique
        logger.debug(f"Using {len(attribute_names)} attribute names by position")
        
        # Convert to strings and ensure uniqueness
        clean_names = []
        seen_names = set()
        for i, name in enumerate(attribute_names):
            str_name = str(name)
            # Make unique if duplicate
            if str_name in seen_names:
                counter = 1
                while f"{str_name}_{counter}" in seen_names:
                    counter += 1
                str_name = f"{str_name}_{counter}"
            clean_names.append(str_name)
            seen_names.add(str_name)
        
        for i, (value, feat_name) in enumerate(zip(row, clean_names)):
            if pd.isna(value):
                note_parts.append(f"The {feat_name} is missing.")
            elif isinstance(value, (int, float)):
                if value == int(value):
                    note_parts.append(f"The {feat_name} is {int(value)}.")
                else:
                    note_parts.append(f"The {feat_name} is {value:.2f}.")
            else:
                note_parts.append(f"The {feat_name} is {value}.")
    
    else:
        # Fallback to Feature_i format
        logger.warning(f"Feature count mismatch - Row: {feature_count}, Semantic: {len(semantic_features)}, Names: {len(attribute_names)}")
        logger.debug(f"Using Feature_i fallback for {feature_count} features")
        for i, value in enumerate(row):
            feat_desc = f"Feature {i}"
            if pd.isna(value):
                note_parts.append(f"The {feat_desc} is missing.")
            elif isinstance(value, (int, float)):
                if value == int(value):
                    note_parts.append(f"The {feat_desc} is {int(value)}.")
                else:
                    note_parts.append(f"The {feat_desc} is {value:.2f}.")
            else:
                note_parts.append(f"The {feat_desc} is {value}.")
    
    return " ".join(note_parts)


def estimate_note_tokens(note: str, tokenizer) -> int:
    """Estimate the number of tokens in a note using existing MARVIS utilities."""
    try:
        if tokenizer is not None:
            return len(tokenizer.encode(note))
        else:
            # Fallback: rough estimation when tokenizer unavailable (API models)
            return len(note.split()) * 1.3  # ~1.3 tokens per word on average
    except Exception:
        # Fallback: rough estimation
        return len(note.split()) * 1.3  # ~1.3 tokens per word on average


def estimate_prompt_tokens(few_shot_examples: List[Tuple[str, str]], test_note: str, 
                          question: str, task_description: str, tokenizer) -> int:
    """Estimate total tokens for the complete prompt."""
    total_tokens = 0
    
    # Task description
    total_tokens += estimate_note_tokens(task_description, tokenizer)
    
    # Few-shot examples
    for note, label in few_shot_examples:
        example_text = f"{note}\n\n{question}\nAnswer: {label}"
        total_tokens += estimate_note_tokens(example_text, tokenizer)
    
    # Test query
    test_query = f"{test_note}\n\n{question}\nAnswer:"
    total_tokens += estimate_note_tokens(test_query, tokenizer)
    
    return total_tokens


def truncate_few_shot_examples_for_context(few_shot_examples: List[Tuple[str, str]], 
                                         test_note: str, question: str, task_description: str,
                                         tokenizer, max_context_length: int) -> List[Tuple[str, str]]:
    """
    Truncate few-shot examples to fit within context length limits.
    Uses existing MARVIS utilities for token estimation and context management.
    """
    logger = logging.getLogger(__name__)
    
    if not few_shot_examples:
        return few_shot_examples
    
    # Reserve tokens for test query and task description
    test_query = f"{test_note}\n\n{question}\nAnswer:"
    reserved_tokens = (estimate_note_tokens(test_query, tokenizer) + 
                      estimate_note_tokens(task_description, tokenizer) + 
                      100)  # Safety buffer
    
    available_tokens = max_context_length - reserved_tokens
    
    if available_tokens <= 0:
        logger.warning(f"No tokens available for few-shot examples after reserving {reserved_tokens} tokens")
        return []
    
    # Estimate tokens per example and select as many as fit
    selected_examples = []
    used_tokens = 0
    
    for note, label in few_shot_examples:
        example_text = f"{note}\n\n{question}\nAnswer: {label}"
        example_tokens = estimate_note_tokens(example_text, tokenizer)
        
        if used_tokens + example_tokens <= available_tokens:
            selected_examples.append((note, label))
            used_tokens += example_tokens
        else:
            # Can't fit this example
            break
    
    if len(selected_examples) < len(few_shot_examples):
        logger.info(f"Truncated few-shot examples from {len(few_shot_examples)} to {len(selected_examples)} "
                   f"to fit context limit ({max_context_length} tokens, {used_tokens} used for examples)")
    
    return selected_examples


def save_sample_notes_for_inspection(few_shot_examples: List[Tuple[str, str]], 
                                   dataset: Dict[str, Any], args, semantic_info: Optional[Dict[str, Any]]):
    """
    Save sample generated notes to the default TabLLM output directory for inspection.
    Saves up to 5 notes with metadata for debugging and quality assessment.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Determine output directory (follow existing MARVIS patterns)
        if hasattr(args, 'output_dir') and args.output_dir:
            base_output_dir = args.output_dir
        else:
            # Fallback to current directory
            base_output_dir = "."
        
        # Create TabLLM-specific subdirectory
        tabllm_output_dir = os.path.join(base_output_dir, "tabllm_notes_inspection")
        os.makedirs(tabllm_output_dir, exist_ok=True)
        
        # Create filename with dataset and timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = dataset.get('name', 'unknown_dataset')
        task_id = dataset.get('task_id', dataset.get('id', 'unknown_task'))
        
        notes_filename = f"notes_{dataset_name}_task_{task_id}_{timestamp}.json"
        notes_filepath = os.path.join(tabllm_output_dir, notes_filename)
        
        # Prepare metadata
        num_features = len(dataset.get("attribute_names", []))
        max_context_length = getattr(args, 'max_context_length', None)
        
        inspection_data = {
            "metadata": {
                "dataset_name": dataset_name,
                "task_id": task_id,
                "timestamp": timestamp,
                "num_features_after_reduction": num_features,
                "total_few_shot_examples": len(few_shot_examples),
                "max_context_length": max_context_length,
                "semantic_info_available": semantic_info is not None,
                "generation_method": "online_with_expanded_semantics" if semantic_info else "online_basic_fallback"
            },
            "semantic_expansion_info": {
                "original_semantic_features": None,
                "expanded_to_features": num_features,
                "expansion_applied": False
            },
            "sample_notes": []
        }
        
        # Add semantic expansion details if available
        if semantic_info:
            if 'columns' in semantic_info:
                original_count = len([col for col in semantic_info['columns']])
                inspection_data["semantic_expansion_info"]["original_semantic_features"] = original_count
                inspection_data["semantic_expansion_info"]["expansion_applied"] = original_count < num_features
            elif 'feature_descriptions' in semantic_info:
                original_count = len(semantic_info['feature_descriptions'])
                inspection_data["semantic_expansion_info"]["original_semantic_features"] = original_count
                inspection_data["semantic_expansion_info"]["expansion_applied"] = original_count < num_features
            elif 'feature_description' in semantic_info:
                original_count = len(semantic_info['feature_description'])
                inspection_data["semantic_expansion_info"]["original_semantic_features"] = original_count
                inspection_data["semantic_expansion_info"]["expansion_applied"] = original_count < num_features
        
        # Save up to 5 sample notes
        num_samples = min(5, len(few_shot_examples))
        for i in range(num_samples):
            note, label = few_shot_examples[i]
            
            # Calculate note statistics
            word_count = len(note.split())
            char_count = len(note)
            
            # Count feature mentions in the note
            feature_mentions = 0
            for attr_name in dataset.get("attribute_names", []):
                if attr_name.lower() in note.lower():
                    feature_mentions += 1
            
            sample_data = {
                "index": i,
                "note": note,
                "label": label,
                "statistics": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "feature_mentions_found": feature_mentions,
                    "total_features_available": num_features
                }
            }
            
            inspection_data["sample_notes"].append(sample_data)
        
        # Write to file
        with open(notes_filepath, 'w', encoding='utf-8') as f:
            json.dump(inspection_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {num_samples} sample notes for inspection to: {notes_filepath}")
        
    except Exception as e:
        logger.warning(f"Failed to save sample notes for inspection: {e}")


def evaluate_tabllm(dataset, args):
    """Evaluate TabLLM baseline using proper ICL methodology with log probability computation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating TabLLM on dataset {dataset['name']}")
    logger.info(f"Using num_few_shot_examples={args.num_few_shot_examples}")
    
    # Get original feature count and names before preprocessing
    X, y = dataset["X"], dataset["y"]
    # Feature count NOT including target column
    original_feature_count = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    original_feature_names = dataset.get("attribute_names", [])
    
    # Load TabLLM configuration by OpenML task ID if available
    # Try multiple ways to get task_id (following CLAUDE.md guidelines)
    openml_task_id = None
    if 'task_id' in dataset:
        openml_task_id = dataset['task_id']
    elif hasattr(dataset, 'task_id'):
        openml_task_id = getattr(dataset, 'task_id', None)
    elif 'openml_task_id' in dataset:
        openml_task_id = dataset['openml_task_id']
    elif hasattr(dataset, 'openml_task_id'):
        openml_task_id = getattr(dataset, 'openml_task_id', None)
    elif 'id' in dataset and isinstance(dataset['id'], (int, str)):
        # We have dataset_id, need to resolve to task_id using resource manager
        try:
            dataset_id = int(dataset['id'])
            from marvis.utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            identifiers = rm.resolve_openml_identifiers(dataset_id=dataset_id)
            openml_task_id = identifiers.get('task_id')
            if openml_task_id:
                logger.debug(f"Resolved dataset_id {dataset_id} to task_id {openml_task_id}")
        except (ValueError, TypeError, Exception) as e:
            logger.debug(f"Could not resolve dataset_id to task_id: {e}")
    
    template_data, feature_mapping = None, None
    
    if openml_task_id:
        template_data, feature_mapping = load_tabllm_config_by_openml_id(
            openml_task_id, 
            original_feature_count
        )
        if template_data:
            logger.info(f"Using TabLLM metadata for OpenML task {openml_task_id} (dataset: {dataset['name']})")
        else:
            logger.info(f"No TabLLM metadata found for OpenML task {openml_task_id} (dataset: {dataset['name']}), using default approach")
    else:
        logger.warning(f"No OpenML task ID found for dataset {dataset['name']}, cannot load TabLLM config")
    
    # If no config loaded from task_id, try loading semantic info for online note generation
    semantic_info = None
    if feature_mapping and 'semantic_info' in feature_mapping:
        # Already have semantic info from task_id lookup
        semantic_info = feature_mapping['semantic_info']
        logger.info(f"Using semantic info from task_id {openml_task_id}")
    else:
        # Try to load semantic information for online note generation
        try:
            # Try using the general metadata loader first
            from marvis.utils.metadata_loader import get_metadata_loader
            loader = get_metadata_loader()
            
            # Try with task_id first, then dataset name
            semantic_file = None
            if openml_task_id:
                semantic_file = loader.detect_metadata_file(openml_task_id)
            if semantic_file is None:
                semantic_file = loader.detect_metadata_file(dataset['name'])
            
            if semantic_file and semantic_file.exists():
                with open(semantic_file, 'r') as f:
                    semantic_info = json.load(f)
                logger.info(f"Loaded semantic info from {semantic_file} for online note generation")
            else:
                logger.warning(f"No semantic information found for dataset {dataset['name']} or task {openml_task_id}")
        except Exception as e:
            logger.debug(f"Error loading semantic information: {e}")
            semantic_info = None
    
    # Notes will be generated online during evaluation, no need to load pre-generated ones
    logger.info(f"Will generate TabLLM notes online during evaluation to match current feature set")
    
    # Import required utilities
    from marvis.utils import (
        drop_feature_for_oom,
        is_oom_error,
        apply_feature_reduction,
        unified_llm_predict
    )
    
    # Import regenerate_few_shot_examples from llm_evaluation_utils (not exported in __init__)
    from marvis.utils.llm_evaluation_utils import regenerate_few_shot_examples
    
    # Split data
    X, y = dataset["X"], dataset["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    
    # Limit test samples if specified
    if args.max_test_samples and args.max_test_samples < len(X_test):
        X_test = X_test[:args.max_test_samples]
        y_test = y_test[:args.max_test_samples]
    
    # Apply feature selection using unified approach
    X_train, X_test, dataset, _ = apply_feature_reduction(
        X_train, y_train, X_test, dataset, args, logger
    )
    
    # Update feature mapping after preprocessing
    processed_feature_names = dataset.get("attribute_names", [])
    
    # ONLINE NOTE GENERATION: Expand semantic features to match current feature count
    if semantic_info is not None:
        current_feature_count = len(processed_feature_names)
        logger.info(f"Expanding semantic features for online note generation: {current_feature_count} features after reduction")
        semantic_info = expand_semantic_features(semantic_info, current_feature_count, processed_feature_names)
        
        # Update the feature mapping with expanded semantic info
        if feature_mapping:
            feature_mapping = feature_mapping.copy()
            feature_mapping['semantic_info'] = semantic_info
    
    processed_feature_mapping = create_feature_mapping_after_preprocessing(
        original_feature_names, processed_feature_names, feature_mapping
    )
    
    start_time = time.time()
    
    try:
        # Use modern model instead of T0
        # Determine which model to use based on API arguments
        if hasattr(args, 'openai_model') and args.openai_model:
            model_name = args.openai_model
            backend = "openai"
            logger.info(f"Using OpenAI API model: {model_name}")
        elif hasattr(args, 'gemini_model') and args.gemini_model:
            model_name = args.gemini_model
            backend = "gemini"
            logger.info(f"Using Gemini API model: {model_name}")
        else:
            model_name = args.tabllm_model if hasattr(args, 'tabllm_model') else "Qwen/QwQ-32B-Preview"
            backend = "auto"
            logger.info(f"Using local/HF model: {model_name}")
        
        # Load model using centralized model loader
        try:
            if backend in ["openai", "gemini"]:
                # API models don't need device/dtype configuration
                model_kwargs = {}
            else:
                # Configure model loading parameters for local models
                model_kwargs = {
                    'low_cpu_mem_usage': True,
                    'use_cache': False  # Disable KV cache to save memory
                }
                
                # Configure device and dtype for local models
                if torch.cuda.is_available() and args.device != "cpu":
                    model_kwargs.update({
                        'torch_dtype': torch.float16,
                        'device_map': "auto" if args.gpu_index == 0 else None
                    })
                else:
                    model_kwargs.update({
                        'torch_dtype': torch.float32
                    })
                
                # For VLLM, add tensor parallel size if using multiple GPUs
                if hasattr(args, 'tensor_parallel_size'):
                    model_kwargs['tensor_parallel_size'] = args.tensor_parallel_size
            
            # Load using model loader
            model_wrapper = model_loader.load_llm(
                model_name, 
                backend=backend,
                device=args.device if backend not in ["openai", "gemini"] else None,
                **model_kwargs
            )
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            # Try without hyphen for backward compatibility (only for local models)
            if backend == "auto" and "Qwen-2.5" in model_name:
                model_name = model_name.replace("Qwen-2.5", "Qwen2.5")
                logger.info(f"Retrying with corrected model name: {model_name}")
                
                model_wrapper = model_loader.load_llm(
                    model_name, 
                    backend=backend,
                    device=args.device,
                    **model_kwargs
                )
            else:
                raise
        
        # Load tokenizer for prompt formatting (not needed for API models)
        if backend in ["openai", "gemini"]:
            tokenizer = None  # API models handle tokenization internally
            logger.info("Skipping tokenizer loading for API model")
        else:
            # For local models, load tokenizer separately for prompt formatting
            # (VLLM handles tokenization internally, but we need it for prompt construction)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Set up device with GPU index
        if torch.cuda.is_available() and args.device != "cpu":
            device = torch.device(f"cuda:{args.gpu_index}")
        else:
            device = torch.device("cpu")
        
        # Note: model_wrapper handles device placement internally
        
        # Detect task type (classification vs regression)
        from marvis.utils.task_detection import detect_task_type, get_target_statistics
        task_id = dataset.get('task_id') or dataset.get('id')
        task_type, detection_method = detect_task_type(
            y=y_train,
            task_id=task_id,
            dataset=dataset,
            classification_threshold=getattr(args, 'regression_bins', 10) * 2  # Use 2x bins as threshold
        )
        logger.info(f"Detected task type: {task_type} (method: {detection_method})")
        
        # Handle regression vs classification
        if task_type == 'regression':
            logger.info("Using regression mode with binned classification strategy")
            
            # Get number of bins from args (default: 10)
            n_bins = getattr(args, 'regression_bins', 10)
            
            # Create bins for regression
            bin_edges, bin_labels, min_val, max_val = create_regression_bins(y_train, n_bins)
            logger.info(f"Created {n_bins} bins for regression: range [{min_val:.3g}, {max_val:.3g}]")
            
            # Convert targets to bins
            y_train_binned = convert_targets_to_bins(y_train, bin_edges, bin_labels)
            y_test_binned = convert_targets_to_bins(y_test, bin_edges, bin_labels)
            
            # Store original targets for final evaluation
            y_train_original = y_train.copy()
            y_test_original = y_test.copy()
            
            # Use binned labels as classes
            answer_choices = bin_labels
            unique_classes = np.array(bin_labels)
            
            # Create simple question for regression
            question = f"Which range does the target value fall into: {', '.join(answer_choices)}?"
            
            # Override y_train and y_test with binned versions for ICL
            y_train = np.array([bin_labels.index(label) for label in y_train_binned])
            y_test = np.array([bin_labels.index(label) for label in y_test_binned])
            
        else:
            logger.info("Using classification mode")
            
            # Get unique classes and create answer choices
            unique_classes = np.unique(y_train)
        
        # Create TabLLM-style template (skip for regression since we already set question)
        if task_type == 'classification' and template_data and 'templates' in template_data:
            template_info = next(iter(template_data['templates'].values()))
            answer_choices_str = template_info.get('answer_choices', " ||| ".join([str(cls) for cls in unique_classes]))
            answer_choices = [choice.strip() for choice in answer_choices_str.split('|||')]
            
            logger.info(f"TabLLM template found: answer_choices = {answer_choices}")
            
            # Create mapping from dataset classes to meaningful names
            class_to_name = {}
            if len(answer_choices) == len(unique_classes):
                # Map dataset classes to meaningful names in order
                # Sort unique classes to ensure consistent mapping
                sorted_classes = sorted(unique_classes)
                for i, cls in enumerate(sorted_classes):
                    if i < len(answer_choices):
                        class_to_name[cls] = answer_choices[i]
                        logger.info(f"TabLLM mapping: {cls} -> {answer_choices[i]}")
            
            # If we couldn't create a mapping, fall back to default
            if not class_to_name:
                logger.warning(f"TabLLM mapping failed: {len(answer_choices)} answer choices vs {len(unique_classes)} unique classes")
                class_to_name = {cls: str(cls) for cls in unique_classes}
                answer_choices = [str(cls) for cls in unique_classes]
            
            # Extract question from jinja template
            jinja_template = template_info.get('jinja', '')
            if 'Which of the following classes does this instance belong to' in jinja_template:
                # Extract the question part
                question_start = jinja_template.find('Which of the following')
                question_end = jinja_template.find('?') + 1
                if question_start != -1 and question_end != 0:
                    question = jinja_template[question_start:question_end]
                else:
                    question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
            else:
                question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
        else:
            # Default format for classification or regression already handled above
            if task_type == 'classification':
                answer_choices = [str(cls) for cls in unique_classes]
                class_to_name = {cls: str(cls) for cls in unique_classes}
                question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
            else:
                # Regression: answer_choices, question already set above
                class_to_name = {i: label for i, label in enumerate(answer_choices)}
        
        logger.info(f"Using question: {question}")
        logger.info(f"Answer choices: {answer_choices}")
        
        # Create few-shot examples using TabLLM note format
        max_examples = args.num_few_shot_examples
        n_examples = min(max_examples, len(X_train))
        
        # Use balanced or random selection based on args
        if getattr(args, 'balanced_few_shot', False):
            # Apply balanced few-shot selection
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            
            # Calculate examples per class (as evenly as possible)
            examples_per_class = n_examples // n_classes
            remainder = n_examples % n_classes
            
            example_indices = []
            for i, class_label in enumerate(unique_classes):
                class_mask = y_train == class_label
                class_indices = np.where(class_mask)[0]
                
                # Add one extra example to first 'remainder' classes
                n_select = examples_per_class + (1 if i < remainder else 0)
                n_select = min(n_select, len(class_indices))
                
                if n_select > 0:
                    selected_class_indices = np.random.RandomState(args.seed).choice(
                        class_indices, n_select, replace=False
                    )
                    example_indices.extend(selected_class_indices)
            
            example_indices = np.array(example_indices)
            logger.info(f"Using balanced few-shot selection: {len(example_indices)} examples ({examples_per_class}+ per class)")
        else:
            example_indices = np.random.choice(len(X_train), n_examples, replace=False)
            logger.info(f"Using random few-shot selection: {n_examples} examples")
        
        few_shot_examples = []
        for idx in example_indices:
            x_example = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
            y_example = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
            
            # Generate TabLLM-style note online with expanded semantic information
            semantic_for_note = semantic_info  # Use the expanded semantic info
            if semantic_for_note is not None:
                # Debug: check alignment between row and semantic info
                current_feature_names = dataset["attribute_names"]
                if hasattr(x_example, 'index'):
                    row_feature_names = list(x_example.index)
                else:
                    row_feature_names = current_feature_names
                
                # Generate note with proper feature alignment
                note = generate_note_from_row(x_example, semantic_for_note, current_feature_names, exclude_target=True)
                
                # Debug empty notes
                if not note.strip():
                    logger.warning(f"Empty note generated - debugging:")
                    logger.warning(f"  Row type: {type(x_example)}")
                    logger.warning(f"  Row shape/len: {len(x_example) if hasattr(x_example, '__len__') else 'N/A'}")
                    logger.warning(f"  Current feature names ({len(current_feature_names)}): {current_feature_names[:5]}...")
                    logger.warning(f"  Row feature names ({len(row_feature_names)}): {row_feature_names[:5]}...")
                    if 'columns' in semantic_for_note:
                        semantic_names = [col['name'] for col in semantic_for_note['columns']]
                        logger.warning(f"  Semantic feature names ({len(semantic_names)}): {semantic_names[:5]}...")
                    # Use fallback for debugging
                    note = " ".join([f"Feature {i} is {val}" for i, val in enumerate(x_example)])
            else:
                # Fallback to basic note if no semantic info available
                note = " ".join([f"Feature {i} is {val}" for i, val in enumerate(x_example)])
                logger.warning(f"No semantic info available, using basic note format")
            # Use meaningful class name if available, otherwise use the original label
            class_label = class_to_name.get(y_example, str(y_example))
            few_shot_examples.append((note, class_label))
        
        logger.info(f"Created {len(few_shot_examples)} few-shot examples (requested: {args.num_few_shot_examples})")
        
        # Apply context-aware truncation to few-shot examples
        max_context_length = getattr(args, 'max_context_length', None)
        if max_context_length and tokenizer is not None and len(X_test) > 0:
            logger.info(f"Applying context-aware truncation with max_context_length={max_context_length}")
            
            # Use a sample test note to estimate truncation (use first test sample)
            sample_test = X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0]
            if semantic_info is not None:
                sample_test_note = generate_note_from_row(sample_test, semantic_info, dataset["attribute_names"], exclude_target=True)
            else:
                sample_test_note = " ".join([f"Feature {i} is {val}" for i, val in enumerate(sample_test)])
            
            # Create task description
            task_description = f"Task: Given tabular data examples, classify each instance into one of the following categories: {', '.join(answer_choices)}.\n\nExamples:\n\n"
            
            # Estimate tokens before truncation
            original_tokens = estimate_prompt_tokens(few_shot_examples, sample_test_note, question, task_description, tokenizer)
            logger.info(f"Estimated {original_tokens} tokens for full prompt with {len(few_shot_examples)} examples")
            
            # Apply context-aware truncation using the existing utility functions
            few_shot_examples = truncate_few_shot_examples_for_context(
                few_shot_examples, 
                sample_test_note, 
                question, 
                task_description,
                tokenizer, 
                max_context_length
            )
            
            # Log results
            if len(few_shot_examples) == 0:
                logger.warning("All few-shot examples were truncated due to context length limits. Using zero-shot approach.")
            else:
                final_tokens = estimate_prompt_tokens(few_shot_examples, sample_test_note, question, task_description, tokenizer)
                logger.info(f"After truncation: {len(few_shot_examples)} examples, ~{final_tokens} tokens")
        elif max_context_length is None:
            logger.info("No max_context_length specified, using all few-shot examples")
        elif tokenizer is None:
            logger.info("No tokenizer available (API model), skipping context-aware truncation")
        else:
            logger.info("No test samples available for context estimation")
        
        # Save sample notes for inspection
        save_sample_notes_for_inspection(few_shot_examples, dataset, args, semantic_info)
        
        # Make predictions using unified LLM prediction function with memory optimization
        predictions = []
        all_class_log_probs = []  # Store log probabilities for ROC AUC calculation
        completed_samples = 0
        example_inputs_outputs = []  # Store example inputs and outputs for debugging
        
        # Feature dropping mechanism for OOM handling
        dropped_features = set()  # Track which features have been dropped
        
        # Note: model_wrapper handles eval mode internally
        
        for i in range(len(X_test)):
            # Clear GPU cache periodically to prevent memory buildup, with error handling
            if i % 10 == 0 and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError as cache_error:
                    if "assert" in str(cache_error).lower():
                        logger.warning(f"CUDA cache clear failed due to assertion error: {cache_error}. Continuing without cache clearing.")
                    else:
                        logger.warning(f"CUDA cache clear failed: {cache_error}")
                        # Continue execution anyway
            test_sample = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
            
            # Generate TabLLM-style note for test sample online with expanded semantic information
            semantic_for_note = semantic_info  # Use the expanded semantic info
            if semantic_for_note is not None:
                test_note = generate_note_from_row(test_sample, semantic_for_note, dataset["attribute_names"], exclude_target=True)
            else:
                # Fallback to basic note if no semantic info available
                test_note = " ".join([f"Feature {i} is {val}" for i, val in enumerate(test_sample)])
                logger.warning(f"No semantic info available for test sample, using basic note format")
            
            # Create TabLLM-style prompt following the template format with task description
            # Add task description for better ICL
            task_description = f"Task: Given tabular data examples, classify each instance into one of the following categories: {', '.join(answer_choices)}.\n\nExamples:\n\n"
            
            # Prepare the test instance (must always be included)
            test_prompt = f"{test_note}\n\n{question}\nAnswer:"
            
            # Tokenize test prompt to ensure we have room for it (skip for API models)
            if tokenizer is not None:
                test_tokens = tokenizer.encode(test_prompt)
                answer_tokens_estimate = 10  # Reserve tokens for the answer
                reserved_tokens = len(test_tokens) + answer_tokens_estimate + 50  # Buffer
                
                # Calculate available tokens for few-shot examples including task description
                task_description_tokens = len(tokenizer.encode(task_description))
                available_tokens = args.max_context_length - reserved_tokens - task_description_tokens
            else:
                # For API models, use approximate token estimation (4 chars ≈ 1 token)
                test_tokens_estimate = len(test_prompt) // 4
                answer_tokens_estimate = 10
                reserved_tokens = test_tokens_estimate + answer_tokens_estimate + 50
                
                task_description_tokens_estimate = len(task_description) // 4
                available_tokens = args.max_context_length - reserved_tokens - task_description_tokens_estimate
            
            # Add few-shot examples that fit within the limit
            example_parts = []
            total_example_tokens = 0
            
            num_shots = min(args.num_few_shot_examples, len(few_shot_examples))
            selected_examples = few_shot_examples[:num_shots]
            
            for note, label in selected_examples:
                example_prompt = f"{note}\n\n{question}\nAnswer: {label}"
                if tokenizer is not None:
                    example_tokens = len(tokenizer.encode(example_prompt))
                else:
                    # For API models, use approximate token estimation
                    example_tokens = len(example_prompt) // 4
                
                # Check if adding this example would exceed our limit
                if total_example_tokens + example_tokens <= available_tokens:
                    example_parts.append(example_prompt)
                    total_example_tokens += example_tokens
                else:
                    # Stop adding examples if we're out of space
                    break
            
            # Construct the full prompt with task description and clear sections
            if example_parts:
                examples_section = "\n\n".join(example_parts)
                full_prompt = f"{task_description}{examples_section}\n\nNew Instance:\n{test_prompt}"
            else:
                # Fallback with no examples but still include task description
                full_prompt = f"{task_description}New Instance:\n{test_prompt}"
            
            # Log if we had to reduce examples
            if i == 0 and len(example_parts) < num_shots:
                print(f"Sample {i}: Reduced few-shot examples from {num_shots} to {len(example_parts)} due to context length limit")
            
            # Use unified prediction function with automatic fallback chain
            try:
                # Get few-shot examples in the right format for the unified function
                selected_examples = few_shot_examples[:num_shots]
                
                # Use model wrapper for prediction (supports both local and API models)
                if backend in ["openai", "gemini"]:
                    # For API models, use the wrapper's generate method directly
                    generation_config = GenerationConfig(
                        max_new_tokens=16384,  # Generous limit for thinking and classification
                        temperature=0.0,    # Deterministic for classification
                        top_p=1.0,
                        do_sample=False,
                        enable_thinking=getattr(args, 'enable_thinking', True)
                    )
                    
                    # Generate response using API model
                    generated_text = model_wrapper.generate(
                        full_prompt, 
                        generation_config
                    )
                    
                    # Parse response to get predicted class
                    predicted_class_name = generated_text.strip()
                    
                    # Clean up the response (remove any extra text)
                    for choice in answer_choices:
                        if choice.lower() in predicted_class_name.lower():
                            predicted_class_name = choice
                            break
                    
                    prediction_result = {
                        'predicted_class': predicted_class_name,
                        'method': f'{backend}_api',
                        'class_log_probs': {},  # API models don't provide log probs
                        'generated_text': generated_text
                    }
                else:
                    # For local models, use unified prediction function
                    underlying_model = model_wrapper.get_model()
                    
                    prediction_result = unified_llm_predict(
                        full_prompt=full_prompt.replace(" Answer:", ""),  # Remove " Answer:" as unified function adds it
                        answer_choices=answer_choices,
                        tokenizer=tokenizer,
                        model=underlying_model,  # Pass the underlying model for compatibility
                        args=args,
                        logger=logger,
                        selected_examples=selected_examples,
                        question=question,
                        test_first_sample=(i == 0)  # Only test methods on the first sample
                    )
                
                predicted_class_name = prediction_result['predicted_class']
                prediction_method = prediction_result['method']
                class_log_probs = prediction_result.get('class_log_probs', {})
                generated_text = prediction_result.get('generated_text', None)
                
                # Map back to original class value
                predicted_class = predicted_class_name  # Default fallback
                
                # Find the original class that maps to this class name
                name_to_class = {name: cls for cls, name in class_to_name.items()}
                if predicted_class_name in name_to_class:
                    predicted_class = name_to_class[predicted_class_name]
                else:
                    # Fallback: try direct string match
                    for cls in unique_classes:
                        if str(cls) == predicted_class_name:
                            predicted_class = cls
                            break
                
                predictions.append(predicted_class)
                all_class_log_probs.append(class_log_probs)
                completed_samples = i + 1
                
                # Store example inputs and outputs for first 20 samples  
                if i < 20:
                    true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                    example_inputs_outputs.append({
                        'sample_index': i,
                        'test_note': test_note,
                        'few_shot_examples': [(note, label) for note, label in selected_examples],
                        'num_shots_used': len(selected_examples),
                        'full_prompt': full_prompt if len(full_prompt) < 5000 else full_prompt[:5000] + "... [truncated]",
                        'question': question,
                        'method': prediction_method,
                        'class_log_probs': class_log_probs,
                        'generated_text': generated_text,
                        'predicted_class_name': predicted_class_name,
                        'predicted_class': predicted_class,
                        'true_class': true_label,
                        'true_class_name': class_to_name.get(true_label, str(true_label)),
                        'class_mapping': class_to_name,
                        'correct': predicted_class == true_label
                    })
                    
            except Exception as e:
                # Check for critical errors that should be raised immediately
                error_message = str(e).lower()
                if any(critical_error in error_message for critical_error in [
                    'invalid api key', 'incorrect api key', 'api key', 'authentication', 
                    'unauthorized', '401', '403', 'quota', 'rate limit', 'billing'
                ]):
                    logger.error(f"Critical API error on sample {i}: {e}")
                    logger.error("This error indicates a configuration problem that needs to be fixed")
                    raise e  # Re-raise API configuration errors
                
                # Check if this is an OOM error
                elif is_oom_error(e):
                    logger.warning(f"Unified prediction failed for sample {i} due to OOM: {e}")
                    
                    # Try dropping a feature and retry this sample
                    if drop_feature_for_oom(dropped_features, len(dataset["attribute_names"]), logger):
                        logger.info(f"Retrying sample {i} with {len(dropped_features)} dropped features")
                        # Need to regenerate few-shot examples with dropped features
                        semantic_for_regenerate = processed_feature_mapping.get('semantic_info') if processed_feature_mapping else None
                        if semantic_for_regenerate is None and feature_mapping:
                            semantic_for_regenerate = feature_mapping.get('semantic_info')
                        few_shot_examples = regenerate_few_shot_examples(
                            X_train, y_train, example_indices, 
                            dataset["attribute_names"], dataset['name'],
                            semantic_for_regenerate, dropped_features, class_to_name
                        )
                        # Continue to next iteration which will retry this sample
                        continue
                    else:
                        logger.error("No more features to drop, cannot continue")
                        # Return partial results
                        break
                else:
                    # For other non-critical errors, log and continue with fallback
                    logger.warning(f"Unified prediction failed for sample {i}: {e}")
                    logger.info("Using fallback prediction and continuing evaluation")
                    # Use default prediction and initialize missing variables
                    predicted_class = unique_classes[0]
                    predicted_class_name = str(predicted_class)
                    class_log_probs = {}
                    generated_text = None
                    prediction_method = "default_fallback"
                    predictions.append(predicted_class)
                    all_class_log_probs.append(class_log_probs)
                    completed_samples = i + 1
            
            # Log first few predictions for debugging
            if i < 10:
                if class_log_probs:
                    print(f"Sample {i}: {dict(sorted(class_log_probs.items(), key=lambda x: x[1], reverse=True))}")
                true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                print(f"Sample {i} ({prediction_method}): Predicted: {predicted_class_name} -> {predicted_class}, True: {class_to_name.get(true_label, str(true_label))} -> {true_label}")
            elif i == 10:
                print("Done with few shot logging. Console output will be quiet.")
        
        # Calculate timing - LLMs don't have separate training, so only prediction_time and total_time
        total_time = time.time() - start_time
        prediction_time = total_time  # For LLMs, prediction time includes model loading and inference
        
        # Calculate metrics on completed samples
        if completed_samples > 0:
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            if task_type == 'regression':
                # For regression: convert bin predictions back to continuous values
                logger.info("Computing regression metrics from binned predictions")
                
                # Convert predictions from bin indices to bin labels
                predicted_bin_labels = [answer_choices[pred] for pred in predictions]
                
                # Convert bin labels back to continuous values
                predicted_values = convert_bin_predictions_to_values(predicted_bin_labels, bin_edges, bin_labels)
                
                # Use original continuous targets for evaluation
                y_test_continuous = y_test_original[:completed_samples]
                
                # Calculate regression metrics
                r2 = r2_score(y_test_continuous, predicted_values)
                mae = mean_absolute_error(y_test_continuous, predicted_values)
                mse = mean_squared_error(y_test_continuous, predicted_values)
                rmse = np.sqrt(mse)
                
                logger.info(f"Regression metrics: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
                
                # Set classification metrics to None for regression
                accuracy = None
                balanced_acc = None
                roc_auc = None
                f1_macro = f1_micro = f1_weighted = None
                precision_macro = recall_macro = None
                
                # Add regression-specific results
                regression_results = {
                    'r2_score': float(r2),
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'predicted_values': predicted_values.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'bin_labels': bin_labels,
                    'n_bins': n_bins
                }
            else:
                # For classification: use standard metrics
                # Import shared metric calculation function
                from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
                
                # Resolve task_id using resource manager as per CLAUDE.md guidelines
                task_id = dataset.get('task_id')
                if task_id is None:
                    # Try to resolve task_id from dataset_id using resource manager
                    try:
                        from marvis.utils.resource_manager import get_resource_manager
                        rm = get_resource_manager()
                        dataset_id = dataset.get('id')
                        if dataset_id:
                            identifiers = rm.resolve_openml_identifiers(dataset_id=dataset_id)
                            task_id = identifiers.get('task_id')
                    except Exception as e:
                        logger.warning(f"Could not resolve task_id from dataset_id {dataset.get('id')}: {e}")
                
                # Calculate all metrics using shared function
                calculated_metrics = calculate_llm_metrics(
                    y_test_partial, predictions, unique_classes, 
                    all_class_log_probs, logger, task_id=task_id, dataset=dataset
                )
                
                # Extract individual metrics for backward compatibility
                accuracy = calculated_metrics['accuracy']
                balanced_acc = calculated_metrics['balanced_accuracy']
                roc_auc = calculated_metrics['roc_auc']
                f1_macro = calculated_metrics['f1_macro']
                f1_micro = calculated_metrics['f1_micro']
                f1_weighted = calculated_metrics['f1_weighted']
                precision_macro = calculated_metrics['precision_macro']
                recall_macro = calculated_metrics['recall_macro']
                
                regression_results = {}
        else:
            accuracy = 0.0
            balanced_acc = 0.0
            roc_auc = None
            f1_macro = f1_micro = f1_weighted = None
            precision_macro = recall_macro = None
            regression_results = {}
        
        results = {
            'model_name': 'TabLLM',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id'],  # For consistency with MARVIS extraction logic
            'task_type': task_type,  # Add task type to results
            'detection_method': detection_method,  # Add detection method
            'accuracy': float(accuracy) if accuracy is not None else None,
            'balanced_accuracy': float(balanced_acc) if balanced_acc is not None else None,
            'prediction_time': float(prediction_time),  # Time for inference (includes model loading for LLMs)
            'total_time': float(total_time),  # Same as prediction_time for LLMs (no separate training phase)
            'num_test_samples': len(X_test),
            'num_samples': len(X_train) + len(X_test),  # Total dataset size
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train.shape[1],  # Use X_train to get actual feature count after reduction
            'num_classes': len(unique_classes),
            'predictions': predictions,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics to match evaluate_on_dataset
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'f1_macro': float(f1_macro) if f1_macro is not None else None,
            'f1_micro': float(f1_micro) if f1_micro is not None else None,
            'f1_weighted': float(f1_weighted) if f1_weighted is not None else None,
            'precision_macro': float(precision_macro) if precision_macro is not None else None,
            'recall_macro': float(recall_macro) if recall_macro is not None else None,
            # TabLLM-specific metadata
            'used_template': template_data is not None,
            'used_tabllm_config': feature_mapping is not None,
            'openml_task_id': openml_task_id,
            'model_used': model_name,
            'class_mapping': class_to_name if template_data else None,
            'example_inputs_outputs': example_inputs_outputs,
            'prediction_method': 'unified',
            'feature_mapping_preserved': processed_feature_mapping is not None,
            **regression_results  # Add regression-specific results if any
        }
        
        if task_type == 'regression':
            logger.info(f"TabLLM R² on {dataset['name']}: {regression_results.get('r2_score', 'N/A'):.4f}")
        else:
            logger.info(f"TabLLM accuracy on {dataset['name']}: {accuracy:.4f}")
        
        # Log where inspection files can be found
        if hasattr(args, 'output_dir') and args.output_dir:
            inspection_dir = os.path.join(args.output_dir, "tabllm_notes_inspection")
            logger.info(f"Sample generated notes saved for inspection in: {inspection_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating TabLLM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'TabLLM',
            'dataset_name': dataset['name'],
            'error': str(e)
        }