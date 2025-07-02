#!/usr/bin/env python
"""
JOLT baseline evaluation module.
Contains functions for evaluating the JOLT baseline on tabular datasets.
"""

import os
import sys
import numpy as np
import torch
import json
import time
import logging
import importlib.util
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List, Tuple
from marvis.utils.model_loader import model_loader, GenerationConfig

def load_jolt_config_by_openml_id(openml_task_id, original_feature_count=None):
    """Load JOLT configuration by OpenML task ID with feature count validation.
    
    Args:
        openml_task_id: OpenML task ID to look up
        original_feature_count: Number of features in dataset before preprocessing
        
    Returns:
        Tuple of (config_data, feature_mapping) or (None, None)
    """
    logger = logging.getLogger(__name__)
    
    # Use resource manager for path resolution
    try:
        from marvis.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        jolt_dir = resource_manager.path_resolver.get_config_path('jolt', '')
        
        if jolt_dir and jolt_dir.exists():
            # Try direct task ID lookup first (new approach)
            jolt_config_path = jolt_dir / f"jolt_config_task_{openml_task_id}.json"
        else:
            jolt_config_path = None
            
    except Exception as e:
        logger.warning(f"Could not use resource manager for JOLT config lookup: {e}")
        # Fallback to relative path from current script location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jolt_dir = os.path.join(current_dir, "jolt")
        jolt_config_path = os.path.join(jolt_dir, f"jolt_config_task_{openml_task_id}.json")
    
    try:
        if jolt_config_path and jolt_config_path.exists():
            with open(jolt_config_path, 'r') as f:
                config_data = json.load(f)
            
            dataset_name = config_data.get('dataset_name', f'task_{openml_task_id}')
            logger.info(f"Found JOLT config for OpenML task {openml_task_id} (dataset: {dataset_name})")
            
            # Validate feature count if provided
            if original_feature_count is not None:
                # JOLT config stores feature count WITHOUT target, compare directly
                config_feature_count = config_data.get('num_features')
                if config_feature_count is not None:
                    # Allow higher actual feature count (due to text preprocessing/encoding)
                    # but warn if actual is significantly lower (potential data loss)
                    if original_feature_count < config_feature_count:
                        error_msg = (
                            f"Feature count mismatch for OpenML task {openml_task_id}: "
                            f"dataset has {original_feature_count} features but JOLT config expects {config_feature_count} features. "
                            f"This indicates potential data loss or version mismatch."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    elif original_feature_count > config_feature_count:
                        logger.info(
                            f"Feature count expanded: dataset has {original_feature_count} features "
                            f"vs {config_feature_count} in config (likely due to text preprocessing/encoding)"
                        )
                    else:
                        logger.info(f"Feature count validation passed: {original_feature_count} features")
                else:
                    logger.info(f"No feature count in config to validate against")
            
            # Create feature mapping to preserve semantic descriptions
            feature_mapping = None
            if 'column_descriptions' in config_data:
                feature_mapping = {
                    'descriptions': config_data['column_descriptions'],
                    'task_id': openml_task_id,
                    'dataset_name': dataset_name
                }
            
            return config_data, feature_mapping
        
        # Fallback: try old dataset name-based approach for backward compatibility
        if isinstance(jolt_dir, Path):
            jolt_dir_str = str(jolt_dir)
        else:
            jolt_dir_str = jolt_dir
        return _load_jolt_config_by_dataset_name_fallback(openml_task_id, jolt_dir_str, original_feature_count)
        
    except ValueError:
        # Re-raise validation errors (like feature count mismatch)
        raise
    except Exception as e:
        logger.error(f"Error loading JOLT config for OpenML task {openml_task_id}: {e}")
        return None, None


def _load_jolt_config_by_dataset_name_fallback(openml_task_id, jolt_dir, original_feature_count=None):
    """Fallback method using the old dataset name mapping approach."""
    logger = logging.getLogger(__name__)
    
    # Load OpenML task mapping
    mapping_path = os.path.join(jolt_dir, "openml_task_mapping.json")
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
        logger.debug(f"No JOLT config found for OpenML task ID {openml_task_id}")
        return None, None
    
    # Load the corresponding config file
    jolt_config_path = os.path.join(jolt_dir, f"jolt_config_{dataset_name}.json")
    
    try:
        if os.path.exists(jolt_config_path):
            with open(jolt_config_path, 'r') as f:
                config_data = json.load(f)
            
            logger.info(f"Found JOLT config for OpenML task {openml_task_id} (dataset: {dataset_name}) via fallback")
            
            # Feature validation and mapping creation (same as main function)
            if original_feature_count is not None:
                config_feature_count = config_data.get('num_features')
                if config_feature_count is not None:
                    # Allow higher actual feature count (due to text preprocessing/encoding)
                    # but warn if actual is significantly lower (potential data loss)
                    if original_feature_count < config_feature_count:
                        error_msg = (
                            f"Feature count mismatch for OpenML task {openml_task_id}: "
                            f"dataset has {original_feature_count} features but JOLT config expects {config_feature_count} features. "
                            f"This indicates potential data loss or version mismatch."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    elif original_feature_count > config_feature_count:
                        logger.info(
                            f"Feature count expanded: dataset has {original_feature_count} features "
                            f"vs {config_feature_count} in config (likely due to text preprocessing/encoding)"
                        )
                    else:
                        logger.info(f"Feature count validation passed: {original_feature_count} features")
            
            feature_mapping = None
            if 'column_descriptions' in config_data:
                feature_mapping = {
                    'descriptions': config_data['column_descriptions'],
                    'task_id': openml_task_id,
                    'dataset_name': dataset_name
                }
            
            return config_data, feature_mapping
        
        logger.debug(f"JOLT config file not found: {jolt_config_path}")
        return None, None
        
    except ValueError:
        # Re-raise validation errors (like feature count mismatch)
        raise
    except Exception as e:
        logger.debug(f"Error loading JOLT config via fallback for OpenML task {openml_task_id}: {e}")
        return None, None

def create_feature_mapping_after_preprocessing(original_feature_names, processed_feature_names, feature_mapping):
    """Create mapping from processed feature names to original semantic descriptions.
    
    Args:
        original_feature_names: List of original feature names from dataset
        processed_feature_names: List of feature names after preprocessing
        feature_mapping: Original feature mapping from JOLT config
        
    Returns:
        Dictionary mapping processed feature names to descriptions
    """
    if feature_mapping is None or 'descriptions' not in feature_mapping:
        return {}
    
    original_descriptions = feature_mapping['descriptions']
    processed_mapping = {}
    
    # Create mapping from original to processed names
    # This is a simple approach - could be enhanced for more complex preprocessing
    for i, processed_name in enumerate(processed_feature_names):
        if i < len(original_feature_names):
            original_name = original_feature_names[i]
            if original_name in original_descriptions:
                processed_mapping[processed_name] = original_descriptions[original_name]
            # Also try to match by name similarity for robustness
            else:
                # Look for partial matches in original descriptions
                for orig_key, description in original_descriptions.items():
                    if orig_key.lower() in processed_name.lower() or processed_name.lower() in orig_key.lower():
                        processed_mapping[processed_name] = description
                        break
    
    return processed_mapping

def get_model_max_length(tokenizer, model_name):
    """Get the actual maximum sequence length for a model."""
    # Try to get from tokenizer config first
    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1e6:
        return tokenizer.model_max_length
    
    # Fallback to common model lengths
    model_max_lengths = {
        'microsoft/DialoGPT-medium': 1024,
        'google/gemma-2-27b-it': 8192,
        'mlfoundations/tabula-8b': 8192,
    }
    
    # Try to match model name
    for model_key, max_len in model_max_lengths.items():
        if model_key in model_name:
            return max_len
    
    # Conservative fallback
    return 512

def evaluate_jolt_legacy(dataset, args):
    """Evaluate JOLT baseline on a dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating JOLT on dataset {dataset['name']}")
    
    # Get original feature count and names before preprocessing
    X, y = dataset["X"], dataset["y"]
    # Feature count NOT including target column
    original_feature_count = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    original_feature_names = dataset.get("attribute_names", [])
    
    # Load JOLT configuration by OpenML ID if available
    openml_task_id = dataset.get('id')
    jolt_config, feature_mapping = None, None
    
    if openml_task_id:
        jolt_config, feature_mapping = load_jolt_config_by_openml_id(
            openml_task_id, 
            original_feature_count
        )
        if jolt_config:
            logger.info(f"Using JOLT metadata for OpenML task {openml_task_id} (dataset: {dataset['name']})")
        else:
            logger.info(f"No JOLT metadata found for OpenML task {openml_task_id} (dataset: {dataset['name']}), using default approach")
    else:
        logger.warning(f"No OpenML task ID found for dataset {dataset['name']}, cannot load JOLT config")
    
    # Import feature selection utilities
    from marvis.utils import apply_feature_reduction, unified_llm_predict
    
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
    processed_feature_mapping = create_feature_mapping_after_preprocessing(
        original_feature_names, processed_feature_names, feature_mapping
    )
    
    start_time = time.time()
    
    try:
        # Load model and tokenizer for JOLT approach
        tokenizer = AutoTokenizer.from_pretrained(args.jolt_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set up device with GPU index
        if torch.cuda.is_available() and args.device != "cpu":
            device = torch.device(f"cuda:{args.gpu_index}")
        else:
            device = torch.device("cpu")
        
        # Load model using centralized model loader with VLLM support
        model_kwargs = {
            'low_cpu_mem_usage': True,
            'use_cache': False  # Disable KV cache to save memory
        }
        
        if torch.cuda.is_available() and args.device != "cpu":
            model_kwargs.update({
                'torch_dtype': torch.float16,
                'device_map': "auto" if args.gpu_index == 0 else None
            })
        
        try:
            # Load using model loader (prefers VLLM for speed)
            model_wrapper = model_loader.load_llm(
                args.jolt_model,
                backend="auto",
                device=args.device,
                **model_kwargs
            )
            
            # For compatibility, get the underlying model
            model = model_wrapper.get_model()
            
        except Exception as e:
            logger.warning(f"Error loading model with model loader: {e}. Falling back to direct loading...")
            # Fallback to direct loading
            model = AutoModelForCausalLM.from_pretrained(
                args.jolt_model,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            model = model.to(device)
        
        # Create few-shot examples with joint probabilistic framing
        n_examples = min(args.num_few_shot_examples, len(X_train))
        unique_classes = np.unique(y_train)
        
        # Use balanced or random selection based on args
        if getattr(args, 'balanced_few_shot', False):
            # Apply balanced few-shot selection
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
        
        # Make predictions using joint probabilistic approach with memory optimization
        predictions = []
        all_class_log_probs = []  # Store log probabilities for ROC AUC calculation
        completed_samples = 0
        
        # Set model to eval mode for inference
        if hasattr(model, 'eval'):
            model.eval()
        
        # Create few-shot examples string (compact version)
        few_shot_examples = []
        for idx in example_indices:
            x_example = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
            y_example = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
            
            example_features = []
            # Use JOLT config column descriptions if available (now using processed mapping)
            if processed_feature_mapping:
                for attr_name, value in zip(dataset["attribute_names"], x_example):
                    # Use descriptive name from processed JOLT config mapping if available
                    display_name = processed_feature_mapping.get(attr_name, attr_name)
                    attr_short = str(display_name)[:15] if len(str(display_name)) > 15 else str(display_name)
                    val_short = str(value)[:25] if len(str(value)) > 25 else str(value)
                    example_features.append(f"{attr_short}={val_short}")
            else:
                for attr_name, value in zip(dataset["attribute_names"], x_example):
                    # Make examples more compact
                    attr_short = str(attr_name)[:15] if len(str(attr_name)) > 15 else str(attr_name)
                    val_short = str(value)[:25] if len(str(value)) > 25 else str(value)
                    example_features.append(f"{attr_short}={val_short}")
            
            example_text = f"[{','.join(example_features)}] → {y_example}"
            few_shot_examples.append(example_text)
        
        for i in range(len(X_test)):
            # Clear GPU cache periodically to prevent memory buildup
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            test_sample = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
            
            # Get actual model max length for JOLT
            model_max_length = get_model_max_length(tokenizer, args.jolt_model)
            effective_max_length = min(args.max_context_length, model_max_length)
            
            # Create compact prompt for JOLT with task context if available
            if jolt_config and 'task_prefix' in jolt_config:
                base_prompt = f"{jolt_config['task_prefix']}\\n\\nExamples:\\n\\n"
            else:
                base_prompt = "Given examples, predict the class:\\n\\n"
            
            test_features = []
            # Use JOLT config column descriptions if available (now using processed mapping)
            if processed_feature_mapping:
                for attr_name, value in zip(dataset["attribute_names"], test_sample):
                    # Use descriptive name from processed JOLT config mapping if available
                    display_name = processed_feature_mapping.get(attr_name, attr_name)
                    attr_short = str(display_name)[:20] if len(str(display_name)) > 20 else str(display_name)
                    val_short = str(value)[:30] if len(str(value)) > 30 else str(value)
                    test_features.append(f"{attr_short}={val_short}")
            else:
                for attr_name, value in zip(dataset["attribute_names"], test_sample):
                    # Truncate long attribute names/values for JOLT too
                    attr_short = str(attr_name)[:20] if len(str(attr_name)) > 20 else str(attr_name)
                    val_short = str(value)[:30] if len(str(value)) > 30 else str(value)
                    test_features.append(f"{attr_short}={val_short}")
            test_text = f"Predict: [{','.join(test_features)}] → Class:"
            
            # Calculate how many examples we can fit
            base_tokens = len(tokenizer.encode(base_prompt + test_text))
            remaining_tokens = effective_max_length - base_tokens - 50
            
            # Add examples until we run out of space
            selected_examples = []
            current_tokens = 0
            for example in few_shot_examples:
                # Further truncate examples if needed
                example_compact = example[:150] if len(example) > 150 else example
                example_tokens = len(tokenizer.encode(example_compact + "\\n"))
                if current_tokens + example_tokens <= remaining_tokens:
                    selected_examples.append(example_compact)
                    current_tokens += example_tokens
                else:
                    break
            
            # Create final prompt
            if selected_examples:
                full_prompt = base_prompt + "\\n".join(selected_examples) + "\\n\\n" + test_text
            else:
                # Fallback with no examples
                full_prompt = test_text
            
            # Use unified prediction function with automatic fallback chain
            try:
                # Convert JOLT answer choices to string format
                answer_choices = [str(cls) for cls in unique_classes]
                
                # Use unified prediction function
                prediction_result = unified_llm_predict(
                    full_prompt=full_prompt.replace(" Class:", ""),  # Remove " Class:" as unified function handles this
                    answer_choices=answer_choices,
                    tokenizer=tokenizer,
                    model=model,
                    args=args,
                    logger=logger,
                    selected_examples=None,  # JOLT doesn't use the same few-shot format
                    question=None,  # JOLT constructs prompts differently
                    test_first_sample=(i == 0)  # Only test methods on the first sample
                )
                
                predicted_class_str = prediction_result['predicted_class']
                prediction_method = prediction_result['method']
                
                # Convert prediction back to original type
                predicted_class = predicted_class_str
                for cls in unique_classes:
                    if str(cls) == predicted_class_str:
                        predicted_class = cls
                        break
                
                predictions.append(predicted_class)
                completed_samples = i + 1
                
            except Exception as e:
                logger.warning(f"Unified prediction failed for sample {i}: {e}")
                # Fallback: use most common class
                predicted_class = unique_classes[0]
                predictions.append(predicted_class)
                completed_samples = i + 1
        
        # Calculate metrics on completed samples
        # Convert predictions to same type as ground truth
        predictions_converted = []
        if completed_samples > 0:
            target_type = type(y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0])
            
            for pred in predictions:
                try:
                    # Try to convert prediction to target type
                    if target_type == int:
                        # Handle common string representations
                        if isinstance(pred, str):
                            # Remove any extra whitespace
                            pred = pred.strip()
                            # Try to convert to int
                            converted_pred = int(float(pred))  # Use float first to handle "1.0" -> 1
                        else:
                            converted_pred = int(pred)
                    elif target_type == float:
                        converted_pred = float(pred)
                    elif target_type == str:
                        converted_pred = str(pred)
                    else:
                        # For other types, try direct conversion
                        converted_pred = target_type(pred)
                    
                    predictions_converted.append(converted_pred)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert prediction '{pred}' to type {target_type}: {e}")
                    # Keep original prediction as fallback
                    predictions_converted.append(pred)
            
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            # Import shared metric calculation function
            from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
            
            # Calculate all metrics using shared function (no log probabilities available for JOLT)
            calculated_metrics = calculate_llm_metrics(
                y_test_partial, predictions_converted, unique_classes, 
                all_class_log_probs=None, logger=logger
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
        else:
            predictions_converted = predictions
            accuracy = 0.0
            balanced_acc = 0.0
            roc_auc = None
            f1_macro = f1_micro = f1_weighted = None
            precision_macro = recall_macro = None
        
        # Calculate timing - LLMs don't have separate training, so only prediction_time and total_time
        total_time = time.time() - start_time
        prediction_time = total_time  # For LLMs, prediction time includes model loading and inference
        
        results = {
            'model_name': 'JOLT',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id'],  # For consistency with MARVIS extraction logic
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'prediction_time': float(prediction_time),  # Time for inference (includes model loading for LLMs)
            'total_time': float(total_time),  # Same as prediction_time for LLMs (no separate training phase)
            'num_test_samples': len(X_test),
            'num_samples': len(X_train) + len(X_test),  # Total dataset size
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train.shape[1],  # Use X_train to get actual feature count after reduction
            'num_classes': len(unique_classes),
            'predictions': predictions_converted,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics to match evaluate_on_dataset
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'f1_macro': float(f1_macro) if f1_macro is not None else None,
            'f1_micro': float(f1_micro) if f1_micro is not None else None,
            'f1_weighted': float(f1_weighted) if f1_weighted is not None else None,
            'precision_macro': float(precision_macro) if precision_macro is not None else None,
            'recall_macro': float(recall_macro) if recall_macro is not None else None,
            # JOLT-specific metadata
            'used_jolt_config': jolt_config is not None,
            'openml_task_id': openml_task_id
        }
        
        logger.info(f"JOLT accuracy on {dataset['name']}: {accuracy:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating JOLT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'JOLT',
            'dataset_name': dataset['name'],
            'error': str(e)
        }

def evaluate_jolt(dataset, args):
    """Evaluate JOLT baseline using the official implementation."""
    logger = logging.getLogger(__name__)
    try:
        # Import the official JOLT wrapper using relative path
        import sys
        import os
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jolt_wrapper_path = os.path.join(current_dir, "jolt")
        
        if jolt_wrapper_path not in sys.path:
            sys.path.insert(0, jolt_wrapper_path)
        
        # Check if the wrapper file exists
        wrapper_file = os.path.join(jolt_wrapper_path, "official_jolt_wrapper.py")
        if not os.path.exists(wrapper_file):
            logger.error(f"Official JOLT wrapper not found at {wrapper_file}")
            raise FileNotFoundError(f"Official JOLT wrapper not found at {wrapper_file}")
        
        # Import using the full module path
        import importlib.util
        spec = importlib.util.spec_from_file_location("official_jolt_wrapper", wrapper_file)
        official_jolt_wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official_jolt_wrapper)
        
        return official_jolt_wrapper.evaluate_jolt_official(dataset, args)
        
    except Exception as e:
        logger.error(f"Error importing or running official JOLT: {e}")
        logger.info("Falling back to legacy JOLT implementation")
        return evaluate_jolt_legacy(dataset, args)