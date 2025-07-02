#!/usr/bin/env python
"""
Utilities for intelligent feature selection based on token limits.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def estimate_tokens_for_feature(feature_name: str, feature_values: np.ndarray, tokenizer, num_examples: int = 5) -> float:
    """Estimate average tokens needed for a feature across examples."""
    # Sample some values to estimate token count
    sample_size = min(num_examples, len(feature_values))
    if sample_size == 0:
        return 0
    
    sample_indices = np.random.choice(len(feature_values), sample_size, replace=False)
    total_tokens = 0
    
    for idx in sample_indices:
        value = feature_values[idx]
        # Format as it would appear in the prompt
        feature_text = f"{feature_name}: {value}"
        tokens = len(tokenizer.encode(feature_text))
        total_tokens += tokens
    
    return total_tokens / sample_size

def categorize_features(X, feature_names, categorical_indicator=None) -> Dict[str, List[int]]:
    """Categorize features into semantic (categorical) and numeric."""
    semantic_indices = []
    numeric_indices = []
    
    for i, name in enumerate(feature_names):
        if categorical_indicator is not None and i < len(categorical_indicator):
            if categorical_indicator[i]:
                semantic_indices.append(i)
            else:
                numeric_indices.append(i)
        else:
            # Heuristic: if unique values < 20% of total values, likely categorical
            if hasattr(X, 'iloc'):
                unique_ratio = len(X.iloc[:, i].unique()) / len(X)
            else:
                unique_ratio = len(np.unique(X[:, i])) / len(X)
            
            if unique_ratio < 0.2:
                semantic_indices.append(i)
            else:
                numeric_indices.append(i)
    
    return {
        'semantic': semantic_indices,
        'numeric': numeric_indices
    }

def select_features_for_token_limit(
    X, 
    y,
    feature_names,
    tokenizer,
    num_few_shot_examples: int,
    max_tokens: int = 8192,
    categorical_indicator=None,
    prioritize_semantic: bool = True
) -> Tuple[List[int], int]:
    """
    Select features that fit within token limit, prioritizing semantic features.
    
    Returns:
        - List of selected feature indices
        - Estimated total tokens
    """
    logger.info(f"Selecting features for {num_few_shot_examples} few-shot examples with {max_tokens} max tokens")
    
    # Validate input dimensions
    num_features = X.shape[1]
    num_feature_names = len(feature_names)
    
    if num_features != num_feature_names:
        logger.warning(f"Dimension mismatch: X has {num_features} features but {num_feature_names} feature names provided")
        # Use the minimum to avoid index errors
        max_valid_index = min(num_features, num_feature_names) - 1
        feature_names = feature_names[:max_valid_index + 1]
        logger.warning(f"Truncated feature names to {len(feature_names)} to match data dimensions")
    
    # Categorize features
    feature_categories = categorize_features(X, feature_names, categorical_indicator)
    semantic_indices = feature_categories['semantic']
    numeric_indices = feature_categories['numeric']
    
    logger.info(f"Found {len(semantic_indices)} semantic and {len(numeric_indices)} numeric features")
    
    # Estimate tokens per feature
    feature_token_estimates = []
    for i, name in enumerate(feature_names):
        if hasattr(X, 'iloc'):
            feature_values = X.iloc[:, i].values
        else:
            feature_values = X[:, i]
        
        avg_tokens = estimate_tokens_for_feature(name, feature_values, tokenizer)
        feature_token_estimates.append((i, name, avg_tokens))
    
    # Sort by priority: semantic features first, then by token efficiency
    if prioritize_semantic:
        # Separate into two groups and sort each by token count
        semantic_features = [(i, n, t) for i, n, t in feature_token_estimates if i in semantic_indices]
        numeric_features = [(i, n, t) for i, n, t in feature_token_estimates if i in numeric_indices]
        
        semantic_features.sort(key=lambda x: x[2])  # Sort by token count
        numeric_features.sort(key=lambda x: x[2])   # Sort by token count
        
        sorted_features = semantic_features + numeric_features
    else:
        # Just sort by token efficiency
        sorted_features = sorted(feature_token_estimates, key=lambda x: x[2])
    
    # Estimate base prompt overhead
    base_prompt = "Task: Classify based on examples.\n\nExamples:\n"
    base_tokens = len(tokenizer.encode(base_prompt))
    
    # Estimate tokens for labels
    unique_labels = np.unique(y)
    label_text = " ".join([f"Label: {label}" for label in unique_labels[:num_few_shot_examples]])
    label_tokens = len(tokenizer.encode(label_text))
    
    # Estimate tokens per example (excluding features)
    example_overhead = "Sample: . Target: \n"
    overhead_per_example = len(tokenizer.encode(example_overhead))
    
    # Calculate available tokens for features
    total_overhead = base_tokens + (overhead_per_example + label_tokens // num_few_shot_examples) * num_few_shot_examples
    available_for_features = max_tokens - total_overhead - 100  # Leave some buffer
    
    logger.info(f"Token budget: {max_tokens} total, {total_overhead} overhead, {available_for_features} for features")
    
    # Select features that fit
    selected_indices = []
    total_feature_tokens = 0
    
    for idx, name, avg_tokens in sorted_features:
        # Estimate total tokens if we add this feature
        estimated_tokens = total_feature_tokens + (avg_tokens * num_few_shot_examples)
        
        if estimated_tokens <= available_for_features:
            selected_indices.append(idx)
            total_feature_tokens = estimated_tokens
            logger.debug(f"Selected feature {name} (idx={idx}, ~{avg_tokens:.1f} tokens/example)")
        else:
            logger.debug(f"Skipping feature {name} (would exceed limit: {estimated_tokens} > {available_for_features})")
    
    estimated_total = total_overhead + total_feature_tokens
    logger.info(f"Selected {len(selected_indices)} features, estimated {estimated_total} total tokens")
    
    return selected_indices, estimated_total

def create_reduced_dataset(X, feature_indices: List[int], feature_names: List[str]) -> Tuple[Any, List[str]]:
    """Create a reduced dataset with only selected features."""
    # Validate indices are within bounds
    max_data_index = X.shape[1] - 1
    max_name_index = len(feature_names) - 1
    
    valid_indices = [i for i in feature_indices if 0 <= i <= max_data_index and i <= max_name_index]
    
    if len(valid_indices) != len(feature_indices):
        invalid_indices = [i for i in feature_indices if i < 0 or i > max_data_index or i > max_name_index]
        logger.warning(f"Skipping invalid feature indices: {invalid_indices}")
        logger.warning(f"Valid range: 0-{max_data_index} (data), 0-{max_name_index} (names)")
    
    if hasattr(X, 'iloc'):
        # DataFrame
        X_reduced = X.iloc[:, valid_indices]
    else:
        # NumPy array
        X_reduced = X[:, valid_indices]
    
    reduced_feature_names = [feature_names[i] for i in valid_indices]
    
    return X_reduced, reduced_feature_names

def test_feature_selection(X, y, feature_names, tokenizer, num_few_shot_examples, categorical_indicator=None):
    """Test different token limits and report results."""
    token_limits = [2048, 4096, 8192]
    
    results = []
    for limit in token_limits:
        selected_indices, estimated_tokens = select_features_for_token_limit(
            X, y, feature_names, tokenizer, 
            num_few_shot_examples=num_few_shot_examples,
            max_tokens=limit,
            categorical_indicator=categorical_indicator
        )
        
        results.append({
            'token_limit': limit,
            'num_features_selected': len(selected_indices),
            'estimated_tokens': estimated_tokens,
            'selected_features': [feature_names[i] for i in selected_indices[:5]],  # First 5 for display
            'utilization': estimated_tokens / limit
        })
    
    return results