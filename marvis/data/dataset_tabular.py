"""
Tabular dataset processing and preprocessing functions.

This module provides reusable functions for processing and preprocessing
tabular datasets, including feature preprocessing, label mapping, and
frequency analysis.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_features(X: np.ndarray, categorical_indicator: List[bool], preserve_categorical: bool = False) -> np.ndarray:
    """
    Preprocess features, converting string features to numeric values 
    and handling missing values.
    
    Args:
        X: Feature matrix
        categorical_indicator: Boolean list indicating categorical features
        preserve_categorical: If True, keep categorical features as strings for CatBoost
        
    Returns:
        Processed feature matrix
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X)
    
    # Process each column
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx]
        is_categorical = categorical_indicator[col_idx] if col_idx < len(categorical_indicator) else False
        
        # Check if column has object/string data
        if col.dtype == 'object' or is_categorical:
            if preserve_categorical:
                # For CatBoost, keep categorical features as strings
                logger.info(f"Preserving categorical feature at column {col_idx}")
                # Just handle missing values
                col_filled = col.fillna('missing')
                # Ensure we can assign string values by converting the column dtype first
                df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(object)
                df.iloc[:, col_idx] = col_filled.astype(str)
                logger.info(f"  Preserved categorical feature with {col_filled.nunique()} unique values")
            else:
                logger.info(f"Converting feature at column {col_idx} to numeric")
                
                # For categorical features, use label encoding
                try:
                    from sklearn.preprocessing import LabelEncoder
                    # Handle missing values first
                    col_filled = col.fillna('missing')
                    
                    # Use label encoder
                    encoder = LabelEncoder()
                    encoded_values = encoder.fit_transform(col_filled)
                    
                    # Ensure column can accept the encoded values
                    # Convert to appropriate dtype that can hold the encoded values
                    if encoded_values.max() <= 127 and encoded_values.min() >= -128:
                        target_dtype = 'int8'
                    elif encoded_values.max() <= 32767 and encoded_values.min() >= -32768:
                        target_dtype = 'int16'
                    else:
                        target_dtype = 'int32'
                    
                    # Assign the encoded values directly with proper dtype
                    df.iloc[:, col_idx] = pd.Series(encoded_values, index=df.index, dtype=target_dtype)
                    logger.info(f"  Encoded {len(encoder.classes_)} unique categories for column {col_idx}")
                except Exception as e:
                    logger.warning(f"  Error encoding column {col_idx}: {e}")
                    # If encoding fails, replace with zeros
                    df.iloc[:, col_idx] = pd.Series(np.zeros(len(df), dtype='int32'), index=df.index)
        else:
            # For numeric features
            if preserve_categorical:
                # For CatBoost, we can keep NaN values as it handles them natively
                # Just ensure the column is numeric type
                if col.dtype == 'object':
                    # Try to convert to numeric
                    df.iloc[:, col_idx] = pd.to_numeric(col, errors='coerce')
                    logger.info(f"  Converted object column {col_idx} to numeric for CatBoost")
                # CatBoost handles NaN values, so we don't fill them
            else:
                # For other models, fill NaN values
                if col.isna().any():
                    # If more than 75% of the values are NaN, fill with zeros
                    if col.isna().mean() > 0.75:
                        fill_value = 0
                    # Otherwise, use the median
                    else:
                        fill_value = col.median() if not np.isnan(col.median()) else 0
                    
                    # Use Series constructor to ensure type compatibility
                    filled_col = col.fillna(fill_value)
                    # Ensure compatible dtype before assignment
                    if df.iloc[:, col_idx].dtype != filled_col.dtype:
                        df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(filled_col.dtype)
                    df.iloc[:, col_idx] = pd.Series(filled_col, index=df.index)
                    logger.info(f"  Filled {col.isna().sum()} missing values in column {col_idx}")
    
    # Convert back to numpy array
    X_processed = df.values
    
    return X_processed


def compute_frequency_distribution(labels: np.ndarray, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute frequency distribution for labels.
    
    Args:
        labels: Label array
        label_names: Optional list of label names for display
        
    Returns:
        Dictionary containing frequency information
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # Create distribution dict with numeric keys
    distribution = {}
    for label, count in zip(unique_labels, counts):
        distribution[int(label)] = {
            'count': int(count),
            'percentage': float(count / total_samples * 100),
            'name': label_names[int(label)] if label_names and int(label) < len(label_names) else f"Class_{int(label)}"
        }
    
    return {
        'total_samples': total_samples,
        'num_classes': len(unique_labels),
        'distribution': distribution,
        'is_balanced': np.std(counts) / np.mean(counts) < 0.5  # Simple balance check
    }


def compute_label_frequency_mapping(true_labels: np.ndarray, predicted_labels: np.ndarray, threshold: float = 0.05) -> Optional[Dict[int, int]]:
    """
    Compute a frequency-based mapping between predicted and true labels.
    
    This function finds an optimal assignment of predicted labels to true labels
    based on frequency matching when predictions don't directly match ground truth labels.
    
    Args:
        true_labels: Array of ground truth labels
        predicted_labels: Array of predicted labels  
        threshold: Minimum frequency threshold for creating mappings
        
    Returns:
        Dictionary mapping predicted labels to true labels, or None if no good mapping found
    """
    from scipy.optimize import linear_sum_assignment
    
    # Get frequency distributions
    true_freq = compute_frequency_distribution(true_labels)
    pred_freq = compute_frequency_distribution(predicted_labels)
    
    true_counts = {k: v['count'] for k, v in true_freq['distribution'].items()}
    pred_counts = {k: v['count'] for k, v in pred_freq['distribution'].items()}
    
    # Create cost matrix for assignment
    true_classes = sorted(true_counts.keys())
    pred_classes = sorted(pred_counts.keys())
    
    if len(true_classes) != len(pred_classes):
        logger.warning(f"Different number of classes: true={len(true_classes)}, pred={len(pred_classes)}")
        return None
    
    # Build cost matrix based on frequency differences
    cost_matrix = np.zeros((len(pred_classes), len(true_classes)))
    
    for i, pred_class in enumerate(pred_classes):
        for j, true_class in enumerate(true_classes):
            pred_count = pred_counts[pred_class]
            true_count = true_counts[true_class]
            
            # Cost is the absolute difference in frequencies
            cost_matrix[i, j] = abs(pred_count - true_count)
    
    # Solve assignment problem
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)
    
    # Create mapping
    label_mapping = {}
    total_cost = 0
    
    for pred_idx, true_idx in zip(pred_indices, true_indices):
        pred_class = pred_classes[pred_idx]
        true_class = true_classes[true_idx]
        label_mapping[pred_class] = true_class
        total_cost += cost_matrix[pred_idx, true_idx]
    
    # Check if mapping is reasonable (low total cost relative to dataset size)
    avg_cost_per_class = total_cost / len(pred_classes)
    total_samples = len(true_labels)
    
    if avg_cost_per_class / total_samples > threshold:
        logger.warning(f"Label mapping has high cost ({avg_cost_per_class:.2f} avg per class), may not be reliable")
        return None
    
    logger.info(f"Created label mapping with average cost {avg_cost_per_class:.2f} per class")
    return label_mapping


def apply_label_mapping(predictions: np.ndarray, label_mapping: Dict[int, int]) -> np.ndarray:
    """
    Apply label mapping to predictions.
    
    Args:
        predictions: Array of predicted labels
        label_mapping: Dictionary mapping original labels to new labels
        
    Returns:
        Array of remapped predictions
    """
    mapped_predictions = predictions.copy()
    
    for original_label, new_label in label_mapping.items():
        mask = predictions == original_label
        mapped_predictions[mask] = new_label
        logger.debug(f"Mapped {np.sum(mask)} predictions from {original_label} to {new_label}")
    
    return mapped_predictions


def compute_baseline_probabilities(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    strategy: str = "frequency"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute baseline probability predictions based on training label distribution.
    
    Args:
        train_labels: Training set labels for computing class frequencies
        test_labels: Test set labels (for shape information)
        strategy: Baseline strategy ("frequency", "uniform", or "majority")
        
    Returns:
        Tuple of (baseline_predictions, baseline_info)
    """
    unique_classes = np.unique(train_labels)
    num_classes = len(unique_classes)
    num_test_samples = len(test_labels)
    
    if strategy == "frequency":
        # Predict based on class frequencies in training set
        class_counts = np.bincount(train_labels)
        class_probs = class_counts / len(train_labels)
        
        # Generate predictions based on these probabilities
        baseline_predictions = np.random.choice(
            unique_classes, 
            size=num_test_samples, 
            p=class_probs[unique_classes]
        )
        
    elif strategy == "uniform":
        # Uniform random predictions
        baseline_predictions = np.random.choice(
            unique_classes, 
            size=num_test_samples
        )
        
    elif strategy == "majority":
        # Always predict the majority class
        majority_class = np.bincount(train_labels).argmax()
        baseline_predictions = np.full(num_test_samples, majority_class)
        
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
    
    # Compute baseline info
    baseline_info = {
        'strategy': strategy,
        'num_classes': num_classes,
        'unique_classes': unique_classes.tolist(),
        'class_distribution': compute_frequency_distribution(train_labels)
    }
    
    return baseline_predictions, baseline_info


def process_tabular_dataset_for_training(
    dataset: Dict[str, Any], 
    embedding_size: int = 1000,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    sampling_strategy: str = "balanced",
    test_size: float = 0.2,
    val_size: float = 0.5,
    preserve_regression: bool = False,
    task_type: Optional[str] = None,
    seed: int = 42,
    force_recompute_embeddings: bool = False,
    embedding_cache_dir: Optional[str] = None,
    compute_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Process a tabular dataset for training, including preprocessing, splitting, and embedding computation.
    
    Args:
        dataset: Dictionary with dataset information including 'X', 'y', 'name', 'id'
        embedding_size: Size for TabPFN embeddings
        max_train_samples: Maximum number of training samples to use
        max_test_samples: Maximum number of test samples to use
        sampling_strategy: Strategy for limiting training samples ("balanced" or "random")
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation (after test split)
        preserve_regression: Whether to preserve regression tasks
        task_type: Manual task type override ("classification" or "regression")
        seed: Random seed for reproducible splits
        force_recompute_embeddings: Whether to force recomputation of embeddings
        embedding_cache_dir: Directory for caching embeddings
        compute_embeddings: Whether to compute TabPFN embeddings
        
    Returns:
        Processed dataset dictionary with additional fields
    """
    # Get the dataset attributes
    X = dataset["X"]
    y = dataset["y"]
    categorical_indicator = dataset.get("categorical_indicator", [False] * X.shape[1])
    
    # Store the categorical indicator for CatBoost preprocessing later
    dataset["categorical_indicator_raw"] = (
        categorical_indicator.copy() if hasattr(categorical_indicator, 'copy') 
        else list(categorical_indicator)
    )
    
    # Preprocess features to convert strings to numeric
    X = preprocess_features(X, categorical_indicator)
    
    # Determine task type
    is_classification = _determine_task_type(
        dataset, y, task_type, preserve_regression
    )
    
    # Process labels based on task type
    y = _process_labels(y, is_classification, dataset)
    
    # Create dataset-specific random seed for reproducible but different splits
    dataset_seed = _create_dataset_specific_seed(dataset["id"], seed)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = _split_dataset(
        X, y, test_size, val_size, dataset_seed, is_classification
    )
    
    logger.info(f"Dataset {dataset['name']} shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Apply training sample limits
    if max_train_samples and max_train_samples < len(X_train):
        X_train, y_train = _apply_training_sample_limit(
            X_train, y_train, max_train_samples, sampling_strategy, dataset_seed
        )
    
    # Apply test sample limits
    if max_test_samples:
        if max_test_samples < len(X_test):
            X_test = X_test[:max_test_samples]
            y_test = y_test[:max_test_samples]
        if max_test_samples < len(X_val):
            X_val = X_val[:max_test_samples]
            y_val = y_val[:max_test_samples]
    
    # Update dataset with processed data
    dataset.update({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "is_classification": is_classification
    })
    
    # Compute embeddings if requested
    if compute_embeddings:
        dataset = _compute_tabpfn_embeddings(
            dataset, embedding_size, embedding_cache_dir, 
            force_recompute_embeddings, dataset_seed
        )
    
    return dataset


def _determine_task_type(dataset: Dict[str, Any], y: np.ndarray, 
                        task_type: Optional[str], preserve_regression: bool) -> bool:
    """Determine whether this is a classification or regression task."""
    # Use manual override if provided
    if task_type:
        is_classification = (task_type.lower() == 'classification')
        logger.info(f"Dataset {dataset['name']}: Using manual task type override: {task_type}")
        return is_classification
    
    # Try using task detection if available
    try:
        from marvis.utils.task_detection import detect_task_type
        
        task_id = dataset.get('task_id')
        if task_id is None and 'id' in dataset:
            try:
                task_id = int(dataset['id'])
            except (ValueError, TypeError):
                task_id = None
        
        detected_task_type, detection_method = detect_task_type(
            dataset=dataset,
            y=y,
            manual_override=task_type,
            task_id=task_id
        )
        
        is_classification = (detected_task_type == 'classification')
        logger.info(f"Dataset {dataset['name']}: Detected {detected_task_type} task using {detection_method}")
        return is_classification
        
    except Exception as e:
        logger.warning(f"Task detection failed for dataset {dataset['name']}: {e}")
        logger.info("Falling back to heuristic task type detection")
    
    # Fallback heuristic detection
    if preserve_regression:
        is_classification = False
        unique_vals = np.unique(y)
        
        # Override to classification only if clearly discrete
        if len(unique_vals) <= 10 and (
            y.dtype.kind == 'O' or  # String labels
            (y.dtype.kind in ('i', 'u')) or  # Integer with few values
            (y.dtype.kind == 'f' and all(float(val).is_integer() for val in unique_vals))  # Integer-like floats
        ):
            is_classification = True
            logger.info(f"Overriding to classification due to {len(unique_vals)} discrete values")
        else:
            logger.info(f"Preserving as regression task with {len(unique_vals)} unique values")
    else:
        # Default to classification (legacy behavior)
        is_classification = True
        logger.info("Defaulting to classification task (legacy behavior)")
    
    return is_classification


def _process_labels(y: np.ndarray, is_classification: bool, dataset: Dict[str, Any]) -> np.ndarray:
    """Process labels based on task type."""
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if is_classification:
        # Classification task processing
        if y.dtype.kind == 'O':
            logger.info(f"Dataset {dataset['name']} has string labels. Encoding to integers.")
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            logger.info(f"Encoded {len(encoder.classes_)} unique classes")
        elif y.dtype.kind == 'f':  # Float type for classification
            unique_vals = np.unique(y)
            if len(unique_vals) <= 10 and all(float(val).is_integer() for val in unique_vals):
                # Convert to integers for classification
                logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} discrete float values. Converting to integers.")
                y = y.astype(int)
            else:
                # Bin continuous values for classification
                logger.info(f"Dataset {dataset['name']} has continuous target. Converting to classification by binning.")
                from sklearn.preprocessing import KBinsDiscretizer
                n_bins = min(10, len(np.unique(y)))
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                y = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
                logger.info(f"Binned continuous target into {n_bins} classes")
                dataset["target_discretizer"] = discretizer
        else:
            # Integer classification - keep as is
            unique_vals = np.unique(y)
            logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} unique integer values for classification")
    else:
        # Regression task processing
        if y.dtype.kind == 'O':
            # String labels for regression - try to convert to numeric
            logger.warning(f"Dataset {dataset['name']} has string labels but is detected as regression. Attempting conversion.")
            try:
                import pandas as pd
                y = pd.to_numeric(y, errors='coerce')
                if np.isnan(y).any():
                    logger.error(f"Cannot convert string labels to numeric for regression. Falling back to classification.")
                    from sklearn.preprocessing import LabelEncoder
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y)
            except Exception:
                logger.error(f"Failed to convert string labels for regression. Falling back to classification.")
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
        else:
            # Numeric regression - keep as is
            unique_vals = np.unique(y)
            logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} unique values for regression (range: {y.min():.3f} to {y.max():.3f})")
    
    return y


def _create_dataset_specific_seed(dataset_id: Union[str, int], base_seed: int) -> int:
    """Create a dataset-specific but reproducible random seed."""
    import hashlib
    dataset_id_bytes = str(dataset_id).encode('utf-8')
    dataset_id_hash = int(hashlib.md5(dataset_id_bytes).hexdigest()[:8], 16) % 10000
    return base_seed + dataset_id_hash


def _split_dataset(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float, 
                  random_state: int, is_classification: bool) -> Tuple[np.ndarray, ...]:
    """Split dataset into train, validation, and test sets."""
    # First split: separate test set
    if is_classification:
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # Fall back to non-stratified if stratification fails
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Second split: separate validation set from remaining data
    if is_classification:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
            )
        except ValueError:
            # Fall back to non-stratified if stratification fails
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=random_state
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _apply_training_sample_limit(X_train: np.ndarray, y_train: np.ndarray, 
                               max_samples: int, strategy: str, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sampling strategy to limit training data."""
    logger.info(f"Limiting training data to {max_samples} samples (from {len(X_train)} available) using {strategy} sampling")
    
    # Set random seed for reproducible sampling
    np.random.seed(random_state)
    
    if strategy == "balanced":
        # Balanced sampling: equal samples per class
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        
        # Calculate how many examples to take from each class
        examples_per_class = max_samples // num_classes
        remainder = max_samples % num_classes
        
        # Initialize list to hold selected indices
        selected_indices = []
        
        for i, class_label in enumerate(unique_classes):
            # Get indices for this class
            class_indices = np.where(y_train == class_label)[0]
            
            # Determine how many samples to take from this class
            samples_from_class = examples_per_class + (1 if i < remainder else 0)
            samples_from_class = min(samples_from_class, len(class_indices))
            
            # Randomly sample from this class
            if samples_from_class > 0:
                chosen_indices = np.random.choice(class_indices, samples_from_class, replace=False)
                selected_indices.extend(chosen_indices)
                
            logger.info(f"  Class {class_label}: selected {samples_from_class} out of {len(class_indices)} samples")
        
        # Convert to numpy array and sort for consistency
        indices = np.array(selected_indices)
        indices.sort()
    else:
        # Random sampling: simple random selection
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        indices.sort()  # Sort to maintain some consistency
    
    # Apply the selected indices
    X_train_sampled = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
    y_train_sampled = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    
    logger.info(f"Training data limited to {len(X_train_sampled)} samples")
    return X_train_sampled, y_train_sampled


def _compute_tabpfn_embeddings(dataset: Dict[str, Any], embedding_size: int, 
                             cache_dir: Optional[str], force_recompute: bool, 
                             random_state: int) -> Dict[str, Any]:
    """Compute TabPFN embeddings for the dataset."""
    logger.info(f"Computing TabPFN embeddings for dataset {dataset['name']} with size {embedding_size}")
    
    # Import required functions
    from marvis.data import get_tabpfn_embeddings
    
    # Handle cache directory
    if cache_dir and cache_dir.lower() != 'none':
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None
    
    # Use dataset ID or name as cache identifier
    dataset_identifier = str(dataset["id"])
    
    # Compute embeddings
    try:
        train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
            dataset["X_train"], dataset["y_train"], dataset["X_val"], dataset["X_test"],
            embedding_size=embedding_size,
            cache_dir=cache_dir,
            dataset_name=dataset_identifier,
            force_recompute=force_recompute,
            seed=random_state
        )
        
        # Add embeddings to dataset
        dataset.update({
            "train_embeddings": train_embeddings,
            "val_embeddings": val_embeddings,
            "test_embeddings": test_embeddings,
            "y_train_sample": y_train_sample,
            "tabpfn_model": tabpfn
        })
        
        logger.info(f"Computed embeddings - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Failed to compute TabPFN embeddings for dataset {dataset['name']}: {e}")
        raise
    
    return dataset