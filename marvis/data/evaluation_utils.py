"""
Dataset loading and processing utilities for evaluation workflows.

This module provides functions to load multiple datasets, handle preprocessing,
and prepare data for evaluation with consistent sampling and splitting strategies.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split

from .dataset import load_dataset, clear_failed_dataset_cache

logger = logging.getLogger(__name__)


def load_datasets_for_evaluation(args) -> List[Dict[str, Any]]:
    """
    Load multiple datasets based on the provided arguments for evaluation.
    
    Args:
        args: Command line arguments with dataset source specification
        
    Returns:
        List of dictionaries with dataset information
    """
    datasets = []

    # Clear the _FAILED_DATASET_CACHE to always try loading the dataset
    clear_failed_dataset_cache()
    logger.info("Cleared _FAILED_DATASET_CACHE to ensure dataset loading is attempted")
    
    # Case 1: Single dataset name provided
    if args.dataset_name:
        logger.info(f"Loading single dataset: {args.dataset_name}")
        try:
            preserve_regression = getattr(args, 'preserve_regression', False)
            X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(args.dataset_name, bypass_size_check=True, preserve_regression=preserve_regression)
            
            # Resolve the dataset name to get proper OpenML identifiers
            from marvis.utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            
            # Try to resolve identifiers using the dataset name
            try:
                identifiers = rm.resolve_openml_identifiers(dataset_name=dataset_name)
                dataset_id = identifiers.get('dataset_id', args.dataset_name)
                task_id = identifiers.get('task_id')
            except Exception as resolve_error:
                logger.warning(f"Could not resolve OpenML identifiers for {dataset_name}: {resolve_error}")
                dataset_id = args.dataset_name
                task_id = None
            
            datasets.append({
                "id": dataset_id,
                "name": dataset_name,
                "task_id": task_id,
                "X": X,
                "y": y,
                "categorical_indicator": categorical_indicator,
                "attribute_names": attribute_names
            })
            logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id}, task_id: {task_id})")
        except Exception as e:
            logger.error(f"Failed to load dataset {args.dataset_name}: {e}")
            raise ValueError(f"Failed to load dataset {args.dataset_name}: {e}")
    
    # Case 2: Load from dataset IDs
    elif args.dataset_ids:
        dataset_ids = [id.strip() for id in args.dataset_ids.split(",")]
        logger.info(f"Loading {len(dataset_ids)} datasets from provided IDs: {dataset_ids}")
        preserve_regression = getattr(args, 'preserve_regression', False)
        
        for dataset_id in dataset_ids:
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(dataset_id, bypass_size_check=True, preserve_regression=preserve_regression)
                datasets.append({
                    "id": dataset_id,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id})")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
    
    # Case 2b: Load from task IDs
    elif args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        logger.info(f"Loading {len(task_ids)} datasets from provided task IDs: {task_ids}")
        preserve_regression = getattr(args, 'preserve_regression', False)
        
        # Get resource manager to resolve task IDs to dataset IDs
        from marvis.utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        
        for task_id in task_ids:
            try:
                # Resolve task_id to dataset_id
                identifiers = rm.resolve_openml_identifiers(task_id=task_id)
                dataset_id = identifiers.get('dataset_id')
                
                if dataset_id is None:
                    logger.error(f"Could not resolve task ID {task_id} to dataset ID")
                    continue
                    
                logger.info(f"Task ID {task_id} resolves to dataset ID {dataset_id}")
                
                X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(dataset_id), bypass_size_check=True, preserve_regression=preserve_regression)
                datasets.append({
                    "id": str(dataset_id),
                    "name": dataset_name,
                    "task_id": task_id,  # Include task_id in the dataset dict
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (task ID: {task_id}, dataset ID: {dataset_id})")
            except Exception as e:
                logger.error(f"Failed to load dataset for task ID {task_id}: {e}")
    
    # Case 3: Load from directory of CSV files
    elif args.data_dir:
        logger.info(f"Loading datasets from directory: {args.data_dir}")
        csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {args.data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
                logger.info(f"Loading dataset from CSV: {csv_file}")
                
                # Load CSV using pandas
                df = pd.read_csv(csv_file)
                
                # Assume the last column is the target
                if len(df.columns) < 2:
                    logger.warning(f"Dataset {csv_file} has fewer than 2 columns, skipping")
                    continue
                
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                # Convert to numpy arrays
                X = X.values
                y = y.values
                
                # Create categorical indicator (assume all non-numeric columns are categorical)
                categorical_indicator = []
                attribute_names = df.columns[:-1].tolist()
                
                for col in df.columns[:-1]:
                    # Check if column is numeric
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    categorical_indicator.append(not is_numeric)
                
                datasets.append({
                    "id": dataset_name,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} from CSV")
                
            except Exception as e:
                logger.error(f"Failed to load CSV file {csv_file}: {e}")
    
    # Case 4: Sample random datasets from OpenML
    elif args.num_datasets:
        logger.info(f"Sampling {args.num_datasets} random datasets from OpenML")
        
        # Import OpenML dataset listing function
        try:
            from .dataset import list_available_datasets
            available_datasets = list_available_datasets()
            
            # Convert to list and shuffle
            dataset_items = list(available_datasets.items())
            np.random.shuffle(dataset_items)
            
            # Try to load the requested number of datasets
            preserve_regression = getattr(args, 'preserve_regression', False)
            loaded_count = 0
            for dataset_name, dataset_id in dataset_items:
                if loaded_count >= args.num_datasets:
                    break
                    
                try:
                    X, y, categorical_indicator, attribute_names, loaded_name = load_dataset(str(dataset_id), bypass_size_check=True, preserve_regression=preserve_regression)
                    datasets.append({
                        "id": str(dataset_id),
                        "name": loaded_name,
                        "X": X,
                        "y": y,
                        "categorical_indicator": categorical_indicator,
                        "attribute_names": attribute_names
                    })
                    logger.info(f"Successfully loaded random dataset {loaded_name} (ID: {dataset_id})")
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load random dataset {dataset_id}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {loaded_count} out of {args.num_datasets} requested random datasets")
            
        except Exception as e:
            logger.error(f"Failed to sample random datasets: {e}")
            raise ValueError(f"Failed to sample random datasets: {e}")
    
    if not datasets:
        raise ValueError("No datasets were successfully loaded")
    
    logger.info(f"Total datasets loaded: {len(datasets)}")
    return datasets


def apply_train_test_split(dataset: Dict[str, Any], test_size: float = 0.2, random_state: int = 42, is_classification: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply train-test split to a dataset.
    
    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
        is_classification: Whether this is a classification task (if None, will be inferred)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = dataset['X']
    y = dataset['y']
    
    # Determine task type if not provided
    if is_classification is None:
        # Check if this is a regression task by examining y
        # If y has many unique values (more than 20% of samples), assume regression
        unique_values = len(np.unique(y))
        is_classification = unique_values <= len(y) * 0.2
    
    if is_classification:
        # For classification, use stratification
        try:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        except ValueError:
            # Fall back to non-stratified if stratification fails
            logger.warning(f"Stratification failed, using non-stratified split")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        # For regression, don't use stratification
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def apply_sampling_strategy(X_train: np.ndarray, y_train: np.ndarray, max_samples: int, 
                          strategy: str = "balanced", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sampling strategy to limit training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_samples: Maximum number of samples to keep
        strategy: Sampling strategy ("balanced" or "random")
        random_state: Random seed for reproducible sampling
        
    Returns:
        Tuple of (X_train_sampled, y_train_sampled)
    """
    if max_samples >= len(X_train):
        return X_train, y_train
    
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
    if hasattr(X_train, 'iloc'):  # pandas DataFrame
        X_train_sampled = X_train.iloc[indices]
        y_train_sampled = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    else:  # numpy array
        X_train_sampled = X_train[indices]
        y_train_sampled = y_train[indices]
    
    logger.info(f"Training data limited to {len(X_train_sampled)} samples")
    return X_train_sampled, y_train_sampled


def validate_training_sample_args(args) -> None:
    """
    Validate training sample arguments for early error detection.
    
    Args:
        args: Command line arguments with sampling parameters
        
    Raises:
        ValueError: If parameter combination is invalid
    """
    balanced_few_shot = getattr(args, 'balanced_few_shot', False)
    num_few_shot_examples = getattr(args, 'num_few_shot_examples', None)
    max_train_samples = getattr(args, 'max_train_samples', None)
    
    if balanced_few_shot and num_few_shot_examples is None:
        raise ValueError("--balanced_few_shot requires --num_few_shot_examples to be specified")
    
    if num_few_shot_examples is not None and num_few_shot_examples <= 0:
        raise ValueError(f"--num_few_shot_examples must be positive, got {num_few_shot_examples}")
    
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError(f"--max_train_samples must be positive, got {max_train_samples}")


def standardize_training_samples(X_train: np.ndarray, y_train: np.ndarray, args, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize training sample selection across evaluation scripts.
    
    Handles both old-style (max_train_samples + sampling_strategy) and 
    new-style (num_few_shot_examples + balanced_few_shot) parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        args: Command line arguments with sampling parameters
        random_state: Random seed for reproducible sampling
        
    Returns:
        Tuple of (X_train_sampled, y_train_sampled)
    """
    # Check for new-style parameters (take priority)
    balanced_few_shot = getattr(args, 'balanced_few_shot', False)
    num_few_shot_examples = getattr(args, 'num_few_shot_examples', None)
    
    # Check for old-style parameters  
    max_train_samples = getattr(args, 'max_train_samples', None)
    sampling_strategy = getattr(args, 'sampling_strategy', 'balanced')
    
    # Parameter validation
    if balanced_few_shot and num_few_shot_examples is None:
        raise ValueError("--balanced_few_shot requires --num_few_shot_examples to be specified")
    
    if num_few_shot_examples is not None and num_few_shot_examples <= 0:
        raise ValueError(f"--num_few_shot_examples must be positive, got {num_few_shot_examples}")
    
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError(f"--max_train_samples must be positive, got {max_train_samples}")
    
    # Warn about conflicting parameters
    if balanced_few_shot and max_train_samples is not None:
        logger.warning("Both --balanced_few_shot and --max_train_samples specified. Using --balanced_few_shot approach.")
    
    if num_few_shot_examples is not None and not balanced_few_shot:
        logger.warning("--num_few_shot_examples specified without --balanced_few_shot. Parameter will be ignored. Use --balanced_few_shot to enable per-class sampling.")
    
    # Determine effective parameters
    if balanced_few_shot and num_few_shot_examples is not None:
        # New-style: per-class sampling
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        effective_max_samples = num_few_shot_examples * num_classes
        effective_strategy = "balanced"
        
        logger.info(f"Using balanced few-shot: {num_few_shot_examples} examples per class ({num_classes} classes) = {effective_max_samples} total samples")
        
    elif max_train_samples is not None:
        # Old-style: total sample count
        effective_max_samples = max_train_samples
        effective_strategy = sampling_strategy
        
        logger.info(f"Using max_train_samples: {effective_max_samples} total samples with {effective_strategy} strategy")
        
    else:
        # No sampling limits specified
        logger.info("No training sample limits specified, using all available training data")
        return X_train, y_train
    
    # Apply sampling using existing function
    return apply_sampling_strategy(X_train, y_train, effective_max_samples, effective_strategy, random_state)


def preprocess_dataset_for_evaluation(dataset: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Preprocess a single dataset for evaluation including train/test split and sampling.
    
    Args:
        dataset: Dataset dictionary with raw data
        args: Command line arguments with preprocessing parameters
        
    Returns:
        Dataset dictionary with processed data including train/test splits
    """
    logger.info(f"Preprocessing dataset {dataset['name']} for evaluation")
    
    # Extract data
    X = dataset['X']
    y = dataset['y']
    
    # Determine task type using task detection
    from marvis.utils.task_detection import detect_task_type
    
    # Get task_id if available for better detection
    task_id = dataset.get('task_id', None)
    manual_task_type = getattr(args, 'task_type', None)
    
    task_type, detection_method = detect_task_type(
        dataset=dataset,
        y=y,
        manual_override=manual_task_type,
        task_id=task_id
    )
    
    is_classification = (task_type == 'classification')
    logger.info(f"Detected task type: {task_type} using {detection_method}")
    
    # Apply train-test split with proper task type
    X_train, X_test, y_train, y_test = apply_train_test_split(
        dataset, test_size=0.2, random_state=args.seed, is_classification=is_classification
    )
    
    # Create validation split from training data with proper task type
    if is_classification:
        # For classification, use stratification
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.5, random_state=args.seed, stratify=y_train
            )
        except ValueError:
            # Fall back to non-stratified if stratification fails
            logger.warning(f"Validation split stratification failed, using non-stratified split")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.5, random_state=args.seed
            )
    else:
        # For regression, don't use stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.5, random_state=args.seed
        )
    
    logger.info(f"Dataset {dataset['name']} shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Apply training data sampling using standardized function
    X_train, y_train = standardize_training_samples(X_train, y_train, args, random_state=args.seed)
    
    # Limit test samples if specified
    if args.max_test_samples and args.max_test_samples < len(X_test):
        X_test = X_test[:args.max_test_samples]
        y_test = y_test[:args.max_test_samples]
        logger.info(f"Limited test set to {args.max_test_samples} samples")
    
    # Update dataset with processed data
    processed_dataset = dataset.copy()
    processed_dataset.update({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "is_classification": is_classification,
        "task_type": task_type,
        "detection_method": detection_method
    })
    
    return processed_dataset


def preprocess_datasets_for_evaluation(datasets: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """
    Preprocess multiple datasets for evaluation.
    
    Args:
        datasets: List of raw dataset dictionaries
        args: Command line arguments with preprocessing parameters
        
    Returns:
        List of processed dataset dictionaries
    """
    logger.info(f"Preprocessing {len(datasets)} datasets for evaluation")
    
    processed_datasets = []
    for dataset in datasets:
        try:
            processed_dataset = preprocess_dataset_for_evaluation(dataset, args)
            processed_datasets.append(processed_dataset)
        except Exception as e:
            logger.error(f"Failed to preprocess dataset {dataset['name']}: {e}")
            # Continue with other datasets
            continue
    
    logger.info(f"Successfully preprocessed {len(processed_datasets)} out of {len(datasets)} datasets")
    return processed_datasets


def validate_dataset_for_evaluation(dataset: Dict[str, Any]) -> bool:
    """
    Validate that a dataset is suitable for evaluation.
    
    Args:
        dataset: Dataset dictionary to validate
        
    Returns:
        True if dataset is valid, False otherwise
    """
    required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'name']
    
    # Check for required keys
    for key in required_keys:
        if key not in dataset:
            logger.warning(f"Dataset missing required key: {key}")
            return False
    
    # Check for minimum data size
    if len(dataset['X_train']) < 10:
        logger.warning(f"Dataset {dataset['name']} has too few training samples: {len(dataset['X_train'])}")
        return False
    
    if len(dataset['X_test']) < 5:
        logger.warning(f"Dataset {dataset['name']} has too few test samples: {len(dataset['X_test'])}")
        return False
    
    # Check for class balance
    unique_classes = np.unique(dataset['y_train'])
    if len(unique_classes) < 2:
        logger.warning(f"Dataset {dataset['name']} has only {len(unique_classes)} class(es)")
        return False
    
    return True