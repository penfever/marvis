"""
Tabular, image, and audio embedding extraction and processing utilities.
"""

import numpy as np
import torch
import logging
import os
import hashlib
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional, Union, Any, Dict, Callable
from PIL import Image
import torchvision.transforms as transforms

__all__ = [
    'generate_dataset_hash',
    'get_tabpfn_embeddings',
    'get_embeddings_in_chunks',
    'resize_embeddings',
    'prepare_tabpfn_embeddings_for_prefix',
    'get_dinov2_embeddings',
    'load_dinov2_model',
    'prepare_image_for_dinov2'
]

# Audio embeddings are in separate module to avoid heavy dependencies
# from marvis.data.audio_embeddings import get_whisper_embeddings, load_whisper_model

logger = logging.getLogger(__name__)

def generate_dataset_hash(X: np.ndarray, y: np.ndarray, embedding_size: int, dataset_name: str = None) -> str:
    """
    Generate a unique hash for a dataset configuration based on its properties.

    Args:
        X: Feature matrix
        y: Labels
        embedding_size: Size of the embeddings to be generated
        dataset_name: Optional name of the dataset

    Returns:
        hash_str: A unique string hash for this dataset configuration
    """
    # Get data properties
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Convert inputs to numpy arrays if they aren't already
    # This handles pandas Series, DataFrame, Categorical, etc.
    if not isinstance(X, np.ndarray):
        X_np = np.array(X)
    else:
        X_np = X
        
    if not isinstance(y, np.ndarray):
        y_np = np.array(y)
    else:
        y_np = y
    
    # Create a dictionary with all relevant properties
    config = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_np, return_counts=True))},
        "embedding_size": int(embedding_size),
        # Convert to float32 to ensure consistent representation across platforms
        "data_hash": hashlib.md5(X_np[:100].astype(np.float32).tobytes()).hexdigest()[:10],
        # Use string representation to avoid tobytes() issues with categorical types
        "label_hash": hashlib.md5(str(y_np[:100].tolist()).encode()).hexdigest()[:10],
    }

    if dataset_name:
        config["dataset_name"] = dataset_name

    # Convert to a string and hash it
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    hash_str = hash_obj.hexdigest()

    return hash_str

def get_tabpfn_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    max_samples: int = 3000,
    embedding_size: int = 1000,
    cache_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    force_recompute: bool = False,
    task_type: str = "classification",
    seed: int = 42
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Any, np.ndarray]:
    """
    Get TabPFN embeddings with improved class balance checking and caching.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        max_samples: Maximum number of samples to use from training set
        embedding_size: Size of the output embeddings
        cache_dir: Directory to store cached embeddings. If None, caching is disabled.
        dataset_name: Optional name of the dataset for the cache filename
        force_recompute: If True, ignore cache and recompute embeddings
        task_type: Type of task - "classification" or "regression"
        seed: Random seed for reproducible feature selection and sampling

    Returns:
        train_embeddings: TabPFN embeddings for training set
        val_embeddings: Always None (validation embeddings no longer generated)
        test_embeddings: TabPFN embeddings for test set
        tabpfn: Fitted TabPFN model
        y_train_sample: Labels for the sampled training set
    """
    try:
        if task_type == "regression":
            from tabpfn import TabPFNRegressor
            TabPFNModel = TabPFNRegressor
        else:
            from tabpfn import TabPFNClassifier  
            TabPFNModel = TabPFNClassifier
    except ImportError:
        raise ImportError("TabPFN package is required to get embeddings. Install it with 'pip install tabpfn'.")

    # Setup caching if enabled
    cache_file = None
    cache_metadata = {
        "dataset_name": dataset_name or "unknown",
        "embedding_size": embedding_size,
        "max_samples": max_samples,
        "task_type": task_type,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1]
    }

    if cache_dir:
        # Try to use the new resource manager for caching
        try:
            from ..utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            use_managed_cache = True
        except Exception:
            use_managed_cache = False
        
        if use_managed_cache:
            # Generate cache key using resource manager
            cache_key = rm.cache_manager.get_cache_key(
                dataset_name=dataset_name or "unknown",
                embedding_size=embedding_size,
                max_samples=max_samples,
                task_type=task_type,
                n_features=X_train.shape[1],
                data_hash=generate_dataset_hash(X_train, y_train, embedding_size, dataset_name)
            )
        else:
            # Legacy caching approach
            # Generate a unique hash for this dataset configuration
            dataset_hash = generate_dataset_hash(X_train, y_train, embedding_size, dataset_name)
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

        if use_managed_cache:
            # Check managed cache
            if not force_recompute and rm.cache_manager.cache_exists('embeddings', cache_key, '.npz'):
                logger.info(f"Loading cached TabPFN embeddings from managed cache")
                try:
                    cache_data = rm.cache_manager.load_from_cache('embeddings', cache_key, '.npz')
                    
                    if cache_data and isinstance(cache_data, dict):
                        # Extract embeddings and metadata
                        train_embeddings = cache_data["train_embeddings"]
                        # val_embeddings removed - skip loading it
                        test_embeddings = cache_data["test_embeddings"]
                        y_train_sample = cache_data["y_train_sample"]
                        cache_metadata = cache_data.get("metadata", {})

                        logger.info(f"Loaded embeddings from managed cache - Train: {train_embeddings.shape}, Val: None, Test: {test_embeddings.shape}")
                        logger.info(f"Cache metadata: {cache_metadata}")

                        # Create a dummy TabPFN model since we don't need to fit anymore
                        if task_type == "regression":
                            tabpfn = TabPFNModel(
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                n_estimators=8,
                                ignore_pretraining_limits=True
                            )
                        else:
                            tabpfn = TabPFNModel(
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                n_estimators=8,
                                ignore_pretraining_limits=True
                            )

                        # Check if the loaded embeddings match the expected embedding size
                        if train_embeddings.shape[1] != embedding_size:
                            logger.warning(f"Cached embeddings size ({train_embeddings.shape[1]}) doesn't match requested size ({embedding_size}). Resizing...")
                            # Resize them (using the same resize code as below)
                            train_embeddings, test_embeddings = resize_embeddings(
                                train_embeddings, test_embeddings, embedding_size
                            )

                        return train_embeddings, None, test_embeddings, tabpfn, y_train_sample

                except Exception as e:
                    logger.warning(f"Error loading cached embeddings from managed cache: {e}. Recomputing...")
                    # Continue with normal computation if loading fails
        else:
            # Legacy cache approach
            # Define cache filename
            prefix = f"{dataset_name}_" if dataset_name else ""
            cache_file = os.path.join(cache_dir, f"{prefix}tabpfn_embeddings_{dataset_hash}.npz")

            # Check if cache file exists and we're not forcing recomputation
            if os.path.exists(cache_file) and not force_recompute:
                logger.info(f"Loading cached TabPFN embeddings from {cache_file}")
                try:
                    cache = np.load(cache_file, allow_pickle=True)

                    # Extract embeddings and metadata
                    train_embeddings = cache["train_embeddings"]
                    # val_embeddings removed - skip loading it
                    test_embeddings = cache["test_embeddings"]
                    y_train_sample = cache["y_train_sample"]
                    cache_metadata = cache["metadata"].item() if "metadata" in cache else {}

                    logger.info(f"Loaded embeddings from cache - Train: {train_embeddings.shape}, Val: None, Test: {test_embeddings.shape}")
                    logger.info(f"Cache metadata: {cache_metadata}")

                    # Create a dummy TabPFN model since we don't need to fit anymore
                    tabpfn = TabPFNModel(
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        n_estimators=8,
                        ignore_pretraining_limits=True
                    )

                    # Check if the loaded embeddings match the expected embedding size
                    if train_embeddings.shape[1] != embedding_size:
                        logger.warning(f"Cached embeddings size ({train_embeddings.shape[1]}) doesn't match requested size ({embedding_size}). Resizing...")
                        # Resize them (using the same resize code as below)
                        train_embeddings, test_embeddings = resize_embeddings(
                            train_embeddings, test_embeddings, embedding_size
                        )

                    return train_embeddings, None, test_embeddings, tabpfn, y_train_sample

                except Exception as e:
                    logger.warning(f"Error loading cached embeddings: {e}. Recomputing...")
                    # Continue with normal computation if loading fails

    logger.info("Fitting TabPFN and extracting embeddings")

    # Log original target distribution
    if task_type == "classification":
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique_classes, class_counts))
        logger.info(f"Original train set class distribution: {class_dist}")
        
        # Validate class distribution for stratified sampling
        MIN_SAMPLES_PER_CLASS = 2
        min_class_count = min(class_counts)
        if min_class_count < MIN_SAMPLES_PER_CLASS:
            raise ValueError(f"Minimum class count is {min_class_count}, which is less than {MIN_SAMPLES_PER_CLASS} required for stratified sampling")
        
        if len(unique_classes) < 2:
            raise ValueError(f"Only {len(unique_classes)} unique classes found, need at least 2 classes")
    else:
        # For regression, log target statistics
        logger.info(f"Original train set target statistics: min={y_train.min():.3f}, max={y_train.max():.3f}, mean={y_train.mean():.3f}, std={y_train.std():.3f}")
        unique_classes = None  # Not needed for regression

    # Handle dimensionality reduction if feature count is very high
    MAX_FEATURES_FOR_TABPFN = 1000  # TabPFN typically works well with less than 1000 features
    if X_train.shape[1] > MAX_FEATURES_FOR_TABPFN:
        # As a last resort, just select a random subset of features
        n_components = min(MAX_FEATURES_FOR_TABPFN, X_train.shape[1])
        from ..utils import create_random_state
        rng = create_random_state(seed)
        selected_features = rng.choice(range(X_train.shape[1]), n_components, replace=False)
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    # Sample if dataset is too large
    if len(X_train) > max_samples:
        from sklearn.model_selection import train_test_split
        
        sample_ratio = max_samples / len(X_train)
        
        if task_type == "classification":
            # Check if we need to adjust sample size to ensure at least 2 samples per class
            min_expected_samples = int(min_class_count * sample_ratio)
            
            if min_expected_samples < MIN_SAMPLES_PER_CLASS:
                # Adjust sample size to ensure minimum class representation
                required_ratio = MIN_SAMPLES_PER_CLASS / min_class_count
                adjusted_max_samples = min(len(X_train), int(len(X_train) * required_ratio * 1.1))  # Add 10% buffer
                logger.warning(f"Adjusting sample size from {max_samples} to {adjusted_max_samples} to ensure minimum class representation")
                sample_ratio = adjusted_max_samples/len(X_train)
            
            try:
                # Split with stratification to maintain class distribution
                _, X_train_sample, _, y_train_sample = train_test_split(
                    X_train, y_train,
                    test_size=sample_ratio,
                    stratify=y_train,
                    random_state=42
                )
            except ValueError as e:
                # If stratified sampling fails, fall back to random sampling
                logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
                _, X_train_sample, _, y_train_sample = train_test_split(
                    X_train, y_train,
                    test_size=sample_ratio,
                    random_state=42
                )
        else:
            # For regression, use simple random sampling
            _, X_train_sample, _, y_train_sample = train_test_split(
                X_train, y_train,
                test_size=sample_ratio,
                random_state=42
            )
            
        logger.info(f"Sampled {len(X_train_sample)} training samples from {len(X_train)} total samples")
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    # Verify the sampled distribution for classification
    if task_type == "classification":
        sampled_classes, sampled_counts = np.unique(y_train_sample, return_counts=True)
        logger.info(f"Sampled train set class distribution: {dict(zip(sampled_classes, sampled_counts))}")
    else:
        logger.info(f"Sampled train set target statistics: min={y_train_sample.min():.3f}, max={y_train_sample.max():.3f}, mean={y_train_sample.mean():.3f}")

    # Initialize and fit TabPFN
    N_ensemble = 8
    tabpfn = TabPFNModel(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_estimators=N_ensemble,
        ignore_pretraining_limits=True
    )
    tabpfn.fit(X_train_sample, y_train_sample)

    # Extract embeddings - Process X_train_sample normally, use chunks for test set
    train_embeddings_raw = tabpfn.get_embeddings(X_train_sample)
    val_embeddings_raw = None  # No validation embeddings generated
    test_embeddings_raw = get_embeddings_in_chunks(tabpfn, X_test, dataset_name="test")

    logger.info(f"Raw embedding shapes - Train: {train_embeddings_raw.shape}, Val: None, Test: {test_embeddings_raw.shape}")

    # Fix test embeddings shape if it's inconsistent (TabPFN single sample issue)
    if len(test_embeddings_raw.shape) == 2 and len(train_embeddings_raw.shape) == 3:
        # Test embeddings are missing the sample dimension, add it back
        test_embeddings_raw = test_embeddings_raw[:, np.newaxis, :]  # (8, 192) -> (8, 1, 192)
        logger.info(f"Fixed test embeddings shape: {test_embeddings_raw.shape}")

    # Process embeddings - average across ensemble members if available
    if len(train_embeddings_raw.shape) == 3 and train_embeddings_raw.shape[0] > 1:
        logger.info("Averaging embeddings across ensemble members")
        train_embeddings = np.mean(train_embeddings_raw, axis=0)
        val_embeddings = None
        test_embeddings = np.mean(test_embeddings_raw, axis=0)
    else:
        # For backward compatibility, handle original format
        if len(train_embeddings_raw.shape) == 3:
            train_embeddings = train_embeddings_raw[0]
            val_embeddings = None
            test_embeddings = test_embeddings_raw[0]
        else:
            train_embeddings = train_embeddings_raw
            val_embeddings = None
            test_embeddings = test_embeddings_raw

    logger.info(f"Processed embedding shapes - Train: {train_embeddings.shape}, Val: None, Test: {test_embeddings.shape}")

    # Standardize embeddings using RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    # Reshape to 2D if needed for RobustScaler
    original_train_shape = train_embeddings.shape
    original_test_shape = test_embeddings.shape

    # Check if we need to reshape for standardization
    if len(train_embeddings.shape) > 2:
        train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)

    # Fit on train embeddings and transform all sets
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)
    val_embeddings = None  # No validation embeddings to transform
    
    # Log scaling statistics
    logger.info(f"RobustScaler statistics - Center (median): {scaler.center_[:5]}... Scale (IQR): {scaler.scale_[:5]}...")

    # Reshape back if needed
    if len(original_train_shape) > 2:
        train_embeddings = train_embeddings.reshape(original_train_shape)
        test_embeddings = test_embeddings.reshape(original_test_shape)

    # Get the embedding dimension (the last dimension)
    embedding_dim = train_embeddings.shape[-1]

    # Resize embeddings to target size if needed
    if embedding_dim != embedding_size:
        train_embeddings, test_embeddings = resize_embeddings(
            train_embeddings, test_embeddings, embedding_size
        )

    # Cache the embeddings if cache_dir is provided
    if cache_dir:
        # Add timestamp to metadata
        import time
        cache_metadata.update({
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        if use_managed_cache:
            # Save to managed cache
            logger.info(f"Saving TabPFN embeddings to managed cache")
            try:
                cache_data = {
                    "train_embeddings": train_embeddings,
                    "val_embeddings": val_embeddings,
                    "test_embeddings": test_embeddings,
                    "y_train_sample": y_train_sample,
                    "metadata": cache_metadata
                }
                
                success = rm.cache_manager.save_to_cache('embeddings', cache_key, cache_data, '.npz')
                if success:
                    logger.info(f"Successfully saved embeddings to managed cache")
                else:
                    logger.warning(f"Failed to save embeddings to managed cache")
            except Exception as e:
                logger.warning(f"Error saving embeddings to managed cache: {e}")
        else:
            # Legacy cache saving
            logger.info(f"Saving TabPFN embeddings to cache: {cache_file}")
            try:
                # Save to cache
                np.savez(
                    cache_file,
                    train_embeddings=train_embeddings,
                    test_embeddings=test_embeddings,
                    y_train_sample=y_train_sample,
                    metadata=cache_metadata
                )
            except Exception as e:
                logger.warning(f"Error saving embeddings to cache: {e}")

    logger.info(f"Final embedding shapes - Train: {train_embeddings.shape}, Val: None, Test: {test_embeddings.shape}")

    return train_embeddings, None, test_embeddings, tabpfn, y_train_sample

def get_embeddings_in_chunks(
    model: Any, 
    X: np.ndarray, 
    dataset_name: str = "dataset", 
    max_chunk_size: int = 3000, 
    embedding_method: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Process embeddings for larger datasets in chunks to avoid memory issues.
    
    This function is designed to process large datasets in manageable chunks,
    reducing memory usage and preventing out-of-memory errors. It's particularly
    useful for validation and test sets that might be too large to process at once.
    
    Args:
        model: Fitted model with get_embeddings method or custom function
        X: Feature matrix to get embeddings for
        dataset_name: Name of the dataset for logging (for informational purposes)
        max_chunk_size: Maximum number of samples to process in a single chunk
        embedding_method: Optional custom function to get embeddings (default: model.get_embeddings)
            If provided, will be called as: embedding_method(model, X_chunk)
        
    Returns:
        Processed embeddings as a numpy array
    
    Example:
        ```python
        # Using default method (model.get_embeddings)
        test_embeddings = get_embeddings_in_chunks(tabpfn, X_test, "test", max_chunk_size=2000)
        
        # Using custom embedding function
        def custom_embedder(model, data):
            return model.embed_data(data, special_flag=True)
            
        test_embeddings = get_embeddings_in_chunks(model, X_test, "test", 
                                                 max_chunk_size=1500,
                                                 embedding_method=custom_embedder)
        ```
    """
    total_samples = len(X)
    
    # Define the embedding function
    if embedding_method is None:
        def get_emb(model, data):
            return model.get_embeddings(data)
    else:
        get_emb = embedding_method
    
    # If the dataset is small enough, process it all at once
    if total_samples <= max_chunk_size:
        return get_emb(model, X)
    
    # Otherwise, process in chunks
    logger.info(f"Processing {dataset_name} embeddings in chunks (total: {total_samples} samples)")
    
    # Calculate number of chunks needed
    n_chunks = (total_samples + max_chunk_size - 1) // max_chunk_size  # Ceiling division
    chunk_embeddings = []
    
    for i in range(n_chunks):
        start_idx = i * max_chunk_size
        end_idx = min((i + 1) * max_chunk_size, total_samples)
        current_chunk_size = end_idx - start_idx
        
        logger.info(f"Processing {dataset_name} chunk {i+1}/{n_chunks}: samples {start_idx} to {end_idx-1}")
        X_chunk = X[start_idx:end_idx]
        
        # Get embeddings using either model.get_embeddings or the custom method
        chunk_embedding = get_emb(model, X_chunk)
        chunk_embeddings.append(chunk_embedding)
        
        logger.info(f"Processed chunk {i+1} with shape: {chunk_embedding.shape}")
    
    # Concatenate chunks along the appropriate axis
    # The embeddings array shape can be either (n_samples, emb_size) or (n_ensemble, n_samples, emb_size)
    if len(chunk_embeddings[0].shape) == 3:  # Shape is (n_ensemble, n_samples, emb_size)
        # Concatenate along axis 1 (sample dimension)
        concatenated = np.concatenate(chunk_embeddings, axis=1)
    else:  # Shape is (n_samples, emb_size)
        # Concatenate along axis 0 (sample dimension)
        concatenated = np.concatenate(chunk_embeddings, axis=0)
        
    logger.info(f"Final concatenated {dataset_name} embeddings shape: {concatenated.shape}")
    return concatenated

def resize_embeddings(train_embeddings, test_embeddings, embedding_size):
    """Helper function to resize embeddings to target size."""
    embedding_dim = train_embeddings.shape[-1]
    logger.info(f"Resizing embeddings from {embedding_dim} to {embedding_size}")

    # Create new arrays with target size
    train_embeddings_resized = np.zeros((train_embeddings.shape[0], embedding_size))
    test_embeddings_resized = np.zeros((test_embeddings.shape[0], embedding_size))

    # Copy over the data (either truncate or zero-pad)
    copy_dim = min(embedding_dim, embedding_size)
    train_embeddings_resized[:, :copy_dim] = train_embeddings[:, :copy_dim]
    test_embeddings_resized[:, :copy_dim] = test_embeddings[:, :copy_dim]

    return train_embeddings_resized, test_embeddings_resized


def prepare_tabpfn_embeddings_for_prefix(
    train_embeddings: np.ndarray, 
    y_train: np.ndarray, 
    num_embeddings: int = 100,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare TabPFN embeddings to be used as prefix, sampling from all classes.
    Ensures the returned class labels match the embeddings.
    
    Args:
        train_embeddings: TabPFN embeddings for train set with shape [n_samples, embedding_dim]
        y_train: Labels for the corresponding train embeddings
        num_embeddings: Total number of embeddings to include (not per class)
        random_state: Random seed for reproducible sampling
        
    Returns:
        embeddings: Array of shape [num_embeddings, embedding_dim]
        class_labels: Array of shape [num_embeddings] with corresponding labels
    """
    # Make sure y_train matches the length of train_embeddings
    if len(y_train) != len(train_embeddings):
        raise ValueError(f"Length of y_train ({len(y_train)}) doesn't match train_embeddings ({len(train_embeddings)})")
    
    # Initialize random state for reproducible sampling
    rng = np.random.RandomState(random_state)
    
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    
    # Calculate how many examples to take from each class
    embeddings_per_class = num_embeddings // num_classes
    remainder = num_embeddings % num_classes
    
    # Initialize arrays to hold selected embeddings and their labels
    selected_embeddings = []
    selected_labels = []
    
    # Sample from each class
    for class_idx, class_label in enumerate(unique_classes):
        # Get indices of samples from this class within the embeddings array
        class_indices = np.where(y_train == class_label)[0]
        
        # Determine how many to take from this class (distribute remainder)
        n_to_take = embeddings_per_class + (1 if class_idx < remainder else 0)
        
        # If we don't have enough samples in this class, take all and repeat
        if len(class_indices) < n_to_take:
            # Take all available samples
            selected_indices = class_indices
            # Repeat to reach desired count
            repeats_needed = int(np.ceil(n_to_take / len(class_indices)))
            selected_indices = np.tile(selected_indices, repeats_needed)[:n_to_take]
        else:
            # Random sample without replacement
            selected_indices = rng.choice(class_indices, n_to_take, replace=False)
        
        # Add selected embeddings and their labels
        selected_embeddings.append(train_embeddings[selected_indices])
        selected_labels.extend([class_label] * len(selected_indices))
    
    # Concatenate all selected embeddings
    combined_embeddings = np.vstack(selected_embeddings)
    combined_labels = np.array(selected_labels)
    
    # Verify shapes
    assert len(combined_embeddings) == len(combined_labels), f"Mismatch between embeddings ({len(combined_embeddings)}) and labels ({len(combined_labels)})"
    
    return combined_embeddings, combined_labels


def load_dinov2_model(model_name: str = "dinov2_vitb14", device: Optional[str] = None) -> torch.nn.Module:
    """
    Load a DINOV2 model for image embedding extraction.
    
    Args:
        model_name: Name of the DINOV2 model to load. Options:
            Standard models:
            - "dinov2_vits14": ViT-Small with patch size 14
            - "dinov2_vitb14": ViT-Base with patch size 14  
            - "dinov2_vitl14": ViT-Large with patch size 14
            - "dinov2_vitg14": ViT-Giant with patch size 14
            
            Models with registers (improved training stability):
            - "dinov2_vits14_reg": ViT-Small with registers
            - "dinov2_vitb14_reg": ViT-Base with registers
            - "dinov2_vitl14_reg": ViT-Large with registers  
            - "dinov2_vitg14_reg": ViT-Giant with registers
            
            Linear classifier variants (with classification head):
            - "dinov2_vits14_lc": ViT-Small with linear classifier
            - "dinov2_vitb14_lc": ViT-Base with linear classifier
            - "dinov2_vitl14_lc": ViT-Large with linear classifier
            - "dinov2_vitg14_lc": ViT-Giant with linear classifier
            
        device: Device to load the model on. If None, auto-detects optimal device
        
    Returns:
        Loaded DINOV2 model
    """
    from ..utils.device_utils import configure_device_for_model, log_device_usage, setup_device_environment
    
    # Configure device for DINOV2
    device, _ = configure_device_for_model('dinov2', device)
    setup_device_environment(device)
    log_device_usage(f"DINOV2 {model_name}", device)
    
    try:
        # Load model from torch.hub
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded DINOV2 model {model_name} on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load DINOV2 model {model_name}: {e}")
        raise


def prepare_image_for_dinov2(image_path: str, size: int = 224) -> torch.Tensor:
    """
    Prepare an image for DINOV2 processing.
    
    Args:
        image_path: Path to the image file
        size: Target size for the image (DINOV2 expects 224x224)
        
    Returns:
        Preprocessed image tensor ready for DINOV2
    """
    # Define the transforms (same as used in DINOV2 training)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def get_dinov2_embeddings(
    image_paths: List[str],
    model_name: str = "dinov2_vitb14",
    embedding_size: int = 1000,
    cache_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    force_recompute: bool = False,
    batch_size: int = 32,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Extract DINOV2 embeddings from a list of image paths.
    
    Args:
        image_paths: List of paths to image files
        model_name: DINOV2 model variant to use
        embedding_size: Target size for output embeddings (will resize if needed)
        cache_dir: Directory to store cached embeddings
        dataset_name: Name for caching purposes
        force_recompute: Whether to ignore cache and recompute
        batch_size: Number of images to process at once
        device: Device to run inference on
        
    Returns:
        embeddings: Array of shape [n_images, embedding_size]
    """
    from ..utils.device_utils import configure_device_for_model, log_device_usage
    
    # Configure device and batch size for DINOV2
    device, batch_size = configure_device_for_model('dinov2', device, batch_size)
    log_device_usage(f"DINOV2 embeddings {model_name}", device, {'batch_size': batch_size})
    
    # Setup caching if enabled
    cache_file = None
    if cache_dir and dataset_name:
        # Generate hash based on image paths and model
        paths_str = ''.join(sorted(image_paths))
        paths_hash = hashlib.md5(paths_str.encode()).hexdigest()[:10]
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{dataset_name}_dinov2_{model_name}_{paths_hash}.npz")
        
        # Check cache
        if os.path.exists(cache_file) and not force_recompute:
            logger.info(f"Loading cached DINOV2 embeddings from {cache_file}")
            try:
                cache = np.load(cache_file)
                embeddings = cache["embeddings"]
                
                # Resize if needed
                if embeddings.shape[1] != embedding_size:
                    logger.warning(f"Cached embeddings size ({embeddings.shape[1]}) doesn't match requested size ({embedding_size}). Resizing...")
                    embeddings = resize_single_embedding_set(embeddings, embedding_size)
                
                logger.debug(f"Loaded embeddings from cache: {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {e}. Recomputing...")
    
    # Load model
    model = load_dinov2_model(model_name, device)
    
    # Process images in batches
    logger.info(f"Extracting DINOV2 embeddings for {len(image_paths)} images using {model_name}")
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        if i % 20 == 0:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        # Prepare batch
        batch_tensors = []
        for img_path in batch_paths:
            try:
                img_tensor = prepare_image_for_dinov2(img_path)
                batch_tensors.append(img_tensor)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}. Using zero tensor.")
                # Create a zero tensor as fallback
                batch_tensors.append(torch.zeros(1, 3, 224, 224))
        
        # Stack batch and move to device
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            batch_embeddings = model(batch_tensor)
            
        # Move to CPU and convert to numpy
        batch_embeddings = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Extracted raw embeddings: {embeddings.shape}")
    
    # Resize if needed
    if embeddings.shape[1] != embedding_size:
        embeddings = resize_single_embedding_set(embeddings, embedding_size)
        logger.info(f"Resized embeddings to: {embeddings.shape}")
    
    # Cache the embeddings
    if cache_file:
        logger.info(f"Saving DINOV2 embeddings to cache: {cache_file}")
        try:
            np.savez(cache_file, embeddings=embeddings)
        except Exception as e:
            logger.warning(f"Error saving embeddings to cache: {e}")
    
    return embeddings


def resize_single_embedding_set(embeddings: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize a single set of embeddings to target size.
    
    Args:
        embeddings: Input embeddings of shape [n_samples, original_size]
        target_size: Target embedding size
        
    Returns:
        Resized embeddings of shape [n_samples, target_size]
    """
    original_size = embeddings.shape[1]
    resized = np.zeros((embeddings.shape[0], target_size))
    
    # Copy over the data (either truncate or zero-pad)
    copy_size = min(original_size, target_size)
    resized[:, :copy_size] = embeddings[:, :copy_size]
    
    return resized