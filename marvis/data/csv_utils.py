"""
Utilities for loading and handling CSV datasets with precomputed embeddings.
"""

import os
import logging
import json
import glob
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def is_csv_dataset(dataset_id: str) -> bool:
    """
    Determine if a dataset ID corresponds to a CSV dataset rather than an OpenML dataset.
    
    Args:
        dataset_id: The dataset identifier
        
    Returns:
        True if this appears to be a CSV dataset, False if it's likely an OpenML dataset
    """
    # OpenML datasets are typically numeric IDs
    if dataset_id.isdigit():
        return False
    
    # Check for common patterns in CSV dataset names
    csv_patterns = [
        "episode" in dataset_id.lower(),
        "_" in dataset_id,
        "-" in dataset_id,
        dataset_id.lower().endswith('.csv'),
    ]
    
    return any(csv_patterns)


def find_csv_file(dataset_id: str, search_dirs: List[str]) -> Optional[str]:
    """
    Search for a CSV file in multiple directories.
    Uses robust resource manager search strategy with fallback to legacy method.
    
    Args:
        dataset_id: The dataset identifier
        search_dirs: List of directories to search in
        
    Returns:
        Path to the CSV file if found, None otherwise
    """
    # Try resource manager first for robust search
    try:
        from ..utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        csv_path = resource_manager.find_csv_file(dataset_id, search_dirs)
        if csv_path:
            return str(csv_path)
    except Exception as e:
        logger.debug(f"Resource manager CSV search failed, falling back to legacy method: {e}")
    
    # Fallback to legacy search method
    filename_patterns = [
        f"{dataset_id}_X.csv",
        f"{dataset_id}.csv",
        f"{dataset_id}_data.csv",
        f"{dataset_id}_train.csv",
    ]
    
    # Also check for files with (1), (2), etc. suffixes
    extended_patterns = []
    for pattern in filename_patterns:
        extended_patterns.append(pattern)
        # Add patterns with parentheses suffixes
        base, ext = os.path.splitext(pattern)
        for suffix in ['(1)', '(2)', '(3)']:
            extended_patterns.append(f"{base}{suffix}{ext}")
    
    for search_dir in search_dirs:
        if not search_dir or not os.path.exists(search_dir):
            continue
            
        for pattern in extended_patterns:
            file_path = os.path.join(search_dir, pattern)
            if os.path.exists(file_path):
                logger.info(f"Found CSV file: {file_path}")
                return file_path
                
        # Also try glob patterns in case of different naming conventions
        for base_pattern in [f"{dataset_id}_X*.csv", f"{dataset_id}*.csv"]:
            matches = glob.glob(os.path.join(search_dir, base_pattern))
            if matches:
                # Return the first match, preferring _X files
                x_files = [m for m in matches if '_X' in m and not '_y' in m]
                if x_files:
                    logger.info(f"Found CSV file via glob: {x_files[0]}")
                    return x_files[0]
                else:
                    non_y_files = [m for m in matches if not '_y' in m and not '_Y' in m]
                    if non_y_files:
                        logger.info(f"Found CSV file via glob: {non_y_files[0]}")
                        return non_y_files[0]
    
    return None


def load_csv_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[bool], List[str], str]:
    """
    Load a dataset from a CSV file.
    Handles both single CSV files with features+labels and separate X/y files.

    Args:
        file_path: Path to the CSV file

    Returns:
        X: Features
        y: Labels
        categorical_indicator: Boolean list indicating categorical features
        attribute_names: List of feature names
        dataset_name: Name of the dataset (derived from file name)
    """
    logger.info(f"Loading CSV dataset from: {file_path}")
    
    # Extract dataset name from file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Check if this is an X file with a corresponding y file
    if '_X' in base_name:
        dataset_name = base_name.replace('_X', '')
        y_file_path = file_path.replace('_X.csv', '_y.csv')
        
        if os.path.exists(y_file_path):
            logger.info(f"Loading separate X and y files for dataset: {dataset_name}")
            
            try:
                # Load X and y from separate files
                X = pd.read_csv(file_path, header=None).values
                y = pd.read_csv(y_file_path, header=None).values.ravel()
                
                # Validate that X and y have the same number of samples
                if X.shape[0] != y.shape[0]:
                    logger.warning(f"Mismatch in sample count: X has {X.shape[0]} samples, y has {y.shape[0]} samples")
                    # Trim to the smaller size
                    min_samples = min(X.shape[0], y.shape[0])
                    logger.info(f"Trimming both to {min_samples} samples")
                    X = X[:min_samples]
                    y = y[:min_samples]
                
                # Create default attribute names
                attribute_names = [f"feature_{i}" for i in range(X.shape[1])]
                
                # Determine categorical features (simple heuristic)
                categorical_indicator = []
                for col_idx in range(X.shape[1]):
                    # Consider a column categorical if it has fewer than 10 unique values
                    unique_vals = len(np.unique(X[:, col_idx]))
                    is_categorical = unique_vals < 10
                    categorical_indicator.append(is_categorical)
                
                logger.info(f"Successfully loaded dataset: {dataset_name}")
                logger.info(f"Dataset info: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
                
                return X, y, categorical_indicator, attribute_names, dataset_name
                
            except Exception as e:
                logger.error(f"Error loading separate X/y files: {e}")
                raise ValueError(f"Failed to load dataset from {file_path}: {e}")
    
    # If not a separate X file, try standard loading
    dataset_name = base_name
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Assume the last column is the target
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        
        # Get feature names
        attribute_names = df.columns[:-1].tolist()
        
        # Determine categorical features (simple heuristic)
        categorical_indicator = []
        for col in df.iloc[:, :-1].columns:
            # Consider a column categorical if it has fewer than 10 unique values
            # or if it contains string values
            unique_vals = df[col].nunique()
            is_categorical = unique_vals < 10 or df[col].dtype == 'object'
            categorical_indicator.append(is_categorical)
            
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        logger.info(f"Dataset info: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        return X, y, categorical_indicator, attribute_names, dataset_name
    
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise ValueError(f"Failed to load dataset from {file_path}: {e}")


def load_dataset_with_metadata(dataset_dir: str) -> Dict[str, Any]:
    """
    Load dataset metadata from a directory containing precomputed embeddings.
    
    Args:
        dataset_dir: Directory containing dataset files and metadata
        
    Returns:
        Dictionary containing dataset metadata including ID, name, is_csv flag, etc.
    """
    dataset_info = {}
    
    # Extract dataset ID from directory name
    dataset_id = os.path.basename(dataset_dir).replace("dataset_", "")
    dataset_info['id'] = dataset_id
    dataset_info['embed_dir'] = dataset_dir
    
    # Try to load metadata from dataset_info.json
    dataset_info_file = os.path.join(dataset_dir, "dataset_info.json")
    
    if os.path.exists(dataset_info_file):
        try:
            with open(dataset_info_file, 'r') as f:
                saved_info = json.load(f)
                dataset_info.update(saved_info)
                
        except Exception as e:
            logger.warning(f"Error reading dataset info file: {e}")
    
    # If is_csv is not explicitly set, try to infer it
    if 'is_csv' not in dataset_info:
        dataset_info['is_csv'] = is_csv_dataset(dataset_id)
        
        # Also check for source_type field
        if dataset_info.get('source_type') == 'csv':
            dataset_info['is_csv'] = True
    
    # Default name to ID if not provided
    if 'name' not in dataset_info:
        dataset_info['name'] = dataset_id
    
    # Check for precomputed embeddings
    prefix_data_file = os.path.join(dataset_dir, "prefix_data.npz")
    embedding_files = glob.glob(os.path.join(dataset_dir, "*tabpfn_embeddings*.npz"))
    
    dataset_info['has_embeddings'] = (
        os.path.exists(prefix_data_file) or 
        len(embedding_files) > 0
    )
    
    return dataset_info


def find_csv_with_fallbacks(dataset_id: str, 
                          primary_dir: Optional[str] = None,
                          data_dir: Optional[str] = None,
                          embed_dir: Optional[str] = None) -> Optional[str]:
    """
    Find a CSV file using multiple fallback directories.
    Uses the new resource manager for robust path resolution.
    
    Args:
        dataset_id: The dataset identifier
        primary_dir: Primary directory to search (e.g., current embedding directory)
        data_dir: Data directory specified by user (e.g., --data_dir)
        embed_dir: Embedding directory
        
    Returns:
        Path to the CSV file if found, None otherwise
    """
    try:
        # Use the new resource manager for robust path resolution
        from ..utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        
        # Build additional search directories from the arguments
        additional_dirs = []
        for d in [primary_dir, data_dir, embed_dir]:
            if d:
                additional_dirs.append(d)
                # Also add parent directories
                parent = os.path.dirname(d)
                if parent and parent != d:
                    additional_dirs.append(parent)
        
        # Use resource manager's robust CSV finding
        csv_path = rm.find_csv_file(dataset_id, additional_dirs)
        
        if csv_path:
            return str(csv_path)
    
    except Exception as e:
        logger.debug(f"Resource manager CSV search failed: {e}, falling back to legacy method")
    
    # Fallback to legacy method
    search_dirs = []
    
    # Add directories in order of preference
    if primary_dir:
        search_dirs.append(primary_dir)
        # Also check parent directory
        search_dirs.append(os.path.dirname(primary_dir))
    
    if embed_dir and embed_dir != primary_dir:
        search_dirs.append(embed_dir)
        search_dirs.append(os.path.dirname(embed_dir))
    
    if data_dir:
        search_dirs.append(data_dir)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dirs = []
    for d in search_dirs:
        if d and d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    
    return find_csv_file(dataset_id, unique_dirs)