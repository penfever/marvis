"""
Dataset loading and preparation utilities.
"""

import numpy as np
import logging
import os
import random
import json
import warnings
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from .embeddings import prepare_tabpfn_embeddings_for_prefix

# Utility function to silence specific warnings
def silence_pandas_warning(func, *args, **kwargs):
    """Execute a function with pandas warnings temporarily silenced."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*factorize.*')
        return func(*args, **kwargs)

logger = logging.getLogger(__name__)

# Import the new resource management system
from ..utils.resource_manager import get_resource_manager, DatasetMetadata

# Global cache to store dataset IDs that have failed to load (legacy)
_FAILED_DATASET_CACHE: Set[int] = set()

def _load_failed_dataset_cache() -> Set[int]:
    """Load the cache of failed dataset IDs using the new resource manager."""
    global _FAILED_DATASET_CACHE
    
    try:
        rm = get_resource_manager()
        cache_key = rm.cache_manager.get_cache_key(type='failed_datasets')
        cached_data = rm.cache_manager.load_from_cache('system', cache_key, '.json')
        
        if cached_data and isinstance(cached_data, list):
            _FAILED_DATASET_CACHE = set(int(did) for did in cached_data)
            logger.info(f"Loaded {len(_FAILED_DATASET_CACHE)} failed dataset IDs from managed cache")
            return _FAILED_DATASET_CACHE
    except Exception as e:
        logger.debug(f"Could not load from managed cache: {e}")
    
    # Fallback to legacy cache file
    legacy_cache_path = os.path.expanduser("~/.marvis_failed_datasets.json")
    if os.path.exists(legacy_cache_path):
        try:
            with open(legacy_cache_path, 'r') as f:
                cache_data = json.load(f)
                if isinstance(cache_data, list):
                    _FAILED_DATASET_CACHE = set(int(did) for did in cache_data)
                    logger.info(f"Loaded {len(_FAILED_DATASET_CACHE)} failed dataset IDs from legacy cache")
                    # Migrate to new cache
                    _save_failed_dataset_cache()
                    return _FAILED_DATASET_CACHE
        except Exception as e:
            logger.warning(f"Error loading legacy failed dataset cache: {e}")
    
    # If we couldn't load any cache
    _FAILED_DATASET_CACHE = set()
    return _FAILED_DATASET_CACHE

def _save_failed_dataset_cache() -> None:
    """Save the cache of failed dataset IDs using the new resource manager."""
    try:
        rm = get_resource_manager()
        cache_key = rm.cache_manager.get_cache_key(type='failed_datasets')
        success = rm.cache_manager.save_to_cache('system', cache_key, list(_FAILED_DATASET_CACHE), '.json')
        
        if success:
            logger.info(f"Saved {len(_FAILED_DATASET_CACHE)} failed dataset IDs to managed cache")
        else:
            # Fallback to legacy file
            legacy_cache_path = os.path.expanduser("~/.marvis_failed_datasets.json")
            with open(legacy_cache_path, 'w') as f:
                json.dump(list(_FAILED_DATASET_CACHE), f)
            logger.info(f"Saved {len(_FAILED_DATASET_CACHE)} failed dataset IDs to legacy cache")
    except Exception as e:
        logger.warning(f"Error saving failed dataset cache: {e}")

# Initialize the cache
_load_failed_dataset_cache()

def list_available_datasets() -> Dict[str, int]:
    """
    Returns a dictionary of predefined OpenML dataset names and their IDs.
    These datasets are known to work well with MARVIS.
    
    Returns:
        Dictionary mapping dataset names to their OpenML IDs
    """
    return {
        'airlines': 1169,        # Binary classification, 539k samples, 7 features
        'albert': 189356,        # Multi-class, 425k samples, 79 features
        'volkert': 41166,        # Multi-class, 58k samples, 181 features
        'higgs': 44129,          # Binary classification, 98k samples, 29 features
        'har': 1478,             # Multi-class, 10k samples, 562 features
        'adult': 1590,           # Binary classification, 48k samples, 14 features
        'bank-marketing': 1461,  # Binary classification, 45k samples, 16 features
        'car': 40975,            # Multi-class, 1.7k samples, 21 features
        'connect-4': 40668,      # Multi-class, 67k samples, 42 features
        'credit-g': 31,          # Binary classification, 1k samples, 20 features
        'diabetes': 37,          # Binary classification, 768 samples, 8 features
        'fashion-mnist': 40996,  # Multi-class, 70k samples, 784 features
        'Jannis': 168911,        # Multi-class, 83k samples, 54 features
        'jungle_chess': 167149,  # Multi-class, 44k samples, 6 features
        'mnist': 554,            # Multi-class, 70k samples, 784 features
        'nomao': 1486,           # Binary classification, 34k samples, 118 features
        'phoneme': 1489,         # Binary classification, 5.4k samples, 5 features
        'Shuttle': 40685,        # Multi-class, 43.5k samples, 9 features
        'vehicle': 54,           # Multi-class, 846 samples, 18 features
        'wine-quality-white': 40498  # Multi-class, 4.9k samples, 11 features
    }

def get_dataset_info(dataset_id_or_name: Union[str, int]) -> Dict[str, Any]:
    """
    Get information about a dataset from OpenML without loading the actual data.
    
    Args:
        dataset_id_or_name: ID or name of the dataset to get info about
    
    Returns:
        Dictionary with dataset information including name, ID, feature count,
        sample count, target feature, etc.
    """
    try:
        import openml
    except ImportError:
        raise ImportError("OpenML package is required. Install it with 'pip install openml'.")
    
    # Resolve dataset ID from name if needed
    dataset_id = None
    
    # If input is already an integer, use it directly
    if isinstance(dataset_id_or_name, int):
        dataset_id = dataset_id_or_name
    # If input is a string that looks like a number, convert to int
    elif isinstance(dataset_id_or_name, str) and dataset_id_or_name.isdigit():
        dataset_id = int(dataset_id_or_name)
    # Check if it's a known dataset name
    elif isinstance(dataset_id_or_name, str):
        dataset_ids = list_available_datasets()
        if dataset_id_or_name in dataset_ids:
            dataset_id = dataset_ids[dataset_id_or_name]
        else:
            # Try to search by name using dataframe format
            try:
                datasets_df = openml.datasets.list_datasets(output_format="dataframe", search=dataset_id_or_name)
                if not datasets_df.empty:
                    # Get the first match
                    first_match = datasets_df.iloc[0]
                    dataset_id = first_match['did']
                else:
                    raise ValueError(f"Could not find any dataset matching '{dataset_id_or_name}' on OpenML.")
            except Exception as e:
                logger.error(f"Error searching for dataset: {e}")
                raise ValueError(f"Could not find dataset '{dataset_id_or_name}'. Please provide a valid OpenML dataset ID or name.")
    
    # Now get the dataset info
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        
        # Check if dataset loaded properly
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} could not be loaded from OpenML")
        
        if dataset.features is None:
            logger.warning(f"Dataset {dataset_id} has no features information available")
        
        # Build info dictionary
        info = {
            'id': dataset.dataset_id,
            'name': dataset.name,
            'version': dataset.version,
            'num_features': len(dataset.features) if dataset.features is not None else 0,
            'num_samples': dataset.qualities.get('NumberOfInstances', 'unknown'),
            'num_classes': dataset.qualities.get('NumberOfClasses', 'unknown'),
            'num_missing_values': dataset.qualities.get('NumberOfMissingValues', 'unknown'),
            'target': dataset.default_target_attribute,
            'url': f"https://www.openml.org/d/{dataset.dataset_id}",
            'description': dataset.description[:200] + '...' if dataset.description and len(dataset.description) > 200 else (dataset.description or 'No description available')
        }
        
        # Get class distribution if available
        class_distribution = dataset.qualities.get('ClassDistribution', None)
        if class_distribution:
            try:
                import ast
                # Parse the class distribution string
                distribution = ast.literal_eval(class_distribution)
                info['class_distribution'] = distribution
            except:
                pass
        
        return info
    
    except Exception as e:
        logger.error(f"Error getting dataset info for ID {dataset_id}: {e}")
        raise ValueError(f"Failed to get dataset info for ID {dataset_id}: {e}")

def load_dataset(dataset_name: str, bypass_size_check: bool = False, preserve_regression: bool = False) -> Tuple[np.ndarray, np.ndarray, List[bool], List[str], str]:
    """
    Load a dataset from OpenML. Can load datasets by name or ID.

    Args:
        dataset_name: Name or ID of the dataset to load. If using a name, it can be one of the
                    predefined datasets (use list_available_datasets() to see options)
                    or any other OpenML dataset name. If using an ID, it should be a string representation
                    of the OpenML dataset ID (e.g., '1169' for Airlines).
        bypass_size_check: If True, bypasses the minimum 1000 samples requirement. Default is False.
        preserve_regression: If True, preserves continuous targets for regression tasks instead of 
                           converting them to classification. Default is False.

    Returns:
        X: Features
        y: Labels
        categorical_indicator: Boolean list indicating categorical features
        attribute_names: List of feature names
        dataset_name: Full name of the dataset
    """
    try:
        import openml
    except ImportError:
        raise ImportError("OpenML package is required to load datasets. Install it with 'pip install openml'.")

    # Get dataset mappings from the available datasets function
    dataset_ids = list_available_datasets()

    # Check if dataset_name is an ID
    if dataset_name.isdigit():
        dataset_id = int(dataset_name)
        logger.info(f"Loading dataset with ID {dataset_id} from OpenML")
    # Check if it's a known dataset name
    elif dataset_name in dataset_ids:
        dataset_id = dataset_ids[dataset_name]
        logger.info(f"Loading {dataset_name} dataset (ID: {dataset_id}) from OpenML")
    # Try to find by name
    else:
        try:
            # Search for dataset by name using dataframe format
            datasets_df = openml.datasets.list_datasets(output_format="dataframe", search=dataset_name)
            if not datasets_df.empty:
                # Get the first match
                first_match = datasets_df.iloc[0]
                dataset_id = first_match['did']
                logger.info(f"Found dataset matching '{dataset_name}': ID {dataset_id} - {first_match['name']}")
            else:
                raise ValueError(f"Could not find any dataset matching '{dataset_name}' on OpenML.")
        except Exception as e:
            logger.error(f"Error searching for dataset: {e}")
            raise ValueError(f"Could not find or load dataset '{dataset_name}'. Please provide a valid OpenML dataset name or ID.")
    
    # Check if the dataset ID is in the failed dataset cache
    if dataset_id in _FAILED_DATASET_CACHE:
        logger.warning(f"Dataset ID {dataset_id} has previously failed to load and will be skipped.")
        available_datasets = list_available_datasets()
        suggestion_msg = ""
        
        # Provide a suggestion for alternative datasets
        if len(available_datasets) > 0:
            suggestion_msg = "\n\nTry one of these known working datasets instead:\n"
            # Pick 5 random datasets to suggest
            suggestions = random.sample(list(available_datasets.items()), min(5, len(available_datasets)))
            for name, did in suggestions:
                suggestion_msg += f"  - {name} (ID: {did})\n"
            suggestion_msg += f"\nUse list_available_datasets() to see all {len(available_datasets)} predefined datasets."
        
        raise ValueError(f"Dataset with ID {dataset_id} was previously marked as failed. {suggestion_msg}")
        
    # Pre-check the dataset size if possible to skip small datasets
    try:
        dataset_info = get_dataset_info(dataset_id)
        num_samples = dataset_info.get('num_samples')
        if num_samples != 'unknown' and int(num_samples) < 1000:
            if bypass_size_check:
                logger.warning(f"Dataset ID {dataset_id} has only {num_samples} samples, which is fewer than the minimum 1000 required, but bypassing size check.")
            else:
                logger.warning(f"Dataset ID {dataset_id} has only {num_samples} samples, which is fewer than the minimum 1000 required.")
                # Add to failed dataset cache
                _FAILED_DATASET_CACHE.add(dataset_id)
                _save_failed_dataset_cache()
                available_datasets = list_available_datasets()
                suggestion_msg = ""
                
                # Provide a suggestion for alternative datasets
                if len(available_datasets) > 0:
                    suggestion_msg = "\n\nTry one of these known working datasets instead:\n"
                    # Pick 5 random datasets to suggest
                    suggestions = random.sample(list(available_datasets.items()), min(5, len(available_datasets)))
                    for name, did in suggestions:
                        suggestion_msg += f"  - {name} (ID: {did})\n"
                    suggestion_msg += f"\nUse list_available_datasets() to see all {len(available_datasets)} predefined datasets."
                
                raise ValueError(f"Dataset ID {dataset_id} has only {num_samples} samples (minimum 1000 required). {suggestion_msg}")
    except Exception as e:
        # If we can't check the size beforehand, we'll check after loading
        logger.warning(f"Could not pre-check dataset size: {e}")

    try:
        # Get the dataset by ID
        dataset = openml.datasets.get_dataset(dataset_id)
        logger.info(f"Successfully loaded dataset: {dataset.name} (ID: {dataset_id})")
        
        # First try getting the data without specifying a target
        try:
            # Use dataframe format to handle string values properly
            data_df, _, categorical_indicator, attribute_names = silence_pandas_warning(
                dataset.get_data,
                dataset_format="dataframe", target=None
            )
            
            # Log the default target attribute if it exists
            if dataset.default_target_attribute:
                logger.info(f"Dataset has default target attribute: '{dataset.default_target_attribute}'")
                
                # Check if the default target is actually a column in the dataframe
                if dataset.default_target_attribute in data_df.columns:
                    target_column = dataset.default_target_attribute
                else:
                    # The default target might be specified differently from the actual column name
                    # Use the last column as a fallback
                    target_column = data_df.columns[-1]
                    logger.info(f"Default target not found in columns, using last column '{target_column}' as target")
            else:
                # If no default target is specified, use the last column
                target_column = data_df.columns[-1]
                logger.info(f"No default target attribute specified, using last column '{target_column}' as target")
            
            # Extract features and target
            y_series = data_df[target_column]
            
            # Handle categorical dtypes in target
            if hasattr(y_series.dtype, 'name') and 'category' in str(y_series.dtype).lower():
                logger.info(f"Converting categorical target to numeric codes")
                # Convert to category codes which are numeric
                y_series = y_series.astype('category').cat.codes
            
            # Convert target to numpy array
            y = y_series.values
            
            # Fix sparse type issues in the entire dataframe before splitting
            # This is specifically to handle the Sparse[float32, 0] type
            sparse_columns = []
            for col in data_df.columns:
                if 'sparse' in str(data_df[col].dtype).lower():
                    sparse_columns.append(col)
            
            if sparse_columns:
                logger.info(f"Found {len(sparse_columns)} sparse columns in the dataset: {', '.join(sparse_columns[:5])}{'...' if len(sparse_columns) > 5 else ''}")
                
                # Process all sparse columns at dataframe level for better efficiency
                for col in sparse_columns:
                    try:
                        # Try to get dense version directly
                        if hasattr(data_df[col], 'sparse') and hasattr(data_df[col].sparse, 'to_dense'):
                            data_df[col] = pd.Series(data_df[col].sparse.to_dense(), index=data_df.index)
                        else:
                            # Direct conversion for Sparse[float32, 0] type
                            try:
                                # Try explicit conversion with pandas API - already imported at top
                                dense_series = pd.Series([float(x) for x in data_df[col]], index=data_df.index)
                                data_df[col] = dense_series
                            except Exception as err:
                                logger.warning(f"Direct sparse conversion failed for column '{col}': {err}. Will try element-wise later.")
                    except Exception as e:
                        logger.warning(f"Issue pre-converting sparse column '{col}': {e}. Will try different method during feature processing.")
            
            # Drop target column and prepare features
            X_df = data_df.drop(columns=[target_column])
            
            # Handle problematic column types (categorical, sparse, etc.)
            for col in X_df.columns:
                col_dtype = X_df[col].dtype
                col_dtype_str = str(col_dtype).lower()
                
                # Handle categorical columns
                if hasattr(col_dtype, 'name') and 'category' in col_dtype_str:
                    logger.info(f"Converting categorical column '{col}' to numeric codes")
                    X_df[col] = pd.Series(X_df[col].astype('category').cat.codes, index=X_df.index)
                
                # Handle sparse columns
                elif 'sparse' in col_dtype_str:
                    logger.info(f"Converting sparse column '{col}' to dense")
                    try:
                        # First check if it's a pandas SparseDtype
                        if hasattr(X_df[col], 'sparse'):
                            try:
                                # Try standard pandas sparse conversion with pd.Series to preserve dtype
                                X_df[col] = pd.Series(X_df[col].sparse.to_dense(), index=X_df.index)
                            except Exception as err1:
                                logger.warning(f"Error using sparse.to_dense() for '{col}': {err1}. Trying alternative approach.")
                                # Try converting manually from sparse array
                                if hasattr(X_df[col], '_values') and hasattr(X_df[col], '_indices'):
                                    # Recreate values for a standard CSR-like format
                                    dense_vals = np.zeros(len(X_df[col]))
                                    dense_vals[X_df[col]._indices] = X_df[col]._values
                                    X_df[col] = pd.Series(dense_vals, index=X_df.index)
                                else:
                                    # Final fallback - try forcing conversion through numpy array
                                    X_df[col] = pd.Series(np.array(X_df[col].tolist(), dtype=np.float32), index=X_df.index)
                        else:
                            # Not a pandas sparse type - might be scipy sparse or a custom sparse format
                            try:
                                # Try to handle scipy sparse format
                                import scipy.sparse as sp
                                if sp.issparse(X_df[col].iloc[0]):
                                    # If each element is a separate sparse matrix (unusual case)
                                    dense_list = [x.toarray()[0,0] if x.shape == (1,1) else x.toarray().mean() 
                                                 for x in X_df[col]]
                                    X_df[col] = pd.Series(np.array(dense_list, dtype=np.float32), index=X_df.index)
                                else:
                                    # Try general conversion to numpy array
                                    X_df[col] = pd.Series(np.array(X_df[col].tolist(), dtype=np.float32), index=X_df.index)
                            except Exception as err2:
                                # Last resort - just assign zeros
                                logger.warning(f"Failed to convert sparse column '{col}' using scipy: {err2}. Using zeros.")
                                X_df[col] = pd.Series(np.zeros(len(X_df)), index=X_df.index)
                    except Exception as sparse_err:
                        logger.warning(f"All sparse conversion methods failed for '{col}': {sparse_err}. Using zeros instead.")
                        X_df[col] = pd.Series(np.zeros(len(X_df)), index=X_df.index)
                
                # Handle other problematic dtypes
                elif not np.issubdtype(col_dtype, np.number) and not np.issubdtype(col_dtype, np.bool_):
                    logger.warning(f"Converting column '{col}' with dtype {col_dtype} to string and then numeric codes")
                    try:
                        # Convert to string then use label encoding
                        X_df[col] = X_df[col].astype(str)
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        X_df[col] = pd.Series(le.fit_transform(X_df[col]), index=X_df.index)
                    except Exception as conv_err:
                        logger.warning(f"Error converting column '{col}': {conv_err}. Using zeros instead.")
                        X_df[col] = pd.Series(np.zeros(len(X_df)), index=X_df.index)
            
            # Convert to numpy array
            X = X_df.values
            
            # Adjust categorical_indicator and attribute_names to match the features (exclude target column)
            if categorical_indicator is not None and len(categorical_indicator) > X.shape[1]:
                # Find the target column index in the original dataframe
                target_index = data_df.columns.get_loc(target_column)
                # Remove the categorical indicator for the target column
                categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if i != target_index]
                logger.debug(f"Adjusted categorical_indicator length from {len(categorical_indicator)+1} to {len(categorical_indicator)}")
            
            # Update attribute_names to exclude target column 
            if attribute_names is not None and len(attribute_names) > X.shape[1]:
                # Find the target column index in attribute_names
                if target_column in attribute_names:
                    attribute_names = [name for name in attribute_names if name != target_column]
                    logger.debug(f"Removed target column '{target_column}' from attribute_names")
                elif len(attribute_names) == len(data_df.columns):
                    # If target is last column, remove last item
                    attribute_names = attribute_names[:-1]
                    logger.debug(f"Removed last column from attribute_names (assuming it was target)")
            
        except Exception as e:
            logger.warning(f"Error getting data in dataframe format: {e}. Trying alternative approach.")
            
            # Fall back to array format if dataframe format fails
            try:
                X, y, categorical_indicator, attribute_names = silence_pandas_warning(
                    dataset.get_data,
                    dataset_format="array", target=dataset.default_target_attribute
                )
                
                # Handle special cases for sparse arrays that might be in the data
                sparse_in_array = False
                if isinstance(X, np.ndarray):
                    # Check for sparse elements in the array
                    try:
                        sparse_check = str(X.dtype).lower()
                        if 'sparse' in sparse_check:
                            sparse_in_array = True
                            logger.warning(f"Detected sparse dtype in numpy array: {X.dtype}")
                    except:
                        # If dtype access fails, check individual elements
                        try:
                            for i in range(min(10, X.shape[0])):
                                for j in range(min(10, X.shape[1])):
                                    if 'sparse' in str(type(X[i, j])).lower():
                                        sparse_in_array = True
                                        logger.warning(f"Detected sparse elements in numpy array at position [{i},{j}]: {type(X[i, j])}")
                                        break
                                if sparse_in_array:
                                    break
                        except:
                            pass
                
                # Special handling for sparse arrays
                if sparse_in_array:
                    logger.info("Converting sparse elements in numpy array to dense values")
                    try:
                        # Create a new array with dense values
                        X_dense = np.zeros_like(X, dtype=np.float32)
                        
                        # Loop through elements and convert sparse to dense
                        import scipy.sparse as sp
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    val = X[i, j]
                                    # Handle different sparse formats
                                    if sp.issparse(val):
                                        X_dense[i, j] = val.toarray()[0, 0] if val.shape == (1, 1) else val.mean()
                                    elif hasattr(val, 'sparse') and hasattr(val.sparse, 'to_dense'):
                                        X_dense[i, j] = val.sparse.to_dense()
                                    elif hasattr(val, '_values') and hasattr(val, '_indices'):
                                        # Manually extract from COO-like format
                                        X_dense[i, j] = val._values[0] if len(val._values) > 0 else 0
                                    else:
                                        # Try direct float conversion
                                        try:
                                            X_dense[i, j] = float(val)
                                        except:
                                            X_dense[i, j] = 0
                                except Exception as elem_err:
                                    logger.debug(f"Error converting element at [{i},{j}]: {elem_err}")
                                    X_dense[i, j] = 0
                        X = X_dense
                    except Exception as conv_err:
                        logger.warning(f"Error during sparse array conversion: {conv_err}")
                
                # Ensure X is a proper numpy array with numeric types
                if not isinstance(X, np.ndarray) or not np.issubdtype(X.dtype, np.number):
                    logger.warning(f"Converting X to numeric array. Original type: {type(X)}, dtype: {X.dtype if hasattr(X, 'dtype') else 'unknown'}")
                    # Try to convert to float array, handling potential issues
                    try:
                        X = np.array(X, dtype=np.float32)
                    except Exception as arr_err:
                        logger.warning(f"Error converting X to float array: {arr_err}")
                        # More aggressive approach: element-wise conversion
                        X_new = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    value = X[i, j]
                                    # Handle various types
                                    if isinstance(value, (int, float, bool, np.number)):
                                        X_new[i, j] = float(value)
                                    elif isinstance(value, str) and value.strip() and value.strip().lower() != 'nan':
                                        try:
                                            X_new[i, j] = float(value)
                                        except:
                                            X_new[i, j] = 0
                                    else:
                                        X_new[i, j] = 0
                                except:
                                    X_new[i, j] = 0
                        X = X_new
                
                # Ensure y is a proper numpy array
                if not isinstance(y, np.ndarray):
                    logger.warning(f"Converting y to numpy array. Original type: {type(y)}")
                    try:
                        # For classification, try to convert to integers
                        if len(np.unique(y)) < 100:  # Assume classification if less than 100 unique values
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        else:
                            y = np.array(y, dtype=np.float32)
                    except Exception as y_err:
                        logger.warning(f"Error converting y: {y_err}")
                        raise
            except Exception as array_err:
                logger.warning(f"Error getting data in array format: {array_err}")
                raise ValueError(f"Could not load dataset {dataset_id} in either dataframe or array format")
        
        # Check if target is continuous (regression) and convert to classification by binning
        # Only do this if preserve_regression is False
        if not preserve_regression and np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 10:
            logger.info(f"Converting continuous target to classification by binning")
            from sklearn.preprocessing import KBinsDiscretizer
            # Use quantile binning to create balanced classes with safety checks
            n_unique_targets = len(np.unique(y))
            n_samples = len(y)
            # Ensure safe quantile computation: need at least 2 samples per bin
            max_safe_bins = min(10, n_samples // 2, n_unique_targets)
            
            if max_safe_bins < 2:
                logger.warning(f"Too few samples ({n_samples}) or unique targets ({n_unique_targets}) for quantile binning, using uniform strategy")
                discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
                n_bins = 2
            else:
                discretizer = KBinsDiscretizer(n_bins=max_safe_bins, encode='ordinal', strategy='quantile')
                n_bins = max_safe_bins
            
            y = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
            logger.info(f"Binned continuous target into {n_bins} classes using {discretizer.strategy} strategy")
        elif preserve_regression and np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 10:
            logger.info(f"Preserving continuous target for regression task (preserve_regression=True)")
        
        # Clean NaN and inf values from dataset
        logger.info("Cleaning NaN and inf values from dataset")
        
        # Clean target variable y
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            before_count = len(y)
            mask_y = ~(np.isnan(y) | np.isinf(y))
            X = X[mask_y]
            y = y[mask_y] 
            after_count = len(y)
            logger.warning(f"Removed {before_count - after_count} samples with NaN/inf in target variable")
        
        # Clean feature matrix X  
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            before_count = len(X)
            mask_X = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[mask_X]
            y = y[mask_X]
            after_count = len(X) 
            logger.warning(f"Removed {before_count - after_count} samples with NaN/inf in features")
        
        logger.info(f"Final dataset size after cleaning: {len(X)} samples")
        
        # Log basic dataset information
        num_features = X.shape[1]
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))
        num_categorical = sum(categorical_indicator) if categorical_indicator else "unknown"
        num_numerical = num_features - num_categorical if categorical_indicator else "unknown"
        
        logger.info(f"Dataset info: {num_samples} samples, {num_features} features "
                    f"({num_categorical} categorical, {num_numerical} numerical), "
                    f"{num_classes} classes")
        
        # Check if dataset has enough samples (minimum 1000)
        if num_samples < 1000:
            if bypass_size_check:
                logger.warning(f"Dataset ID {dataset_id} has only {num_samples} samples, which is fewer than the minimum 1000 required, but bypassing size check.")
            else:
                logger.warning(f"Dataset ID {dataset_id} has only {num_samples} samples, which is fewer than the minimum 1000 required.")
                # Add to failed dataset cache
                _FAILED_DATASET_CACHE.add(dataset_id)
                _save_failed_dataset_cache()
                available_datasets = list_available_datasets()
                suggestion_msg = ""
                
                # Provide a suggestion for alternative datasets
                if len(available_datasets) > 0:
                    suggestion_msg = "\n\nTry one of these known working datasets instead:\n"
                    # Pick 5 random datasets to suggest
                    suggestions = random.sample(list(available_datasets.items()), min(5, len(available_datasets)))
                    for name, did in suggestions:
                        suggestion_msg += f"  - {name} (ID: {did})\n"
                    suggestion_msg += f"\nUse list_available_datasets() to see all {len(available_datasets)} predefined datasets."
                
                raise ValueError(f"Dataset ID {dataset_id} has only {num_samples} samples (minimum 1000 required). {suggestion_msg}")
        
        # Register the successfully loaded dataset
        try:
            rm = get_resource_manager()
            
            # Check if dataset is already registered
            existing = rm.dataset_registry.find_dataset_by_name(dataset.name)
            if not existing:
                # Resolve OpenML identifiers to get proper task_id
                identifiers = rm.resolve_openml_identifiers(dataset_id=dataset_id)
                resolved_task_id = identifiers.get('task_id')
                
                # Create metadata for the dataset
                metadata = DatasetMetadata(
                    id=str(dataset_id),
                    name=dataset.name,
                    source_type='openml',
                    openml_task_id=resolved_task_id,  # Use resolved task_id, not dataset_id
                    num_samples=X.shape[0],
                    num_features=X.shape[1],
                    num_classes=len(np.unique(y)),
                    task_type='regression' if preserve_regression and np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 10 else 'classification'
                )
                
                rm.dataset_registry.register_dataset(metadata)
                logger.debug(f"Registered dataset {dataset.name} in resource manager")
        except Exception as e:
            logger.debug(f"Could not register dataset in resource manager: {e}")
        
        return X, y, categorical_indicator, attribute_names, dataset.name
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {e}")
        # Add the failed dataset ID to the cache
        _FAILED_DATASET_CACHE.add(dataset_id)
        _save_failed_dataset_cache()
        logger.info(f"Added dataset ID {dataset_id} to failed dataset cache")
        
        available_datasets = list_available_datasets()
        suggestion_msg = ""
        
        # Provide a suggestion for alternative datasets
        if len(available_datasets) > 0:
            suggestion_msg = "\n\nTry one of these known working datasets instead:\n"
            # Pick 5 random datasets to suggest
            suggestions = random.sample(list(available_datasets.items()), min(5, len(available_datasets)))
            for name, did in suggestions:
                suggestion_msg += f"  - {name} (ID: {did})\n"
            suggestion_msg += f"\nUse list_available_datasets() to see all {len(available_datasets)} predefined datasets."
        
        raise ValueError(f"Failed to load dataset with ID {dataset_id}: {e}{suggestion_msg}")


def clear_failed_dataset_cache() -> None:
    """Clear the cache of failed dataset IDs."""
    global _FAILED_DATASET_CACHE
    _FAILED_DATASET_CACHE.clear()
    _save_failed_dataset_cache()
    logger.info("Cleared the failed dataset cache")


def get_failed_dataset_ids() -> List[int]:
    """Get the list of dataset IDs that have failed to load."""
    return list(_FAILED_DATASET_CACHE)


def analyze_dataset(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[Dict, Dict, Dict]:
    """
    Analyze the class distribution in the dataset splits.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        train_counts: Class distribution in training set
        val_counts: Class distribution in validation set
        test_counts: Class distribution in test set
    """
    logger.info("Analyzing class distribution in dataset splits:")
    
    # Get unique classes
    all_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
    
    # Check train set
    train_counts = {}
    for cls in all_classes:
        count = np.sum(y_train == cls)
        train_counts[cls] = count
    
    # Check val set
    val_counts = {}
    for cls in all_classes:
        count = np.sum(y_val == cls)
        val_counts[cls] = count
    
    # Check test set
    test_counts = {}
    for cls in all_classes:
        count = np.sum(y_test == cls)
        test_counts[cls] = count
    
    logger.info(f"Train set class distribution: {train_counts}")
    logger.info(f"Val set class distribution: {val_counts}")
    logger.info(f"Test set class distribution: {test_counts}")
    
    return train_counts, val_counts, test_counts


def create_llm_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    tokenizer: Any,
    prefix_start_id: int,
    prefix_end_id: int,
    class_token_ids: List[int],
    output_dir: Optional[str] = None,
    num_few_shot_examples: int = 100,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset, Dataset, LabelEncoder, str]:
    """
    Create LLM datasets with TabPFN embeddings as prefix.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        train_embeddings: TabPFN embeddings for training set
        val_embeddings: TabPFN embeddings for validation set
        test_embeddings: TabPFN embeddings for test set
        tokenizer: Tokenizer for the LLM
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
        output_dir: Directory to save prefix data
        num_few_shot_examples: Number of few-shot examples to include in the prefix
                              (configurable at inference time)
        max_train_samples: Maximum number of training samples to use (None for no limit)
        max_test_samples: Maximum number of test and validation samples to use (None for no limit)

    Returns:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        label_encoder: Label encoder for class labels
        prefix_data_file: Path to the saved prefix data
    """
    logger.info("Creating datasets for LLM training and evaluation")
    
    # Ensure all label arrays are integers for classification
    # TabPFN and the LLM label encoding work expects discrete classes
    def ensure_integer_labels(y_array):
        if not isinstance(y_array, np.ndarray):
            y_array = np.array(y_array)
        
        # If label type is float but all values are integers, convert to int
        if y_array.dtype.kind == 'f':
            unique_vals = np.unique(y_array)
            if len(unique_vals) <= 100 and all(float(val).is_integer() for val in unique_vals[:100]):
                logger.info(f"Converting float labels to integers for LLM dataset")
                return y_array.astype(int)
        
        # Otherwise leave as is - the label encoder will handle it
        return y_array

    # Ensure all labels are in proper format
    y_train = ensure_integer_labels(y_train)
    y_val = ensure_integer_labels(y_val)
    y_test = ensure_integer_labels(y_test)
    
    # Check for continuous labels (rare case, should've been binned earlier)
    if any(arr.dtype.kind == 'f' for arr in [y_train, y_val, y_test]):
        logger.warning("Detected float labels in dataset, will be encoded as distinct classes")
    
    # Encode labels to ensure consistency between TabPFN and LLM
    label_encoder = LabelEncoder()
    try:
        label_encoder.fit(np.concatenate([y_train, y_val, y_test]))
    except Exception as e:
        logger.error(f"Error encoding labels: {e}")
        # Fall back approach if concatenation fails
        unique_labels = set()
        for y_arr in [y_train, y_val, y_test]:
            unique_labels.update(np.unique(y_arr))
        label_encoder.fit(np.array(list(unique_labels)))
        logger.info(f"Used fallback label encoding with {len(unique_labels)} unique classes")
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Log label encoding information
    logger.info(f"Encoded {len(label_encoder.classes_)} unique classes")
    
    # Verify we have enough class tokens
    num_classes = len(label_encoder.classes_)
    if num_classes > len(class_token_ids):
        raise ValueError(f"Not enough class tokens for {num_classes} classes (have {len(class_token_ids)})")
    
    logger.info(f"TabPFN embedding shapes - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
    
    # Prepare prefix embeddings from train set with samples from all classes
    # This returns both embeddings and their corresponding labels
    prefix_embeddings, prefix_class_labels = prepare_tabpfn_embeddings_for_prefix(
        train_embeddings, y_train_encoded, num_embeddings=num_few_shot_examples, random_state=42
    )
    
    logger.info(f"Prepared prefix data - Embeddings: {prefix_embeddings.shape}, Class labels: {prefix_class_labels.shape}")
    
    # Save prefix embeddings and class labels
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    prefix_data_file = os.path.join(output_dir, "prefix_data.npz")
    np.savez(prefix_data_file, embeddings=prefix_embeddings, class_labels=prefix_class_labels)
    logger.info(f"Saved prefix data to {prefix_data_file}")
    
    # Check that length of embeddings and labels match
    if len(val_embeddings) != len(y_val_encoded) or len(test_embeddings) != len(y_test_encoded):
        logger.warning(f"Mismatch between embeddings and labels: val embeddings: {len(val_embeddings)}, val labels: {len(y_val_encoded)}, "
                       f"test embeddings: {len(test_embeddings)}, test labels: {len(y_test_encoded)}")
        
        # Trim the encoded label arrays to match the embedding lengths
        val_length = len(val_embeddings)
        test_length = len(test_embeddings)
        
        # Ensure we don't try to access more elements than available
        if val_length > len(y_val_encoded):
            val_length = len(y_val_encoded)
        
        if test_length > len(y_test_encoded):
            test_length = len(y_test_encoded)
            
        # Use the trimmed lengths
        y_val_encoded = y_val_encoded[:val_length]
        y_test_encoded = y_test_encoded[:test_length]
        
        logger.info(f"Aligned label and embedding counts: now using {val_length} validation samples and {test_length} test samples")
    else:
        val_length = len(val_embeddings)
        test_length = len(test_embeddings)
    
    # Apply max_test_samples limit to validation set if specified
    if max_test_samples is not None and val_length > max_test_samples:
        logger.info(f"Limiting validation set from {val_length} to {max_test_samples} samples")
        val_length = max_test_samples
    
    # Apply max_test_samples limit to test set if specified
    if max_test_samples is not None and test_length > max_test_samples:
        logger.info(f"Limiting test set from {test_length} to {max_test_samples} samples")
        test_length = max_test_samples
    
    # Create validation examples for training
    val_examples = []
    for i in range(len(y_val_encoded)):  # Use the length of y_val_encoded directly
        val_examples.append({
            "label_id": int(y_val_encoded[i]),  # Use numeric label ID instead of string
            "query_embedding": val_embeddings[i].tolist()
        })
    
    # Create test examples
    test_examples = []
    for i in range(len(y_test_encoded)):  # Use the length of y_test_encoded directly
        test_examples.append({
            "label_id": int(y_test_encoded[i]),
            "query_embedding": test_embeddings[i].tolist()
        })
    
    # Split validation examples into train and eval
    train_val_split = int(0.8*len(val_examples))
    train_dataset = val_examples[:train_val_split]
    eval_dataset = val_examples[train_val_split:]
    
    # Apply max_train_samples limit if specified
    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        logger.info(f"Limiting training set from {len(train_dataset)} to {max_train_samples} samples")
        train_dataset = train_dataset[:max_train_samples]
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_examples)}")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_dict({
        "label_id": [item["label_id"] for item in train_dataset],
        "query_embedding": [item["query_embedding"] for item in train_dataset]
    })
    
    eval_dataset = Dataset.from_dict({
        "label_id": [item["label_id"] for item in eval_dataset],
        "query_embedding": [item["query_embedding"] for item in eval_dataset]
    })
    
    test_dataset = Dataset.from_dict({
        "label_id": [item["label_id"] for item in test_examples],
        "query_embedding": [item["query_embedding"] for item in test_examples]
    })
    
    return train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file


def load_datasets(args):
    """Load datasets based on command line arguments."""
    import glob
    
    logger = logging.getLogger(__name__)
    datasets = []

    # Clear the _FAILED_DATASET_CACHE to always try loading the dataset
    clear_failed_dataset_cache()
    logger.info("Cleared _FAILED_DATASET_CACHE to ensure dataset loading is attempted")
    
    # Case 1: Single dataset by name
    if hasattr(args, 'dataset_name') and args.dataset_name:
        logger.info(f"Loading single dataset: {args.dataset_name}")
        try:
            preserve_regression = getattr(args, 'preserve_regression', False)
            X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(args.dataset_name, bypass_size_check=True, preserve_regression=preserve_regression)
            datasets.append({
                "id": args.dataset_name,
                "name": dataset_name,
                "X": X,
                "y": y,
                "categorical_indicator": categorical_indicator,
                "attribute_names": attribute_names
            })
            logger.info(f"Successfully loaded dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {args.dataset_name}: {e}")
    
    # Case 2: Multiple datasets by ID
    elif hasattr(args, 'dataset_ids') and args.dataset_ids:
        dataset_ids = [id.strip() for id in args.dataset_ids.split(',')]
        logger.info(f"Loading datasets with IDs: {dataset_ids}")
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
                
    # Case 2b: Multiple datasets by task ID
    elif hasattr(args, 'task_ids') and args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(',')]
        logger.info(f"Loading datasets with task IDs: {task_ids}")
        preserve_regression = getattr(args, 'preserve_regression', False)
        
        # Get resource manager to resolve task IDs to dataset IDs
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
                    "id": dataset_id,
                    "name": dataset_name,
                    "task_id": task_id,  # Include task_id in the dataset dict
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id})")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
    
    # Case 3: Load from directory of CSV files
    elif hasattr(args, 'data_dir') and args.data_dir:
        from . import load_csv_dataset  # Import here to avoid circular imports
        
        logger.info(f"Loading datasets from directory: {args.data_dir}")
        csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory {args.data_dir}")
            
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Filter to avoid double loading when we have separate X and y files
        x_files = [f for f in csv_files if '_X' in f or '_x' in f]
        y_files = [f for f in csv_files if '_y' in f or '_Y' in f]
        non_xy_files = [f for f in csv_files if f not in x_files and f not in y_files]
        
        # Use X files if available, otherwise use regular CSV files
        files_to_process = x_files if x_files else non_xy_files
        
        logger.info(f"Processing {len(files_to_process)} dataset files")
        
        for file_path in files_to_process:
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_csv_dataset(file_path)
                datasets.append({
                    "id": dataset_name,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names,
                    "is_csv": True
                })
                logger.info(f"Successfully loaded dataset {dataset_name} from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset from {file_path}: {e}")
    
    # Case 4: Sample random datasets from OpenML
    elif hasattr(args, 'num_datasets') and args.num_datasets:
        logger.info(f"Sampling {args.num_datasets} random datasets from OpenML")
        
        try:
            import openml
            
            # Use a curated list of known working datasets
            benchmark_datasets = [
                1590,   # adult (binary)
                1461,   # bank-marketing (binary)
                40975,  # car (4 classes)
                31,     # credit-g (binary)
                37,     # diabetes (binary)
                54,     # vehicle (4 classes)
                1489,   # phoneme (binary)
                40498,  # wine-quality-white (7 classes)
                40701,  # australian (binary)
                40981,  # kr-vs-kp (binary)
            ]
            
            # Create a dedicated random generator with the specified seed
            rng = random.Random(getattr(args, 'seed', 42))
            
            # Sample from benchmark datasets if we have enough
            if len(benchmark_datasets) >= args.num_datasets:
                dataset_ids = rng.sample(benchmark_datasets, args.num_datasets)
                logger.info(f"Sampled {len(dataset_ids)} datasets from benchmark set")
            else:
                dataset_ids = benchmark_datasets.copy()
                logger.warning(f"Using all {len(dataset_ids)} benchmark datasets (less than requested {args.num_datasets})")
            
            # Now try to load each dataset
            preserve_regression = getattr(args, 'preserve_regression', False)
            for dataset_id in dataset_ids:
                try:
                    X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(dataset_id), bypass_size_check=True, preserve_regression=preserve_regression)
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
        
        except ImportError:
            logger.error("OpenML package not found. Please install it with 'pip install openml'.")
            return []
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets