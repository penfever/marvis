"""
Time Series Data Loading and Processing for MARVIS.

This module provides interfaces for loading time series data from the gift-eval 
benchmark and processing it for time series distribution classification.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """
    Wrapper for gift-eval datasets to work with MARVIS time series classification.
    
    This class adapts the gift-eval Dataset interface to provide time series data
    in a format suitable for MARVIS time series distribution visualization.
    """
    
    def __init__(
        self,
        name: str,
        term: str = "short",
        gift_eval_path: Optional[str] = None,
        to_univariate: Optional[bool] = None
    ):
        """
        Initialize time series dataset.
        
        Args:
            name: Dataset name (e.g., 'm4_weekly', 'electricity/H')
            term: Evaluation term ('short', 'medium', 'long')
            gift_eval_path: Path to gift-eval installation
            to_univariate: Whether to convert multivariate to univariate.
                          If None, automatically detect based on dataset dimensionality.
        """
        self.name = name
        self.term = term
        
        # Try to import gift-eval
        if gift_eval_path:
            sys.path.insert(0, gift_eval_path)
        
        try:
            from gift_eval.data import Dataset as GiftEvalDataset
            
            # Auto-detect univariate requirement if not specified
            if to_univariate is None:
                # First, check target dimensionality without transformation
                temp_dataset = GiftEvalDataset(name=name, term=term, to_univariate=False)
                target_dim = temp_dataset.target_dim
                to_univariate = target_dim > 1
                logger.info(f"Auto-detected to_univariate={to_univariate} for {name} (target_dim={target_dim})")
            
            self.to_univariate = to_univariate
            self._gift_eval_dataset = GiftEvalDataset(
                name=name,
                term=term,
                to_univariate=to_univariate
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import gift-eval. Please ensure gift-eval is installed and "
                f"GIFT_EVAL environment variable is set. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load gift-eval dataset {name}: {e}")
        
        # Extract dataset properties
        self.prediction_length = self._gift_eval_dataset.prediction_length
        self.freq = self._gift_eval_dataset.freq
        self.target_dim = self._gift_eval_dataset.target_dim
        self._dataset_info = self._extract_dataset_info()
    
    def _extract_dataset_info(self) -> Dict[str, Any]:
        """Extract dataset information for metadata tracking."""
        return {
            'name': self.name,
            'term': self.term,
            'prediction_length': self.prediction_length,
            'frequency': self.freq,
            'target_dim': self.target_dim,
            'to_univariate': self.to_univariate
        }
    
    @property
    def dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return self._dataset_info
    
    def get_train_data(self) -> List[Dict[str, Any]]:
        """
        Get training data from gift-eval dataset.
        
        Returns:
            List of time series dictionaries with 'target' and 'start' keys
        """
        try:
            # Access the underlying gluonts_dataset directly
            return list(self._gift_eval_dataset.gluonts_dataset)
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Get test data from gift-eval dataset.
        
        Returns:
            List of time series dictionaries with 'target' and 'start' keys
        """
        try:
            # For test data, we also use the gluonts_dataset but may need different handling
            # Let's use the same as training data for now
            return list(self._gift_eval_dataset.gluonts_dataset)
        except Exception as e:
            logger.error(f"Failed to get test data: {e}")
            return []
    
    def get_validation_data(self) -> List[Dict[str, Any]]:
        """
        Get validation data from gift-eval dataset.
        
        Returns:
            List of time series dictionaries with 'target' and 'start' keys
        """
        try:
            return list(self._gift_eval_dataset.validation_dataset)
        except Exception as e:
            logger.error(f"Failed to get validation data: {e}")
            return []
    
    def extract_time_series_array(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract time series values as numpy array.
        
        Args:
            data: List of time series dictionaries from gift-eval
            
        Returns:
            Array of shape [n_series, max_length] with time series values
        """
        if not data:
            return np.array([]).reshape(0, 0)
        
        # Extract target values
        series_list = []
        for item in data:
            target = item.get('target', [])
            if isinstance(target, (list, tuple)):
                series_list.append(np.array(target))
            elif isinstance(target, np.ndarray):
                series_list.append(target)
            elif isinstance(target, (np.float32, np.float64, float, int)):
                # Handle scalar values - convert to single-element array
                series_list.append(np.array([target]))
            else:
                logger.warning(f"Unexpected target type: {type(target)}, value: {target}")
                # Try to convert to array anyway
                try:
                    series_list.append(np.array(target))
                except:
                    continue
        
        if not series_list:
            return np.array([]).reshape(0, 0)
        
        # Handle different length series by padding
        max_length = max(len(series) for series in series_list)
        n_series = len(series_list)
        
        # Create padded array
        result = np.full((n_series, max_length), np.nan)
        for i, series in enumerate(series_list):
            result[i, :len(series)] = series
        
        return result
    
    def prepare_for_marvis(self, split: str = 'train') -> Dict[str, Any]:
        """
        Prepare dataset for MARVIS time series classification.
        
        Args:
            split: Which split to use ('train', 'test', 'validation')
            
        Returns:
            Dictionary with processed data for MARVIS
        """
        if split == 'train':
            data = self.get_train_data()
        elif split == 'test':
            data = self.get_test_data()
        elif split == 'validation':
            data = self.get_validation_data()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Extract time series arrays
        series_array = self.extract_time_series_array(data)
        
        if series_array.size == 0:
            logger.warning(f"No data found for split {split}")
            return {
                'series_data': np.array([]).reshape(0, 0),
                'n_series': 0,
                'max_length': 0,
                'dataset_info': self.dataset_info,
                'raw_data': data
            }
        
        # For univariate, we might want to aggregate multiple series or select one
        if self.to_univariate and series_array.shape[0] > 1:
            # Take the first series or aggregate
            logger.info(f"Converting {series_array.shape[0]} series to univariate using first series")
            series_data = series_array[0:1, :]  # Keep 2D shape
        else:
            series_data = series_array
        
        return {
            'series_data': series_data,
            'n_series': series_data.shape[0],
            'max_length': series_data.shape[1],
            'dataset_info': self.dataset_info,
            'raw_data': data,
            'prediction_length': self.prediction_length
        }


def load_gift_eval_dataset(
    dataset_name: str,
    term: str = "short",
    gift_eval_path: Optional[str] = None,
    to_univariate: Optional[bool] = None
) -> TimeSeriesDataset:
    """
    Load a gift-eval dataset for time series processing.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'm4_weekly', 'electricity/H')
        term: Evaluation term ('short', 'medium', 'long')
        gift_eval_path: Path to gift-eval installation (optional if in PYTHONPATH)
        to_univariate: Whether to convert multivariate to univariate.
                      If None, automatically detect based on dataset dimensionality.
        
    Returns:
        TimeSeriesDataset instance
    """
    return TimeSeriesDataset(
        name=dataset_name,
        term=term,
        gift_eval_path=gift_eval_path,
        to_univariate=to_univariate
    )


def load_multiple_gift_eval_datasets(
    dataset_names: List[str],
    terms: Optional[List[str]] = None,
    gift_eval_path: Optional[str] = None,
    to_univariate: Optional[bool] = None
) -> List[TimeSeriesDataset]:
    """
    Load multiple gift-eval datasets.
    
    Args:
        dataset_names: List of dataset names
        terms: List of terms for each dataset (if None, uses 'short' for all)
        gift_eval_path: Path to gift-eval installation
        to_univariate: Whether to convert multivariate to univariate.
                      If None, automatically detect based on dataset dimensionality.
        
    Returns:
        List of TimeSeriesDataset instances
    """
    if terms is None:
        terms = ['short'] * len(dataset_names)
    elif len(terms) != len(dataset_names):
        raise ValueError("Number of terms must match number of dataset names")
    
    datasets = []
    for name, term in zip(dataset_names, terms):
        try:
            dataset = load_gift_eval_dataset(
                dataset_name=name,
                term=term,
                gift_eval_path=gift_eval_path,
                to_univariate=to_univariate
            )
            datasets.append(dataset)
            logger.info(f"Successfully loaded dataset {name} with term {term}")
        except Exception as e:
            logger.error(f"Failed to load dataset {name} with term {term}: {e}")
    
    return datasets


def create_time_series_train_test_split(
    series_data: np.ndarray,
    prediction_length: int,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split for time series data.
    
    Args:
        series_data: Time series array [n_series, length]
        prediction_length: Length of prediction horizon
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_data, test_targets) where:
        - train_data: Training time series [n_series, train_length]
        - test_targets: True values for forecasting [n_series, prediction_length]
    """
    if series_data.ndim != 2:
        raise ValueError("Series data must be 2D array [n_series, length]")
    
    n_series, total_length = series_data.shape
    
    if total_length < prediction_length + 10:  # Need some minimum training data
        raise ValueError(f"Series too short for prediction length {prediction_length}")
    
    # Calculate split point
    train_length = int((total_length - prediction_length) * train_ratio)
    train_length = max(10, train_length)  # Minimum training length
    
    # Create splits
    train_data = series_data[:, :train_length]
    test_start = train_length
    test_end = test_start + prediction_length
    
    # Make sure we don't exceed array bounds
    test_end = min(test_end, total_length)
    test_targets = series_data[:, test_start:test_end]
    
    return train_data, test_targets


def prepare_time_series_for_visualization(
    series_data: np.ndarray,
    series_index: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prepare a single time series for MARVIS visualization.
    
    Args:
        series_data: Time series array [n_series, length]
        series_index: Index of series to use (for multivariate data)
        
    Returns:
        Tuple of (time_series_values, metadata)
    """
    if series_data.ndim != 2:
        raise ValueError("Series data must be 2D array [n_series, length]")
    
    n_series, length = series_data.shape
    
    if series_index >= n_series:
        raise ValueError(f"Series index {series_index} >= number of series {n_series}")
    
    # Extract single series
    series = series_data[series_index, :]
    
    # Remove NaN values (from padding)
    valid_mask = ~np.isnan(series)
    if not np.any(valid_mask):
        raise ValueError(f"Series {series_index} contains only NaN values")
    
    clean_series = series[valid_mask]
    
    # Create metadata
    metadata = {
        'series_index': series_index,
        'original_length': length,
        'clean_length': len(clean_series),
        'has_missing': np.any(~valid_mask),
        'series_stats': {
            'mean': float(np.mean(clean_series)),
            'std': float(np.std(clean_series)),
            'min': float(np.min(clean_series)),
            'max': float(np.max(clean_series))
        }
    }
    
    return clean_series, metadata


def validate_gift_eval_environment() -> bool:
    """
    Validate that gift-eval is properly installed and configured.
    
    Returns:
        True if gift-eval is available, False otherwise
    """
    try:
        # Check environment variable
        gift_eval_path = os.environ.get('GIFT_EVAL')
        if not gift_eval_path:
            logger.warning("GIFT_EVAL environment variable not set")
            return False
        
        if not os.path.exists(gift_eval_path):
            logger.warning(f"GIFT_EVAL path does not exist: {gift_eval_path}")
            return False
        
        # Try importing gift-eval
        try:
            from gift_eval.data import Dataset
            logger.info("gift-eval successfully imported")
            return True
        except ImportError as e:
            logger.warning(f"Failed to import gift-eval: {e}")
            return False
            
    except Exception as e:
        logger.warning(f"Error validating gift-eval environment: {e}")
        return False


# Test datasets commonly used in gift-eval benchmarks
COMMON_GIFT_EVAL_DATASETS = [
    'm4_yearly', 'm4_quarterly', 'm4_monthly', 'm4_weekly', 'm4_daily', 'm4_hourly',
    'electricity/15T', 'electricity/H', 'electricity/D', 'electricity/W',
    'solar/10T', 'solar/H', 'solar/D', 'solar/W',
    'hospital', 'covid_deaths',
    'us_births/D', 'us_births/M', 'us_births/W',
    'saugeenday/D', 'saugeenday/M', 'saugeenday/W',
    'temperature_rain_with_missing',
    'kdd_cup_2018_with_missing/H', 'kdd_cup_2018_with_missing/D',
    'car_parts_with_missing',
    'restaurant',
    'hierarchical_sales/D', 'hierarchical_sales/W'
]


def get_available_datasets() -> List[str]:
    """
    Get list of commonly available gift-eval datasets.
    
    Returns:
        List of dataset names
    """
    return COMMON_GIFT_EVAL_DATASETS.copy()


def get_dataset_properties(dataset_name: str) -> Dict[str, Any]:
    """
    Get properties for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset properties
    """
    # This is a simplified version - in practice, you might want to load
    # this from a configuration file or query the actual dataset
    properties = {
        'name': dataset_name,
        'domain': 'unknown',
        'num_variates': 1,
        'typical_frequency': 'unknown'
    }
    
    # Add some basic heuristics based on name
    if 'm4_' in dataset_name:
        properties['domain'] = 'M4 Competition'
        properties['num_variates'] = 1
        
        if 'yearly' in dataset_name:
            properties['typical_frequency'] = 'Y'
        elif 'quarterly' in dataset_name:
            properties['typical_frequency'] = 'Q'
        elif 'monthly' in dataset_name:
            properties['typical_frequency'] = 'M'
        elif 'weekly' in dataset_name:
            properties['typical_frequency'] = 'W'
        elif 'daily' in dataset_name:
            properties['typical_frequency'] = 'D'
        elif 'hourly' in dataset_name:
            properties['typical_frequency'] = 'H'
    
    elif 'electricity' in dataset_name:
        properties['domain'] = 'Energy'
        properties['num_variates'] = 'multiple'
        
    elif 'solar' in dataset_name:
        properties['domain'] = 'Energy'
        properties['num_variates'] = 'multiple'
        
    elif 'covid' in dataset_name:
        properties['domain'] = 'Health'
        properties['num_variates'] = 1
        
    elif 'births' in dataset_name:
        properties['domain'] = 'Demographics'
        properties['num_variates'] = 1
    
    return properties