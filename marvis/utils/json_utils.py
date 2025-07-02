#!/usr/bin/env python3
"""
JSON serialization utilities for handling NumPy types and problematic float values.
Provides robust JSON serialization that handles NaN, inf, and various NumPy types.
"""

import json
import math
import logging
import numpy as np
from typing import Any, Dict, List, Union


def convert_for_json_serialization(obj: Any) -> Any:
    """
    Convert numpy types and problematic float values to JSON-serializable types.
    
    Handles:
    - NumPy integers, floats, booleans, arrays
    - NaN values (converted to null)
    - Infinity values (converted to very large numbers)
    - Regular Python floats that may contain NaN/inf
    - Nested dictionaries and lists
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        val = float(obj)
        if math.isnan(val):
            return None  # Convert NaN to null
        elif math.isinf(val):
            return 1e308 if val > 0 else -1e308  # Convert inf to very large numbers
        return val
    
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, (float, int)):
        # Handle regular Python float/int that might contain NaN/inf
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return 1e308 if obj > 0 else -1e308
        return obj
    
    elif isinstance(obj, dict):
        # Convert both keys and values, handling numpy types in keys
        converted_dict = {}
        for key, value in obj.items():
            # Convert numpy keys to native Python types
            if isinstance(key, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                converted_key = int(key)
            elif isinstance(key, (np.floating, np.float16, np.float32, np.float64)):
                key_val = float(key)
                if math.isnan(key_val) or math.isinf(key_val):
                    converted_key = str(key)  # Convert problematic keys to strings
                else:
                    converted_key = key_val
            elif isinstance(key, np.bool_):
                converted_key = bool(key)
            else:
                converted_key = str(key)  # Convert any remaining non-JSON types to string
            
            converted_dict[converted_key] = convert_for_json_serialization(value)
        return converted_dict
    
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json_serialization(item) for item in obj]
    
    else:
        return obj


def safe_json_dump(obj: Any, file_path: str, logger: logging.Logger = None, 
                   minimal_fallback: bool = True, **kwargs) -> bool:
    """
    Safely dump an object to JSON with robust error handling.
    
    Args:
        obj: Object to serialize
        file_path: Path to save JSON file
        logger: Logger instance for error reporting
        minimal_fallback: Whether to save a minimal version if full serialization fails
        **kwargs: Additional arguments passed to json.dump()
        
    Returns:
        True if successful, False if failed (even with fallback)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convert problematic types
    try:
        converted_obj = convert_for_json_serialization(obj)
    except Exception as e:
        logger.error(f"Failed to convert object for JSON serialization: {e}")
        if not minimal_fallback:
            return False
        converted_obj = {'error': f'Conversion failed: {e}', 'conversion_error': True}
    
    # Validate JSON serialization
    try:
        json.dumps(converted_obj)  # Test serialization
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization validation failed: {e}")
        if not minimal_fallback:
            return False
        
        # Create minimal fallback
        converted_obj = {
            'error': f'JSON serialization failed: {e}',
            'serialization_error': True,
            'original_type': str(type(obj)),
            'object_keys': list(obj.keys()) if isinstance(obj, dict) else None
        }
    
    # Write to file
    try:
        with open(file_path, "w") as f:
            json.dump(converted_obj, f, indent=kwargs.get('indent', 2), 
                     **{k: v for k, v in kwargs.items() if k != 'indent'})
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        return False


def safe_json_dumps(obj: Any, logger: logging.Logger = None, **kwargs) -> str:
    """
    Safely serialize an object to JSON string with robust error handling.
    
    Args:
        obj: Object to serialize
        logger: Logger instance for error reporting
        **kwargs: Additional arguments passed to json.dumps()
        
    Returns:
        JSON string, or error message if serialization fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        converted_obj = convert_for_json_serialization(obj)
        return json.dumps(converted_obj, **kwargs)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        error_obj = {
            'error': f'JSON serialization failed: {e}',
            'serialization_error': True,
            'original_type': str(type(obj))
        }
        return json.dumps(error_obj, **kwargs)


def validate_json_serializable(obj: Any, logger: logging.Logger = None) -> bool:
    """
    Check if an object can be JSON serialized after conversion.
    
    Args:
        obj: Object to check
        logger: Logger instance for error reporting
        
    Returns:
        True if serializable, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        converted_obj = convert_for_json_serialization(obj)
        json.dumps(converted_obj)
        return True
    except Exception as e:
        logger.debug(f"Object not JSON serializable: {e}")
        return False


# For backward compatibility, provide the old function name
convert_numpy_types = convert_for_json_serialization


def save_results(results, output_dir, dataset_name, use_unified_manager: bool = False):
    """
    Save evaluation results to JSON files using robust serialization.
    
    Args:
        results: List of result dictionaries or single result dictionary
        output_dir: Output directory for results
        dataset_name: Name of the dataset
        use_unified_manager: Whether to use the new unified results manager
    
    This function maintains backward compatibility while optionally using
    the new unified results management system.
    """
    import os
    
    logger = logging.getLogger(__name__)
    
    # Handle single result dictionary
    if isinstance(results, dict):
        results = [results]
    
    if use_unified_manager:
        # Use the new unified results manager
        try:
            from .results_manager import save_results_unified
            
            for result in results:
                model_name = result.get('model_name', 'unknown_model')
                modality = result.get('modality', 'tabular')  # Default to tabular for backward compatibility
                
                try:
                    save_results_unified(
                        results=result,
                        output_dir=output_dir,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        modality=modality
                    )
                    logger.info(f"Successfully saved {model_name} results using unified manager")
                except Exception as e:
                    logger.warning(f"Failed to save {model_name} results using unified manager: {e}")
                    logger.info("Falling back to legacy save method")
                    # Fall back to legacy method
                    _save_results_legacy(result, output_dir, dataset_name, model_name, logger)
        except ImportError:
            logger.warning("Unified results manager not available, using legacy method")
            # Fall back to legacy method for all results
            for result in results:
                model_name = result.get('model_name', 'unknown_model')
                _save_results_legacy(result, output_dir, dataset_name, model_name, logger)
    else:
        # Use legacy method
        for result in results:
            model_name = result.get('model_name', 'unknown_model')
            _save_results_legacy(result, output_dir, dataset_name, model_name, logger)


def _save_results_legacy(result, output_dir, dataset_name, model_name, logger):
    """Legacy results saving method for backward compatibility."""
    import os
    
    dataset_output_dir = os.path.join(output_dir, f"dataset_{dataset_name}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    model_name_clean = model_name.lower().replace('-', '_')
    results_file = os.path.join(dataset_output_dir, f"{model_name_clean}_results.json")
    
    # Use robust JSON saving with automatic fallback handling
    success = safe_json_dump(
        result, 
        results_file, 
        logger=logger,
        minimal_fallback=True,
        indent=2
    )
    
    if success:
        print(f"Successfully saved {model_name} results to {results_file}")
    else:
        logger.error(f"Failed to save {model_name} results to {results_file}")