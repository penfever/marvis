"""
Image processing utilities for MARVIS.

Provides unified handling of different image input types including file paths,
numpy arrays, PIL Images, and sklearn datasets.
"""

import numpy as np
import logging
from typing import Union
from PIL import Image

logger = logging.getLogger(__name__)


def normalize_image_input(image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    """
    Normalize different image input types to PIL Image.
    
    Args:
        image_input: Can be:
            - str: Path to image file
            - np.ndarray: Image data as numpy array (1D, 2D grayscale, or 3D RGB)
            - PIL.Image: Already a PIL image
            
    Returns:
        PIL.Image: Normalized RGB PIL Image
        
    Raises:
        TypeError: If input type is not supported
        ValueError: If array dimensions are invalid
    """
    if isinstance(image_input, str):
        # Load image from file path
        try:
            image = Image.open(image_input).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from path '{image_input}': {e}")
            
    elif isinstance(image_input, Image.Image):
        # Already a PIL Image, just ensure RGB
        return image_input.convert('RGB')
        
    elif isinstance(image_input, np.ndarray):
        # Handle numpy array input
        return _numpy_to_pil_image(image_input)
        
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}. "
                       "Expected str (file path), np.ndarray, or PIL.Image")


def _numpy_to_pil_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array representing image data
        
    Returns:
        PIL.Image: RGB PIL Image
        
    Raises:
        ValueError: If array dimensions are invalid
    """
    # Handle 1D arrays (e.g., sklearn digits dataset)
    if array.ndim == 1:
        array = _reshape_1d_to_2d(array)
    
    # Handle different array dimensions
    if array.ndim == 2:
        # Grayscale image
        array = _normalize_pixel_values(array)
        image = Image.fromarray(array.astype(np.uint8), mode='L')
        return image.convert('RGB')
        
    elif array.ndim == 3:
        # Multi-channel image
        if array.shape[-1] == 1:
            # Single channel, squeeze to 2D
            array = array.squeeze(-1)
            array = _normalize_pixel_values(array)
            image = Image.fromarray(array.astype(np.uint8), mode='L')
            return image.convert('RGB')
            
        elif array.shape[-1] == 3:
            # RGB image
            array = _normalize_pixel_values(array)
            image = Image.fromarray(array.astype(np.uint8), mode='RGB')
            return image
            
        elif array.shape[-1] == 4:
            # RGBA image, drop alpha channel
            array = array[:, :, :3]
            array = _normalize_pixel_values(array)
            image = Image.fromarray(array.astype(np.uint8), mode='RGB')
            return image
            
        else:
            raise ValueError(f"Unsupported number of channels: {array.shape[-1]}. "
                           "Expected 1, 3, or 4 channels.")
    else:
        raise ValueError(f"Unsupported array dimensions: {array.ndim}. "
                        "Expected 1D, 2D, or 3D arrays.")


def _reshape_1d_to_2d(array: np.ndarray) -> np.ndarray:
    """
    Reshape 1D array to 2D image array.
    
    Args:
        array: 1D numpy array
        
    Returns:
        np.ndarray: 2D array representing image
        
    Raises:
        ValueError: If array cannot be reshaped to a square image
    """
    length = len(array)
    
    # Special case for sklearn digits dataset (8x8 = 64 pixels)
    if length == 64:
        logger.debug("Detected sklearn digits dataset format, reshaping to 8x8")
        return array.reshape(8, 8)
    
    # Try to find square dimensions
    side_length = int(np.sqrt(length))
    if side_length * side_length == length:
        logger.debug(f"Reshaping 1D array of length {length} to {side_length}x{side_length}")
        return array.reshape(side_length, side_length)
    
    # Try common aspect ratios
    common_ratios = [(28, 28), (32, 32), (224, 224), (256, 256)]  # MNIST, CIFAR, ImageNet, etc.
    for height, width in common_ratios:
        if height * width == length:
            logger.debug(f"Reshaping 1D array of length {length} to {height}x{width}")
            return array.reshape(height, width)
    
    raise ValueError(f"Cannot reshape 1D array of length {length} into a valid image. "
                    f"Expected square dimensions or common image sizes.")


def _normalize_pixel_values(array: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to 0-255 range.
    
    Args:
        array: Image array with arbitrary pixel value range
        
    Returns:
        np.ndarray: Array with pixel values in [0, 255] range
    """
    # Handle different value ranges
    if array.max() <= 1.0 and array.min() >= 0.0:
        # Values in [0, 1] range, scale to [0, 255]
        return (array * 255)
    elif array.min() < 0:
        # Values might be centered around 0, normalize to [0, 255]
        array = array - array.min()  # Shift to start at 0
        array = array / array.max()  # Scale to [0, 1]
        return (array * 255)
    elif array.max() > 255:
        # Values larger than 255, scale down
        return (array / array.max() * 255)
    else:
        # Values already in reasonable range
        return array


def validate_image_array(array: np.ndarray) -> bool:
    """
    Validate that a numpy array can be converted to an image.
    
    Args:
        array: Numpy array to validate
        
    Returns:
        bool: True if array is valid for image conversion
    """
    try:
        _numpy_to_pil_image(array.copy())
        return True
    except (ValueError, TypeError):
        return False


def get_image_info(image_input: Union[str, np.ndarray, Image.Image]) -> dict:
    """
    Get information about an image input.
    
    Args:
        image_input: Image input of any supported type
        
    Returns:
        dict: Information about the image including type, shape, etc.
    """
    info = {
        'input_type': type(image_input).__name__,
        'is_valid': False,
        'shape': None,
        'dtype': None,
        'value_range': None
    }
    
    try:
        if isinstance(image_input, str):
            info['is_valid'] = True
            info['path'] = image_input
            
        elif isinstance(image_input, Image.Image):
            info['is_valid'] = True
            info['shape'] = image_input.size  # (width, height)
            info['mode'] = image_input.mode
            
        elif isinstance(image_input, np.ndarray):
            info['shape'] = image_input.shape
            info['dtype'] = str(image_input.dtype)
            info['value_range'] = (float(image_input.min()), float(image_input.max()))
            info['is_valid'] = validate_image_array(image_input)
            
        # Try to normalize to get final image info
        if info['is_valid']:
            pil_image = normalize_image_input(image_input)
            info['final_size'] = pil_image.size
            info['final_mode'] = pil_image.mode
            
    except Exception as e:
        info['error'] = str(e)
        
    return info