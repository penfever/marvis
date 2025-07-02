"""
VLM (Vision Language Model) utilities for MARVIS.

This module provides common functionality for working with VLMs across
different MARVIS implementations, including response parsing, conversation
formatting, and error handling.
"""

import re
import logging
from typing import Any, List, Dict, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


def parse_vlm_response(response: str, unique_classes: np.ndarray, logger_instance: Optional[logging.Logger] = None) -> Any:
    """
    Parse VLM response to extract predicted class.
    
    This function implements a robust parsing strategy that works across
    different VLM response formats and class types.
    
    Args:
        response: Raw VLM response string
        unique_classes: Array of valid class labels
        logger_instance: Logger instance for warnings (optional)
        
    Returns:
        Parsed class label from unique_classes (as Python native type)
    """
    if logger_instance is None:
        logger_instance = logger
        
    response = response.strip()
    
    # Helper function to convert numpy types to Python native types
    def to_native_type(value):
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        return value
    
    # Try to find "Class: X" pattern (most common format)
    class_match = re.search(r'Class:\s*([^\s|]+)', response, re.IGNORECASE)
    if class_match:
        class_str = class_match.group(1).strip()
        
        # Try to convert to appropriate type and match with unique_classes
        for cls in unique_classes:
            try:
                if str(cls) == class_str or cls == class_str:
                    return to_native_type(cls)
                # Try numeric conversion
                if isinstance(cls, (int, float)) or hasattr(cls, 'item'):
                    if float(cls) == float(class_str):
                        return to_native_type(cls)
            except (ValueError, TypeError):
                continue
    
    # Fallback: look for any mention of class labels in the response
    for cls in unique_classes:
        if str(cls) in response:
            return to_native_type(cls)
    
    # Final fallback: return first class
    logger_instance.warning(f"Could not parse class from VLM response: '{response[:100]}...'. Using fallback.")
    fallback_class = unique_classes[0]
    
    # Convert numpy types to Python native types
    if hasattr(fallback_class, 'item'):  # numpy scalar
        return fallback_class.item()
    else:
        return fallback_class




def create_vlm_conversation(image, prompt: str) -> List[Dict]:
    """
    Create standardized VLM conversation format.
    
    Args:
        image: PIL Image object
        prompt: Text prompt for the VLM
        
    Returns:
        Conversation in standard format for VLM processing
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]