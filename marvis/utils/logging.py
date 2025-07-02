"""
Logging utilities for MARVIS.
"""

import logging
import os
import sys
from typing import Optional, Union, Dict, Any

def setup_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
    propagate: bool = False
) -> logging.Logger:
    """
    Configure the logger for script usage.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: File to log to (default: None, log to stdout)
        log_format: Format for log messages
        propagate: Whether to propagate logs to parent loggers
        
    Returns:
        Configured logger
    """
    # Convert string log level to numeric if necessary
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Get the logger
    logger = logging.getLogger("marvis")
    logger.setLevel(log_level)
    logger.propagate = propagate
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        # Make sure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_notebook_logging(
    log_level: Union[int, str] = logging.INFO,
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
    propagate: bool = False
) -> logging.Logger:
    """
    Configure the logger specifically for Jupyter notebook usage.
    
    Args:
        log_level: Logging level (default: INFO)
        log_format: Format for log messages
        propagate: Whether to propagate logs to parent loggers
        
    Returns:
        Configured logger
    """
    # Convert string log level to numeric if necessary
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Get the logger
    logger = logging.getLogger("marvis")
    logger.setLevel(log_level)
    logger.propagate = propagate
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler for notebook display
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger