"""
Device detection utilities for consistent device selection across MARVIS.

This module provides centralized device detection and selection logic to ensure
consistent behavior across all embedding models and neural network operations.
"""

import os
import sys
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def detect_optimal_device(prefer_mps: bool = True, force_cpu: bool = False) -> str:
    """
    Detect the optimal device for PyTorch operations.
    
    This function implements MARVIS's device selection strategy:
    1. If force_cpu=True, always return 'cpu'
    2. On non-Mac systems: prefer CUDA > CPU
    3. On Mac systems: prefer MPS > CPU (if prefer_mps=True)
    
    Args:
        prefer_mps: Whether to prefer MPS over CPU on Mac systems (default: True)
        force_cpu: If True, force CPU usage regardless of available accelerators
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if force_cpu:
        logger.info("Device detection: CPU forced by user")
        return 'cpu'
    
    # Non-Mac systems: prefer CUDA
    if sys.platform != "darwin":
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Device detection: Using CUDA GPU acceleration")
        else:
            device = 'cpu'
            logger.info("Device detection: CUDA not available, using CPU")
        return device
    
    # Mac systems: prefer MPS if available and requested
    if prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Device detection: Using MPS GPU acceleration on Mac")
        # Enable MPS fallback for compatibility
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = 'cpu'
        if prefer_mps:
            logger.info("Device detection: MPS not available on Mac, using CPU")
        else:
            logger.info("Device detection: MPS disabled by user, using CPU on Mac")
    
    return device


def get_device_info(device: str) -> dict:
    """
    Get detailed information about the specified device.
    
    Args:
        device: Device string ('cuda', 'mps', or 'cpu')
        
    Returns:
        Dictionary with device information
    """
    info = {
        'device': device,
        'platform': sys.platform,
        'available_devices': []
    }
    
    # Check available devices
    if torch.cuda.is_available():
        info['available_devices'].append('cuda')
        if device == 'cuda':
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['available_devices'].append('mps')
    
    info['available_devices'].append('cpu')
    
    return info


def configure_device_for_model(model_type: str, device: Optional[str] = None, 
                             batch_size: Optional[int] = None) -> Tuple[str, int]:
    """
    Configure device and batch size for specific model types.
    
    This function applies model-specific optimizations and constraints
    based on the target device and model architecture.
    
    Args:
        model_type: Type of model ('dinov2', 'bioclip', 'whisper', 'tabpfn', etc.)
        device: Preferred device (if None, auto-detect)
        batch_size: Preferred batch size (if None, use model-specific defaults)
        
    Returns:
        Tuple of (device, batch_size)
    """
    if device is None or device == "auto":
        device = detect_optimal_device()
    
    # Model-specific configurations
    if model_type.lower() in ['dinov2', 'bioclip', 'bioclip2']:
        # Vision models
        if batch_size is None:
            if device == 'mps':
                batch_size = 16  # Conservative for MPS
            elif device == 'cuda':
                batch_size = 32  # Standard for CUDA
            else:
                batch_size = 8   # Conservative for CPU
        else:
            # Apply device-specific limits
            if device == 'mps':
                batch_size = min(batch_size, 16)
            elif device == 'cpu':
                batch_size = min(batch_size, 8)
                
    elif model_type.lower() in ['whisper', 'clap']:
        # Audio models
        if batch_size is None:
            if device == 'mps':
                batch_size = 8   # Conservative for audio processing
            elif device == 'cuda':
                batch_size = 16
            else:
                batch_size = 4
        else:
            if device == 'mps':
                batch_size = min(batch_size, 8)
            elif device == 'cpu':
                batch_size = min(batch_size, 4)
                
    elif model_type.lower() == 'tabpfn':
        # TabPFN doesn't use batch processing in the same way
        if batch_size is None:
            batch_size = 1  # TabPFN processes datasets as a whole
            
    else:
        # Default configuration
        if batch_size is None:
            batch_size = 32 if device == 'cuda' else 16 if device == 'mps' else 8
    
    logger.debug(f"Device configuration for {model_type}: device={device}, batch_size={batch_size}")
    return device, batch_size


def log_device_usage(model_name: str, device: str, additional_info: Optional[dict] = None):
    """
    Log device usage information for debugging and monitoring.
    
    Args:
        model_name: Name of the model being loaded
        device: Device being used
        additional_info: Optional additional information to log
    """
    device_info = get_device_info(device)
    
    log_msg = f"Loading {model_name} on {device}"
    if device == 'cuda' and 'cuda_device_name' in device_info:
        log_msg += f" ({device_info['cuda_device_name']})"
    elif device == 'mps':
        log_msg += " (Apple Silicon GPU)"
    
    logger.info(log_msg)
    
    if additional_info:
        for key, value in additional_info.items():
            logger.debug(f"{model_name} {key}: {value}")


def setup_device_environment(device: str):
    """
    Setup environment variables and settings for the specified device.
    
    Args:
        device: Target device ('cuda', 'mps', or 'cpu')
    """
    if device == 'mps':
        # Enable MPS fallback for compatibility
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Disable deterministic algorithms for MPS compatibility
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            # Older PyTorch versions might not have this
            pass
            
    elif device == 'cuda':
        # CUDA-specific optimizations
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    logger.debug(f"Device environment configured for {device}")


def log_platform_info(logger_instance: Optional[logging.Logger] = None) -> dict:
    """
    Log platform and device information.
    
    Args:
        logger_instance: Logger to use (uses module logger if None)
        
    Returns:
        Dictionary with platform information
    """
    if logger_instance is None:
        logger_instance = logger
    
    device = detect_optimal_device()
    device_info = get_device_info(device)
    
    platform_info = {
        'platform': sys.platform,
        'torch_version': torch.__version__,
        'device': device,
        'available_devices': device_info.get('available_devices', []),
    }
    
    if device == 'cuda' and 'cuda_device_name' in device_info:
        platform_info['cuda_device_name'] = device_info['cuda_device_name']
        platform_info['cuda_device_count'] = device_info.get('cuda_device_count', 0)
    
    # Log concise platform info
    log_msg = f"Platform: {platform_info['platform']}, PyTorch: {platform_info['torch_version']}, Device: {device}"
    if device == 'cuda' and 'cuda_device_name' in device_info:
        log_msg += f" ({device_info['cuda_device_name']})"
    logger_instance.info(log_msg)
    
    return platform_info