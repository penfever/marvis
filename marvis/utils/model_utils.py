"""
Model loading utilities for MARVIS models.

This module provides robust functions for loading models with intelligent path
resolution and error handling.
"""

import os
import glob
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def find_best_checkpoint(model_path: str) -> str:
    """
    Find the best checkpoint in a model directory using the following priority:
    1. final_model directory (if exists)
    2. latest checkpoint-* directory (sorted by step number)
    3. best_model directory (if exists)
    4. The original path itself
    
    Args:
        model_path: Base path to search for checkpoints
        
    Returns:
        String path to the best checkpoint found
    """
    # Start with the provided path
    resolved_path = model_path
    
    # Check if this is a directory (not a specific checkpoint)
    if os.path.isdir(model_path):
        # First, check for final_model subdirectory
        final_model_path = os.path.join(model_path, "final_model")
        if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
            resolved_path = final_model_path
            logger.info(f"Found 'final_model' directory at {final_model_path}")
        else:
            # Look for checkpoints
            checkpoint_pattern = os.path.join(model_path, "checkpoint-*")
            checkpoints = sorted(glob.glob(checkpoint_pattern))
            
            if checkpoints:
                # Get the latest checkpoint by sorting
                # We'll try to extract the step number for better sorting
                def get_checkpoint_step(checkpoint_path):
                    try:
                        # Extract the number after "checkpoint-"
                        step = int(os.path.basename(checkpoint_path).split('-')[1])
                        return step
                    except (IndexError, ValueError):
                        # If we can't extract a number, use the path as a fallback
                        return checkpoint_path
                
                # Sort checkpoints by step number (highest = latest)
                checkpoints.sort(key=get_checkpoint_step, reverse=True)
                resolved_path = checkpoints[0]
                logger.info(f"Found latest checkpoint at {resolved_path}")
            else:
                # If no checkpoints found, check for "best_model" subdirectory
                best_model_path = os.path.join(model_path, "best_model")
                if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
                    resolved_path = best_model_path
                    logger.info(f"Found 'best_model' directory at {best_model_path}")
                else:
                    # No special subdirectories found, use the path as-is
                    logger.info(f"No 'final_model', checkpoints, or 'best_model' found in {model_path}, trying path directly")
    
    return resolved_path

def load_pretrained_model(model_path: str, device_map: str = "auto", embedding_size: int = 1000, model_id: str = None) -> Tuple:
    """
    Load a pretrained MARVIS model with smart path resolution.
    
    This function handles finding the best checkpoint in a model directory:
    1. First checks for a "final_model" subdirectory
    2. If not found, looks for latest "checkpoint-*" subdirectory
    3. If neither is found, uses the provided path directly
    4. Raises clear error if no valid model is found
    
    Args:
        model_path: Path to the model or directory containing checkpoints
        device_map: Device mapping strategy for model loading
        embedding_size: Size of the embeddings
        
    Returns:
        Tuple of (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids)
    
    Raises:
        ValueError: If model loading fails with a clear error message
    """
    from marvis.models import load_pretrained_model as core_load_pretrained_model
    
    # First resolve the path to find the best checkpoint
    resolved_path = find_best_checkpoint(model_path)
    
    logger.info(f"Loading pretrained model from {resolved_path}")
    
    try:
        # Use the centralized model loading function
        return core_load_pretrained_model(
            model_path=resolved_path,
            device_map=device_map,
            embedding_size=embedding_size,
            model_id=model_id
        )
    except (FileNotFoundError, ValueError) as e:
        # Re-raise with a clearer error message that includes path resolution info
        if resolved_path != model_path:
            error_msg = f"Failed to load model from resolved path: {resolved_path} (original path: {model_path}). Error: {e}"
        else:
            error_msg = f"Failed to load model from path: {model_path}. Error: {e}"
        
        logger.error(error_msg)
        raise ValueError(error_msg)