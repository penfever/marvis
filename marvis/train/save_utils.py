"""
Model saving utilities for MARVIS models.
"""

import os
import torch
import logging
import shutil
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any, 
    output_dir: str,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    training_state: Optional[Dict[str, Any]] = None,
    class_token_ids: Optional[List[int]] = None,
    is_best: bool = False,
    save_optimizer: bool = True
) -> None:
    """
    Save a model checkpoint including model weights, tokenizer, and training state.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Base directory to save checkpoints in
        step: Current training step
        optimizer: Optional optimizer to save state
        scheduler: Optional scheduler to save state
        training_state: Optional dictionary with additional training state to save
        class_token_ids: Optional list of class token IDs used by the model (for MARVIS)
        is_best: Whether this is the best checkpoint so far
        save_optimizer: Whether to save optimizer state
    """
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model and tokenizer
    if hasattr(model, 'save_pretrained'):
        # For models like QwenWithPrefixEmbedding that have custom save_pretrained
        if class_token_ids is not None:
            model.save_pretrained(checkpoint_path, final_class_token_ids=class_token_ids.copy())
        else:
            model.save_pretrained(checkpoint_path)
    else:
        # Generic model saving
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
    
    # Save tokenizer if it has save_pretrained method
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(checkpoint_path)
    
    # Prepare and save training state
    state_dict = {}
    
    # Add optimizer state if provided and requested
    if optimizer is not None and save_optimizer:
        state_dict['optimizer'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        if hasattr(scheduler, 'state_dict'):
            state_dict['scheduler'] = scheduler.state_dict()
        if hasattr(scheduler, 'current_step'):
            state_dict['scheduler_step'] = scheduler.current_step
    
    # Add global step
    state_dict['global_step'] = step
    
    # Add any other training state if provided
    if training_state:
        state_dict.update(training_state)
    
    # Save the training state
    torch.save(state_dict, os.path.join(checkpoint_path, "training_state.pt"))
    
    # If this is the best model, create a copy in the best_model directory
    if is_best:
        best_path = os.path.join(output_dir, "best_model")
        
        # Remove existing best model directory if it exists
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        
        # Copy the checkpoint to best_model directory
        logger.info(f"Saving new best model to {best_path}")
        shutil.copytree(checkpoint_path, best_path)

def save_final_model(
    model: torch.nn.Module,
    tokenizer: Any, 
    output_dir: str,
    final_class_token_ids: Optional[List[int]] = None,
    original_class_token_ids: Optional[List[int]] = None,
    restore_original_mapping: bool = False
) -> None:
    """
    Save the final model at the end of training.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Base directory to save the model in
        final_class_token_ids: Optional list of final class token IDs used in the last epoch
        original_class_token_ids: Optional list of original class token IDs for restoring mapping
        restore_original_mapping: Whether to restore the original class token mapping before saving
    """
    # Create final output path
    final_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    
    logger.info(f"Saving final model to {final_path}")
    
    # Determine which class token IDs to use
    save_token_ids = None
    if final_class_token_ids is not None:
        if restore_original_mapping and original_class_token_ids is not None:
            logger.info("Restoring original (unpermuted) class token mapping before saving final model")
            save_token_ids = original_class_token_ids
        else:
            logger.info("Saving final model with current (possibly permuted) class token mapping")
            save_token_ids = final_class_token_ids
    
    # Save model with appropriate class token IDs if applicable
    if hasattr(model, 'save_pretrained') and save_token_ids is not None:
        model.save_pretrained(final_path, final_class_token_ids=save_token_ids)
    elif hasattr(model, 'save_pretrained'):
        model.save_pretrained(final_path)
    else:
        # Generic model saving
        torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
    
    # Save tokenizer
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(final_path)
    
    # Log success
    logger.info(f"Final model saved successfully to {final_path}")

def should_save_checkpoint(
    global_step: int,
    save_steps: int,
    total_trained_steps: int = 0,
    save_start: Optional[int] = None
) -> bool:
    """
    Determine if a checkpoint should be saved at the current step.
    
    Args:
        global_step: Current global training step
        save_steps: Frequency of checkpointing in steps
        total_trained_steps: Total number of steps already trained (for resumed training)
        save_start: Minimum number of steps before saving checkpoints (optional)
        
    Returns:
        True if checkpoint should be saved, False otherwise
    """
    actual_step = global_step + total_trained_steps
    
    # Don't save if we haven't reached save_start yet
    if save_start is not None and actual_step < save_start:
        return False
    
    # Otherwise, save on intervals based on save_steps
    return actual_step % save_steps == 0