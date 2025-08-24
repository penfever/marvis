"""
Training and evaluation functions for MARVIS models.

This module provides functionality for:
- Training LLMs with tabular embeddings
- Evaluating models on test datasets
- Saving model checkpoints and final models
- Utility functions for determining when to save checkpoints

It also re-exports the model loading utilities from the utils module
for convenience.
"""

from ..utils.model_utils import find_best_checkpoint, load_pretrained_model
from .evaluator import check_tensor_devices, evaluate_llm_on_test_set
from .save_utils import save_checkpoint, save_final_model, should_save_checkpoint
from .trainer import train_llm_with_tabpfn_embeddings

__all__ = [
    # Training and evaluation
    "train_llm_with_tabpfn_embeddings",
    "evaluate_llm_on_test_set",
    "check_tensor_devices",
    # Model saving
    "save_checkpoint",
    "save_final_model",
    "should_save_checkpoint",
    # Model loading (re-exported from utils)
    "load_pretrained_model",
    "find_best_checkpoint",
]
