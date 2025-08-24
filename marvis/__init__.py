"""
MARVIS: Modality Adaptive Reasoning over VISualizations
A powerful framework for multi-modal classification using Vision Language Models (VLMs) and intelligent visualizations.
"""

__version__ = "0.1.0"

# Import main modules
from . import data, models, train, utils, viz

# Make submodules available directly
from .data import create_llm_dataset, get_tabpfn_embeddings, load_dataset
from .models import prepare_qwen_with_prefix_embedding
from .train import evaluate_llm_on_test_set, train_llm_with_tabpfn_embeddings
from .utils import setup_logging

__all__ = [
    "data",
    "models",
    "train",
    "utils",
    "viz",
    "load_dataset",
    "get_tabpfn_embeddings",
    "create_llm_dataset",
    "prepare_qwen_with_prefix_embedding",
    "train_llm_with_tabpfn_embeddings",
    "evaluate_llm_on_test_set",
    "setup_logging",
]
