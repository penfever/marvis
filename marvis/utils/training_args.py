"""
Shared argument parsing utilities for MARVIS training scripts.

This module provides common argument parsing functions to avoid code duplication
between train_tabular_mix.py and train_tabular_dataset.py.
"""

import argparse
from typing import Optional


def add_model_args(parser: argparse.ArgumentParser):
    """Add model-related arguments."""
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model ID to use from Hugging Face"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1000,
        help="Size of the embeddings"
    )
    parser.add_argument(
        "--unfreeze_last_k_layers",
        type=int,
        default=0,
        help="Number of last layers to keep unfrozen (default: 0, freeze entire LLM)"
    )


def add_vector_quantization_args(parser: argparse.ArgumentParser):
    """Add Vector Quantization (VQ) related arguments."""
    parser.add_argument(
        "--use_vector_quantization",
        action="store_true",
        help="Enable Vector Quantization (VQ) for embeddings"
    )
    parser.add_argument(
        "--vq_num_embeddings",
        type=int,
        default=256,
        help="Size of the VQ codebook (number of discrete codes)"
    )
    parser.add_argument(
        "--vq_commitment_cost",
        type=float,
        default=1.0,
        help="Weight for the VQ commitment loss"
    )
    parser.add_argument(
        "--vq_decay",
        type=float,
        default=0.99,
        help="Decay factor for EMA updates of the codebook"
    )
    parser.add_argument(
        "--vq_debug",
        action="store_true",
        help="Enable additional debugging for the vector quantizer"
    )
    parser.add_argument(
        "--vq_warmup_steps",
        type=int,
        default=100,
        help="Number of steps to warm up the VQ commitment loss"
    )


def add_training_args(parser: argparse.ArgumentParser):
    """Add training-related arguments."""
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_epochs when specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use for evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Mixup alpha parameter for data augmentation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--save_best_model",
        action="store_true",
        help="Save the best model based on validation loss"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save model every N steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep (None to keep all)"
    )


def add_learning_rate_scheduler_args(parser: argparse.ArgumentParser):
    """Add learning rate scheduler arguments."""
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["constant", "linear", "cosine", "exponential", "polynomial", "cosine_with_restarts"],
        default="cosine",
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total steps to use for warmup"
    )
    parser.add_argument(
        "--lr_eta_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine scheduler"
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor for exponential scheduler"
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power for polynomial scheduler"
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=float,
        default=0.5,
        help="Number of cycles for cosine scheduler"
    )
    parser.add_argument(
        "--lr_cycle_length",
        type=int,
        default=None,
        help="Length of each cycle for cosine_with_restarts scheduler (in steps)"
    )


def add_component_specific_lr_args(parser: argparse.ArgumentParser):
    """Add component-specific learning rate arguments."""
    parser.add_argument(
        "--vq_learning_rate",
        type=float,
        default=None,
        help="Learning rate for VQ codebook (defaults to learning_rate)"
    )
    parser.add_argument(
        "--vq_lr_scheduler",
        type=str,
        choices=["constant", "linear", "cosine", "exponential", "polynomial", "cosine_with_restarts"],
        default=None,
        help="Learning rate scheduler for VQ codebook (defaults to lr_scheduler)"
    )
    parser.add_argument(
        "--class_token_learning_rate",
        type=float,
        default=None,
        help="Learning rate for class tokens (defaults to learning_rate)"
    )
    parser.add_argument(
        "--class_token_lr_scheduler",
        type=str,
        choices=["constant", "linear", "cosine", "exponential", "polynomial", "cosine_with_restarts"],
        default=None,
        help="Learning rate scheduler for class tokens (defaults to lr_scheduler)"
    )
    parser.add_argument(
        "--llm_learning_rate",
        type=float,
        default=None,
        help="Learning rate for LLM parameters (defaults to learning_rate)"
    )
    parser.add_argument(
        "--llm_lr_scheduler",
        type=str,
        choices=["constant", "linear", "cosine", "exponential", "polynomial", "cosine_with_restarts"],
        default=None,
        help="Learning rate scheduler for LLM parameters (defaults to lr_scheduler)"
    )


def add_temperature_annealing_args(parser: argparse.ArgumentParser):
    """Add temperature annealing arguments."""
    parser.add_argument(
        "--temperature_initial",
        type=float,
        default=1.0,
        help="Initial temperature for scaling label logits"
    )
    parser.add_argument(
        "--temperature_final",
        type=float,
        default=0.1,
        help="Final temperature for scaling label logits"
    )
    parser.add_argument(
        "--temperature_anneal_steps",
        type=int,
        default=None,
        help="Number of steps to anneal temperature over. If None, uses total_steps"
    )


def add_gradient_penalty_args(parser: argparse.ArgumentParser):
    """Add gradient penalty arguments."""
    parser.add_argument(
        "--gradient_penalty_weight",
        type=float,
        default=0.0,
        help="Weight for gradient penalty on embeddings (default: 0.0, disabled)"
    )
    parser.add_argument(
        "--gradient_penalty_threshold",
        type=float,
        default=10.0,
        help="Gradient norm threshold above which to apply penalty"
    )


def add_data_args(parser: argparse.ArgumentParser):
    """Add data-related arguments."""
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default="./data",
        help="Directory to store cached embeddings. Set to 'none' to disable caching."
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists"
    )


def add_few_shot_args(parser: argparse.ArgumentParser):
    """Add few-shot learning arguments."""
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=100,
        help="Number of few-shot examples to include in the prefix during training and inference"
    )
    parser.add_argument(
        "--permute_examples",
        action="store_true",
        help="Randomly permute the order of few-shot examples each epoch to discourage memorization"
    )
    parser.add_argument(
        "--permute_labels",
        action="store_true",
        help="Randomly permute the class-to-label mapping each epoch to discourage memorization"
    )
    parser.add_argument(
        "--permute_labels_every_k_steps",
        type=int,
        default=None,
        help="When using max_steps, permute labels every k steps instead of every epoch"
    )
    parser.add_argument(
        "--no_permute_last_k",
        type=int,
        default=None,
        help="If set, disables label permutation for the last k steps when permute_labels is True"
    )
    parser.add_argument(
        "--variable_few_shot",
        action="store_true",
        help="Randomly vary the number of few-shot examples during training to improve generalization"
    )
    parser.add_argument(
        "--few_shot_min",
        type=int,
        default=10,
        help="Minimum number of few-shot examples when using variable_few_shot"
    )
    parser.add_argument(
        "--few_shot_max",
        type=int,
        default=None,
        help="Maximum number of few-shot examples when using variable_few_shot (defaults to num_few_shot_examples if not specified)"
    )


def add_regularization_args(parser: argparse.ArgumentParser):
    """Add regularization arguments."""
    parser.add_argument(
        "--min_freq_weight",
        type=float,
        default=0.05,
        help="Weight for minimum frequency regularization (0 to disable)"
    )
    parser.add_argument(
        "--min_freq_target",
        type=float,
        default=0.05,
        help="Target minimum frequency for each class in regularization"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of steps with no improvement after which training will be stopped"
    )


def add_wandb_args(parser: argparse.ArgumentParser, default_project: str = "marvis-training"):
    """Add Weights & Biases arguments."""
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking"
    )
    parser.add_argument(
        "--resume_wandb",
        action="store_true",
        help="Resume Weights & Biases run when resuming from checkpoint"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=default_project,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team) name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments used by both scripts."""
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/marvis_output",
        help="Directory to save model checkpoints and results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Resume optimizer state when resuming from checkpoint"
    )


def create_common_parser(description: str, default_wandb_project: str = "marvis-training") -> argparse.ArgumentParser:
    """
    Create an argument parser with all common arguments for MARVIS training.
    
    Args:
        description: Description for the argument parser
        default_wandb_project: Default project name for W&B
        
    Returns:
        ArgumentParser with all common training arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Add all common argument groups
    add_common_args(parser)
    add_model_args(parser)
    add_vector_quantization_args(parser)
    add_training_args(parser)
    add_learning_rate_scheduler_args(parser)
    add_component_specific_lr_args(parser)
    add_temperature_annealing_args(parser)
    add_gradient_penalty_args(parser)
    add_data_args(parser)
    add_few_shot_args(parser)
    add_regularization_args(parser)
    add_wandb_args(parser, default_wandb_project)
    
    return parser


def create_single_dataset_parser() -> argparse.ArgumentParser:
    """Create argument parser specific to single dataset training."""
    parser = create_common_parser(
        "Train MARVIS on a single tabular dataset from OpenML",
        default_wandb_project="marvis-training"
    )
    
    # Add single-dataset specific arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="har",
        help="Name of the dataset (used for cache filenames)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (ignored if total_steps is provided)"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.5,
        help="Only start tracking early stopping patience when loss is below this threshold"
    )
    
    # Evaluation-specific arguments
    parser.add_argument(
        "--bypass_eval",
        action="store_true",
        help="Skip the final evaluation step (useful for batch training when evaluation will be done separately)"
    )
    parser.add_argument(
        "--evaluate_with_explanations",
        action="store_true",
        help="Use the explanation-based evaluation method when evaluating the model"
    )
    parser.add_argument(
        "--explanation_type",
        type=str,
        choices=["standard", "counterfactual", "feature_importance", "decision_rules"],
        default="standard",
        help="Type of explanation to generate"
    )
    parser.add_argument(
        "--max_explanation_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate for explanations"
    )
    parser.add_argument(
        "--save_explanations",
        action="store_true",
        help="Save all explanations to a separate file when using explanation-based evaluation"
    )
    
    # Override some defaults for single dataset usage
    parser.set_defaults(
        output_dir="./models/marvis_output",
        max_train_samples=None,  # Keep help text simple for single dataset
        permute_labels=False,  # Simpler help text without multi-step mention
        wandb_project="marvis-training",
        wandb_name=None  # Will default to dataset name + timestamp
    )
    
    return parser


def create_multi_dataset_parser() -> argparse.ArgumentParser:
    """Create argument parser specific to multi-dataset training."""
    parser = create_common_parser(
        "Train MARVIS on multiple tabular datasets",
        default_wandb_project="marvis-multi-training"
    )
    
    # Add multi-dataset specific arguments
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default=None,
        help="Comma-separated list of OpenML dataset IDs to use for training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        default=None,
        help="Directory or list of directories containing CSV files to use as datasets"
    )
    parser.add_argument(
        "--num_datasets",
        type=int,
        default=None,
        help="Number of random datasets to sample from OpenML"
    )
    parser.add_argument(
        "--dataset_switch_steps",
        type=int,
        default=None,
        help="[DEPRECATED] Parameter no longer used as all datasets are combined before training"
    )
    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of datasets to hold out for testing"
    )
    
    # Override some defaults for multi-dataset usage
    parser.set_defaults(
        output_dir="./models/marvis_multi_output",
        total_steps=10000,  # Multi-dataset uses step-based training by default
        save_steps=500,
        wandb_project="marvis-multi-training",
        wandb_name=None  # Will default to 'multi_dataset' + timestamp
    )
    
    return parser