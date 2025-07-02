#!/usr/bin/env python
"""
Script for evaluating LLM baselines (TabLLM, Tabula-8B, JOLT, and MARVIS-T-SNe) on tabular datasets.
This script handles:
1. Loading and preprocessing datasets from multiple sources
2. Creating textual serializations for LLMs
3. Evaluating the four LLM baselines on these datasets
4. Optional Weights & Biases logging and visualization

LLM Baselines evaluated:
- TabLLM: Few-shot classification using textual serializations
- Tabula-8B: Large language model fine-tuned for tabular data
- JOLT: Joint probabilistic predictions on tabular data using LLMs
- MARVIS-T-SNe: Vision Language Model classification using t-SNE visualizations of TabPFN embeddings

Usage examples:
    # Basic usage with a single dataset
    python evaluate_llm_baselines.py --dataset_name har --output_dir ./llm_baseline_results
    
    # Evaluating on multiple specific datasets
    python evaluate_llm_baselines.py --dataset_ids 1590,40975,37,54 --output_dir ./llm_baseline_results
    
    # Evaluating on 5 randomly sampled datasets from OpenML
    python evaluate_llm_baselines.py --num_datasets 5 --output_dir ./llm_baseline_results
    
    # Evaluating only specific LLM models
    python evaluate_llm_baselines.py --dataset_name har --models tabula_8b,jolt,marvis_tsne --output_dir ./results
    
    # Using Weights & Biases for experiment tracking
    python evaluate_llm_baselines.py --dataset_ids 1590,40975 --use_wandb --wandb_project llm_baselines
    
    # TabLLM automatically uses semantic templates and meaningful class names when available
    python evaluate_llm_baselines.py --dataset_name adult --models tabllm
    
    # Using 3D t-SNE with multiple viewing angles for MARVIS-T-SNe
    python evaluate_llm_baselines.py --dataset_name diabetes --models marvis_tsne --use_3d --output_dir ./results
    
    # Custom viewing angles for 3D t-SNE
    python evaluate_llm_baselines.py --dataset_name har --models marvis_tsne --use_3d --viewing_angles "20,45;0,0;90,0" --output_dir ./results
    
    # Using KNN connections to show nearest neighbors in embedding space
    python evaluate_llm_baselines.py --dataset_name adult --models marvis_tsne --use_knn_connections --nn_k 7 --output_dir ./results
    
    # Combining 3D t-SNE with KNN connections for maximum information
    python evaluate_llm_baselines.py --dataset_name diabetes --models marvis_tsne --use_3d --use_knn_connections --nn_k 5 --output_dir ./results
    
    # Customizing image size and DPI for VLM compatibility
    python evaluate_llm_baselines.py --dataset_name adult --models marvis_tsne --max_vlm_image_size 1024 --image_dpi 72 --output_dir ./results
    
    # Disable RGB conversion if needed (keeping RGBA mode)
    python evaluate_llm_baselines.py --dataset_name diabetes --models marvis_tsne --no-force_rgb_mode --output_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import glob
import datetime
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from marvis.data import load_datasets_for_evaluation, preprocess_datasets_for_evaluation
from marvis.data.evaluation_utils import validate_training_sample_args
from sklearn.model_selection import train_test_split
from marvis.utils import setup_logging, timeout_context, MetricsLogger

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring and JSON utilities
from marvis.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring, 
    GPUMonitor,
    safe_json_dump, 
    convert_for_json_serialization, 
    save_results
)

# Import centralized argument parser
from marvis.utils.evaluation_args import create_tabular_llm_evaluation_parser

# Import metadata validation utilities
from marvis.utils.metadata_validation import validate_metadata_for_models

# Import LLM baseline evaluation functions  
from examples.tabular.llm_baselines.tabllm_baseline import evaluate_tabllm
from examples.tabular.llm_baselines.tabula_8b_baseline import evaluate_tabula_8b
from examples.tabular.llm_baselines.jolt_baseline import evaluate_jolt
from marvis.models.marvis_tsne import evaluate_marvis_tsne

def parse_args():
    """Parse command line arguments using centralized tabular LLM evaluation parser."""
    parser = create_tabular_llm_evaluation_parser("Evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, MARVIS-T-SNe) on tabular datasets")
    
    # Add metadata validation specific arguments
    parser.add_argument(
        "--check_metadata",
        action="store_true",
        help="Check metadata coverage and exit without running evaluations"
    )
    parser.add_argument(
        "--skip_missing_metadata",
        action="store_true",
        help="Skip models with missing metadata instead of failing"
    )
    
    # Set tabular LLM-specific defaults
    parser.set_defaults(
        output_dir="./llm_baseline_results"
    )
    
    args = parser.parse_args()
    
    # models is already a list from nargs="+" so no conversion needed
    
    return args


def parse_args_old():
    """Legacy argument parser - replaced by centralized parser."""
    parser = argparse.ArgumentParser(description="Evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, MARVIS-T-SNe) on tabular datasets")
    
    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default="tabllm,tabula_8b,jolt,marvis_tsne",
        help="Comma-separated list of models to evaluate: 'tabllm', 'tabula_8b', 'jolt', 'marvis_tsne'"
    )
    
    # Dataset source options (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        help="Name of a single dataset to evaluate on"
    )
    dataset_group.add_argument(
        "--dataset_ids",
        type=str,
        help="Comma-separated list of OpenML dataset IDs to evaluate on"
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing CSV files to evaluate on"
    )
    dataset_group.add_argument(
        "--num_datasets",
        type=int,
        help="Number of random datasets to sample from OpenML for evaluation"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    
    # Model-specific configurations
    parser.add_argument(
        "--tabllm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for TabLLM baseline"
    )
    parser.add_argument(
        "--tabula_model",
        type=str,
        default="mlfoundations/tabula-8b",
        help="HuggingFace model name for Tabula-8B baseline"
    )
    parser.add_argument(
        "--jolt_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for JOLT baseline (should be a generative LLM)"
    )
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="HuggingFace model name for Vision Language Model (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1000,
        help="Size of TabPFN embeddings for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--max_tabpfn_samples",
        type=int,
        default=3000,
        help="Maximum samples for TabPFN fitting in MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache TabPFN embeddings for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Force recomputation of cached embeddings for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles instead of 2D (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--viewing_angles",
        type=str,
        default=None,
        help="Custom viewing angles for 3D t-SNE as 'elev1,azim1;elev2,azim2;...' (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors in embedding space (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--nn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show when using KNN connections (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--max_vlm_image_size",
        type=int,
        default=2048,
        help="Maximum image size (width/height) for VLM compatibility (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--image_dpi",
        type=int,
        default=100,
        help="DPI for saving t-SNE visualizations (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--force_rgb_mode",
        action="store_true",
        default=True,
        help="Convert images to RGB mode to improve VLM processing speed (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--no-force_rgb_mode",
        action="store_false",
        dest="force_rgb_mode",
        help="Disable RGB conversion (keep RGBA mode) for MARVIS-T-SNe baseline"
    )
    parser.add_argument(
        "--save_sample_visualizations",
        action="store_true",
        default=True,
        help="Save sample t-SNE visualizations for debugging and documentation (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--no-save_sample_visualizations",
        action="store_false",
        dest="save_sample_visualizations",
        help="Disable saving of sample t-SNE visualizations (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations (2.0 = 200%% zoom, showing 50%% of the range) (MARVIS-T-SNe baseline)"
    )
    parser.add_argument(
        "--visualization_save_cadence",
        type=int,
        default=10,
        help="Save visualizations for every N samples (default: 10, i.e., save every 10th visualization) (MARVIS-T-SNe baseline)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Number of training examples per class for few-shot learning (for dataset splitting). If None, uses full training set."
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=32,
        help="Number of few-shot examples to use for in-context learning"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use class-balanced few-shot examples in LLM prompts instead of random selection"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=4096,
        help="Maximum context length for LLM models"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--feature_selection_threshold",
        type=int,
        default=500,
        help="Apply feature selection if dataset has more than this many features"
    )
    
    # Hardware and performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU index to use when device is cuda"
    )
    parser.add_argument(
        "--timeout_minutes",
        type=int,
        default=30,
        help="Timeout for each model evaluation in minutes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Backend to use for model loading (auto chooses VLLM if available)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism in VLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for VLLM (0.0-1.0)"
    )
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-baselines",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    # VLM Configuration
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()




def apply_k_shot_split(dataset: Dict, k_shot: int, random_state: int = 42) -> Dict:
    """
    Apply k-shot splitting to a dataset.
    
    Args:
        dataset: Dataset dictionary with X, y, etc.
        k_shot: Number of samples per class
        random_state: Random seed for reproducibility
        
    Returns:
        Modified dataset with k-shot training split
    """
    logger = logging.getLogger(__name__)
    
    X = dataset["X"]
    y = dataset["y"]
    
    # Convert to numpy arrays if needed
    if hasattr(X, 'values'):
        X_array = X.values
        original_columns = X.columns.tolist()
    else:
        X_array = np.array(X)
        original_columns = dataset.get("attribute_names", [f"feature_{i}" for i in range(X_array.shape[1])])
    
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Create k-shot training set (class-balanced)
    X_kshot = []
    y_kshot = []
    
    unique_classes = np.unique(y_array)
    logger.info(f"Applying k-shot split: {k_shot} samples per class, {len(unique_classes)} classes")
    
    for class_label in unique_classes:
        class_mask = y_array == class_label
        class_X = X_array[class_mask]
        class_y = y_array[class_mask]
        
        # Select k samples per class
        n_samples = min(k_shot, len(class_X))
        if n_samples < k_shot:
            logger.warning(f"Class {class_label} only has {n_samples} samples, requested {k_shot}")
        
        selected_idx = np.random.RandomState(random_state).choice(
            len(class_X), n_samples, replace=False
        )
        
        X_kshot.append(class_X[selected_idx])
        y_kshot.append(class_y[selected_idx])
    
    # Combine all selected samples
    X_kshot = np.vstack(X_kshot)
    y_kshot = np.concatenate(y_kshot)
    
    # Convert back to original format
    if hasattr(dataset["X"], 'iloc'):  # DataFrame
        import pandas as pd
        X_kshot = pd.DataFrame(X_kshot, columns=original_columns)
    
    if hasattr(dataset["y"], 'iloc'):  # Series
        import pandas as pd
        y_kshot = pd.Series(y_kshot)
    
    # Create new dataset with k-shot training data
    dataset_kshot = dataset.copy()
    dataset_kshot["X"] = X_kshot
    dataset_kshot["y"] = y_kshot
    
    logger.info(f"K-shot dataset: {len(X_kshot)} total samples ({k_shot} per class * {len(unique_classes)} classes)")
    
    return dataset_kshot


def apply_balanced_few_shot_selection(X_train, y_train, num_examples: int, random_state: int = 42):
    """
    Select few-shot examples with class balance.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        num_examples: Total number of examples to select
        random_state: Random seed
        
    Returns:
        Tuple of (selected_indices, actual_num_selected)
    """
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    
    # Calculate examples per class (as evenly as possible)
    examples_per_class = num_examples // n_classes
    remainder = num_examples % n_classes
    
    selected_indices = []
    
    for i, class_label in enumerate(unique_classes):
        class_mask = y_train == class_label
        class_indices = np.where(class_mask)[0]
        
        # Add one extra example to first 'remainder' classes
        n_select = examples_per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))
        
        if n_select > 0:
            selected_class_indices = np.random.RandomState(random_state).choice(
                class_indices, n_select, replace=False
            )
            selected_indices.extend(selected_class_indices)
    
    return np.array(selected_indices), len(selected_indices)


def main():
    args = parse_args()
    
    # Handle metadata checking mode
    if args.check_metadata:
        from marvis.utils.metadata_validation import generate_metadata_coverage_report, print_metadata_coverage_report
        
        print("Checking metadata coverage for requested models...")
        
        # Get task IDs from datasets if specified
        task_ids = None
        if hasattr(args, 'dataset_ids') and args.dataset_ids:
            task_ids = args.dataset_ids
        elif hasattr(args, 'dataset_name') and args.dataset_name:
            # For single dataset, try to resolve to OpenML task ID
            print(f"Checking metadata for dataset: {args.dataset_name}")
            # This would need dataset loading logic, for now just show general report
        
        report = generate_metadata_coverage_report(task_ids, args.models)
        print_metadata_coverage_report(report)
        
        # Save report
        report_file = os.path.join(args.output_dir, "metadata_coverage_report.json")
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
        return
    
    # Set random seed for reproducibility
    from marvis.utils import set_seed_with_args
    set_seed_with_args(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"llm_baseline_evaluation_{timestamp}.log"
    logger = setup_logging(log_file=os.path.join(args.output_dir, log_filename))
    logger.info(f"Arguments: {args}")
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            if args.wandb_name is None:
                args.wandb_name = f"llm_baselines_{timestamp}"
            
            gpu_monitor = init_wandb_with_gpu_monitoring(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                output_dir=args.output_dir,
                enable_system_monitoring=True,
                gpu_log_interval=30.0,
                enable_detailed_gpu_logging=True
            )
            logger.info(f"Initialized Weights & Biases run with GPU monitoring: {args.wandb_name}")
    
    # Legacy k_shot parameter conversion to unified parameters
    if args.k_shot is not None:
        logger.warning("--k_shot is deprecated. Use --num_few_shot_examples with --balanced_few_shot instead.")
        # Convert k_shot to unified parameters if not already specified
        if args.num_few_shot_examples is None and not args.balanced_few_shot:
            args.num_few_shot_examples = args.k_shot
            args.balanced_few_shot = True
            logger.info(f"Converted --k_shot {args.k_shot} to --num_few_shot_examples {args.k_shot} --balanced_few_shot")
    
    # Validate training sample parameters early
    try:
        validate_training_sample_args(args)
    except ValueError as e:
        logger.error(f"Invalid training sample parameters: {e}")
        return
    
    # Load datasets using standardized preprocessing
    # This now handles unified training sample parameters automatically
    raw_datasets = load_datasets_for_evaluation(args)
    if not raw_datasets:
        logger.error("No datasets loaded successfully. Exiting.")
        return
    
    # Preprocess datasets with standardized train/test split and sampling
    datasets = preprocess_datasets_for_evaluation(raw_datasets, args)
    if not datasets:
        logger.error("No datasets preprocessed successfully. Exiting.")
        return
    
    # Parse models to evaluate (already a list from nargs="+")
    models_to_evaluate = args.models
    logger.info(f"Evaluating models: {models_to_evaluate}")
    
    # Evaluate each model on each dataset
    all_results = []
    
    for dataset in datasets:
        logger.info(f"\\n{'='*50}\\nEvaluating dataset: {dataset['name']}\\n{'='*50}")
        
        # Detect task type and apply filtering if requested
        from marvis.utils.task_detection import detect_task_type
        try:
            # Get target values for task detection
            y_data = dataset.get('y_train', dataset.get('y', []))
            if len(y_data) == 0:
                # If no y_train, try to get from combined data
                if 'X_train' in dataset and 'y_train' in dataset:
                    y_data = dataset['y_train']
                elif 'X' in dataset and 'y' in dataset:
                    y_data = dataset['y']
            
            # Get task_id for proper task type detection
            task_id = dataset.get('task_id') or dataset.get('id')
            task_type, detection_method = detect_task_type(y=y_data, dataset=dataset, task_id=task_id)
            logger.info(f"Detected task type: {task_type}")
            
            # Apply task type filtering
            if args.skip_classification and task_type == 'classification':
                logger.info(f"Skipping classification dataset {dataset['name']} (--skip_classification enabled)")
                continue
            elif args.skip_regression and task_type == 'regression':
                logger.info(f"Skipping regression dataset {dataset['name']} (--skip_regression enabled)")
                continue
                
        except Exception as e:
            logger.warning(f"Could not detect task type for {dataset['name']}: {e}. Proceeding with evaluation.")
            task_type = 'unknown'
        
        dataset_results = []
        
        # Validate metadata for models that require it
        # Check if dataset has task_id or we need to resolve it
        openml_task_id = None
        try:
            # Check various ways the task ID might be stored
            if hasattr(dataset, 'task_id'):
                openml_task_id = getattr(dataset, 'task_id', None)
            elif 'task_id' in dataset:
                openml_task_id = dataset['task_id']
            elif hasattr(dataset, 'openml_task_id'):
                openml_task_id = getattr(dataset, 'openml_task_id', None)
            elif 'openml_task_id' in dataset:
                openml_task_id = dataset['openml_task_id']
            elif 'id' in dataset and isinstance(dataset['id'], (int, str)):
                # We have dataset_id, need to resolve to task_id
                try:
                    dataset_id = int(dataset['id'])
                    # Use resource manager to resolve
                    from marvis.utils.resource_manager import get_resource_manager
                    rm = get_resource_manager()
                    identifiers = rm.resolve_openml_identifiers(dataset_id=dataset_id)
                    openml_task_id = identifiers.get('task_id')
                    if openml_task_id:
                        logger.debug(f"Resolved dataset_id {dataset_id} to task_id {openml_task_id}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse dataset ID as int: {e}")
                    pass
            
            if openml_task_id:
                # Calculate feature count NOT including target column for metadata validation
                X_data = dataset.get('X_train', dataset.get('X', []))
                if hasattr(X_data, 'shape'):
                    feature_count = X_data.shape[1]  # Feature count without target
                else:
                    feature_count = None
                metadata_results = validate_metadata_for_models(openml_task_id, models_to_evaluate, feature_count)
                
                # Log validation results and filter models
                valid_models = []
                for model_name in models_to_evaluate:
                    validation = metadata_results.get(model_name, {'valid': True, 'errors': [], 'warnings': []})
                    if validation['valid']:
                        valid_models.append(model_name)
                        if validation['warnings']:
                            for warning in validation['warnings']:
                                logger.warning(f"{model_name} metadata warning: {warning}")
                    else:
                        if args.skip_missing_metadata:
                            logger.warning(f"Skipping {model_name} on {dataset['name']} due to metadata validation failures:")
                            for error in validation['errors']:
                                logger.warning(f"  - {error}")
                            for missing_file in validation['missing_files']:
                                logger.warning(f"  - Missing file: {missing_file}")
                        else:
                            logger.error(f"Skipping {model_name} on {dataset['name']} due to metadata validation failures:")
                            for error in validation['errors']:
                                logger.error(f"  - {error}")
                            for missing_file in validation['missing_files']:
                                logger.error(f"  - Missing file: {missing_file}")
                
                if len(valid_models) < len(models_to_evaluate):
                    skipped = set(models_to_evaluate) - set(valid_models)
                    if args.skip_missing_metadata:
                        logger.warning(f"Skipped models due to missing metadata: {', '.join(skipped)}")
                    else:
                        logger.warning(f"Skipped models due to missing metadata: {', '.join(skipped)}")
                
                # Update models list to only include valid ones
                models_to_evaluate_filtered = valid_models
            else:
                logger.warning(f"No OpenML task ID found for dataset {dataset['name']}, skipping metadata validation")
                models_to_evaluate_filtered = models_to_evaluate
        except Exception as e:
            logger.error(f"Error during metadata validation: {e}")
            models_to_evaluate_filtered = models_to_evaluate

        for model_name in models_to_evaluate_filtered:
            logger.info(f"Evaluating {model_name} on {dataset['name']}")
            
            try:
                # Set timeout for each model evaluation
                timeout_seconds = args.timeout_minutes * 60
                
                with timeout_context(timeout_seconds):
                    if model_name.lower() == 'tabllm':
                        result = evaluate_tabllm(dataset, args)
                    elif model_name.lower() == 'tabula_8b':
                        result = evaluate_tabula_8b(dataset, args)
                    elif model_name.lower() == 'jolt':
                        result = evaluate_jolt(dataset, args)
                    elif model_name.lower() == 'marvis_tsne':
                        # Cap nn_k based on number of classes and training dataset size to avoid over-averaging
                        # Get training dataset size
                        X_train = dataset.get('X_train', dataset.get('X', []))
                        if hasattr(X_train, 'shape'):
                            train_size = X_train.shape[0]
                        elif hasattr(X_train, '__len__'):
                            train_size = len(X_train)
                        else:
                            train_size = 100  # fallback default
                        
                        # Get number of classes (use 1 for regression)
                        y_train = dataset.get('y_train', dataset.get('y', []))
                        if len(y_train) > 0:
                            # Detect if this is regression or classification
                            from marvis.utils.task_detection import detect_task_type
                            task_id = dataset.get('task_id') or dataset.get('id')
                            task_type = detect_task_type(y_train, task_id=task_id)
                            
                            if task_type == 'regression':
                                n_classes = 1
                            else:
                                unique_classes = np.unique(y_train)
                                n_classes = len(unique_classes)
                        else:
                            n_classes = 1  # fallback for regression
                        
                        # Calculate cap: max(n_classes * 2, 10% of training size)
                        nn_k_cap = max(n_classes * 2, int(train_size * 0.1))
                        original_nn_k = args.nn_k
                        effective_nn_k = min(original_nn_k, nn_k_cap)
                        
                        if effective_nn_k != original_nn_k:
                            logger.info(f"Capping nn_k from {original_nn_k} to {effective_nn_k} (max of {n_classes * 2} classes*2 or 10% of {train_size} training samples)")
                        
                        # Create modified args with capped nn_k
                        modified_args = argparse.Namespace(**vars(args))
                        modified_args.nn_k = effective_nn_k
                        
                        result = evaluate_marvis_tsne(dataset, modified_args)
                    elif model_name.lower() in ['openai_llm', 'api_llm'] and hasattr(args, 'openai_model') and args.openai_model:
                        from examples.tabular.llm_baselines.openai_llm_baseline import evaluate_openai_llm
                        result = evaluate_openai_llm(dataset, args)
                    elif model_name.lower() in ['gemini_llm', 'api_llm'] and hasattr(args, 'gemini_model') and args.gemini_model:
                        from examples.tabular.llm_baselines.gemini_llm_baseline import evaluate_gemini_llm
                        result = evaluate_gemini_llm(dataset, args)
                    else:
                        logger.warning(f"Unknown model: {model_name}. Skipping.")
                        continue
                
            except TimeoutError:
                logger.warning(f"Evaluation of {model_name} on {dataset['name']} timed out after {args.timeout_minutes} minutes")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': f'Timeout after {args.timeout_minutes} minutes',
                    'timeout': True
                }
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset['name']}: {e}")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': str(e)
                }
            
            dataset_results.append(result)
            all_results.append(result)
            
            # Log to wandb using unified metrics system
            if args.use_wandb and WANDB_AVAILABLE:
                # Initialize unified metrics logger for this model/dataset pair
                metrics_logger = MetricsLogger(
                    model_name=model_name,
                    dataset_name=dataset['name'],
                    use_wandb=True,
                    logger=logger
                )
                
                # Add explicit dataset_id to the result before logging
                result_with_dataset_id = result.copy()
                result_with_dataset_id['dataset_id'] = dataset['id']
                result_with_dataset_id['task_id'] = dataset['id']  # For MARVIS compatibility, task_id = dataset_id
                
                # Log all metrics using unified system
                metrics_logger.log_all_metrics(result_with_dataset_id)
        
        # Save results for this dataset
        save_results(dataset_results, args.output_dir, dataset['name'])
    
    # Save aggregated results using robust JSON serialization
    aggregated_file = os.path.join(args.output_dir, "aggregated_results.json")
    success = safe_json_dump(
        all_results, 
        aggregated_file, 
        logger=logger,
        minimal_fallback=True,
        indent=2
    )
    
    if success:
        print(f"Successfully saved aggregated results to {aggregated_file}")
    else:
        logger.error(f"Failed to save aggregated results to {aggregated_file}")
    
    # Log model-level aggregation metrics using unified system
    if args.use_wandb and WANDB_AVAILABLE:
        for model_name in models_to_evaluate:
            # Include results that have either accuracy (classification) or R² score (regression)
            model_results = [r for r in all_results if (
                r.get('model_name') == model_name and 
                ('accuracy' in r or 'r2_score' in r) and 
                not r.get('timeout', False)
            )]
            if model_results:
                # Initialize aggregation metrics logger
                agg_metrics_logger = MetricsLogger(
                    model_name=model_name,
                    dataset_name="aggregated",  # Special dataset name for aggregated metrics
                    use_wandb=True,
                    logger=logger
                )
                
                # Log aggregated metrics
                agg_metrics_logger.log_aggregated_metrics(model_results, prefix="average")
                
                # Log number of valid datasets as a special metric
                agg_metrics_logger._log_metric("num_valid_datasets", len(model_results))
    
    # Print summary
    logger.info(f"\\n{'='*50}\\nEVALUATION SUMMARY\\n{'='*50}")
    
    for model_name in models_to_evaluate:
        model_results = [r for r in all_results if r.get('model_name') == model_name]
        if model_results:
            # Separate classification and regression results
            classification_results = [r for r in model_results if 'accuracy' in r and r['accuracy'] is not None]
            regression_results = [r for r in model_results if 'r2_score' in r and r['r2_score'] is not None]
            
            completion_rates = [r.get('completion_rate', 1.0) for r in model_results]
            timeouts = len([r for r in model_results if r.get('timeout', False)])
            errors = len([r for r in model_results if 'error' in r and not r.get('timeout', False)])
            
            # Report metrics based on task types present
            if classification_results:
                accuracies = [r['accuracy'] for r in classification_results]
                avg_accuracy = np.mean(accuracies)
                avg_completion = np.mean(completion_rates)
                logger.info(f"{model_name}: Average accuracy = {avg_accuracy:.4f} ({len(classification_results)} classification datasets)")
            
            if regression_results:
                r2_scores = [r['r2_score'] for r in regression_results]
                avg_r2 = np.mean(r2_scores)
                logger.info(f"{model_name}: Average R² = {avg_r2:.4f} ({len(regression_results)} regression datasets)")
                
                # Also report MAE if available
                mae_scores = [r['mae'] for r in regression_results if 'mae' in r and r['mae'] is not None]
                if mae_scores:
                    avg_mae = np.mean(mae_scores)
                    logger.info(f"{model_name}: Average MAE = {avg_mae:.4f}")
            
            if classification_results or regression_results:
                avg_completion = np.mean(completion_rates)
                total_valid = len(classification_results) + len(regression_results)
                logger.info(f"{model_name}: Completion rate = {avg_completion:.1%} ({total_valid} total datasets)")
                if timeouts > 0:
                    logger.info(f"  - {timeouts} timeouts")
                if errors > 0:
                    logger.info(f"  - {errors} errors")
    
    logger.info(f"\\nResults saved to: {args.output_dir}")
    logger.info(f"Aggregated results: {aggregated_file}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)

if __name__ == "__main__":
    main()