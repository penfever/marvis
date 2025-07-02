#!/usr/bin/env python
"""
Script to train and evaluate MARVIS models on the New OpenML Suite 2025 regression collection.

This script:
1. Retrieves the New OpenML Suite 2025 regression collection (study_id=455)
2. For each task in the collection:
   a. Trains a MARVIS model on 3 different splits of the dataset
   b. Evaluates the trained model and all baselines on each split
3. Logs the results to Weights & Biases with version control by date

Requirements:
- OpenML installed (pip install openml)
- MARVIS installed and configured
- W&B account for logging results

Usage:
    python run_openml_regression_2025_tabular.py --marvis_repo_path /path/to/marvis --output_dir ./results

The script assumes the MARVIS repo structure.
"""

import os
import argparse
import subprocess
import json
import logging
import openml
from pathlib import Path
import random
import numpy as np
import torch
import time
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to path for centralized parser
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from marvis.utils.evaluation_args import create_dataset_evaluation_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openml_regression_2025_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using centralized dataset evaluation parser."""
    # Create a custom parser that doesn't require dataset arguments
    parser = argparse.ArgumentParser(description="Train and evaluate MARVIS on New OpenML Suite 2025 regression collection")
    
    # Import and add all the standard evaluation args without the required dataset constraint
    from marvis.utils.evaluation_args import (
        add_common_evaluation_args, add_model_args, add_data_processing_args, add_embedding_args, 
        add_baseline_model_args, add_tabpfn_args, add_llm_baseline_args, add_evaluation_wandb_args, 
        add_label_fitting_args, add_calibration_args, add_score_normalization_args, 
        add_minority_class_args, add_evaluation_control_args
    )
    
    # Add all argument groups
    add_common_evaluation_args(parser)
    add_model_args(parser)
    
    # Add dataset source arguments as optional (not in mutually exclusive group)
    parser.add_argument("--task_ids", type=str, help="Comma-separated list of OpenML task IDs to evaluate on. If not provided, runs on all regression 2025 tasks.")
    parser.add_argument("--dataset_name", type=str, help="Name of the OpenML dataset to evaluate on")
    parser.add_argument("--dataset_ids", type=str, help="Comma-separated list of OpenML dataset IDs to evaluate on")
    parser.add_argument("--data_dir", type=str, help="Directory containing CSV files to use as datasets")
    parser.add_argument("--num_datasets", type=int, help="Number of random datasets to sample from OpenML")
    
    # Add all other argument groups
    add_data_processing_args(parser)
    add_embedding_args(parser)
    add_baseline_model_args(parser)
    add_tabpfn_args(parser)
    add_llm_baseline_args(parser)
    add_evaluation_wandb_args(parser, "marvis-openml-regression-2025")
    add_label_fitting_args(parser)
    add_calibration_args(parser)
    add_score_normalization_args(parser)
    add_minority_class_args(parser)
    add_evaluation_control_args(parser)
    
    # Add OpenML regression 2025 orchestration-specific arguments
    parser.add_argument(
        "--marvis_repo_path",
        type=str,
        required=True,
        help="Path to the MARVIS repository"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="Number of different train/test splits to use for each task"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation on existing models"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation and only run training"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start from this task index in the regression collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the regression collection (exclusive)"
    )
    
    # Override some defaults for OpenML regression 2025 context
    parser.set_defaults(
        output_dir="./openml_regression_2025_results",
        wandb_project="marvis-openml-regression-2025",
        model_id="Qwen/Qwen2.5-3B-Instruct"
    )
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_openml_regression_2025_tasks():
    """
    Get the list of tasks in the New OpenML Suite 2025 regression collection (study_id=455).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching New OpenML Suite 2025 regression collection (study_id=455)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch regression suite using newer API method (get_suite)")
        suite = openml.study.get_suite(455)  # 455 is the ID for New_OpenML_Suite_2025_regression
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(455, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # If both methods fail, we'll need to fetch individual tasks or use a hardcoded list
            logger.error("Could not fetch regression suite. Please check study_id=455 exists and contains regression tasks.")
            return []

    logger.info(f"Retrieved {len(task_ids)} tasks from regression collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            # Verify this is a regression task
            if task.task_type.lower() != 'supervised regression':
                logger.warning(f"Task {task_id} is not a regression task (type: {task.task_type}), skipping")
                continue
            tasks.append(task)
            logger.info(f"Retrieved regression task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} regression tasks")
    return tasks

def train_on_task(task, split_idx, args):
    """
    Train a MARVIS model on a specific OpenML regression task and split.
    
    Args:
        task: OpenML task object
        split_idx: Index of the split to use
        args: Command line arguments
    
    Returns:
        Path to the trained model
    """
    task_id = task.task_id
    dataset_id = task.dataset_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Training on regression task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    model_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "model"
    )
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build training command
    train_script = os.path.join(args.marvis_repo_path, "examples", "tabular", "train_tabular_dataset_tabular.py")
    
    cmd = [
        "python", train_script,
        "--task_ids", str(task_id),  # Pass task_id properly
        "--output_dir", model_output_dir,
        "--model_id", args.model_id,
        "--batch_size", "8",
        # Use total_steps instead of num_epochs for regression
        "--total_steps", "2000",
        "--save_steps", "500",
        "--unfreeze_last_k_layers", "1",
        "--gradient_accumulation_steps", "1",
        "--early_stopping_patience", "30",
        "--early_stopping_threshold", "0.1",  # Lower threshold for regression (MSE-based)
        "--learning_rate", "1e-4",
        "--mixup_alpha", "0.0",  # Disable mixup for regression
        "--bypass_eval",  # Skip evaluation in training script as we'll do it separately
        "--use_wandb",
        "--save_best_model",
        "--wandb_entity", "nyu-dice-lab",
        "--wandb_project", wandb_project,
        "--wandb_name", f"train_regression_task{task_id}_split{split_idx}",
        "--seed", str(args.seed + split_idx),  # Vary seed for different splits
        "--feature_selection_threshold", str(args.feature_selection_threshold),
        "--task_type", "regression"  # Explicitly specify regression task type
    ]
    
    # Run training command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Training completed for regression task {task_id}, split {split_idx+1}")
        return model_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for regression task {task_id}, split {split_idx+1}: {e}")
        return None

def evaluate_model(task, split_idx, model_dir, args):
    """
    Evaluate a trained MARVIS model and baselines on a specific OpenML regression task and split.
    
    Args:
        task: OpenML task object
        split_idx: Index of the split to use
        model_dir: Path to the trained model directory
        args: Command line arguments
    
    Returns:
        Path to the evaluation results
    """
    task_id = task.task_id
    dataset_id = task.dataset_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Evaluating on regression task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "evaluation"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Generate version tag based on date for W&B project - keep consistent with training
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command
    eval_script = os.path.join(args.marvis_repo_path, "examples", "tabular", "evaluate_on_dataset_tabular.py")
    
    cmd = [
        "python", eval_script,
        "--task_ids", str(task_id),  # Pass task_id properly
        "--output_dir", eval_output_dir,
        "--model_path", model_dir,  # Use model_path parameter instead of model_dir
        "--model_id", args.model_id,  # Add model_id parameter to specify base model architecture
        "--use_wandb",
        "--wandb_entity", "nyu-dice-lab",
        "--wandb_project", wandb_project,
        "--wandb_name", f"eval_regression_task{task_id}_split{split_idx}",
        "--run_all_baselines",  # Add run_all_baselines option
        "--seed", str(args.seed + split_idx),  # Use same seed as training
        "--feature_selection_threshold", str(args.feature_selection_threshold),
        "--task_type", "regression"  # Explicitly specify regression task type
    ]
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Evaluation completed for regression task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed for regression task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single regression task: train and evaluate on multiple splits.
    
    Args:
        task: OpenML task object
        args: Command line arguments
    """
    task_id = task.task_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Processing regression task {task_id}: {dataset_name}")
    
    # Create task directory
    task_dir = os.path.join(args.output_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)
    
    # Generate version information for tracking
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    run_timestamp = today.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save task metadata
    with open(os.path.join(task_dir, f"task_info_{run_timestamp}.json"), "w") as f:
        task_info = {
            "task_id": task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset_name,
            "task_type": "regression",
            "target_attribute": task.target_name if hasattr(task, "target_name") else None,
            "num_features": len(task.get_dataset().features) if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict) else None,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "training_params": {
                "total_steps": 2000,
                "save_steps": 500,
                "early_stopping_patience": 30,
                "early_stopping_threshold": 0.1,  # MSE-based threshold for regression
                "lr_final": "1e-5",
                "mixup_alpha": "0.0",  # Disabled for regression
                "bypass_eval": True,
                "task_type": "regression"
            }
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split
    for split_idx in range(args.num_splits):
        # Train model
        model_dir = None
        if not args.skip_training:
            model_dir = train_on_task(task, split_idx, args)
            if model_dir is None:
                logger.error(f"Skipping evaluation for regression task {task_id}, split {split_idx+1} due to training failure")
                continue
        else:
            # If skipping training, look for existing model directory
            model_dir = os.path.join(args.output_dir, f"task_{task_id}", f"split_{split_idx}", "model")
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                continue
        
        # Evaluate model
        if not args.skip_evaluation:
            eval_dir = evaluate_model(task, split_idx, model_dir, args)
            if eval_dir is None:
                logger.error(f"Evaluation failed for regression task {task_id}, split {split_idx+1}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML regression 2025 tasks
    tasks = get_openml_regression_2025_tasks()
    
    if not tasks:
        logger.error("No regression tasks found. Exiting.")
        return
    
    # Filter tasks if task_ids is provided, otherwise use all regression 2025 tasks
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified tasks")
    else:
        logger.info(f"No specific tasks provided, using all {len(tasks)} regression 2025 tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing regression tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing regression task {i+1}/{len(tasks)}")
            process_task(task, args)
        except Exception as e:
            logger.error(f"Error processing regression task {task.task_id}: {e}")
    
    logger.info("All regression tasks completed")

if __name__ == "__main__":
    main()