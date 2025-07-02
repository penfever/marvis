#!/usr/bin/env python
"""
Script to evaluate non-LLM, non-MARVIS baselines on the New OpenML Suite 2025 regression collection.

This script:
1. Retrieves the New OpenML Suite 2025 regression collection (study_id=455)
2. For each task in the collection:
   a. Evaluates all traditional ML regression baselines (Random Forest, XGBoost, etc.) on 3 different splits
3. Logs the results to Weights & Biases with version control by date

This is a simplified version of run_openml_regression_2025_tabular.py that focuses only on baseline evaluation,
using the evaluate_on_dataset script with --run_all_baselines and --baselines_only flags.

Requirements:
- OpenML installed (pip install openml)
- MARVIS installed and configured (for baseline evaluation utilities)
- W&B account for logging results

Usage:
    # Basic usage - evaluate all baselines on all regression 2025 tasks (default behavior)
    python run_openml_regression_2025_baselines_tabular.py --marvis_repo_path /path/to/marvis --output_dir ./baseline_results
    
    # Test on specific tasks (optional)
    python run_openml_regression_2025_baselines_tabular.py --marvis_repo_path /path/to/marvis --task_ids "361085,361086" --output_dir ./test_results
    
    # Run with limited test samples for quick testing
    python run_openml_regression_2025_baselines_tabular.py --marvis_repo_path /path/to/marvis --max_test_samples 1000 --output_dir ./quick_test
    
    # Run with both training and test sample limits
    python run_openml_regression_2025_baselines_tabular.py --marvis_repo_path /path/to/marvis --max_train_samples 5000 --max_test_samples 1000 --output_dir ./limited_test
    
    # Run without W&B logging
    python run_openml_regression_2025_baselines_tabular.py --marvis_repo_path /path/to/marvis --no_wandb --output_dir ./local_results
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
        logging.FileHandler("openml_regression_2025_baselines_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using centralized dataset evaluation parser."""
    # Create a custom parser that doesn't require dataset arguments
    parser = argparse.ArgumentParser(description="Evaluate baseline models on New OpenML Suite 2025 regression collection")
    
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
    add_evaluation_wandb_args(parser, "marvis-openml-regression-2025-baselines")
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
        "--start_idx",
        type=int,
        default=0,
        help="Start from this task index in the regression 2025 collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the regression 2025 collection (exclusive)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Override some defaults for OpenML regression 2025 context
    parser.set_defaults(
        output_dir="./openml_regression_2025_baseline_results",
        wandb_project="marvis-openml-regression-2025-baselines",
        preserve_regression=True  # Ensure regression tasks are preserved
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
        logger.info("Attempting to fetch regression 2025 using newer API method (get_suite)")
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
            logger.error("Could not fetch regression 2025 suite. Please check study_id=455 exists and contains regression tasks.")
            return []

    logger.info(f"Retrieved {len(task_ids)} tasks from regression 2025 collection")
    
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

def evaluate_baselines_on_task(task, split_idx, args):
    """
    Evaluate baseline models on a specific OpenML regression task and split.
    
    Args:
        task: OpenML task object
        split_idx: Index of the split to use
        args: Command line arguments
    
    Returns:
        Path to the evaluation results
    """
    task_id = task.task_id
    dataset_id = task.dataset_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Evaluating baselines on regression task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "baselines"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command
    eval_script = os.path.join(args.marvis_repo_path, "examples", "tabular", "evaluate_on_dataset_tabular.py")
    
    cmd = [
        "python", eval_script,
        "--task_ids", str(task_id),  # Pass task_id properly
        "--output_dir", eval_output_dir,
        "--run_all_baselines",  # Run all baseline models
        "--baselines_only",  # Skip MARVIS model evaluation
        "--only_ground_truth_classes",  # Only use ground truth classes
        "--preserve_regression",  # Ensure regression tasks are preserved
        "--task_type", "regression",  # Explicitly specify task type
        "--seed", str(args.seed + split_idx)  # Use different seed for each split
    ]
    
    # Add W&B logging if not disabled
    if not args.no_wandb:
        cmd.extend([
            "--use_wandb",
            "--wandb_entity", "nyu-dice-lab",
            "--wandb_project", wandb_project,
            "--wandb_name", f"baselines_regression_task{task_id}_split{split_idx}"
        ])
    
    # Add sample limits if specified
    if args.max_test_samples:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])
    if args.max_train_samples:
        cmd.extend(["--max_train_samples", str(args.max_train_samples)])
    
    # Add feature selection parameter
    cmd.extend(["--feature_selection_threshold", str(args.feature_selection_threshold)])
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Baseline evaluation completed for regression task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Baseline evaluation failed for regression task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single regression task: evaluate baselines on multiple splits.
    
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
    with open(os.path.join(task_dir, f"task_info_baselines_{run_timestamp}.json"), "w") as f:
        task_info = {
            "task_id": task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset_name,
            "task_type": "regression",
            "target_attribute": task.target_name if hasattr(task, "target_name") else None,
            "num_features": len(task.get_dataset().features) if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict) else None,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "evaluation_type": "baselines_only",
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "num_splits": args.num_splits
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split
    for split_idx in range(args.num_splits):
        # Evaluate baselines
        eval_dir = evaluate_baselines_on_task(task, split_idx, args)
        if eval_dir is None:
            logger.error(f"Baseline evaluation failed for regression task {task_id}, split {split_idx+1}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML regression 2025 tasks
    tasks = get_openml_regression_2025_tasks()
    
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
    
    logger.info("All regression baseline evaluations completed")

if __name__ == "__main__":
    main()