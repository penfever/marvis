#!/usr/bin/env python
"""
Script to evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, and MARVIS-T-SNe) on the OpenML CC18 collection.

This script:
1. Retrieves the OpenML CC18 collection (study_id=99)
2. For each task in the collection:
   a. Evaluates TabLLM, Tabula-8B, JOLT, and MARVIS-T-SNe baselines on multiple splits
3. Logs the results to Weights & Biases with version control by date

Requirements:
- OpenML installed (pip install openml)
- MARVIS installed and configured
- W&B account for logging results
- RTFM package for Tabula-8B (pip install git+https://github.com/penfever/rtfm.git)
- Transformers and torch for LLM baselines
- Vision dependencies for MARVIS-T-SNe: PIL, scikit-learn, matplotlib

Usage:
    # Basic usage with all models
    python run_openml_cc18_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results
    
    # Run only MARVIS-T-SNe with 3D t-SNE and KNN connections
    python run_openml_cc18_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results \
        --models marvis_tsne --use_3d --use_knn_connections --nn_k 7
    
    # Run with custom image settings
    python run_openml_cc18_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results \
        --models marvis_tsne --max_vlm_image_size 1024 --image_dpi 72 --no-force_rgb_mode

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

# Import centralized argument parser
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from marvis.utils.evaluation_args import create_tabular_llm_evaluation_parser
from marvis.utils.metadata_validation import generate_metadata_coverage_report, print_metadata_coverage_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openml_cc18_llm_baselines.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using centralized tabular LLM evaluation parser."""
    parser = create_tabular_llm_evaluation_parser("Evaluate LLM baselines on OpenML CC18 collection")
    
    # Remove the dataset source requirement since we automatically fetch CC18
    # Find the mutually exclusive group and make it not required
    for action_group in parser._mutually_exclusive_groups:
        if any(action.dest in ['dataset_name', 'dataset_ids', 'data_dir', 'num_datasets'] 
               for action in action_group._group_actions):
            action_group.required = False
            break
    
    # Add OpenML CC18 orchestration-specific arguments
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
        help="Start from this task index in the CC18 collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the CC18 collection (exclusive)"
    )
    parser.add_argument(
        "--validate_metadata_only",
        action="store_true",
        help="Only validate metadata coverage, don't run evaluations"
    )
    parser.add_argument(
        "--skip_missing_metadata",
        action="store_true",
        help="Automatically skip tasks with incomplete metadata instead of failing"
    )
    
    # Override some defaults for OpenML CC18 context
    parser.set_defaults(
        output_dir="./openml_cc18_llm_baselines_results",
        wandb_project="marvis-openml-cc18-llm-baselines",
        num_few_shot_examples=16,
        models=["tabllm", "tabula_8b", "jolt", "marvis_tsne"]
    )
    
    args = parser.parse_args()
    
    # Convert models back to comma-separated string for internal processing
    if isinstance(args.models, list):
        args.models = ",".join(args.models)
    
    return args

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_openml_cc18_tasks():
    """
    Get the list of tasks in the OpenML CC18 collection (study_id=99).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching OpenML CC18 collection (study_id=99)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch CC18 using newer API method (get_suite)")
        suite = openml.study.get_suite(99)  # 99 is the ID for CC18
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(99, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # Hardcoded list of CC18 tasks as a last resort
            logger.info("Using hardcoded list of CC18 tasks")
            task_ids = [
                3573, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3917, 3918,
                3950, 3954, 7592, 7593, 9914, 9946, 9957, 9960, 9961, 9962, 9964, 9965, 9966, 9967, 9968,
                9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 9987, 10060, 10061,
                10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10075, 10076,
                10077, 10078, 10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089,
                10090, 10092, 10093, 10096, 10097, 10098, 10099, 10100, 10101, 14954, 14965, 14969, 14970,
                125920, 125921, 125922, 125923, 125928, 125929, 125920, 125921, 125922, 125923, 125928,
                125929, 125930, 125931, 125932, 125933, 125934, 14954, 14965, 14969, 14970, 34536, 34537,
                34539, 146574
            ]
            # Remove duplicates
            task_ids = list(set(task_ids))

    logger.info(f"Retrieved {len(task_ids)} tasks from CC18 collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            tasks.append(task)
            logger.info(f"Retrieved task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} tasks")
    return tasks

def evaluate_llm_baselines_on_task(task, split_idx, args):
    """
    Evaluate LLM baselines on a specific OpenML task and split.
    
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
    
    logger.info(f"Evaluating LLM baselines on task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "llm_baselines"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command using the LLM baselines script
    eval_script = os.path.join(args.marvis_repo_path, "examples", "tabular", "evaluate_llm_baselines_tabular.py")
    
    cmd = [
        "python", eval_script,
        "--task_ids", str(task_id),  # Pass task_id properly
        "--output_dir", eval_output_dir,
        "--models"
    ]
    
    # Add models as separate arguments (space-separated, not comma-separated)
    models_list = [model.strip() for model in args.models.split(',')]
    cmd.extend(models_list)
    
    cmd.extend([
        "--num_few_shot_examples", str(args.num_few_shot_examples),
        "--seed", str(args.seed + split_idx),  # Vary seed for different splits
        "--device", args.device,
    ])
    
    # Add balanced few-shot flag if specified
    if args.balanced_few_shot:
        cmd.append("--balanced_few_shot")
    
    # Add optional parameters
    if args.max_test_samples:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])
    
    # Add feature selection parameter
    cmd.extend(["--feature_selection_threshold", str(args.feature_selection_threshold)])
    
    # Add backend parameters
    cmd.extend([
        "--backend", args.backend,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
    ])
    
    # Add MARVIS-T-SNe specific parameters
    cmd.extend([
        "--vlm_model_id", args.vlm_model_id,
        "--embedding_size", str(args.embedding_size),
        "--tsne_perplexity", str(args.tsne_perplexity),
        "--tsne_max_iter", str(getattr(args, 'tsne_max_iter', getattr(args, 'tsne_n_iter', 1000))),
        "--max_tabpfn_samples", str(args.max_tabpfn_samples),
        "--nn_k", str(args.nn_k),
        "--zoom_factor", str(args.zoom_factor),
        "--max_vlm_image_size", str(args.max_vlm_image_size),
        "--image_dpi", str(args.image_dpi),
    ])
    
    # Add MARVIS-T-SNe boolean flags
    if args.use_3d:
        cmd.append("--use_3d")
    if args.use_knn_connections:
        cmd.append("--use_knn_connections")
    if args.force_rgb_mode:
        cmd.append("--force_rgb_mode")
    else:
        cmd.append("--no-force_rgb_mode")
    if args.save_sample_visualizations:
        cmd.append("--save_sample_visualizations")
    else:
        cmd.append("--no-save_sample_visualizations")
    if args.use_semantic_names:
        cmd.append("--use_semantic_names")
    if args.use_metadata:
        cmd.append("--use_metadata")
    if args.show_test_points:
        cmd.append("--show_test_points")
    
    # Add visualization save cadence
    cmd.extend(["--visualization_save_cadence", str(args.visualization_save_cadence)])
    
    # Add custom viewing angles if specified
    if args.viewing_angles:
        cmd.extend(["--viewing_angles", args.viewing_angles])
    
    # Add model-specific arguments
    cmd.extend([
        "--tabllm_model", args.tabllm_model,
        "--tabula_model", args.tabula_model,
        "--jolt_model", args.jolt_model,
    ])
    
    # Add API model arguments if specified
    if hasattr(args, 'openai_model') and args.openai_model:
        cmd.extend(["--openai_model", args.openai_model])
    if hasattr(args, 'gemini_model') and args.gemini_model:
        cmd.extend(["--gemini_model", args.gemini_model])
    if hasattr(args, 'enable_thinking'):
        if args.enable_thinking:
            cmd.append("--enable_thinking")
        else:
            cmd.append("--disable_thinking")
    
    # Add W&B parameters if we want to track (commented out by default to avoid clutter)
    cmd.extend([
        "--use_wandb",
        "--wandb_project", wandb_project,
        "--wandb_name", f"llm_baselines_task{task_id}_split{split_idx}",
    ])
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Debug - args.use_metadata: {args.use_metadata}")
    logger.info(f"Debug - args.balanced_few_shot: {args.balanced_few_shot}")
    logger.info(f"Debug - args.use_semantic_names: {args.use_semantic_names}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"LLM baseline evaluation completed for task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"LLM baseline evaluation failed for task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single task: evaluate LLM baselines on multiple splits.
    
    Args:
        task: OpenML task object
        args: Command line arguments
    """
    task_id = task.task_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Processing task {task_id}: {dataset_name}")
    
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
            "num_classes": len(task.class_labels) if hasattr(task, "class_labels") else None,
            "num_features": len(task.get_dataset().features) if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict) else None,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "evaluation_params": {
                "models": args.models,
                "num_few_shot_examples": args.num_few_shot_examples,
                "max_test_samples": args.max_test_samples,
                "device": args.device,
                "backend": args.backend,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "marvis_tsne_params": {
                    "vlm_model_id": args.vlm_model_id,
                    "embedding_size": args.embedding_size,
                    "tsne_perplexity": args.tsne_perplexity,
                    "tsne_max_iter": getattr(args, 'tsne_max_iter', getattr(args, 'tsne_n_iter', 1000)),
                    "max_tabpfn_samples": args.max_tabpfn_samples,
                    "use_3d": args.use_3d,
                    "viewing_angles": args.viewing_angles,
                    "use_knn_connections": args.use_knn_connections,
                    "nn_k": args.nn_k,
                    "max_vlm_image_size": args.max_vlm_image_size,
                    "image_dpi": args.image_dpi,
                    "force_rgb_mode": args.force_rgb_mode,
                    "save_sample_visualizations": args.save_sample_visualizations,
                    "visualization_save_cadence": args.visualization_save_cadence
                }
            }
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split
    for split_idx in range(args.num_splits):
        # Evaluate LLM baselines
        eval_dir = evaluate_llm_baselines_on_task(task, split_idx, args)
        if eval_dir is None:
            logger.error(f"LLM baseline evaluation failed for task {task_id}, split {split_idx+1}")

def aggregate_results(args):
    """
    Aggregate results from all tasks and splits into summary files.
    
    Args:
        args: Command line arguments
    """
    logger.info("Aggregating results from all tasks and splits")
    
    all_results = []
    summary_by_model = {}
    
    # Walk through all result directories
    for task_dir in os.listdir(args.output_dir):
        if not task_dir.startswith("task_"):
            continue
        
        task_path = os.path.join(args.output_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        
        task_id = task_dir.replace("task_", "")
        
        for split_dir in os.listdir(task_path):
            if not split_dir.startswith("split_"):
                continue
            
            split_path = os.path.join(task_path, split_dir)
            if not os.path.isdir(split_path):
                continue
            
            split_idx = split_dir.replace("split_", "")
            
            # Look for LLM baseline results
            llm_baselines_path = os.path.join(split_path, "llm_baselines")
            if not os.path.exists(llm_baselines_path):
                continue
            
            # Check for aggregated results file
            aggregated_file = os.path.join(llm_baselines_path, "aggregated_results.json")
            if os.path.exists(aggregated_file):
                try:
                    with open(aggregated_file, 'r') as f:
                        results = json.load(f)
                    
                    for result in results:
                        result['task_id'] = task_id
                        result['split_idx'] = split_idx
                        all_results.append(result)
                        
                        # Update summary by model
                        model_name = result.get('model_name', 'unknown')
                        if model_name not in summary_by_model:
                            summary_by_model[model_name] = {
                                'accuracies': [],
                                'balanced_accuracies': [],
                                'total_tasks': 0,
                                'successful_tasks': 0
                            }
                        
                        summary_by_model[model_name]['total_tasks'] += 1
                        if 'accuracy' in result:
                            summary_by_model[model_name]['accuracies'].append(result['accuracy'])
                            summary_by_model[model_name]['successful_tasks'] += 1
                        if 'balanced_accuracy' in result:
                            summary_by_model[model_name]['balanced_accuracies'].append(result['balanced_accuracy'])
                
                except Exception as e:
                    logger.error(f"Error reading results from {aggregated_file}: {e}")
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save all results
    all_results_file = os.path.join(args.output_dir, f"all_results_{timestamp}.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate and save summary statistics
    summary_stats = {}
    for model_name, stats in summary_by_model.items():
        if stats['accuracies']:
            summary_stats[model_name] = {
                'mean_accuracy': np.mean(stats['accuracies']),
                'std_accuracy': np.std(stats['accuracies']),
                'mean_balanced_accuracy': np.mean(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                'std_balanced_accuracy': np.std(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                'total_evaluations': stats['total_tasks'],
                'successful_evaluations': stats['successful_tasks'],
                'success_rate': stats['successful_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
            }
    
    summary_file = os.path.join(args.output_dir, f"summary_stats_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Log summary
    logger.info(f"Aggregation complete. Found {len(all_results)} total results.")
    for model_name, stats in summary_stats.items():
        logger.info(f"{model_name}: Mean accuracy = {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                   f"({stats['successful_evaluations']}/{stats['total_evaluations']} successful)")
    
    logger.info(f"All results saved to: {all_results_file}")
    logger.info(f"Summary statistics saved to: {summary_file}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML CC18 tasks
    tasks = get_openml_cc18_tasks()
    
    # Filter tasks if task_ids is provided
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
    # Parse models to check
    models_to_check = [model.strip() for model in args.models.split(',')]
    
    # Handle metadata validation
    if args.validate_metadata_only:
        logger.info("Running metadata validation only...")
        task_ids = [task.task_id for task in tasks]
        report = generate_metadata_coverage_report(task_ids, models_to_check)
        print_metadata_coverage_report(report)
        
        # Save detailed report
        report_file = os.path.join(args.output_dir, "metadata_coverage_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed metadata report saved to: {report_file}")
        return
    
    # Filter tasks based on metadata availability if requested
    if args.skip_missing_metadata:
        logger.info("Filtering tasks based on metadata availability...")
        task_ids = [task.task_id for task in tasks]
        report = generate_metadata_coverage_report(task_ids, models_to_check)
        
        # Keep only tasks where at least one model has valid metadata
        valid_task_ids = []
        for task_id, results in report['detailed_results'].items():
            if any(result['valid'] for result in results.values()):
                valid_task_ids.append(task_id)
        
        # Filter tasks list
        original_count = len(tasks)
        tasks = [task for task in tasks if task.task_id in valid_task_ids]
        logger.info(f"Filtered {original_count} tasks to {len(tasks)} tasks with valid metadata")
        
        if len(tasks) == 0:
            logger.error("No tasks have valid metadata for any of the requested models. Exiting.")
            return
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            process_task(task, args)
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
    
    # Aggregate results at the end
    try:
        aggregate_results(args)
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")
    
    logger.info("All tasks completed")

if __name__ == "__main__":
    main()