#!/usr/bin/env python
"""
Script to evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, and MARVIS-T-SNe) on the New OpenML Suite 2025 regression collection.

This script:
1. Retrieves the New OpenML Suite 2025 regression collection (study_id=455)
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
    python run_openml_regression_2025_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results
    
    # Run only MARVIS-T-SNe with 3D t-SNE and KNN connections
    python run_openml_regression_2025_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results \
        --models marvis_tsne --use_3d --use_knn_connections --nn_k 7
    
    # Run with custom image settings for regression visualization
    python run_openml_regression_2025_llm_baselines.py --marvis_repo_path /path/to/marvis --output_dir ./results \
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
        logging.FileHandler("openml_regression_2025_llm_baselines.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using centralized tabular LLM evaluation parser."""
    parser = create_tabular_llm_evaluation_parser("Evaluate LLM baselines on New OpenML Suite 2025 regression collection")
    
    # Remove the dataset source requirement since we automatically fetch regression tasks
    # Find the mutually exclusive group and make it not required
    for action_group in parser._mutually_exclusive_groups:
        if any(action.dest in ['dataset_name', 'dataset_ids', 'data_dir', 'num_datasets'] 
               for action in action_group._group_actions):
            action_group.required = False
            break
    
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
        help="Start from this task index in the regression collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the regression collection (exclusive)"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if output files already exist"
    )
    parser.add_argument(
        "--save_detailed_outputs",
        action="store_true",
        help="Save detailed VLM outputs and visualizations"
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
    
    # Override some defaults for OpenML regression 2025 context
    parser.set_defaults(
        output_dir="./openml_regression_2025_llm_results",
        wandb_project="marvis-regression-llm-baselines-2025",
        models=["marvis_tsne", "tabllm", "jolt"],  # Remove tabula_8b for now as it may need adaptation
        model_id="Qwen/Qwen2.5-3B-Instruct",
        nn_k=7,
        use_3d=False,  # Default to 2D for better performance and compatibility
        preserve_regression=True,  # Keep regression tasks as continuous targets
        num_few_shot_examples=16
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

    logger.info(f"Retrieved {len(task_ids)} task IDs from regression collection")
    
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

def evaluate_llm_baselines_on_task(task, split_idx, args):
    """
    Evaluate LLM baselines on a specific OpenML regression task and split.
    
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
    
    logger.info(f"Evaluating LLM baselines on regression task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "llm_baselines"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Check if results already exist and skip if not forcing rerun
    results_file = os.path.join(eval_output_dir, "aggregated_results.json")
    if os.path.exists(results_file) and not getattr(args, 'force_rerun', False):
        logger.info(f"Results already exist for task {task_id}, split {split_idx+1}. Skipping.")
        return eval_output_dir
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command using the LLM baselines script
    eval_script = os.path.join(args.marvis_repo_path, "examples", "tabular", "evaluate_llm_baselines_tabular.py")
    
    cmd = [
        "python", eval_script,
        "--task_ids", str(task_id),  # Pass task_id properly for regression
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
        "--preserve_regression",  # Ensure regression tasks remain continuous
    ])
    
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
    
    # Add W&B parameters
    cmd.extend([
        "--use_wandb",
        "--wandb_project", wandb_project,
        "--wandb_name", f"llm_baselines_regression_task{task_id}_split{split_idx}",
    ])
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"LLM baseline evaluation completed for regression task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"LLM baseline evaluation failed for regression task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single regression task: evaluate all baselines on multiple splits.
    
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
            "models_evaluated": args.models,
            "evaluation_params": {
                "num_splits": args.num_splits,
                "nn_k": args.nn_k,
                "use_3d": args.use_3d,
                "use_knn_connections": args.use_knn_connections,
                "max_vlm_image_size": args.max_vlm_image_size,
                "seed": args.seed
            }
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split
    for split_idx in range(args.num_splits):
        # Evaluate LLM baselines
        eval_dir = evaluate_llm_baselines_on_task(task, split_idx, args)
        if eval_dir is None:
            logger.error(f"LLM baseline evaluation failed for regression task {task_id}, split {split_idx+1}")

def generate_metadata_report(tasks, args):
    """Generate a metadata coverage report for the regression tasks."""
    logger.info("Generating metadata coverage report for regression tasks...")
    
    # Create a summary of tasks and their metadata
    task_summary = []
    for task in tasks:
        try:
            dataset = task.get_dataset()
            task_summary.append({
                "task_id": task.task_id,
                "dataset_name": dataset.name,
                "task_type": task.task_type,
                "target_attribute": task.target_name if hasattr(task, "target_name") else "unknown",
                "num_features": len(dataset.features) if hasattr(dataset, "features") and isinstance(dataset.features, dict) else 0,
                "num_instances": dataset.qualities.get("NumberOfInstances", 0) if hasattr(dataset, "qualities") else 0
            })
        except Exception as e:
            logger.warning(f"Could not extract metadata for task {task.task_id}: {e}")
    
    # Save metadata report
    report_path = os.path.join(args.output_dir, "regression_tasks_metadata_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_tasks": len(tasks),
            "report_timestamp": datetime.now().isoformat(),
            "tasks": task_summary
        }, f, indent=2)
    
    logger.info(f"Metadata report saved to {report_path}")
    return task_summary

def aggregate_results(args):
    """
    Aggregate results from all tasks and splits into summary files.
    
    Args:
        args: Command line arguments
    """
    logger.info("Aggregating results from all regression tasks and splits")
    
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
                                'r2_scores': [],
                                'mae_scores': [],
                                'mse_scores': [],
                                'total_tasks': 0,
                                'successful_tasks': 0
                            }
                        
                        summary_by_model[model_name]['total_tasks'] += 1
                        # For regression, track R², MAE, and MSE instead of accuracy
                        if 'r2_score' in result:
                            summary_by_model[model_name]['r2_scores'].append(result['r2_score'])
                            summary_by_model[model_name]['successful_tasks'] += 1
                        if 'mae' in result:
                            summary_by_model[model_name]['mae_scores'].append(result['mae'])
                        if 'mse' in result:
                            summary_by_model[model_name]['mse_scores'].append(result['mse'])
                
                except Exception as e:
                    logger.error(f"Error reading results from {aggregated_file}: {e}")
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save all results
    all_results_file = os.path.join(args.output_dir, f"all_regression_results_{timestamp}.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate and save summary statistics
    summary_stats = {}
    for model_name, stats in summary_by_model.items():
        if stats['r2_scores']:
            summary_stats[model_name] = {
                'mean_r2_score': np.mean(stats['r2_scores']),
                'std_r2_score': np.std(stats['r2_scores']),
                'mean_mae': np.mean(stats['mae_scores']) if stats['mae_scores'] else None,
                'std_mae': np.std(stats['mae_scores']) if stats['mae_scores'] else None,
                'mean_mse': np.mean(stats['mse_scores']) if stats['mse_scores'] else None,
                'std_mse': np.std(stats['mse_scores']) if stats['mse_scores'] else None,
                'total_evaluations': stats['total_tasks'],
                'successful_evaluations': stats['successful_tasks'],
                'success_rate': stats['successful_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
            }
    
    summary_file = os.path.join(args.output_dir, f"regression_summary_stats_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Log summary
    logger.info(f"Aggregation complete. Found {len(all_results)} total regression results.")
    for model_name, stats in summary_stats.items():
        logger.info(f"{model_name}: Mean R² = {stats['mean_r2_score']:.4f} ± {stats['std_r2_score']:.4f}")
        if stats['mean_mae']:
            logger.info(f"  Mean MAE = {stats['mean_mae']:.4f} ± {stats['std_mae']:.4f}")
        logger.info(f"  ({stats['successful_evaluations']}/{stats['total_evaluations']} successful)")
    
    logger.info(f"All results saved to: {all_results_file}")
    logger.info(f"Summary statistics saved to: {summary_file}")

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
    
    # Generate metadata report
    generate_metadata_report(tasks, args)
    
    # Filter tasks if task_ids is provided
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified regression tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing regression tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
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
        logger.info("Filtering regression tasks based on metadata availability...")
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
        logger.info(f"Filtered {original_count} regression tasks to {len(tasks)} tasks with valid metadata")
        
        if len(tasks) == 0:
            logger.error("No regression tasks have valid metadata for any of the requested models. Exiting.")
            return
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing regression task {i+1}/{len(tasks)}")
            process_task(task, args)
        except Exception as e:
            logger.error(f"Error processing regression task {task.task_id}: {e}")
    
    # Aggregate results at the end
    try:
        aggregate_results(args)
    except Exception as e:
        logger.error(f"Error aggregating regression results: {e}")
    
    logger.info("All regression tasks completed")

if __name__ == "__main__":
    main()