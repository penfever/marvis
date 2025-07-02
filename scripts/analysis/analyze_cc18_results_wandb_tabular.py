#!/usr/bin/env python
"""
Script to analyze results from OpenML CC18 experiments using Weights & Biases data.

This script:
1. Retrieves experiment results from W&B using the API
2. Filters runs that start with 'eval_'
3. Compiles performance metrics across all tasks and splits
4. Generates summary statistics and comparisons between MARVIS and baselines
5. Creates visualizations of the results
6. Exports the results to CSV/JSON for further analysis

Variable Distinctions:
- dataset_name (str): Human-readable name like "kr-vs-kp", "adult", etc.
- dataset_id (int): OpenML dataset ID (only available in some run types)
- task_id (int): OpenML task ID, used as primary identifier for grouping

Different run types handle these variables differently:
1. Standard baselines (catboost, etc.): dataset_name and dataset_id from summary
2. LLM baselines (marvis_tsne, etc.): dataset_name from metric keys, task_id from mapping
3. MARVIS training runs: dataset_name from config, task_id from run name

Usage:
    python analyze_cc18_results_wandb.py --wandb_entity nyu-dice-lab --wandb_project marvis-openml-cc18-hero1
    python analyze_cc18_results_wandb.py --wandb_entity nyu-dice-lab --wandb_project marvis-openml-cc18-hero1 --all_algs_behavior impute
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import logging
import wandb
import re

# Import W&B extraction utilities from MARVIS utils
from marvis.utils.wandb_extractor import (
    fetch_wandb_data,
    fetch_wandb_train_data,
    extract_task_id_from_dataset_name,  # Keep for backward compatibility
    extract_variables_from_wandb_data,  # New function for proper variable extraction
    extract_split_idx_from_name,
    extract_task_idx_from_name,
    is_numeric,
    safe_float_convert,
    should_exclude_failed_run,
    extract_model_metrics_from_summary,
    extract_results_from_wandb,
    detect_run_types
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openml_cc18_analysis_wandb.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def normalize_task_id(task_id):
    """
    Normalize task ID - simply returns the task_id as-is since we're now using
    OpenML mapping system for consistency.
    
    Args:
        task_id: The task ID to normalize (can be int or str)
        
    Returns:
        Normalized task ID
    """
    try:
        # Convert to int if it's a string number
        if isinstance(task_id, str) and task_id.isdigit():
            task_id = int(task_id)
        
        return task_id
    except (ValueError, TypeError):
        # If conversion fails, return the original
        return task_id

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze OpenML CC18 experiment results from W&B")
    
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=True,
        help="W&B entity name (username or team name)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        nargs="+",
        required=True,
        help="W&B project name(s) - can specify multiple projects"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./openml_cc18_analysis_wandb",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--plot_format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Format for saved plots"
    )
    parser.add_argument(
        "--save_raw_data",
        action="store_true",
        help="Save raw data fetched from W&B to CSV"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--all_algs_behavior",
        type=str,
        default="none",
        choices=["none", "impute", "drop"],
        help="How to handle datasets where algorithms failed. 'none': current behavior, 'impute': assign random chance scores, 'drop': drop datasets with any failures"
    )
    
    return parser.parse_args()




def get_random_chance_score(metric_name, n_classes=None):
    """
    Calculate the random chance score for a given metric.
    
    Args:
        metric_name: Name of the metric (can be a path like "dataset/name/metric" or just "metric")
        n_classes: Number of classes in the dataset (for classification metrics)
        
    Returns:
        Random chance score for the metric
    """
    # Extract the actual metric name from the path if needed
    # Handle cases like "dataset/credit-approval/balanced_accuracy" -> "balanced_accuracy"
    if '/' in metric_name:
        simple_metric = metric_name.split('/')[-1]
    else:
        simple_metric = metric_name
    
    # Also handle cases where the metric name has prefixes like "test_" or suffixes
    simple_metric = simple_metric.replace('test_', '').replace('val_', '').replace('train_', '')
    
    if simple_metric in ["accuracy", "average_accuracy", "balanced_accuracy"]:
        # For accuracy metrics, random chance is 1/n_classes
        if n_classes is not None and n_classes > 0:
            return 1.0 / n_classes
        else:
            # Default to binary classification if n_classes unknown
            return 0.5
    elif simple_metric in ["f1_macro", "f1_weighted", "f1", "f1_score"]:
        # For F1 scores, random chance is approximately 1/n_classes for macro
        # For weighted, it depends on class distribution, but we'll use 1/n_classes as approximation
        if n_classes is not None and n_classes > 0:
            return 1.0 / n_classes
        else:
            return 0.5
    elif simple_metric in ["auc", "roc_auc"]:
        # For AUC, random chance is 0.5
        return 0.5
    elif simple_metric in ["precision", "recall", "precision_macro", "recall_macro"]:
        # For precision/recall, random chance is approximately 1/n_classes
        if n_classes is not None and n_classes > 0:
            return 1.0 / n_classes
        else:
            return 0.5
    elif simple_metric in ["log_loss", "cross_entropy"]:
        # For log loss, random chance depends on n_classes
        # Random chance log loss = -log(1/n_classes) = log(n_classes)
        if n_classes is not None and n_classes > 0:
            return np.log(n_classes)
        else:
            return np.log(2)  # Binary case
    else:
        # For metrics that are not performance-related (like time, counts, etc.), return 0
        non_performance_metrics = [
            "training_time", "prediction_time", "total_time", "timeout",
            "num_classes", "num_features", "num_samples", "num_test_samples", 
            "completed_samples", "total_samples", "completion_rate",
            "dmft_total_samples", "dmft_completed_samples", "dmft_completion_rate",
            "gpu_avg_power_watts", "gpu_total_energy_wh"
        ]
        
        if simple_metric in non_performance_metrics:
            return 0.0
        else:
            # For unknown performance metrics, use a conservative estimate
            logger.debug(f"Unknown performance metric {simple_metric} (from {metric_name}), using 0 as random chance score")
            return 0.0


def sanitize_filename(filename):
    """
    Sanitize a filename by replacing characters that could cause problems.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized filename that's safe to use
    """
    # Replace slashes and other problematic characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Also replace spaces with underscores for better filenames
    sanitized = sanitized.replace(' ', '_')
    # Remove redundant text
    sanitized = sanitized.replace('average_', '')
    # Limit the length to avoid overly long filenames
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def get_dataset_n_classes(results, task_id):
    """
    Try to extract the number of classes for a dataset from the results.
    
    Args:
        results: Dictionary mapping task IDs to evaluation results
        task_id: The task ID to get n_classes for
        
    Returns:
        Number of classes if found, None otherwise
    """
    if task_id not in results:
        return None
    
    task_data = results[task_id]
    
    # Try to get n_classes from task info
    if "info" in task_data:
        if "n_classes" in task_data["info"]:
            return task_data["info"]["n_classes"]
        elif "num_classes" in task_data["info"]:
            return task_data["info"]["num_classes"]
    
    # Try to infer from the data structure if available
    for split_idx, split_data in task_data.get("splits", {}).items():
        for model_name, model_data in split_data.items():
            if isinstance(model_data, dict):
                # Check in info section
                if "info" in model_data:
                    if "n_classes" in model_data["info"]:
                        return model_data["info"]["n_classes"]
                    elif "num_classes" in model_data["info"]:
                        return model_data["info"]["num_classes"]
                # Also check in metrics section
                if "metrics" in model_data:
                    if "num_classes" in model_data["metrics"]:
                        return int(model_data["metrics"]["num_classes"])
                    elif "n_classes" in model_data["metrics"]:
                        return int(model_data["metrics"]["n_classes"])
    
    # Default to None if we can't find it
    return None


def aggregate_splits(results, all_algs_behavior="none"):
    """
    Aggregate metrics across splits for each task and model.
    Uses only task_id for split detection and failure logic.
    
    Args:
        results: Dictionary mapping task IDs to evaluation results with splits
        all_algs_behavior: How to handle datasets where algorithms failed
            - "none": current behavior (default)
            - "impute": assign random chance scores for failed dataset/metric pairs
            - "drop": drop datasets where any algorithm failed to report
        
    Returns:
        Dictionary with aggregated results
    """
    logger.info(f"Aggregating results across splits with behavior: {all_algs_behavior}")
    
    # Since we're using OpenML mapping system, task IDs should already be consistent
    # No need for complex normalization - just work with results directly
    aggregated = {}
    
    # First, discover all models across all datasets
    all_models = set()
    for task_id, task_data in results.items():
        for split_idx, split_data in task_data.get("splits", {}).items():
            for model_name in split_data.keys():
                all_models.add(model_name)
    
    logger.info(f"Found {len(all_models)} unique models across all datasets: {sorted(all_models)}")
    
    # If drop behavior is enabled, identify tasks to drop
    tasks_to_drop = set()
    dataset_fate_report = {}  # Track fate of each dataset
    
    if all_algs_behavior == "drop":
        # For each task, check if all models reported results - use only task_id for identification
        for task_id, task_data in results.items():
            # Use task_id as primary identifier, fallback to dataset_name if available
            task_name = task_data.get("info", {}).get("dataset_name", f"task_{task_id}")
            models_in_task = set()
            for split_idx, split_data in task_data.get("splits", {}).items():
                for model_name in split_data.keys():
                    models_in_task.add(model_name)
            
            # Decision based solely on task_id and model presence
            if models_in_task != all_models:
                missing_models = all_models - models_in_task
                logger.info(f"Dropping task_id {task_id} ({task_name}) - missing models: {missing_models}")
                tasks_to_drop.add(task_id)
                
                # Record in fate report using task_id as key
                dataset_fate_report[task_id] = {
                    "task_name": task_name,
                    "fate": "dropped",
                    "reason": "missing_models",
                    "models_present": sorted(list(models_in_task)),
                    "models_missing": sorted(list(missing_models)),
                    "total_models_expected": len(all_models),
                    "total_models_present": len(models_in_task),
                    "original_task_ids": [str(task_id)]
                }
            else:
                # Record kept datasets using task_id as key
                dataset_fate_report[task_id] = {
                    "task_name": task_name,
                    "fate": "kept",
                    "reason": "all_models_present",
                    "models_present": sorted(list(models_in_task)),
                    "models_missing": [],
                    "total_models_expected": len(all_models),
                    "total_models_present": len(models_in_task),
                    "original_task_ids": [str(task_id)]
                }
        
        logger.info(f"Dropping {len(tasks_to_drop)} tasks out of {len(results)} due to missing models")
    
    # If impute behavior is enabled, we need to identify all metrics
    all_metrics = set()
    if all_algs_behavior == "impute":
        # Collect all unique metrics across all tasks
        for task_id, task_data in results.items():
            for split_idx, split_data in task_data.get("splits", {}).items():
                for model_name, model_data in split_data.items():
                    metrics = model_data.get("metrics", {})
                    for metric_name in metrics.keys():
                        all_metrics.add(metric_name)
    
    for task_id, task_data in results.items():
        # Skip tasks marked for dropping based on task_id
        if task_id in tasks_to_drop:
            continue
            
        # Use task_id as primary identifier
        task_name = task_data.get("info", {}).get("dataset_name", f"task_{task_id}")
        
        # Initialize aggregated task data using task_id as key
        if task_id not in aggregated:
            aggregated[task_id] = {
                "info": {"dataset_name": task_name, "task_id": task_id},
                "models": {}
            }
            
        # Collect metrics for each model across splits
        model_metrics = defaultdict(lambda: defaultdict(list))
        
        for split_idx, split_data in task_data.get("splits", {}).items():
            for model_name, model_data in split_data.items():
                metrics = model_data.get("metrics", {})
                
                for metric_name, metric_value in metrics.items():
                    # Skip None or non-numeric values
                    if metric_value is not None and is_numeric(metric_value):
                        try:
                            # Convert to float to ensure numeric values
                            model_metrics[model_name][metric_name].append(float(metric_value))
                        except (ValueError, TypeError):
                            # Skip values that can't be converted to float
                            logger.warning(f"Skipping non-numeric value for {model_name}/{metric_name}: {metric_value}")
        
        # If impute behavior is enabled, fill in missing model/metric combinations for COMPLETE FAILURES only
        if all_algs_behavior == "impute":
            n_classes = get_dataset_n_classes(results, task_id)
            
            # Identify which models completely failed (no data at all for this task)
            models_with_data = set(model_metrics.keys())
            completely_failed_models = all_models - models_with_data
            
            if completely_failed_models:
                logger.debug(f"Task {task_id}: Models with data: {models_with_data}")
                logger.debug(f"Task {task_id}: Completely failed models: {completely_failed_models}")
            
            # For completely failed models, add random chance scores for core performance metrics only
            for model_name in completely_failed_models:
                # Only add the most basic performance metrics that are commonly available
                core_performance_metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'precision', 'recall']
                
                for metric_name in core_performance_metrics:
                    random_score = get_random_chance_score(metric_name, n_classes)
                    if random_score > 0:  # Only apply non-zero random scores
                        # Get the number of splits for this task to replicate the random score
                        num_splits = len(task_data.get("splits", {}))
                        if num_splits == 0:
                            num_splits = 1  # Default to 1 if no splits found
                        
                        # Add the random score for each split
                        if model_name not in model_metrics:
                            model_metrics[model_name] = defaultdict(list)
                        model_metrics[model_name][metric_name] = [random_score] * num_splits
                        
                        logger.debug(f"Imputed random chance score {random_score} for COMPLETELY FAILED {model_name}/{metric_name} on task {task_id}")
            
            # NOTE: We are NOT filling in partial failures to avoid the complexity of 
            # distinguishing between different metric naming conventions and actual failures
        
        # Calculate mean and std for each model and metric
        # IMPORTANT: Only include models that have all 3 splits available
        for model_name, metrics in model_metrics.items():
            # Check if this model has all 3 splits for at least one metric
            required_splits = 3
            has_all_splits = False
            
            for metric_name, values in metrics.items():
                if len(values) >= required_splits:
                    has_all_splits = True
                    break
            
            if not has_all_splits:
                logger.warning(f"Excluding {model_name} from task {task_id} - insufficient splits (need {required_splits}, found max {max([len(values) for values in metrics.values()]) if metrics else 0})")
                continue
            
            aggregated[task_id]["models"][model_name] = {
                "metrics": {},
                "metrics_std": {}
            }
            
            for metric_name, values in metrics.items():
                # Only include metrics that have all required splits
                if values and len(values) >= required_splits:  # Check if there are enough valid numeric values
                    # Use only the first 3 splits in case there are more
                    values_to_use = values[:required_splits]
                    try:
                        aggregated[task_id]["models"][model_name]["metrics"][metric_name] = np.mean(values_to_use)
                        aggregated[task_id]["models"][model_name]["metrics_std"][metric_name] = np.std(values_to_use)
                    except TypeError as e:
                        logger.error(f"Error calculating stats for {model_name}/{metric_name}: {e}")
                        logger.error(f"Values: {values_to_use}")
                else:
                    logger.debug(f"Skipping {model_name}/{metric_name} on task {task_id} - insufficient splits ({len(values) if values else 0}/{required_splits})")
    
    # Return aggregated results and dataset fate report (if drop behavior was used)
    if all_algs_behavior == "drop":
        return aggregated, dataset_fate_report
    else:
        return aggregated

def create_performance_dataframe(results, aggregated=True):
    """
    Create a DataFrame with performance metrics for all tasks and models.
    
    Args:
        results: Dictionary mapping task IDs to evaluation results
        aggregated: Whether the results are already aggregated across splits
        
    Returns:
        DataFrame with performance metrics
    """
    logger.info("Creating performance DataFrame")
    
    rows = []
    
    if aggregated:
        # Process aggregated results
        for task_id, task_data in results.items():
            task_name = task_data.get("info", {}).get("dataset_name", f"task_{task_id}")
            
            for model_name, model_data in task_data.get("models", {}).items():
                metrics = model_data.get("metrics", {})
                metrics_std = model_data.get("metrics_std", {})
                
                row = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "model": model_name
                }
                
                # Add metrics with normalization of metric names
                for metric_name, value in metrics.items():
                    if is_numeric(value):
                        # Handle aggregated metrics - these are preferred for baseline models
                        if metric_name.startswith("aggregated_"):
                            # Remove the "aggregated_" prefix to get the simple metric name
                            simple_metric = metric_name[len("aggregated_"):]
                            row[simple_metric] = value
                            row[f"{simple_metric}_std"] = metrics_std.get(metric_name)
                            continue
                            
                        # Extract the base metric name from various formats
                        # Handle dataset-prefixed metrics like "adult_accuracy"
                        if task_name and metric_name.startswith(f"{task_name}_"):
                            # Remove the dataset prefix
                            simple_metric = metric_name[len(f"{task_name}_"):]
                        elif '/' in metric_name:
                            # Handle path-like metrics
                            simple_metric = metric_name.split('/')[-1]
                        else:
                            # First, check if this is already a standard metric name (no prefix extraction needed)
                            standard_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
                                              'precision_macro', 'recall_macro', 'precision_weighted', 'recall_weighted',
                                              'roc_auc', 'training_time', 'prediction_time', 'num_features', 'num_samples', 
                                              'num_classes', 'num_test_samples', 'num_train_samples', 'precision', 'recall']
                            
                            if metric_name in standard_metrics:
                                # This is a standard metric, use as-is
                                simple_metric = metric_name
                            else:
                                # Check if it's a dataset_metric format by looking for known metric suffixes
                                # Only look for basic metrics that could have dataset prefixes
                                basic_metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1_score']
                                simple_metric = metric_name
                                for basic_metric in basic_metrics:
                                    if metric_name.endswith(f"_{basic_metric}") and metric_name != basic_metric:
                                        # This is likely a dataset_metric format, extract just the metric
                                        simple_metric = basic_metric
                                        break
                        
                        # Normalize metric names for consistency across model types
                        if simple_metric == 'average_accuracy':
                            simple_metric = 'accuracy'  # Unify traditional model metric with LLM baseline metric
                        
                        # Only add non-aggregated metrics if we don't already have the aggregated version
                        if simple_metric not in row:
                            row[simple_metric] = value
                            row[f"{simple_metric}_std"] = metrics_std.get(metric_name)
                
                rows.append(row)
    else:
        # Process non-aggregated results with splits
        for task_id, task_data in results.items():
            task_name = task_data.get("info", {}).get("dataset_name", f"task_{task_id}")
            
            for split_idx, split_data in task_data.get("splits", {}).items():
                for model_name, model_data in split_data.items():
                    # Skip non-model entries
                    if not isinstance(model_data, dict) or "metrics" not in model_data:
                        continue
                        
                    # Extract metrics
                    metrics = model_data.get("metrics", {})
                    
                    # Add row to DataFrame
                    row = {
                        "task_id": task_id,
                        "task_name": task_name,
                        "split": split_idx,
                        "model": model_name,
                    }
                    
                    # Add all available metrics with normalization
                    for metric_name, value in metrics.items():
                        if is_numeric(value):
                            # Handle aggregated metrics - these are preferred for baseline models
                            if metric_name.startswith("aggregated_"):
                                # Remove the "aggregated_" prefix to get the simple metric name
                                simple_metric = metric_name[len("aggregated_"):]
                                row[simple_metric] = value
                                continue
                                
                            # Extract the base metric name from various formats
                            # Handle dataset-prefixed metrics like "adult_accuracy"
                            if task_name and metric_name.startswith(f"{task_name}_"):
                                # Remove the dataset prefix
                                simple_metric = metric_name[len(f"{task_name}_"):]
                            elif '/' in metric_name:
                                # Handle path-like metrics
                                simple_metric = metric_name.split('/')[-1]
                            else:
                                # Check if it's a dataset_metric format by looking for known metric suffixes
                                known_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
                                               'precision', 'recall', 'roc_auc', 'training_time', 'prediction_time',
                                               'num_classes', 'num_features', 'num_samples', 'num_test_samples', 'num_train_samples']
                                simple_metric = metric_name
                                for known_metric in known_metrics:
                                    if metric_name.endswith(f"_{known_metric}"):
                                        # This is likely a dataset_metric format, extract just the metric
                                        simple_metric = known_metric
                                        break
                            
                            # Normalize metric names for consistency across model types
                            if simple_metric == 'average_accuracy':
                                simple_metric = 'accuracy'  # Unify traditional model metric with LLM baseline metric
                            
                            # Only add non-aggregated metrics if we don't already have the aggregated version
                            if simple_metric not in row:
                                row[simple_metric] = value
                    
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure model names are never empty and "." is replaced with "marvis"
    df["model"] = df["model"].apply(lambda x: "marvis" if pd.isna(x) or x == "" or x == "." else x)
    
    logger.info(f"Created DataFrame with {len(df)} rows")
    return df

def compute_summary_statistics(df):
    """
    Compute summary statistics across all tasks.
    
    Args:
        df: DataFrame with performance metrics
        
    Returns:
        DataFrame with summary statistics
    """
    logger.info("Computing summary statistics")
    
    if df.empty:
        logger.warning("DataFrame is empty, cannot compute summary statistics")
        return pd.DataFrame(), pd.DataFrame()
    
    # Find available metrics (columns that are not metadata)
    # Also filter out numbered class-specific metrics and boolean metrics
    metric_cols = []
    for col in df.columns:
        # Skip metadata columns
        if col in ["task_id", "task_name", "model", "split"] or col.endswith("_std"):
            continue
        # Skip numbered class-specific metrics (like "0", "1", "2", etc.)
        if col.isdigit():
            continue
        # Skip boolean metrics (like "True", "False")
        if col in ["True", "False"]:
            continue
        metric_cols.append(col)
    
    # Group by model and compute mean and std of metrics
    summary_dict = {}
    for metric in metric_cols:
        summary_dict[metric] = ["mean", "std", "min", "max", "count"]
    
    summary = df.groupby("model").agg(summary_dict)
    
    # Compute win rates
    models = df["model"].unique()
    win_rates = {}
    
    for metric in metric_cols:
        if metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc"]:  # Focus on common metrics
            win_rates[metric] = {}
            
            # Compute wins for each model per task
            for task_id in df["task_id"].unique():
                task_df = df[df["task_id"] == task_id]
                if task_df.empty or task_df[metric].isna().all():
                    continue
                    
                best_model = task_df.loc[task_df[metric].idxmax(), "model"]
                if best_model not in win_rates[metric]:
                    win_rates[metric][best_model] = 0
                win_rates[metric][best_model] += 1
    
    # Create win rate DataFrame
    win_rate_rows = []
    total_tasks = len(df["task_id"].unique())
    
    # Ensure all models have entries for all metrics (even with 0 wins)
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc"]:
        for model in models:
            win_count = win_rates.get(metric, {}).get(model, 0)
            win_rate_rows.append({
                "metric": metric,
                "model": model,
                "wins": win_count,
                "win_rate": win_count / total_tasks if total_tasks > 0 else 0
            })
    
    win_rate_df = pd.DataFrame(win_rate_rows)
    
    return summary, win_rate_df

def create_visualizations(df, summary, win_rate_df, output_dir, plot_format):
    """
    Create focused visualizations matching the key statistics logged to console.
    
    Args:
        df: DataFrame with performance metrics
        summary: DataFrame with summary statistics
        win_rate_df: DataFrame with win rates
        output_dir: Directory to save visualizations
        plot_format: Format for saved plots
    """
    logger.info("Creating focused visualizations for key metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        logger.warning("DataFrame is empty, skipping visualizations")
        return
    
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 100,
        "font.size": 12
    })
    
    # Focus on the key metrics that are logged to console and used in win rates
    key_metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc"]
    
    # Also include key timing metrics if available
    timing_metrics = ["training_time", "prediction_time", "total_time"]
    
    # Filter to only include key metrics that exist in the DataFrame
    performance_metrics = [col for col in key_metrics if col in df.columns and df[col].notna().sum() > 0]
    timing_metrics_available = [col for col in timing_metrics if col in df.columns and df[col].notna().sum() > 0]
    
    logger.info(f"Creating visualizations for performance metrics: {performance_metrics}")
    logger.info(f"Creating visualizations for timing metrics: {timing_metrics_available}")
    
    # 1. Boxplots for key performance metrics only
    for metric in performance_metrics:
        plt.figure(figsize=(12, 8))
        # Make sure all the model names are present and not blank or "."
        plot_df = df.copy()
        plot_df["model"] = plot_df["model"].apply(lambda x: "marvis" if pd.isna(x) or x == "" or x == "." else x)
        sns.boxplot(x="model", y=metric, data=plot_df)
        plt.title(f"{metric.replace('_', ' ').title()} by Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Create a safe filename
        safe_metric_name = sanitize_filename(metric)
        output_path = os.path.join(output_dir, f"{safe_metric_name}_boxplot.{plot_format}")
        
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
        finally:
            plt.close()
    
    # 2. Combined timing metrics plot (if any timing data is available)
    if timing_metrics_available:
        plt.figure(figsize=(15, 10))
        plot_df = df.copy()
        plot_df["model"] = plot_df["model"].apply(lambda x: "marvis" if pd.isna(x) or x == "" or x == "." else x)
        
        # Create subplots for timing metrics
        n_timing = len(timing_metrics_available)
        fig, axes = plt.subplots(1, n_timing, figsize=(5*n_timing, 8))
        if n_timing == 1:
            axes = [axes]
        
        for i, metric in enumerate(timing_metrics_available):
            sns.boxplot(x="model", y=metric, data=plot_df, ax=axes[i])
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].tick_params(axis='x', rotation=45)
            # Set log scale for y-axis to better visualize timing data
            axes[i].set_yscale('log')
        
        plt.suptitle("Timing Metrics by Model (Log Scale)", fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"timing_metrics.{plot_format}")
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
        finally:
            plt.close()
    
    # 3. Win rates bar chart (focused on the 4 key metrics)
    if not win_rate_df.empty:
        plt.figure(figsize=(12, 8))
        # Make sure all models have entries for all metrics
        win_pivot = win_rate_df.pivot(index="model", columns="metric", values="win_rate").fillna(0)
        win_pivot.plot(kind="bar", figsize=(12, 8))
        plt.title("Win Rate by Model and Metric")
        plt.ylabel("Win Rate")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"win_rates.{plot_format}")
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
        finally:
            plt.close()
    
    # 4. Overall performance summary (mean accuracy across all models)
    if "accuracy" in performance_metrics and not summary.empty:
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract mean accuracy for each model
            model_names = []
            accuracies = []
            accuracy_stds = []
            
            for model in summary.index:
                display_model = "marvis" if model == "." else model
                if ("accuracy", "mean") in summary.columns:
                    mean_acc = summary.loc[model, ("accuracy", "mean")]
                    std_acc = summary.loc[model, ("accuracy", "std")] if ("accuracy", "std") in summary.columns else 0
                    
                    if not np.isnan(mean_acc):
                        model_names.append(display_model)
                        accuracies.append(mean_acc)
                        accuracy_stds.append(std_acc if not np.isnan(std_acc) else 0)
            
            if model_names:
                # Create bar plot with error bars
                plt.bar(model_names, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.7)
                plt.title("Mean Accuracy Across All Tasks")
                plt.ylabel("Accuracy")
                plt.xlabel("Model")
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                output_path = os.path.join(output_dir, f"mean_accuracy_summary.{plot_format}")
                plt.savefig(output_path)
        except Exception as e:
            logger.error(f"Error creating accuracy summary plot: {e}")
        finally:
            plt.close()
    
    logger.info(f"Saved focused visualizations to {output_dir}")

def flatten_multiindex_dataframe(df):
    """
    Convert a DataFrame with MultiIndex columns to a flat structure for JSON serialization.
    
    Args:
        df: DataFrame with MultiIndex columns
        
    Returns:
        DataFrame with flattened columns suitable for JSON export
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Convert the multi-index columns to string with underscore separator
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    
    return df

def detect_failed_dataset_metrics(df):
    """
    Detect failed dataset/metric pairs for each model using only task_id for identification.
    
    A dataset/metric pair is considered "failed" if a model has NaN or null values for a metric
    that other models have successfully computed for the same task_id, OR if the model
    doesn't appear in the results at all for a task_id (indicating complete failure).
    
    Args:
        df: DataFrame with performance metrics
        
    Returns:
        Dictionary mapping model names to lists of failed dataset/metric pairs
    """
    logger.info("Detecting failed dataset/metric pairs using task_id")
    
    if df.empty:
        logger.warning("DataFrame is empty, no failed pairs to detect")
        return {}
    
    # Find available metrics (columns that are not metadata)
    # Also filter out numbered class-specific metrics and boolean metrics
    metric_cols = []
    for col in df.columns:
        # Skip metadata columns
        if col in ["task_id", "task_name", "model", "split"] or col.endswith("_std"):
            continue
        # Skip numbered class-specific metrics (like "0", "1", "2", etc.)
        if col.isdigit():
            continue
        # Skip boolean metrics (like "True", "False")
        if col in ["True", "False"]:
            continue
        metric_cols.append(col)
    
    # Initialize failed pairs dictionary
    failed_pairs = defaultdict(list)
    
    # Get unique models and task_ids
    all_models = df["model"].unique().tolist()  # Convert to Python list
    all_task_ids = df["task_id"].unique().tolist()  # Convert to Python list
    
    # Check for each task_id and metric if any model has a value while others don't
    for task_id in all_task_ids:
        task_df = df[df["task_id"] == task_id]
        task_name = task_df["task_name"].iloc[0] if "task_name" in task_df.columns else f"task_{task_id}"
        
        # Convert task_name to string to ensure JSON serialization
        if not isinstance(task_name, str):
            task_name = str(task_name)
        
        # Get models that have data for this task_id
        models_with_data = task_df["model"].unique().tolist()
        
        # Check for models that have no data at all for this task_id
        for model in all_models:
            if model not in models_with_data:
                # This model completely failed on this task_id
                normalized_task_id = task_id
                if not isinstance(normalized_task_id, (int, str, float, bool)) or normalized_task_id is None:
                    normalized_task_id = int(normalized_task_id) if isinstance(normalized_task_id, np.integer) else str(normalized_task_id)
                
                # Add failure entries for all common metrics
                for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
                    if metric in metric_cols:
                        failed_pairs[model].append({
                            "task_id": normalized_task_id,
                            "task_name": task_name,
                            "metric": metric,
                            "failure_type": "complete_failure"  # Model didn't run at all
                        })
        
        # Check for partial failures (model ran but missing specific metrics)
        for metric in metric_cols:
            # Skip metrics that don't exist for this task_id
            if metric not in task_df.columns or task_df[metric].count() == 0:
                continue
                
            # If at least one model has a value for this metric on this task_id
            if task_df[metric].notna().any():
                # Check which models are missing this metric
                for model in models_with_data:
                    model_df = task_df[task_df["model"] == model]
                    
                    # If the model is missing this metric or value is NaN
                    if metric not in model_df.columns or model_df[metric].isna().all():
                        # Convert all values to Python native types
                        normalized_task_id = task_id
                        if not isinstance(normalized_task_id, (int, str, float, bool)) or normalized_task_id is None:
                            normalized_task_id = int(normalized_task_id) if isinstance(normalized_task_id, np.integer) else str(normalized_task_id)
                        
                        failed_pairs[model].append({
                            "task_id": normalized_task_id,
                            "task_name": task_name,
                            "metric": metric,
                            "failure_type": "metric_failure"  # Model ran but this metric failed
                        })
    
    # Convert defaultdict to regular dict
    return dict(failed_pairs)

def export_dataset_fate_report(dataset_fate_report, output_dir):
    """
    Export a report describing the fate of each dataset when using drop behavior.
    
    Args:
        dataset_fate_report: Dictionary with dataset fate information
        output_dir: Directory to save the report
    """
    logger.info("Exporting dataset fate report")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    fate_rows = []
    for task_id, fate_info in dataset_fate_report.items():
        fate_rows.append({
            "normalized_task_id": task_id,
            "original_task_ids": ", ".join(fate_info.get("original_task_ids", [str(task_id)])),
            "task_name": fate_info["task_name"],
            "fate": fate_info["fate"],
            "reason": fate_info["reason"],
            "models_present": ", ".join(fate_info["models_present"]),
            "models_missing": ", ".join(fate_info["models_missing"]),
            "total_models_expected": fate_info["total_models_expected"],
            "total_models_present": fate_info["total_models_present"]
        })
    
    fate_df = pd.DataFrame(fate_rows)
    
    # Sort by fate (dropped first) and then by task_id
    fate_df = fate_df.sort_values(["fate", "normalized_task_id"], ascending=[False, True])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "dataset_fate_report.csv")
    fate_df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset fate report to {csv_path}")
    
    # Save as JSON with more detailed information
    json_path = os.path.join(output_dir, "dataset_fate_report.json")
    with open(json_path, "w") as f:
        json.dump(dataset_fate_report, f, indent=2)
    logger.info(f"Saved detailed dataset fate report to {json_path}")
    
    # Create a summary text file
    summary_path = os.path.join(output_dir, "dataset_fate_summary.txt")
    with open(summary_path, "w") as f:
        # Write header
        f.write("Dataset Fate Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        total_datasets = len(dataset_fate_report)
        kept_datasets = sum(1 for d in dataset_fate_report.values() if d["fate"] == "kept")
        dropped_datasets = sum(1 for d in dataset_fate_report.values() if d["fate"] == "dropped")
        
        f.write(f"Total datasets: {total_datasets}\n")
        f.write(f"Kept datasets: {kept_datasets} ({kept_datasets/total_datasets*100:.1f}%)\n")
        f.write(f"Dropped datasets: {dropped_datasets} ({dropped_datasets/total_datasets*100:.1f}%)\n\n")
        
        # List all models
        all_models = set()
        for fate_info in dataset_fate_report.values():
            all_models.update(fate_info["models_present"])
            all_models.update(fate_info["models_missing"])
        
        f.write(f"All models ({len(all_models)}): {', '.join(sorted(all_models))}\n\n")
        
        # Dropped datasets details
        f.write("DROPPED DATASETS\n")
        f.write("-" * 80 + "\n")
        dropped_tasks = [(tid, info) for tid, info in dataset_fate_report.items() if info["fate"] == "dropped"]
        if dropped_tasks:
            for task_id, info in sorted(dropped_tasks, key=lambda x: str(x[0])):
                original_ids = info.get("original_task_ids", [str(task_id)])
                if len(original_ids) > 1:
                    f.write(f"\nTask {task_id} (original IDs: {', '.join(original_ids)}): {info['task_name']}\n")
                else:
                    f.write(f"\nTask {task_id}: {info['task_name']}\n")
                f.write(f"  Missing models: {', '.join(info['models_missing'])}\n")
                f.write(f"  Present models: {', '.join(info['models_present'])}\n")
        else:
            f.write("None\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEPT DATASETS\n")
        f.write("-" * 80 + "\n")
        kept_tasks = [(tid, info) for tid, info in dataset_fate_report.items() if info["fate"] == "kept"]
        if kept_tasks:
            for task_id, info in sorted(kept_tasks, key=lambda x: str(x[0])):
                original_ids = info.get("original_task_ids", [str(task_id)])
                if len(original_ids) > 1:
                    f.write(f"\nTask {task_id} (original IDs: {', '.join(original_ids)}): {info['task_name']}\n")
                else:
                    f.write(f"\nTask {task_id}: {info['task_name']}\n")
                f.write(f"  All {info['total_models_expected']} models present\n")
        else:
            f.write("None\n")
    
    logger.info(f"Saved dataset fate summary to {summary_path}")
    
    # Print summary to console
    print(f"\nDataset Fate Summary (drop behavior):")
    print("-" * 80)
    print(f"Total datasets: {total_datasets}")
    print(f"Kept datasets: {kept_datasets} ({kept_datasets/total_datasets*100:.1f}%)")
    print(f"Dropped datasets: {dropped_datasets} ({dropped_datasets/total_datasets*100:.1f}%)")
    print(f"Detailed report saved to: {output_dir}")
    print("-" * 80)


def export_results(df, summary, win_rate_df, output_dir):
    """
    Export results to CSV and JSON for further analysis.
    
    Args:
        df: DataFrame with performance metrics
        summary: DataFrame with summary statistics
        win_rate_df: DataFrame with win rates
        output_dir: Directory to save exported results
    """
    logger.info("Exporting results")
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        logger.warning("DataFrame is empty, exporting empty files")
    
    # Export raw data
    df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    
    # Export summary statistics - flatten the multi-index columns first
    if not summary.empty:
        flat_summary = flatten_multiindex_dataframe(summary.copy())
        flat_summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
    
    # Export win rates
    if not win_rate_df.empty:
        win_rate_df.to_csv(os.path.join(output_dir, "win_rates.csv"), index=False)
    
    # Find available metrics (columns that are not metadata)
    # Also filter out numbered class-specific metrics and boolean metrics
    metric_cols = []
    for col in df.columns:
        # Skip metadata columns
        if col in ["task_id", "task_name", "model", "split"] or col.endswith("_std"):
            continue
        # Skip numbered class-specific metrics (like "0", "1", "2", etc.)
        if col.isdigit():
            continue
        # Skip boolean metrics (like "True", "False")
        if col in ["True", "False"]:
            continue
        metric_cols.append(col)
    
    # Detect and export failed dataset/metric pairs
    failed_pairs = detect_failed_dataset_metrics(df)
    
    # Convert to a format suitable for CSV export
    failed_rows = []
    for model, pairs in failed_pairs.items():
        for pair in pairs:
            failed_rows.append({
                "model": model,
                "task_id": pair["task_id"],
                "task_name": pair["task_name"],
                "metric": pair["metric"]
            })
    
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(os.path.join(output_dir, "failed_dataset_metrics.csv"), index=False)
    
    # Helper function to convert NumPy/Pandas types to Python native types for JSON serialization
    def json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        else:
            return obj
    
    # Export as JSON for web visualization
    # We need to convert the summary DataFrame to a serializable format
    summary_dict = []
    if not summary.empty:
        # Create a flattened version of the summary for JSON serialization
        flat_summary = flatten_multiindex_dataframe(summary.reset_index())
        summary_dict = flat_summary.to_dict(orient="records")
        
        # Convert any NumPy types to native Python types
        for item in summary_dict:
            for key, value in list(item.items()):
                item[key] = json_serializable(value)
    
    # Convert models, metrics, and tasks to native Python types
    models = [str(model) for model in df["model"].unique().tolist()] if not df.empty else []
    tasks = [str(task) for task in df["task_name"].unique().tolist()] if not df.empty else []
    
    # Convert win_rates to JSON-serializable format
    win_rates_json = []
    if not win_rate_df.empty:
        win_rates_dict = win_rate_df.to_dict(orient="records")
        for item in win_rates_dict:
            serialized_item = {}
            for key, value in item.items():
                serialized_item[key] = json_serializable(value)
            win_rates_json.append(serialized_item)
    
    # Make sure failed_pairs is JSON serializable
    serialized_failed_pairs = {}
    for model, pairs in failed_pairs.items():
        serialized_pairs = []
        for pair in pairs:
            serialized_pair = {}
            for key, value in pair.items():
                serialized_pair[key] = json_serializable(value)
            serialized_pairs.append(serialized_pair)
        serialized_failed_pairs[model] = serialized_pairs
    
    result_json = {
        "models": models,
        "metrics": metric_cols,
        "tasks": tasks,
        "summary": summary_dict,
        "win_rates": win_rates_json,
        "failed_pairs": serialized_failed_pairs
    }
    
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(result_json, f, indent=2)
    
    logger.info(f"Exported results to {output_dir}")

def main():
    args = parse_args()
    
    # Set debug level if specified
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Log all_algs_behavior mode
    if args.all_algs_behavior != "none":
        logger.info(f"All-algs-behavior mode: {args.all_algs_behavior}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fetch evaluation data from W&B
    wandb_df = fetch_wandb_data(args.wandb_entity, args.wandb_project)
    
    # Save raw W&B data if requested
    if args.save_raw_data:
        project_names = "_".join(args.wandb_project) if isinstance(args.wandb_project, list) else args.wandb_project
        raw_data_path = os.path.join(args.output_dir, f"{project_names}_raw_data.csv")
        # Flatten the config and summary columns for CSV export
        export_df = wandb_df.copy()
        export_df["config"] = export_df["config"].apply(lambda x: json.dumps(x))
        export_df["summary"] = export_df["summary"].apply(lambda x: json.dumps(x))
        # Use proper CSV quoting to handle embedded commas and quotes
        export_df.to_csv(raw_data_path, index=False, quoting=1)  # 1 = csv.QUOTE_ALL
        logger.info(f"Saved raw W&B data to {raw_data_path}")
    
    # First detect run types to see what we have
    run_types = detect_run_types(wandb_df)
    eval_runs = run_types['eval_runs']
    llm_baseline_runs = run_types['llm_baseline_runs'] 
    baseline_runs = run_types.get('baseline_runs', pd.DataFrame())  # Traditional baseline runs (sklearn, TabPFN, CatBoost)
    train_dataset_runs = run_types['train_dataset_runs']
    
    # Check if we have any processable runs
    total_processable_runs = len(eval_runs) + len(llm_baseline_runs) + len(baseline_runs) + len(train_dataset_runs)
    if total_processable_runs == 0:
        logger.error("No evaluation runs found in the W&B project")
        return
    
    logger.info(f"Detected {len(eval_runs)} traditional eval runs, {len(llm_baseline_runs)} LLM baseline runs, {len(baseline_runs)} traditional baseline runs, and {len(train_dataset_runs)} train_tabular_dataset runs")
    
    # Fetch runtime information from training runs (mainly for traditional eval runs)
    runtime_info = fetch_wandb_train_data(args.wandb_entity, args.wandb_project)
    
    # Extract results from W&B data
    results = extract_results_from_wandb(wandb_df, runtime_info=runtime_info, debug=args.debug)
    
    if not results:
        logger.error("No results could be extracted from W&B runs")
        return
    
    # Aggregate results across splits
    aggregate_output = aggregate_splits(results, all_algs_behavior=args.all_algs_behavior)
    
    # Handle different return types based on behavior
    dataset_fate_report = None
    if args.all_algs_behavior == "drop":
        aggregated_results, dataset_fate_report = aggregate_output
    else:
        aggregated_results = aggregate_output
    
    # Create performance DataFrame from aggregated results
    df = create_performance_dataframe(aggregated_results, aggregated=True)
    
    # Also save detailed DataFrame with split-level data
    df_detailed = create_performance_dataframe(results, aggregated=False)
    df_detailed.to_csv(os.path.join(args.output_dir, "detailed_metrics.csv"), index=False)
    
    # Compute summary statistics
    summary, win_rate_df = compute_summary_statistics(df)
    
    # Create visualizations
    viz_dir = os.path.join(args.output_dir, "visualizations")
    create_visualizations(df, summary, win_rate_df, viz_dir, args.plot_format)
    
    # Export results
    export_dir = os.path.join(args.output_dir, "exported")
    export_results(df, summary, win_rate_df, export_dir)
    
    # Export dataset fate report if using drop behavior
    if args.all_algs_behavior == "drop" and dataset_fate_report:
        export_dataset_fate_report(dataset_fate_report, export_dir)
    
    logger.info("Analysis completed")
    
    if df.empty:
        logger.error("No data to display")
        return
    
    # Print summary to console
    print("\nPerformance Summary (mean ± std):")
    print("-" * 80)
    project_names = ", ".join(args.wandb_project) if isinstance(args.wandb_project, list) else args.wandb_project
    print(f"Analyzed data from projects: {project_names}")
    print(f"Total: {len(eval_runs)} traditional eval runs, {len(llm_baseline_runs)} LLM baseline runs, and {len(train_dataset_runs)} train_tabular_dataset runs")
    print("-" * 80)
    
    # Find available metrics (columns that are not metadata)
    # Also filter out numbered class-specific metrics, boolean metrics, and count metrics with numeric suffixes
    metric_cols = []
    import re
    for col in df.columns:
        # Skip metadata columns
        if col in ["task_id", "task_name", "model", "split"] or col.endswith("_std"):
            continue
        # Skip numbered class-specific metrics (like "0", "1", "2", etc.)
        if col.isdigit():
            continue
        # Skip boolean metrics (like "True", "False")
        if col in ["True", "False"]:
            continue
        # Skip metrics ending in /<NUM> (like "pred_count/1", "gt_count/2")
        if re.search(r'/\d+$', col):
            continue
        metric_cols.append(col)
    
    for model in summary.index:
        # Ensure model names are properly displayed
        display_model = "marvis" if model == "." else model
        print(f"Model: {display_model}")
        for metric in metric_cols:
            if (metric, "mean") in summary.columns and (metric, "std") in summary.columns:
                mean = summary.loc[model, (metric, "mean")]
                std = summary.loc[model, (metric, "std")]
                count = summary.loc[model, (metric, "count")]
                if not np.isnan(mean) and not np.isnan(std):
                    print(f"  {metric}: {mean:.4f} ± {std:.4f} (n={int(count)})")
        print("-" * 80)
    
    # Print win rates
    if not win_rate_df.empty:
        print("\nWin Rates:")
        print("-" * 80)
        for metric in win_rate_df["metric"].unique():
            print(f"Metric: {metric}")
            metric_df = win_rate_df[win_rate_df["metric"] == metric].sort_values("win_rate", ascending=False)
            for _, row in metric_df.iterrows():
                # Ensure model names are properly displayed
                display_model = "marvis" if row['model'] == "." else row['model']
                print(f"  {display_model}: {row['wins']} wins ({row['win_rate']:.2%})")
            print("-" * 80)
    
    # Print failed dataset/metric pairs summary
    failed_pairs = detect_failed_dataset_metrics(df)
    if failed_pairs:
        print("\nFailed Dataset/Metric Pairs Summary:")
        print("-" * 80)
        for model, pairs in sorted(failed_pairs.items(), key=lambda x: len(x[1]), reverse=True):
            # Ensure model names are properly displayed
            display_model = "marvis" if model == "." else model
            print(f"Model: {display_model}")
            print(f"  Total failures: {len(pairs)}")
            
            # Group failures by type and metric
            complete_failures = [p for p in pairs if p.get("failure_type") == "complete_failure"]
            metric_failures = [p for p in pairs if p.get("failure_type") == "metric_failure"]
            
            if complete_failures:
                complete_tasks = set(p["task_name"] for p in complete_failures)
                print(f"  Complete failures (model didn't run): {len(complete_tasks)} datasets")
                for task in sorted(list(complete_tasks))[:5]:
                    print(f"    - {task}")
                if len(complete_tasks) > 5:
                    print(f"    - ... and {len(complete_tasks) - 5} more")
            
            if metric_failures:
                print(f"  Metric failures (model ran but metrics missing): {len(metric_failures)} cases")
                # Group metric failures by metric to make the output more readable
                failures_by_metric = defaultdict(list)
                for pair in metric_failures:
                    failures_by_metric[pair["metric"]].append(pair["task_name"])
                
                # Print the failures by metric
                for metric, tasks in failures_by_metric.items():
                    unique_tasks = list(set(tasks))
                    print(f"    {metric}: {len(unique_tasks)} datasets")
                    # Print at most 3 example tasks for metric failures
                    for task in sorted(unique_tasks)[:3]:
                        print(f"      - {task}")
                    if len(unique_tasks) > 3:
                        print(f"      - ... and {len(unique_tasks) - 3} more")
            
            print("-" * 80)

if __name__ == "__main__":
    main()