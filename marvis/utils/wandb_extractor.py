#!/usr/bin/env python
"""
Weights & Biases data extraction utilities for MARVIS evaluation results.

This module provides functionality to extract and parse evaluation results from W&B
across different experiment formats:
1. Traditional eval_ runs (original OpenML CC18 format)
2. LLM baseline runs (from evaluate_llm_baselines.py)
3. Train tabular dataset runs (from train_tabular_dataset.py)

The extracted data is normalized into a common format for analysis.
"""

import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import wandb

logger = logging.getLogger(__name__)

def fetch_wandb_data(entity, projects):
    """
    Fetch experiment data from multiple Weights & Biases projects.
    
    Args:
        entity: W&B entity name
        projects: W&B project name(s) - can be string or list
        
    Returns:
        DataFrame with run data from W&B
    """
    if isinstance(projects, str):
        projects = [projects]
    
    logger.info(f"Fetching data from W&B projects: {projects}")
    
    api = wandb.Api()
    all_runs = []
    
    summary_list, config_list, name_list, state_list = [], [], [], []
    eval_runs_count = 0
    llm_baseline_runs_count = 0
    baseline_runs_count = 0
    train_dataset_runs_count = 0
    
    for project in projects:
        try:
            logger.info(f"Fetching runs from {entity}/{project}")
            runs = api.runs(f"{entity}/{project}")
            all_runs.extend(runs)
        except Exception as e:
            logger.error(f"Failed to fetch runs from {entity}/{project}: {e}")
            continue
    
    logger.info(f"Total runs fetched from all projects: {len(all_runs)}")
    
    for run in all_runs: 
        # Include runs that start with 'eval_' (original format), have LLM baseline metrics (new format),
        # have traditional baseline metrics (new unified format), or have test metrics from train_tabular_dataset.py
        is_eval_run = run.name.startswith("eval_")
        is_llm_baseline_run = any(key for key in run.summary._json_dict.keys() 
                                 if (
                                     # Model hierarchical format: model/{model}/dataset/{dataset}/{metric}
                                     (any(key.startswith(f"model/{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                                      any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                                     or
                                     # New hierarchical format: {model}/dataset/{dataset}/{metric}
                                     (any(key.startswith(f"{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                                      any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                                     or
                                     # Old format: {model}_{dataset}_{metric}
                                     (any(model in key.lower() for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) 
                                      and any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                                 ))
        
        # Detect traditional baseline runs (scikit-learn models + TabPFN + CatBoost)
        baseline_model_names = ['catboost', 'tabpfn_v2', 'random_forest', 'gradient_boosting', 'logistic_regression', 
                               'xgboost', 'lightgbm', 'svm', 'knn', 'naive_bayes']
        is_baseline_run = any(key for key in run.summary._json_dict.keys() 
                             if (
                                 # New unified hierarchical format: model/{model_name}/dataset/{dataset_name}/{metric}
                                 (key.startswith('model/') and 
                                  any(model in key.lower() for model in baseline_model_names) and
                                  any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_macro']))
                                 or
                                 # Old format: {model_name}_{dataset_name}_{metric}
                                 (any(model in key.lower() for model in baseline_model_names) and 
                                  any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score']))
                             ))
        
        is_train_dataset_run = any(key for key in run.summary._json_dict.keys()
                                  if key.startswith('test/') and any(metric in key for metric in ['accuracy', 'f1_score', 'precision', 'recall']))
        
        if not (is_eval_run or is_llm_baseline_run or is_baseline_run or is_train_dataset_run):
            continue
            
        if is_eval_run:
            eval_runs_count += 1
        if is_llm_baseline_run:
            llm_baseline_runs_count += 1
        if is_baseline_run:
            baseline_runs_count += 1
        if is_train_dataset_run:
            train_dataset_runs_count += 1
            
        # .summary contains the output keys/values for metrics like accuracy.
        # We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)
        
        # .config contains the hyperparameters.
        # We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
              if not k.startswith('_')})
        
        # .name is the human-readable name of the run.
        name_list.append(run.name)
        
        # .state indicates if the run completed successfully
        state_list.append(run.state)
    
    runs_df = pd.DataFrame({
        "name": name_list,
        "state": state_list,
        "summary": summary_list,
        "config": config_list
    })
    
    logger.info(f"Fetched {len(runs_df)} total runs from W&B ({eval_runs_count} eval runs, {llm_baseline_runs_count} LLM baseline runs, {baseline_runs_count} traditional baseline runs, {train_dataset_runs_count} train_tabular_dataset runs)")
    return runs_df

def fetch_wandb_train_data(entity, projects):
    """
    Fetch training run data from multiple Weights & Biases projects to extract runtime information.
    
    Args:
        entity: W&B entity name
        projects: W&B project name(s) - can be string or list
        
    Returns:
        Dictionary mapping task IDs and splits to runtime information
    """
    if isinstance(projects, str):
        projects = [projects]
    
    logger.info(f"Fetching training runtime data from W&B projects: {projects}")
    
    api = wandb.Api()
    all_runs = []
    
    # Dictionary to store runtime information by task ID and split
    runtime_info = {}
    
    for project in projects:
        try:
            logger.info(f"Fetching training runs from {entity}/{project}")
            runs = api.runs(f"{entity}/{project}")
            all_runs.extend(runs)
        except Exception as e:
            logger.error(f"Failed to fetch training runs from {entity}/{project}: {e}")
            continue
    
    for run in all_runs:
        # Look for training runs (not starting with 'eval_')
        if run.name.startswith("eval_"):
            continue
            
        # Determine the model type based on run name prefix
        model_name = None
        if run.name.startswith("train_task"):
            model_name = "marvis"
        elif run.name.startswith("baselines_task"):
            # Skip baseline training runs as they don't provide MARVIS runtime data
            continue
        else:
            # Skip other types of runs that don't provide MARVIS runtime
            continue
            
        # Extract task ID from run name or config
        task_id = None
        split_idx = 0  # Default split
        
        # Try to extract task ID from run name
        task_id_match = re.search(r'task[_-]?(\d+)', run.name)
        if task_id_match:
            task_id = int(task_id_match.group(1))
        
        # Try to extract split index from run name
        split_match = re.search(r'split[_-]?(\d+)', run.name)
        if split_match:
            split_idx = int(split_match.group(1))
        
        # If we couldn't find a task ID in the name, try the config
        if task_id is None and 'dataset_name' in run.config:
            dataset_name = run.config['dataset_name']
            if dataset_name:
                matches = re.findall(r'\d+', dataset_name)
                if matches:
                    task_id = int(matches[0])
        
        # If we couldn't find a split in the name, try the config
        if 'split_idx' in run.config:
            split_idx = int(run.config['split_idx'])
        
        # Skip if we still don't have a task ID
        if task_id is None:
            continue
        
        # Extract runtime information from summary
        runtime = None
        prediction_time = None
        
        # Look for runtime metrics in the summary
        if '_runtime' in run.summary:
            runtime = run.summary['_runtime']
        
        # Look for prediction time metrics in the summary
        if 'prediction_time' in run.summary:
            prediction_time = run.summary['prediction_time']
        
        # Store runtime information
        if task_id not in runtime_info:
            runtime_info[task_id] = {}
        
        if split_idx not in runtime_info[task_id]:
            runtime_info[task_id][split_idx] = {}
        
        if model_name not in runtime_info[task_id][split_idx]:
            runtime_info[task_id][split_idx][model_name] = {}
        
        # Add runtime information
        if runtime is not None:
            runtime_info[task_id][split_idx][model_name]['training_time'] = float(runtime)
        
        if prediction_time is not None:
            runtime_info[task_id][split_idx][model_name]['prediction_time'] = float(prediction_time)
    
    logger.info(f"Extracted runtime information for {len(runtime_info)} tasks")
    return runtime_info

def extract_variables_from_wandb_data(summary, config, run_name, run_type="unknown"):
    """
    Extract dataset_name, dataset_id, and task_id from wandb data based on run type.
    
    Args:
        summary: The W&B run summary dictionary
        config: The W&B run config dictionary  
        run_name: The name of the W&B run
        run_type: Type of run ("standard_baseline", "llm_baseline", "marvis_training", "eval")
        
    Returns:
        dict with keys: dataset_name (str), dataset_id (int), task_id (int)
        Only returns values that are actually found in the data - never imputes
    """
    result = {
        "dataset_name": None,
        "dataset_id": None, 
        "task_id": None
    }
    
    if run_type == "standard_baseline":
        # Format: model/{model_name}/dataset/{dataset_name}/{metric}
        # Look for dataset_name and dataset_id in summary
        for key, value in summary.items():
            if "/dataset/" in key and "/dataset_name" in key:
                result["dataset_name"] = value
            elif "/dataset/" in key and "/dataset_id" in key:
                try:
                    result["dataset_id"] = int(value) if value is not None else None
                except (ValueError, TypeError):
                    result["dataset_id"] = None
        
        # Map dataset_name to task_id using consistent mapping function
        if result["dataset_name"]:
            result["task_id"] = _map_dataset_name_to_task_id(result["dataset_name"])
        elif result["dataset_id"] is not None:
            # Fallback: use dataset_id as task_id if no dataset_name mapping available
            result["task_id"] = result["dataset_id"]
    
    elif run_type == "llm_baseline":
        # Format can be:
        # 1. Old: {model_name}_{dataset_name}_{metric}
        # 2. New: {model_name}/dataset/{dataset_name}/{metric}
        # Extract dataset_name from metric keys
        llm_models = ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']
        
        # Try model hierarchical format first (most specific)
        for key in summary.keys():
            for model in llm_models:
                if key.startswith(f"model/{model}/dataset/") and key.endswith("/accuracy"):
                    # Extract dataset name from hierarchical path
                    # Format: model/model_name/dataset/dataset_name/metric
                    parts = key.split('/')
                    if len(parts) >= 5:
                        dataset_name = parts[3]
                        result["dataset_name"] = dataset_name
                        break
            if result["dataset_name"]:
                break
        
        # Try new hierarchical format if model format not found
        if not result["dataset_name"]:
            for key in summary.keys():
                for model in llm_models:
                    if key.startswith(f"{model}/dataset/") and key.endswith("/accuracy"):
                        # Extract dataset name from hierarchical path
                        # Format: model/dataset/dataset_name/metric
                        parts = key.split('/')
                        if len(parts) >= 4:
                            dataset_name = parts[2]
                            result["dataset_name"] = dataset_name
                            break
                if result["dataset_name"]:
                    break
        
        # Fallback to old format if new format not found
        if not result["dataset_name"]:
            for key in summary.keys():
                for model in llm_models:
                    if key.startswith(f"{model}_") and key.endswith("_accuracy"):
                        # Extract dataset name from middle part
                        parts = key.split('_')
                        if len(parts) >= 3:
                            if model == 'marvis_tsne' and len(parts) >= 4:
                                # Handle marvis_tsne specially
                                dataset_name = parts[2]
                            else:
                                dataset_name = parts[1]
                            result["dataset_name"] = dataset_name
                            break
                if result["dataset_name"]:
                    break
        
        # For LLM baselines, try to map dataset_name to task_id using known mappings
        if result["dataset_name"]:
            result["task_id"] = _map_dataset_name_to_task_id(result["dataset_name"])
    
    elif run_type == "marvis_training":
        # Format: test/{metric} or direct metrics
        # Get dataset_name from config
        if "dataset_name" in config:
            result["dataset_name"] = config["dataset_name"]
        
        # Extract task_id from run name first (more reliable)
        task_id_from_name = extract_task_idx_from_name(run_name)
        if task_id_from_name is not None:
            result["task_id"] = task_id_from_name
        elif result["dataset_name"]:
            # Fallback to extracting from dataset_name
            result["task_id"] = _extract_numeric_from_string(result["dataset_name"])
    
    elif run_type == "eval":
        # Original eval_ format
        if "dataset_name" in config:
            result["dataset_name"] = config["dataset_name"]
        
        # Try to extract task_id from dataset_name or run_name
        if result["dataset_name"]:
            result["task_id"] = _map_dataset_name_to_task_id(result["dataset_name"])
        
        if result["task_id"] is None:
            result["task_id"] = extract_task_idx_from_name(run_name)
    
    # Special imputation for task_id when missing - try multiple approaches
    if result["task_id"] is None:
        # Try to impute from dataset_name if available
        if result["dataset_name"]:
            result["task_id"] = _map_dataset_name_to_task_id(result["dataset_name"])
        
        # Try to impute from dataset_id if available
        if result["task_id"] is None and result["dataset_id"] is not None:
            try:
                from .openml_mapping import impute_task_id_from_dataset_id
                result["task_id"] = impute_task_id_from_dataset_id(result["dataset_id"])
            except Exception as e:
                logger.debug(f"Failed to impute task_id from dataset_id {result['dataset_id']}: {e}")
        
        # Finally, try extracting from run name
        if result["task_id"] is None:
            result["task_id"] = extract_task_idx_from_name(run_name)
    
    return result


def _map_dataset_name_to_task_id(dataset_name):
    """
    Map dataset names to OpenML task IDs using the OpenML mapping system.
    
    Args:
        dataset_name: The dataset name string
        
    Returns:
        task_id (int) if mapping found, otherwise None
    """
    if dataset_name is None:
        return None
    
    try:
        from .openml_mapping import impute_task_id_from_dataset_name
        return impute_task_id_from_dataset_name(dataset_name)
    except Exception as e:
        logger.warning(f"OpenML mapping failed for {dataset_name}: {e}")
        return None


def _extract_numeric_from_string(text):
    """
    Extract the first numeric value from a string.
    
    Args:
        text: String that may contain numeric values
        
    Returns:
        int if found, otherwise None
    """
    if text is None:
        return None
    
    matches = re.findall(r'\d+', str(text))
    if matches:
        return int(matches[0])
    return None


def extract_task_id_from_dataset_name(dataset_name):
    """
    Legacy function for backward compatibility.
    Extract the task ID from the dataset name.
    
    Args:
        dataset_name: The dataset name, which may contain a task ID
        
    Returns:
        task_id (int) if found, otherwise None
    """
    # First try mapping
    task_id = _map_dataset_name_to_task_id(dataset_name)
    if task_id is not None:
        return task_id
    
    # Then try extracting numeric
    return _extract_numeric_from_string(dataset_name)

def extract_split_idx_from_name(run_name):
    """
    Extract the split index from the run name.
    
    Args:
        run_name: The name of the W&B run
        
    Returns:
        split_idx (int) if found, otherwise None
    """
    # Try to find split index in patterns like "eval_task3_split1" or "eval_task_3_split_1"
    patterns = [
        r'split(\d+)',      # matches split1, split2, etc.
        r'split_(\d+)',     # matches split_1, split_2, etc.
        r'split[_-]?(\d+)', # matches split1, split_1, split-1, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, run_name)
        if match:
            return int(match.group(1))
            
    return None

def extract_task_idx_from_name(run_name):
    """
    Extract the task index from the run name.
    
    Args:
        run_name: The name of the W&B run
        
    Returns:
        task_idx (int) if found, otherwise None
    """
    # Try to find task index in patterns like "eval_task3_split1" or "eval_task_3_split_1"
    patterns = [
        r'task(\d+)',      # matches task1, task2, etc.
        r'task_(\d+)',     # matches task_1, task_2, etc.
        r'task[_-]?(\d+)', # matches task1, task_1, task-1, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, run_name)
        if match:
            return int(match.group(1))
            
    return None

def is_numeric(value):
    """
    Check if a value is numeric (int, float, etc.).
    
    Args:
        value: The value to check
        
    Returns:
        True if the value is numeric, False otherwise
    """
    if value is None:
        return False
        
    # Try to convert to float
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False
        
def safe_float_convert(value):
    """
    Safely convert a value to float, returning None if conversion fails.
    
    Args:
        value: The value to convert
        
    Returns:
        float value if conversion succeeds, None otherwise
    """
    if value is None:
        return None
        
    # Try to convert to float
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def should_exclude_failed_run(summary, model_name=None, dataset_name=None, debug=False):
    """
    Check if a run should be excluded because it failed.
    
    A run is considered failed if:
    1. completed_samples = 0 (no samples were processed)
    2. accuracy = 0 AND other metrics are also 0 or missing (indicates failure, not legitimate poor performance)
    3. Has explicit error/timeout indicators in the summary
    
    Args:
        summary: The W&B run summary dictionary
        model_name: Model name (for LLM baseline format)
        dataset_name: Dataset name (for LLM baseline format)
        debug: Enable debug logging
        
    Returns:
        True if the run failed and should be excluded, False otherwise
    """
    # Check for error/timeout indicators first (these are definitive failures)
    for key, value in summary.items():
        if isinstance(value, str) and value.strip():  # Only check non-empty strings
            value_lower = value.lower()
            if any(error_word in value_lower for error_word in ['timeout', 'timed out', 'operation timed out', 'error', 'failed', 'exception']):
                if debug:
                    logger.debug(f"Excluding run due to error indicator: {key}={value}")
                return True
        
        # Check key names for error indicators, but only if the value indicates an actual error
        if any(error_word in key.lower() for error_word in ['error', 'failed', 'exception']):
            # Only exclude if the error value is non-empty and not a success indicator
            if isinstance(value, str) and value.strip() and not any(success_word in value.lower() for success_word in ['success', 'completed', 'none']):
                if debug:
                    logger.debug(f"Excluding run due to error key: {key}={value}")
                return True
    
    # For traditional format, check completed_samples directly
    if 'completed_samples' in summary:
        completed_samples = summary.get('completed_samples', 0)
        if completed_samples == 0:
            if debug:
                logger.debug(f"Excluding run due to completed_samples = 0")
            return True
    
    # For LLM baseline format, check model-specific completed_samples (all formats)
    if model_name and dataset_name:
        # Check model hierarchical format: model/model_name/dataset/dataset_name/completed_samples
        completed_samples_key_model = f"model/{model_name}/dataset/{dataset_name}/completed_samples"
        if completed_samples_key_model in summary:
            completed_samples = summary.get(completed_samples_key_model, 0)
            if completed_samples == 0:
                if debug:
                    logger.debug(f"Excluding run due to {completed_samples_key_model} = 0")
                return True
        
        # Check new hierarchical format: model/dataset/dataset_name/completed_samples
        completed_samples_key_new = f"{model_name}/dataset/{dataset_name}/completed_samples"
        if completed_samples_key_new in summary:
            completed_samples = summary.get(completed_samples_key_new, 0)
            if completed_samples == 0:
                if debug:
                    logger.debug(f"Excluding run due to {completed_samples_key_new} = 0")
                return True
        
        # Check old format: model_dataset_completed_samples
        completed_samples_key_old = f"{model_name}_{dataset_name}_completed_samples"
        if completed_samples_key_old in summary:
            completed_samples = summary.get(completed_samples_key_old, 0)
            if completed_samples == 0:
                if debug:
                    logger.debug(f"Excluding run due to {completed_samples_key_old} = 0")
                return True
        
        # Additional check for failed runs: accuracy = 0 with missing other metrics
        # Check all format patterns
        accuracy_key_model = f"model/{model_name}/dataset/{dataset_name}/accuracy"
        accuracy_key_new = f"{model_name}/dataset/{dataset_name}/accuracy"
        accuracy_key_old = f"{model_name}_{dataset_name}_accuracy"
        
        accuracy = None
        accuracy_key = None
        if accuracy_key_model in summary:
            accuracy = summary.get(accuracy_key_model, None)
            accuracy_key = accuracy_key_model
        elif accuracy_key_new in summary:
            accuracy = summary.get(accuracy_key_new, None)
            accuracy_key = accuracy_key_new
        elif accuracy_key_old in summary:
            accuracy = summary.get(accuracy_key_old, None)
            accuracy_key = accuracy_key_old
        
        if accuracy is not None and accuracy == 0.0:
            # Check if other metrics are also missing/zero, indicating a failed run
            # Define metric keys for all formats
            if accuracy_key == accuracy_key_model:
                # Model hierarchical format
                other_metric_keys = [
                    f"model/{model_name}/dataset/{dataset_name}/f1_macro",
                    f"model/{model_name}/dataset/{dataset_name}/f1_weighted",
                    f"model/{model_name}/dataset/{dataset_name}/precision_macro", 
                    f"model/{model_name}/dataset/{dataset_name}/recall_macro",
                    f"model/{model_name}/dataset/{dataset_name}/balanced_accuracy",
                    f"model/{model_name}/dataset/{dataset_name}/roc_auc"
                ]
            elif accuracy_key == accuracy_key_new:
                # New hierarchical format
                other_metric_keys = [
                    f"{model_name}/dataset/{dataset_name}/f1_macro",
                    f"{model_name}/dataset/{dataset_name}/f1_weighted",
                    f"{model_name}/dataset/{dataset_name}/precision_macro", 
                    f"{model_name}/dataset/{dataset_name}/recall_macro",
                    f"{model_name}/dataset/{dataset_name}/balanced_accuracy",
                    f"{model_name}/dataset/{dataset_name}/roc_auc"
                ]
            else:
                # Old format
                other_metric_keys = [
                    f"{model_name}_{dataset_name}_f1_macro",
                    f"{model_name}_{dataset_name}_f1_weighted",
                    f"{model_name}_{dataset_name}_precision_macro", 
                    f"{model_name}_{dataset_name}_recall_macro",
                    f"{model_name}_{dataset_name}_balanced_accuracy",
                    f"{model_name}_{dataset_name}_roc_auc"
                ]
            
            # Count how many other metrics exist and are non-zero
            valid_metrics = 0
            for metric_key in other_metric_keys:
                if metric_key in summary:
                    value = summary.get(metric_key, 0)
                    if value is not None and value > 0:
                        valid_metrics += 1
            
            # If accuracy is 0 and no other valid metrics exist, it's likely a failed run
            if valid_metrics == 0:
                if debug:
                    logger.debug(f"Excluding run {model_name}_{dataset_name} due to accuracy=0 with no other valid metrics")
                return True
    
    # For traditional format, also check for accuracy = 0 with missing other metrics
    if model_name is None and dataset_name is None:
        # Check traditional format (direct accuracy key)
        accuracy = summary.get('accuracy', None)
        if accuracy == 0.0:
            # Check if other metrics are also missing/zero
            other_metrics = ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'balanced_accuracy']
            valid_metrics = 0
            for metric in other_metrics:
                if metric in summary:
                    value = summary.get(metric, 0)
                    if value is not None and value > 0:
                        valid_metrics += 1
            
            # If accuracy is 0 and no other valid metrics exist, it's likely a failed run
            if valid_metrics == 0:
                if debug:
                    logger.debug(f"Excluding traditional run due to accuracy=0 with no other valid metrics")
                return True
        
        # Check train_tabular_dataset format (test/ prefix)
        test_accuracy = summary.get('test/accuracy', None)
        if test_accuracy == 0.0:
            # Check if other test metrics are also missing/zero
            other_test_metrics = ['test/f1_score', 'test/precision', 'test/recall']
            valid_metrics = 0
            for metric in other_test_metrics:
                if metric in summary:
                    value = summary.get(metric, 0)
                    if value is not None and value > 0:
                        valid_metrics += 1
            
            # If test accuracy is 0 and no other valid test metrics exist, it's likely a failed run
            if valid_metrics == 0:
                if debug:
                    logger.debug(f"Excluding test run due to test/accuracy=0 with no other valid metrics")
                return True
    
    return False

def analyze_zero_accuracy_run(summary, run_name, config, model_name=None, dataset_name=None):
    """
    Analyze a run with 0 accuracy to determine if it's legitimate or a failure.
    
    Args:
        summary: W&B run summary
        run_name: Name of the run
        config: Run configuration
        model_name: Model name (for LLM baseline format)
        dataset_name: Dataset name (for LLM baseline format)
        
    Returns:
        Dictionary with detailed analysis of the zero accuracy run
    """
    analysis = {
        'run_name': run_name,
        'model_name': model_name or 'unknown',
        'dataset_name': dataset_name or config.get('dataset_name', 'unknown'),
        'config': dict(config) if config else {},
        'all_metrics': {},
        'accuracy_sources': [],
        'other_performance_metrics': {},
        'failure_indicators': [],
        'is_legitimate': False,
        'reasoning': ''
    }
    
    # Check different accuracy sources
    if model_name and dataset_name:
        # LLM baseline format
        accuracy_key = f"{model_name}_{dataset_name}_accuracy"
        if accuracy_key in summary and summary[accuracy_key] == 0.0:
            analysis['accuracy_sources'].append(accuracy_key)
            
            # Check other metrics for this model-dataset combination
            metric_prefixes = [f"{model_name}_{dataset_name}_"]
            for key, value in summary.items():
                if any(key.startswith(prefix) for prefix in metric_prefixes):
                    analysis['all_metrics'][key] = value
                    if 'accuracy' not in key and value is not None and value > 0:
                        analysis['other_performance_metrics'][key] = value
    else:
        # Traditional or train_tabular_dataset format
        if 'accuracy' in summary and summary['accuracy'] == 0.0:
            analysis['accuracy_sources'].append('accuracy')
            
        if 'test/accuracy' in summary and summary['test/accuracy'] == 0.0:
            analysis['accuracy_sources'].append('test/accuracy')
            
        # Get all metrics
        for key, value in summary.items():
            analysis['all_metrics'][key] = value
            # Look for other performance metrics
            if key in ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'balanced_accuracy', 
                      'test/f1_score', 'test/precision', 'test/recall', 'roc_auc'] and value is not None and value > 0:
                analysis['other_performance_metrics'][key] = value
    
    # Check for failure indicators
    if 'completed_samples' in summary and summary.get('completed_samples', 0) == 0:
        analysis['failure_indicators'].append('completed_samples = 0')
        
    if model_name and dataset_name:
        completed_samples_key = f"{model_name}_{dataset_name}_completed_samples"
        if completed_samples_key in summary and summary.get(completed_samples_key, 0) == 0:
            analysis['failure_indicators'].append(f'{completed_samples_key} = 0')
    
    # Check for error indicators in the summary
    error_keys = [key for key in summary.keys() if 'error' in key.lower() or 'failed' in key.lower()]
    if error_keys:
        analysis['failure_indicators'].extend([f'{key}: {summary[key]}' for key in error_keys])
    
    # Check for timeout errors in summary values
    for key, value in summary.items():
        if isinstance(value, str) and ('timeout' in value.lower() or 'timed out' in value.lower()):
            analysis['failure_indicators'].append(f'{key}: {value}')
    
    # Determine if this is legitimate
    has_other_metrics = len(analysis['other_performance_metrics']) > 0
    has_failure_indicators = len(analysis['failure_indicators']) > 0
    
    if has_failure_indicators:
        analysis['is_legitimate'] = False
        analysis['reasoning'] = f"Has failure indicators: {', '.join(analysis['failure_indicators'])}"
    elif has_other_metrics:
        analysis['is_legitimate'] = True
        analysis['reasoning'] = f"Has other valid metrics: {list(analysis['other_performance_metrics'].keys())}"
    else:
        analysis['is_legitimate'] = False
        analysis['reasoning'] = "No other performance metrics found, likely a failed run"
    
    return analysis

def extract_model_metrics_from_summary(summary, debug=False):
    """
    Extract model metrics from the run summary.
    
    Args:
        summary: The W&B run summary dictionary
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    model_metrics = {}
    
    # Standard metrics that directly appear in the summary
    metric_keys = ["accuracy", "f1_macro", "f1_micro", "f1_weighted", 
                 "precision_macro", "recall_macro", "auc", "balanced_accuracy", "completion_rate",
                 "training_time", "prediction_time", "total_time", 
                 "gpu_avg_power_watts", "gpu_total_energy_wh"]
    
    # Check if this is a failed run for traditional format
    if should_exclude_failed_run(summary, debug=debug):
        # Return empty metrics for failed runs
        if debug:
            logger.debug(f"Excluding failed run from metric extraction")
        return model_metrics
    
    # Extract MARVIS metrics (these should be directly in the summary)
    marvis_metrics = {}
    for key in metric_keys:
        if key in summary and is_numeric(summary[key]):
            marvis_metrics[key] = safe_float_convert(summary[key])
    
    if marvis_metrics:
        model_metrics["marvis"] = marvis_metrics
    
    # Extract baseline model metrics
    # These appear in patterns like "model/{model_name}/dataset/{dataset_name}/{metric}"
    # New unified hierarchical format
    hierarchical_pattern = r'^model/([^/]+)/dataset/([^/]+)/(.+)$'
    
    # Old format pattern like "model/random_forest/accuracy" (kept for backward compatibility)
    old_baseline_pattern = r'^model/([^/]+)/(.+)$'
    
    baseline_metrics = defaultdict(lambda: defaultdict(dict))
    for key, value in summary.items():
        # Try hierarchical pattern first
        match = re.match(hierarchical_pattern, key)
        if match and (is_numeric(value) or isinstance(value, dict)):  # Allow dict for classification_report
            model_name = match.group(1)
            dataset_name = match.group(2)
            metric_name = match.group(3)
            
            if debug:
                logger.debug(f"Matched hierarchical pattern: {key} -> model={model_name}, dataset={dataset_name}, metric={metric_name}, value_type={type(value)}")
            
            # Check if this model-dataset combination failed
            if should_exclude_failed_run(summary, model_name, dataset_name, debug=debug):
                if debug:
                    logger.debug(f"Skipping {model_name}_{dataset_name} metrics due to failed run detection")
                continue  # Skip metrics for failed runs
            
            # Handle classification report as a dictionary object
            if metric_name == "classification_report" and isinstance(value, dict):
                # Extract metrics from the nested classification report dictionary
                if debug:
                    logger.debug(f"Processing classification report for {model_name}/{dataset_name}")
                
                # Extract weighted averages
                if "weighted avg" in value:
                    weighted_avg = value["weighted avg"]
                    if "f1-score" in weighted_avg:
                        baseline_metrics[model_name][dataset_name]["f1_weighted"] = safe_float_convert(weighted_avg["f1-score"])
                        if debug:
                            logger.debug(f"Extracted f1_weighted from classification report: {model_name}/{dataset_name} = {weighted_avg['f1-score']}")
                    if "precision" in weighted_avg:
                        baseline_metrics[model_name][dataset_name]["precision_weighted"] = safe_float_convert(weighted_avg["precision"])
                        if debug:
                            logger.debug(f"Extracted precision_weighted from classification report: {model_name}/{dataset_name} = {weighted_avg['precision']}")
                    if "recall" in weighted_avg:
                        baseline_metrics[model_name][dataset_name]["recall_weighted"] = safe_float_convert(weighted_avg["recall"])
                        if debug:
                            logger.debug(f"Extracted recall_weighted from classification report: {model_name}/{dataset_name} = {weighted_avg['recall']}")
                
                # Extract macro averages
                if "macro avg" in value:
                    macro_avg = value["macro avg"]
                    if "f1-score" in macro_avg:
                        baseline_metrics[model_name][dataset_name]["f1_macro"] = safe_float_convert(macro_avg["f1-score"])
                        if debug:
                            logger.debug(f"Extracted f1_macro from classification report: {model_name}/{dataset_name} = {macro_avg['f1-score']}")
                    if "precision" in macro_avg:
                        baseline_metrics[model_name][dataset_name]["precision_macro"] = safe_float_convert(macro_avg["precision"])
                        if debug:
                            logger.debug(f"Extracted precision_macro from classification report: {model_name}/{dataset_name} = {macro_avg['precision']}")
                    if "recall" in macro_avg:
                        baseline_metrics[model_name][dataset_name]["recall_macro"] = safe_float_convert(macro_avg["recall"])
                        if debug:
                            logger.debug(f"Extracted recall_macro from classification report: {model_name}/{dataset_name} = {macro_avg['recall']}")
                
                # Skip storing the raw classification report
                continue
            
            # Handle flattened classification report metrics (dot notation format)
            elif metric_name.startswith("classification_report."):
                # Extract metrics from flattened classification report keys
                # e.g., "classification_report.macro avg.f1-score" or "classification_report.weighted avg.f1-score"
                report_suffix = metric_name[len("classification_report."):]
                
                if debug:
                    logger.debug(f"Processing flattened classification report metric for {model_name}/{dataset_name}: {report_suffix}")
                
                # Handle weighted averages
                if report_suffix == "weighted avg.f1-score":
                    baseline_metrics[model_name][dataset_name]["f1_weighted"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted f1_weighted from flattened classification report: {model_name}/{dataset_name} = {value}")
                elif report_suffix == "weighted avg.precision":
                    baseline_metrics[model_name][dataset_name]["precision_weighted"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted precision_weighted from flattened classification report: {model_name}/{dataset_name} = {value}")
                elif report_suffix == "weighted avg.recall":
                    baseline_metrics[model_name][dataset_name]["recall_weighted"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted recall_weighted from flattened classification report: {model_name}/{dataset_name} = {value}")
                
                # Handle macro averages
                elif report_suffix == "macro avg.f1-score":
                    baseline_metrics[model_name][dataset_name]["f1_macro"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted f1_macro from flattened classification report: {model_name}/{dataset_name} = {value}")
                elif report_suffix == "macro avg.precision":
                    baseline_metrics[model_name][dataset_name]["precision_macro"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted precision_macro from flattened classification report: {model_name}/{dataset_name} = {value}")
                elif report_suffix == "macro avg.recall":
                    baseline_metrics[model_name][dataset_name]["recall_macro"] = safe_float_convert(value)
                    if debug:
                        logger.debug(f"Extracted recall_macro from flattened classification report: {model_name}/{dataset_name} = {value}")
                
                # Skip storing the raw flattened classification report metric
                continue
            else:
                # Store regular metrics directly
                baseline_metrics[model_name][dataset_name][metric_name] = safe_float_convert(value)
                if debug and metric_name == "balanced_accuracy":
                    logger.debug(f"Extracted balanced_accuracy: {model_name}/{dataset_name} = {value}")
        
        # Try old baseline pattern for backward compatibility
        else:
            match = re.match(old_baseline_pattern, key)
            if match and is_numeric(value):
                model_name = match.group(1)
                # Fix empty or "." model names
                if not model_name or model_name == ".":
                    model_name = "marvis"
                metric_name = match.group(2)
                # For old format, use a default dataset name
                baseline_metrics[model_name]["default"][metric_name] = safe_float_convert(value)
    
    # Extract train_tabular_dataset metrics (test/ prefix format)
    # These appear in patterns like "test/accuracy", "test/f1_score", etc.
    train_dataset_metrics = {}
    for key, value in summary.items():
        if key.startswith('test/') and is_numeric(value):
            metric_name = key[5:]  # Remove 'test/' prefix
            # Map test metric names to standard names
            if metric_name == 'f1_score':
                metric_name = 'f1_weighted'  # Use weighted F1 as the standard F1 metric
            train_dataset_metrics[metric_name] = safe_float_convert(value)
    
    # Only add train_dataset_metrics if they exist and the run didn't fail
    if train_dataset_metrics and not should_exclude_failed_run(summary, debug=debug):
        model_metrics["marvis"] = train_dataset_metrics
    elif train_dataset_metrics and debug:
        logger.debug(f"Excluding train_dataset_metrics due to failed run detection")

    # Extract LLM baseline metrics (new format from evaluate_llm_baselines.py)
    # These can appear in three patterns:
    # 1. Old format: "tabllm_adult_accuracy", "jolt_har_balanced_accuracy", "marvis_tsne_adult_accuracy", etc.
    # 2. New hierarchical format: "marvis_tsne/dataset/kr-vs-kp/balanced_accuracy", etc.
    # 3. Model hierarchical format: "model/marvis_tsne/dataset/kr-vs-kp/balanced_accuracy", etc.
    llm_baseline_pattern_old = r'^(tabllm|tabula_8b|jolt|marvis_tsne)_([^_]+)_(.+)$'
    llm_baseline_pattern_new = r'^(tabllm|tabula_8b|jolt|marvis_tsne)/dataset/([^/]+)/(.+)$'
    llm_baseline_pattern_model = r'^model/(tabllm|tabula_8b|jolt|marvis_tsne)/dataset/([^/]+)/(.+)$'
    
    llm_baseline_metrics = defaultdict(lambda: defaultdict(dict))
    for key, value in summary.items():
        # Try model hierarchical pattern first (most specific)
        match = re.match(llm_baseline_pattern_model, key)
        if match and is_numeric(value):
            model_name = match.group(1)
            dataset_name = match.group(2)
            metric_name = match.group(3)
            
            if debug:
                logger.debug(f"Matched model LLM pattern: {key} -> model={model_name}, dataset={dataset_name}, metric={metric_name}")
            
            # Check if this model-dataset combination failed
            if should_exclude_failed_run(summary, model_name, dataset_name, debug=debug):
                if debug:
                    logger.debug(f"Skipping {model_name}_{dataset_name} metrics due to failed run detection")
                continue  # Skip metrics for failed runs
            
            # Store metric value
            llm_baseline_metrics[model_name][dataset_name][metric_name] = safe_float_convert(value)
        else:
            # Try new hierarchical pattern
            match = re.match(llm_baseline_pattern_new, key)
            if match and is_numeric(value):
                model_name = match.group(1)
                dataset_name = match.group(2)
                metric_name = match.group(3)
                
                if debug:
                    logger.debug(f"Matched new LLM pattern: {key} -> model={model_name}, dataset={dataset_name}, metric={metric_name}")
                
                # Check if this model-dataset combination failed
                if should_exclude_failed_run(summary, model_name, dataset_name, debug=debug):
                    if debug:
                        logger.debug(f"Skipping {model_name}_{dataset_name} metrics due to failed run detection")
                    continue  # Skip metrics for failed runs
                
                # Store metric value
                llm_baseline_metrics[model_name][dataset_name][metric_name] = safe_float_convert(value)
            else:
                # Try old pattern for backward compatibility
                match = re.match(llm_baseline_pattern_old, key)
                if match and is_numeric(value):
                    model_name = match.group(1)
                    dataset_name = match.group(2)
                    metric_name = match.group(3)
                    
                    if debug:
                        logger.debug(f"Matched old LLM pattern: {key} -> model={model_name}, dataset={dataset_name}, metric={metric_name}")
                    
                    # Check if this model-dataset combination failed
                    if should_exclude_failed_run(summary, model_name, dataset_name, debug=debug):
                        if debug:
                            logger.debug(f"Skipping {model_name}_{dataset_name} metrics due to failed run detection")
                        continue  # Skip metrics for failed runs
                    
                    # Store metric value
                    llm_baseline_metrics[model_name][dataset_name][metric_name] = safe_float_convert(value)
    
    # Add LLM baseline models to the results if we found metrics
    for model_name, dataset_metrics in llm_baseline_metrics.items():
        if dataset_metrics:  # Only add if we have metrics
            # For LLM baselines, we'll aggregate across datasets later
            # For now, store all dataset-specific metrics
            all_metrics = {}
            for dataset_name, metrics in dataset_metrics.items():
                for metric_name, value in metrics.items():
                    # Use dataset-specific metric names to avoid conflicts
                    dataset_metric_key = f"{dataset_name}_{metric_name}"
                    all_metrics[dataset_metric_key] = value
            
            if all_metrics:
                model_metrics[model_name] = all_metrics
    
    # Add baseline models to the results
    for model_name, dataset_metrics in baseline_metrics.items():
        if dataset_metrics:  # Only add if we have metrics
            # For baseline models with hierarchical metrics, we need to decide how to aggregate
            # For this analysis, we'll store each dataset's metrics with dataset-specific keys
            # but also try to extract common metrics that can be averaged across datasets
            all_metrics = {}
            
            # First pass: collect all dataset-specific metrics  
            for dataset_name, metrics in dataset_metrics.items():
                for metric_name, value in metrics.items():
                    if dataset_name == "default":
                        # Old format - use metric name directly
                        all_metrics[metric_name] = value
                    else:
                        # New hierarchical format - use dataset-specific metric names
                        dataset_metric_key = f"{dataset_name}_{metric_name}"
                        all_metrics[dataset_metric_key] = value
            
            # Second pass: For baseline models, also compute aggregated common metrics
            # This helps with analysis scripts that expect standard metric names
            common_metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
            
            for common_metric in common_metrics:
                # Collect values for this metric across all datasets
                metric_values = []
                for dataset_name, metrics in dataset_metrics.items():
                    if common_metric in metrics and metrics[common_metric] is not None:
                        metric_values.append(metrics[common_metric])
                
                # If we have values, compute the mean and store it
                if metric_values:
                    all_metrics[f"aggregated_{common_metric}"] = sum(metric_values) / len(metric_values)
                    if debug:
                        logger.debug(f"Computed aggregated {common_metric} for {model_name}: {all_metrics[f'aggregated_{common_metric}']} from {len(metric_values)} datasets")
            
            if all_metrics:
                model_metrics[model_name] = all_metrics
    
    return model_metrics

def extract_results_from_wandb(wandb_df, runtime_info=None, debug=False, save_zero_accuracy_details=True):
    """
    Extract and structure results from W&B run data.
    
    Args:
        wandb_df: DataFrame with W&B run data
        runtime_info: Optional dictionary with runtime information from training runs
        debug: Enable debugging output
        save_zero_accuracy_details: Whether to save detailed info about zero accuracy runs
        
    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    logger.info("Extracting results from W&B data")
    
    results = {}
    tasks_without_id = 0
    zero_accuracy_details = []  # Track all runs with 0 accuracy for analysis
    
    for _, row in wandb_df.iterrows():
        config = row["config"]
        summary = row["summary"]
        run_name = row["name"]
        
        if debug:
            logger.info(f"Processing run: {run_name}")
            logger.info(f"Config: {config}")
            logger.info(f"Summary keys: {list(summary.keys())}")
        
        # FIRST: Check for zero accuracy runs and analyze them regardless of filtering
        if save_zero_accuracy_details:
            # Check all possible zero accuracy scenarios
            zero_accuracy_found = False
            
            # Check traditional format
            if 'accuracy' in summary and summary['accuracy'] == 0.0:
                analysis = analyze_zero_accuracy_run(summary, run_name, config)
                analysis['would_be_filtered'] = should_exclude_failed_run(summary, debug=debug)
                zero_accuracy_details.append(analysis)
                zero_accuracy_found = True
                
            # Check train_tabular_dataset format
            if 'test/accuracy' in summary and summary['test/accuracy'] == 0.0:
                analysis = analyze_zero_accuracy_run(summary, run_name, config)
                analysis['would_be_filtered'] = should_exclude_failed_run(summary, debug=debug)
                zero_accuracy_details.append(analysis)
                zero_accuracy_found = True
            
            # Check LLM baseline format (both old and new hierarchical)
            for key, value in summary.items():
                # Check model hierarchical format: model/model_name/dataset/dataset_name/accuracy
                if key.endswith('/accuracy') and value == 0.0:
                    parts = key.split('/')
                    if len(parts) >= 5 and parts[0] == 'model' and parts[2] == 'dataset':  # model/model_name/dataset/dataset_name/accuracy
                        model_name = parts[1]
                        dataset_name = parts[3]
                        if model_name in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']:
                            analysis = analyze_zero_accuracy_run(summary, run_name, config, model_name, dataset_name)
                            analysis['would_be_filtered'] = should_exclude_failed_run(summary, model_name, dataset_name, debug=debug)
                            zero_accuracy_details.append(analysis)
                            zero_accuracy_found = True
                    elif len(parts) >= 4 and parts[1] == 'dataset':  # model_name/dataset/dataset_name/accuracy
                        model_name = parts[0]
                        dataset_name = parts[2]
                        if model_name in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']:
                            analysis = analyze_zero_accuracy_run(summary, run_name, config, model_name, dataset_name)
                            analysis['would_be_filtered'] = should_exclude_failed_run(summary, model_name, dataset_name, debug=debug)
                            zero_accuracy_details.append(analysis)
                            zero_accuracy_found = True
                
                # Check old format: model_dataset_accuracy
                elif key.endswith('_accuracy') and value == 0.0:
                    # Extract model and dataset names
                    parts = key.split('_')
                    if len(parts) >= 3:  # e.g., marvis_tsne_adult_accuracy
                        if parts[0] in ['tabllm', 'tabula', 'jolt', 'marvis']:
                            if parts[0] == 'marvis' and parts[1] == 'tsne':
                                model_name = 'marvis_tsne'
                                dataset_name = parts[2]
                            else:
                                model_name = parts[0]
                                dataset_name = parts[1] if len(parts) > 2 else 'unknown'
                            
                            analysis = analyze_zero_accuracy_run(summary, run_name, config, model_name, dataset_name)
                            # Add filtering information
                            analysis['would_be_filtered'] = should_exclude_failed_run(summary, model_name, dataset_name, debug=debug)
                            zero_accuracy_details.append(analysis)
                            zero_accuracy_found = True
            
            if zero_accuracy_found and debug:
                logger.info(f"Found zero accuracy in run: {run_name}")
        
        # Determine run type for proper variable extraction
        is_llm_baseline_run = any(key for key in summary.keys() 
                                 if (
                                     # Model hierarchical format: model/{model}/dataset/{dataset}/{metric}
                                     (any(key.startswith(f"model/{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                                      any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy']))
                                     or
                                     # New hierarchical format: {model}/dataset/{dataset}/{metric}
                                     (any(key.startswith(f"{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                                      any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy']))
                                     or
                                     # Old format: {model}_{dataset}_{metric}
                                     (any(model in key.lower() for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) 
                                      and any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy']))
                                 ))
        
        is_standard_baseline_run = any(key for key in summary.keys() 
                                      if key.startswith('model/') and '/dataset/' in key
                                      and any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy']))
        
        is_train_dataset_run = any(key for key in summary.keys()
                                  if key.startswith('test/') and any(metric in key for metric in ['accuracy', 'f1_score', 'precision', 'recall']))
        
        is_eval_run = run_name.startswith("eval_")
        
        # Determine run type
        if is_llm_baseline_run:
            run_type = "llm_baseline"
        elif is_standard_baseline_run:
            run_type = "standard_baseline"
        elif is_train_dataset_run:
            run_type = "marvis_training"
        elif is_eval_run:
            run_type = "eval"
        else:
            run_type = "unknown"
        
        # Extract variables using the new function
        variables = extract_variables_from_wandb_data(summary, config, run_name, run_type)
        dataset_name = variables["dataset_name"]
        dataset_id = variables["dataset_id"] 
        task_id = variables["task_id"]
        
        if debug:
            logger.info(f"Run {run_name} - Type: {run_type}, dataset_name: {dataset_name}, dataset_id: {dataset_id}, task_id: {task_id}")
        
        if is_llm_baseline_run:
            # Handle LLM baseline format
            model_metrics = extract_model_metrics_from_summary(summary, debug=debug)
            
            # For LLM baseline runs, extract datasets from metric names
            llm_datasets = set()
            for model_name, metrics in model_metrics.items():
                for metric_key in metrics.keys():
                    # Extract dataset name from metric key like "adult_accuracy"
                    parts = metric_key.split('_')
                    if len(parts) >= 2:
                        dataset_name_from_key = parts[0]
                        llm_datasets.add(dataset_name_from_key)
            
            # Process each dataset found in the metrics
            for dataset_name_from_key in llm_datasets:
                # Use the extracted task_id from our new function for this dataset
                # Re-extract variables for this specific dataset to get accurate task_id
                temp_variables = extract_variables_from_wandb_data(summary, config, run_name, run_type)
                if temp_variables["dataset_name"] == dataset_name_from_key:
                    use_task_id = temp_variables["task_id"]
                    use_dataset_name = temp_variables["dataset_name"]
                else:
                    # Fallback: map this dataset name to task_id
                    use_task_id = _map_dataset_name_to_task_id(dataset_name_from_key)
                    use_dataset_name = dataset_name_from_key
                
                if use_task_id is None:
                    # Use dataset name as task ID if no numeric ID found
                    use_task_id = use_dataset_name
                
                # Extract split index from run name if available, otherwise use 0
                split_idx = extract_split_idx_from_name(run_name)
                if split_idx is None:
                    split_idx = 0
                
                # Initialize task data if needed
                if use_task_id not in results:
                    results[use_task_id] = {
                        "info": {
                            "dataset_name": use_dataset_name,
                            "dataset_id": None,  # LLM baselines don't have explicit dataset_id
                            "task_id": use_task_id
                        },
                        "splits": {}
                    }
                
                # Initialize split data if needed
                if split_idx not in results[use_task_id]["splits"]:
                    results[use_task_id]["splits"][split_idx] = {}
                
                # Extract metrics for this dataset from each model
                for model_name, all_metrics in model_metrics.items():
                    dataset_metrics = {}
                    
                    # Extract metrics specific to this dataset
                    for metric_key, value in all_metrics.items():
                        if metric_key.startswith(f"{use_dataset_name}_"):
                            metric_name = metric_key[len(f"{use_dataset_name}_"):]
                            dataset_metrics[metric_name] = value
                    
                    if dataset_metrics:
                        results[use_task_id]["splits"][split_idx][model_name] = {
                            "metrics": dataset_metrics
                        }
        
        elif is_train_dataset_run:
            # Handle train_tabular_dataset format (test/ prefix metrics)
            if dataset_name is None:
                logger.warning(f"No dataset_name found in train_tabular_dataset run: {run_name}")
                continue
            
            if task_id is None:
                logger.warning(f"No task_id could be extracted for train_tabular_dataset run: {run_name}")
                # Use dataset name as fallback
                task_id = dataset_name
            
            # Get split index from run name first, then config
            split_idx = extract_split_idx_from_name(run_name)
            if split_idx is None:
                split_idx = config.get("split_idx", 0)
            
            # Initialize task data if needed
            if task_id not in results:
                results[task_id] = {
                    "info": {
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "task_id": task_id
                    },
                    "splits": {}
                }
            
            # Initialize split data if needed
            if split_idx not in results[task_id]["splits"]:
                results[task_id]["splits"][split_idx] = {}
            
            # Extract model metrics from summary
            model_metrics = extract_model_metrics_from_summary(summary, debug=debug)
            
            # Store model results
            for model_name, metrics in model_metrics.items():
                results[task_id]["splits"][split_idx][model_name] = {
                    "metrics": metrics
                }
            
            # If no model metrics were found, log a warning
            if not model_metrics:
                logger.warning(f"No model metrics found in summary for train_tabular_dataset run: {run_name}")
                if debug:
                    logger.info(f"Run summary: {summary}")
        
        elif is_standard_baseline_run:
            # Handle standard baseline format (model/{model}/dataset/{dataset}/{metric})
            if dataset_name is None:
                logger.warning(f"No dataset_name found in standard baseline run: {run_name}")
                continue
            
            if task_id is None:
                logger.warning(f"No task_id could be extracted for standard baseline run: {run_name}")
                # Use dataset name as fallback
                task_id = dataset_name
            
            # Get split index from run name or config
            split_idx = extract_split_idx_from_name(run_name)
            if split_idx is None:
                split_idx = config.get("split_idx", 0)
            
            # Initialize task data if needed
            if task_id not in results:
                results[task_id] = {
                    "info": {
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "task_id": task_id
                    },
                    "splits": {}
                }
            
            # Initialize split data if needed
            if split_idx not in results[task_id]["splits"]:
                results[task_id]["splits"][split_idx] = {}
            
            # Extract model metrics from summary
            model_metrics = extract_model_metrics_from_summary(summary, debug=debug)
            
            # Store model results
            for model_name, metrics in model_metrics.items():
                results[task_id]["splits"][split_idx][model_name] = {
                    "metrics": metrics
                }
            
            # If no model metrics were found, log a warning
            if not model_metrics:
                logger.warning(f"No model metrics found in summary for standard baseline run: {run_name}")
                if debug:
                    logger.info(f"Run summary: {summary}")
        
        else:
            # Handle original eval_ format or unknown format
            if dataset_name is None:
                logger.warning(f"No dataset_name found in run: {run_name}")
                continue
                
            if task_id is None:
                # Use a unique identifier for tasks without numeric IDs
                task_id = f"unnamed_task_{tasks_without_id}"
                tasks_without_id += 1
                logger.warning(f"Could not extract task ID from dataset name or run name: {dataset_name}, {run_name}. Using {task_id}")
            
            # Get split index from run name or config
            split_idx = extract_split_idx_from_name(run_name)
            if split_idx is None:
                split_idx = config.get("split_idx")
                
            if split_idx is None:
                # Default to 0 if we can't find a split index
                split_idx = 0
                logger.info(f"No split_idx found in run {run_name}, using default 0")
                
            # Initialize task data if needed
            if task_id not in results:
                results[task_id] = {
                    "info": {
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "task_id": task_id
                    },
                    "splits": {}
                }
                
            # Initialize split data if needed
            if split_idx not in results[task_id]["splits"]:
                results[task_id]["splits"][split_idx] = {}
                
            # Extract model metrics from summary
            model_metrics = extract_model_metrics_from_summary(summary, debug=debug)
            
            # If no model metrics were found, check if MARVIS metrics are in the summary directly
            if not model_metrics:
                # Extract metrics from summary
                marvis_metrics = {}
                metric_keys = ["accuracy", "f1_macro", "f1_micro", "f1_weighted", 
                             "precision_macro", "recall_macro", "auc", "balanced_accuracy"]
                             
                for key in metric_keys:
                    if key in summary and is_numeric(summary[key]):
                        marvis_metrics[key] = safe_float_convert(summary[key])
                        
                if marvis_metrics:
                    model_metrics["marvis"] = marvis_metrics
            
            # Store model results
            for model_name, metrics in model_metrics.items():
                results[task_id]["splits"][split_idx][model_name] = {
                    "metrics": metrics
                }
                
            # If we couldn't determine any model metrics, log a warning
            if not model_metrics:
                logger.warning(f"No model metrics found in summary for run: {run_name}")
                if debug:
                    logger.info(f"Run summary: {summary}")
    
    # Add runtime information from training runs if available
    if runtime_info:
        logger.info("Adding runtime information from training runs")
        for task_id, task_runtime in runtime_info.items():
            # Skip if task not in results
            if task_id not in results:
                continue
                
            for split_idx, split_runtime in task_runtime.items():
                # Skip if split not in results
                if split_idx not in results[task_id]["splits"]:
                    continue
                    
                for model_name, model_runtime in split_runtime.items():
                    # Skip if model not in results
                    if model_name not in results[task_id]["splits"][split_idx]:
                        continue
                        
                    # Add runtime metrics to model metrics
                    for metric_name, metric_value in model_runtime.items():
                        if metric_value is not None and is_numeric(metric_value):
                            # Initialize metrics dict if not exists
                            if "metrics" not in results[task_id]["splits"][split_idx][model_name]:
                                results[task_id]["splits"][split_idx][model_name]["metrics"] = {}
                                
                            results[task_id]["splits"][split_idx][model_name]["metrics"][metric_name] = float(metric_value)
                            if debug:
                                logger.debug(f"Added runtime metric {metric_name}={metric_value} to task {task_id}, split {split_idx}, model {model_name}")
    
    # Save zero accuracy details if requested
    if save_zero_accuracy_details and zero_accuracy_details:
        logger.info(f"Found {len(zero_accuracy_details)} runs with zero accuracy")
        
        # Save detailed analysis
        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zero_accuracy_analysis_{timestamp}.json"
        
        # Convert to JSON serializable format
        serializable_details = []
        for detail in zero_accuracy_details:
            serializable_detail = {}
            for key, value in detail.items():
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_detail[key] = value
                else:
                    serializable_detail[key] = str(value)
            serializable_details.append(serializable_detail)
        
        try:
            with open(filename, 'w') as f:
                legitimate_runs = [d for d in serializable_details if d.get('is_legitimate', False)]
                failed_runs = [d for d in serializable_details if not d.get('is_legitimate', False)]
                filtered_runs = [d for d in serializable_details if d.get('would_be_filtered', False)]
                unfiltered_runs = [d for d in serializable_details if not d.get('would_be_filtered', False)]
                
                json.dump({
                    'timestamp': timestamp,
                    'summary': {
                        'total_zero_accuracy_runs': len(zero_accuracy_details),
                        'legitimate_runs_count': len(legitimate_runs),
                        'failed_runs_count': len(failed_runs),
                        'filtered_runs_count': len(filtered_runs),
                        'unfiltered_runs_count': len(unfiltered_runs),
                        'unfiltered_failed_runs_count': len([d for d in unfiltered_runs if not d.get('is_legitimate', False)])
                    },
                    'legitimate_runs': legitimate_runs,
                    'failed_runs': failed_runs,
                    'filtered_runs': filtered_runs,
                    'unfiltered_runs': unfiltered_runs,
                    'problematic_runs': [d for d in unfiltered_runs if not d.get('is_legitimate', False)],
                    'all_runs': serializable_details
                }, f, indent=2)
            logger.info(f"Saved zero accuracy analysis to {filename}")
        except Exception as e:
            logger.error(f"Failed to save zero accuracy analysis: {e}")
    
    logger.info(f"Extracted results for {len(results)} tasks")
    return results

def detect_run_types(wandb_df):
    """
    Detect the types of runs in the W&B data.
    
    Args:
        wandb_df: DataFrame with W&B run data
        
    Returns:
        Dictionary with counts of each run type
    """
    # Ensure name column is string type to use .str accessor
    wandb_df = wandb_df.copy()
    wandb_df['name'] = wandb_df['name'].astype(str)
    
    eval_runs = wandb_df[wandb_df['name'].str.startswith('eval_')]
    
    # Detect LLM baseline runs (old format and new hierarchical formats)
    llm_baseline_runs = wandb_df[wandb_df['summary'].apply(
        lambda x: any(key for key in x.keys() 
                     if (
                         # Model hierarchical format: model/{model}/dataset/{dataset}/{metric}
                         (any(key.startswith(f"model/{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                          any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                         or
                         # New hierarchical format: {model}/dataset/{dataset}/{metric}
                         (any(key.startswith(f"{model}/dataset/") for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) and
                          any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                         or
                         # Old format: {model}_{dataset}_{metric}
                         (any(model in key.lower() for model in ['tabllm', 'tabula_8b', 'jolt', 'marvis_tsne']) 
                          and any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']))
                     ))
    )]
    
    # Detect traditional baseline runs (scikit-learn models + TabPFN + CatBoost)
    # Look for both old format and new unified hierarchical format
    baseline_model_names = ['catboost', 'tabpfn_v2', 'random_forest', 'gradient_boosting', 'logistic_regression', 
                           'xgboost', 'lightgbm', 'svm', 'knn', 'naive_bayes']
    
    baseline_runs = wandb_df[wandb_df['summary'].apply(
        lambda x: any(key for key in x.keys() 
                     if (
                         # New unified hierarchical format: model/{model_name}/dataset/{dataset_name}/{metric}
                         (key.startswith('model/') and 
                          any(model in key.lower() for model in baseline_model_names) and
                          any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_macro']))
                         or
                         # Old format: {model_name}_{dataset_name}_{metric}
                         (any(model in key.lower() for model in baseline_model_names) and 
                          any(metric in key.lower() for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score']))
                     ))
    )]
    
    train_dataset_runs = wandb_df[wandb_df['summary'].apply(
        lambda x: any(key for key in x.keys()
                     if key.startswith('test/') and any(metric in key for metric in ['accuracy', 'f1_score', 'precision', 'recall']))
    )]
    
    return {
        'eval_runs': eval_runs,
        'llm_baseline_runs': llm_baseline_runs,
        'baseline_runs': baseline_runs,
        'train_dataset_runs': train_dataset_runs,
        'counts': {
            'eval': len(eval_runs),
            'llm_baseline': len(llm_baseline_runs),
            'baseline': len(baseline_runs),
            'train_dataset': len(train_dataset_runs)
        }
    }