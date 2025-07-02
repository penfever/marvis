#!/usr/bin/env python3
"""
Script to parse OpenML CC18 results from tar archives and generate summary spreadsheets.

This script:
1. Extracts and parses results from multiple tar archives containing OpenML CC18 runs
2. Generates two summary spreadsheets:
   - Aggregated performance across all datasets with 95% confidence intervals
   - Per-dataset performance with confidence intervals
3. Handles missing results by imputing chance-level performance
4. Rounds all values to 3 significant digits

Usage:
    python parse_openml_cc18_results.py --input_dir /path/to/results --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def round_to_n_significant_digits(x: float, n: int = 3) -> float:
    """Round a number to n significant digits."""
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values using normal approximation."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean
    
    # Use t-distribution for small samples
    if len(values) < 30:
        dof = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, dof)
        margin_error = t_critical * std_err
    else:
        # Use normal distribution for large samples
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical * std_err
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return ci_lower, ci_upper


def extract_results_from_retry_dir(retry_dir: str) -> List[Dict[str, Any]]:
    """Extract and parse results from retry directory."""
    logger = logging.getLogger(__name__)
    results = []
    
    if not os.path.exists(retry_dir):
        logger.info(f"Retry directory {retry_dir} does not exist, skipping")
        return results
    
    # Look for JSON result files in retry directory
    for file_name in os.listdir(retry_dir):
        if file_name.endswith('.json') and 'all_evaluation_results_' in file_name:
            file_path = os.path.join(retry_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process retry results similar to tabular baseline format
                processed_results = process_tabular_baseline_results(data, file_name, 'retry_results')
                results.extend(processed_results)
                logger.info(f"Loaded {len(processed_results)} retry results from {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing retry file {file_name}: {e}")
                continue
    
    return results


def extract_results_from_tar(tar_path: str, temp_dir: str) -> List[Dict[str, Any]]:
    """Extract and parse results from a tar archive."""
    logger = logging.getLogger(__name__)
    results = []
    archive_name = os.path.basename(tar_path).replace('.tar', '')
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Look for all_results_*.json files first (consolidated format)
            all_results_files = [name for name in tar.getnames() 
                               if name.endswith('.json') and 'all_results_' in name]
            
            # If no consolidated files, look for individual aggregated_results.json files
            if not all_results_files:
                aggregated_files = [name for name in tar.getnames() 
                                  if name.endswith('aggregated_results.json')]
                if aggregated_files:
                    logger.info(f"Found {len(aggregated_files)} individual aggregated_results.json files in {archive_name}")
                    all_results_files = aggregated_files
                else:
                    # Look for tabular baseline files (all_evaluation_results_*.json)
                    eval_results_files = [name for name in tar.getnames() 
                                        if name.endswith('.json') and 'all_evaluation_results_' in name]
                    if eval_results_files:
                        logger.info(f"Found {len(eval_results_files)} tabular baseline result files in {archive_name}")
                        all_results_files = eval_results_files
                    else:
                        logger.warning(f"No recognizable result files found in {tar_path}")
                        return results
            
            for file_name in all_results_files:
                try:
                    # Extract the file to temp directory
                    tar.extract(file_name, temp_dir)
                    extracted_path = os.path.join(temp_dir, file_name)
                    
                    # Load and parse JSON
                    with open(extracted_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle tabular baseline format
                    if 'all_evaluation_results_' in file_name:
                        # Process tabular baseline results
                        processed_results = process_tabular_baseline_results(data, file_name, archive_name)
                        results.extend(processed_results)
                        logger.info(f"Loaded {len(processed_results)} tabular baseline results from {file_name}")
                    else:
                        # Handle existing formats
                        # Add archive source information to each result
                        if isinstance(data, list):
                            for result in data:
                                if isinstance(result, dict):
                                    result['_archive_source'] = archive_name
                            results.extend(data)
                        elif isinstance(data, dict):
                            data['_archive_source'] = archive_name
                            results.append(data)
                        
                        logger.info(f"Loaded {len(data) if isinstance(data, list) else 1} results from {file_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error extracting {tar_path}: {e}")
    
    return results


def process_tabular_baseline_results(data: List[Dict[str, Any]], file_name: str, archive_name: str) -> List[Dict[str, Any]]:
    """Process tabular baseline results into the standard format."""
    logger = logging.getLogger(__name__)
    processed_results = []
    
    # Extract task information from file path
    # Format: project/tabular_baselines/task_XXXXX/split_X/baselines/all_evaluation_results_TIMESTAMP.json
    # For retry results, task_id and split_id will be extracted from individual results
    path_parts = file_name.split('/')
    task_id = None
    split_id = None
    
    for part in path_parts:
        if part.startswith('task_'):
            task_id = part.replace('task_', '')
        elif part.startswith('split_'):
            split_id = part.replace('split_', '')
    
    # For retry results, we'll extract task_id and split_id from the individual results
    if archive_name == 'retry_results':
        task_id = None  # Will be extracted from individual results
        split_id = None  # Will be extracted from individual results
    else:
        if not task_id:
            logger.warning(f"Could not extract task_id from {file_name}")
            task_id = 'unknown'
        
        if not split_id:
            logger.warning(f"Could not extract split_id from {file_name}")
            split_id = '0'
    
    # Process each model result
    for result in data:
        if not isinstance(result, dict):
            continue
        
        # For retry results, extract task_id and split_id from the result itself
        if archive_name == 'retry_results':
            result_task_id = str(result.get('task_id', result.get('dataset_id', 'unknown')))
            result_split_id = str(result.get('split_id', '0'))
        else:
            result_task_id = task_id
            result_split_id = split_id
            
        # Convert to standard format
        processed_result = {
            'model_name': result.get('model_name', 'unknown'),
            'dataset_name': result.get('dataset_name', 'unknown'), 
            'dataset_id': result.get('dataset_id', result_task_id),
            'task_id': result_task_id,
            'split_id': result_split_id,
            'task_type': result.get('task_type', 'classification'),
            'num_classes': result.get('num_classes', 2),
            'accuracy': result.get('accuracy'),
            'balanced_accuracy': result.get('balanced_accuracy'),
            'roc_auc': result.get('roc_auc'),
            'training_time': result.get('training_time'),
            'prediction_time': result.get('prediction_time'),
            'total_time': result.get('total_time'),
            'num_train_samples': result.get('num_train_samples'),
            'num_test_samples': result.get('num_test_samples'),
            'num_features': result.get('num_features'),
            '_archive_source': archive_name,
            '_file_source': file_name
        }
        
        # Calculate additional metrics if not present
        if 'f1_macro' not in result and 'classification_report' in result:
            try:
                # Try to extract F1 macro from classification report
                classification_report = result['classification_report']
                if isinstance(classification_report, dict) and 'macro avg' in classification_report:
                    processed_result['f1_macro'] = classification_report['macro avg'].get('f1-score')
            except Exception as e:
                logger.debug(f"Could not extract F1 macro from classification report: {e}")
        
        processed_results.append(processed_result)
    
    return processed_results


def get_chance_level_performance(n_classes: int) -> Dict[str, float]:
    """Calculate chance-level performance metrics for a given number of classes."""
    chance_accuracy = 1.0 / n_classes
    chance_balanced_accuracy = 1.0 / n_classes
    chance_f1_macro = 1.0 / n_classes  # Simplified approximation
    # ROC AUC is not imputed as requested
    
    return {
        'accuracy': chance_accuracy,
        'balanced_accuracy': chance_balanced_accuracy,
        'f1_macro': chance_f1_macro,
        'f1_micro': chance_accuracy,  # Same as accuracy for balanced case
        'f1_weighted': chance_f1_macro,
        'precision_macro': chance_accuracy,
        'recall_macro': chance_accuracy,
        'roc_auc': None  # Do not impute ROC AUC as requested
    }


def normalize_model_name(model_name: str) -> str:
    """Normalize model names for consistency."""
    model_name = model_name.lower().strip()
    
    # Map variations to standard names
    name_mapping = {
        'marvis-t-sne-tabular': 'marvis_tsne',
        'marvis_t_sne_tabular': 'marvis_tsne',
        'marvis-tsne': 'marvis_tsne',
        'marvis-t-sne': 'marvis_tsne',
        'marvis_tsne': 'marvis_tsne',
        'jolt': 'jolt',
        'tabllm': 'tabllm',
        'tabula-8b': 'tabula_8b',
        'tabula_8b': 'tabula_8b',
        # Tabular baseline models
        'catboost': 'catboost',
        'tabpfn_v2': 'tabpfn_v2',
        'random_forest': 'random_forest',
        'gradient_boosting': 'gradient_boosting',
        'logistic_regression': 'logistic_regression',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'svm': 'svm',
        'naive_bayes': 'naive_bayes',
        'knn': 'knn'
    }
    
    return name_mapping.get(model_name, model_name)


def create_unique_model_identifier(model_name: str, archive_source: str, model_used: str = None) -> str:
    """Create a unique model identifier based on model name, archive source, and backend."""
    normalized_name = normalize_model_name(model_name)
    
    # For tabllm, try to extract backend information from model_used field
    if normalized_name == 'tabllm' and model_used:
        if 'qwen' in model_used.lower():
            return 'tabllm_qwen'
        elif 'llama' in model_used.lower():
            return 'tabllm_llama'
        elif 'mistral' in model_used.lower():
            return 'tabllm_mistral'
        elif 'gemma' in model_used.lower():
            return 'tabllm_gemma'
        elif 'gpt' in model_used.lower():
            return 'tabllm_gpt'
        elif 'gemini' in model_used.lower():
            return 'tabllm_gemini'
    
    # If we can't distinguish by model_used, use archive source as disambiguator
    # Extract meaningful parts from archive name
    archive_lower = archive_source.lower()
    
    # Look for LLM indicators in archive name
    if normalized_name == 'tabllm':
        if 'qwen' in archive_lower:
            return 'tabllm_qwen'
        elif 'llama' in archive_lower:
            return 'tabllm_llama' 
        elif 'mistral' in archive_lower:
            return 'tabllm_mistral'
        elif 'gemma' in archive_lower:
            return 'tabllm_gemma'
        elif 'gpt' in archive_lower or 'openai' in archive_lower:
            return 'tabllm_gpt'
        elif 'gemini' in archive_lower:
            return 'tabllm_gemini'
        elif '32b' in archive_lower:
            return 'tabllm_32b'
        elif '8b' in archive_lower:
            return 'tabllm_8b'
        elif '3b' in archive_lower:
            return 'tabllm_3b'
        else:
            # Fall back to archive-based suffix
            return f'tabllm_{archive_lower.replace("_", "").replace("-", "")}'
    
    # Map marvis_tsne to MARVIS for display
    if normalized_name == 'marvis_tsne':
        return 'MARVIS'
    
    return normalized_name


def detect_model_conflicts(all_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Detect potential model name conflicts across archives."""
    model_archive_mapping = defaultdict(set)
    
    for result in all_results:
        model_name = normalize_model_name(result.get('model_name', 'unknown'))
        archive_source = result.get('_archive_source', 'unknown')
        model_archive_mapping[model_name].add(archive_source)
    
    # Find models that appear in multiple archives
    conflicts = {model: list(archives) for model, archives in model_archive_mapping.items() 
                if len(archives) > 1}
    
    return conflicts


def merge_results_superset(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge results taking the superset of valid predictions for each dataset/split/algorithm triple."""
    logger = logging.getLogger(__name__)
    
    # Group results by unique key: (model_name, dataset_id, split_id, archive_source)
    # Note: We include archive_source to distinguish between different variants of the same model
    # We use dataset_id instead of task_id for matching since retry results only have dataset_id
    result_groups = defaultdict(list)
    
    for result in all_results:
        model_name = result.get('model_name', 'unknown')
        dataset_id = str(result.get('dataset_id', result.get('task_id', 'unknown')))
        split_id = str(result.get('split_id', '0'))
        archive_source = result.get('_archive_source', 'unknown')
        
        # Use dataset_id as primary key for matching across retry and original results
        # Include archive_source to prevent merging different model variants
        key = (model_name, dataset_id, split_id, archive_source)
        result_groups[key].append(result)
    
    # For each group, prefer retry results over original results if they have valid metrics
    merged_results = []
    retry_count = 0
    for key, group in result_groups.items():
        if len(group) == 1:
            merged_results.append(group[0])
        else:
            # Multiple results for the same key - prioritize retry results with valid metrics
            retry_results = [r for r in group if r.get('_archive_source') == 'retry_results']
            original_results = [r for r in group if r.get('_archive_source') != 'retry_results']
            
            chosen_result = None
            
            # Always prefer retry results if they exist and have valid metrics
            for retry_result in retry_results:
                if (retry_result.get('accuracy') is not None or 
                    retry_result.get('balanced_accuracy') is not None):
                    chosen_result = retry_result
                    retry_count += 1
                    logger.debug(f"Using retry result for {key[0]} on dataset {key[1]} (split {key[2]}, archive {key[3]})")
                    break
            
            # If no valid retry result, use original results
            if chosen_result is None:
                for original_result in original_results:
                    if (original_result.get('accuracy') is not None or 
                        original_result.get('balanced_accuracy') is not None):
                        chosen_result = original_result
                        break
            
            # If still no valid result, take the first available
            if chosen_result is None:
                chosen_result = group[0]
            
            # Mark as merged result
            chosen_result['_merged_from'] = len(group)
            merged_results.append(chosen_result)
    
    logger.info(f"Merged {len(all_results)} results into {len(merged_results)} unique dataset/model combinations")
    logger.info(f"Used {retry_count} retry results to replace original results with missing metrics")
    return merged_results


def process_results(all_results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process results and generate summary dataframes."""
    logger = logging.getLogger(__name__)
    
    # Merge results to create superset
    merged_results = merge_results_superset(all_results)
    
    # Detect and report model conflicts
    conflicts = detect_model_conflicts(merged_results)
    if conflicts:
        logger.info("Detected model name conflicts across archives:")
        for model, archives in conflicts.items():
            logger.info(f"  {model}: found in archives {', '.join(archives)}")
        logger.info("Using unique identifiers to distinguish models...")
    
    # Organize results by model, dataset, and task
    organized_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dataset_info = {}  # Store dataset metadata
    
    # Define datasets to exclude
    excluded_datasets = {'CIFAR_10', 'Devnagari-Script', 'jungle_chess_2pcs_raw_endgame_complete'}
    
    for result in merged_results:
        original_model_name = result.get('model_name', 'unknown')
        archive_source = result.get('_archive_source', 'unknown')
        model_used = result.get('model_used', None)
        
        # Create unique model identifier
        unique_model_name = create_unique_model_identifier(
            original_model_name, archive_source, model_used
        )
        
        # Log unique identifier creation for first occurrence  
        if unique_model_name != normalize_model_name(original_model_name):
            logger.debug(f"Created unique identifier: {original_model_name} -> {unique_model_name} (archive: {archive_source})")
        
        # Additional debug for TabLLM to track all variants
        if 'tabllm' in original_model_name.lower():
            logger.debug(f"TabLLM processing: {original_model_name} -> {unique_model_name} (archive: {archive_source}, model_used: {model_used})")
        
        dataset_name = result.get('dataset_name', 'unknown')
        task_id = str(result.get('task_id', result.get('dataset_id', 'unknown')))
        
        # Skip excluded datasets
        if dataset_name in excluded_datasets:
            continue
        
        # Store dataset info
        if task_id not in dataset_info:
            dataset_info[task_id] = {
                'dataset_name': dataset_name,
                'n_classes': result.get('num_classes', 2),
                'task_type': result.get('task_type', 'classification')
            }
        
        # Only process classification tasks
        task_type = result.get('task_type', 'classification')
        if task_type != 'classification':
            continue
            
        # Extract key metrics
        metrics = {
            'accuracy': result.get('accuracy'),
            'balanced_accuracy': result.get('balanced_accuracy'),
            'f1_macro': result.get('f1_macro'),
            'f1_micro': result.get('f1_micro'),
            'f1_weighted': result.get('f1_weighted'),
            'precision_macro': result.get('precision_macro'),
            'recall_macro': result.get('recall_macro'),
            'roc_auc': result.get('roc_auc'),
            'completion_rate': result.get('completion_rate', 1.0)
        }
        
        organized_results[unique_model_name][task_id]['metrics'].append(metrics)
        
        # Store result details for per-dataset analysis (include model metadata)
        result_with_metadata = result.copy()
        result_with_metadata['_unique_model_name'] = unique_model_name
        result_with_metadata['_original_model_name'] = original_model_name
        organized_results[unique_model_name][task_id]['raw_results'].append(result_with_metadata)
    
    logger.info(f"Processed {len(merged_results)} results across {len(organized_results)} models and {len(dataset_info)} datasets")
    
    # Generate aggregated summary
    aggregated_data = []
    per_dataset_data = []
    
    for model_name, model_results in organized_results.items():
        # Collect all metrics across datasets for this model
        all_model_metrics = defaultdict(list)
        
        # Extract metadata from first result to get original model name and archive source
        first_result = None
        for task_results in model_results.values():
            if task_results['raw_results']:
                first_result = task_results['raw_results'][0]
                break
        
        original_model_name = first_result.get('_original_model_name', model_name) if first_result else model_name
        archive_source = first_result.get('_archive_source', 'unknown') if first_result else 'unknown'
        model_used = first_result.get('model_used', None) if first_result else None
        
        for task_id, task_results in model_results.items():
            dataset_name = dataset_info[task_id]['dataset_name']
            n_classes = dataset_info[task_id]['n_classes']
            
            # Get metrics for this dataset
            metrics_list = task_results['metrics']
            
            if not metrics_list:
                # No results for this dataset - use chance level
                chance_metrics = get_chance_level_performance(n_classes)
                metrics_list = [chance_metrics]
                logger.warning(f"No results for {model_name} on {dataset_name} (task {task_id}), using chance level")
            
            # Calculate per-dataset statistics
            dataset_stats = {}
            for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 
                              'f1_weighted', 'precision_macro', 'recall_macro', 'roc_auc']:
                values = [m.get(metric_name) for m in metrics_list if m.get(metric_name) is not None]
                
                if values:
                    mean_val = np.mean(values)
                    ci_lower, ci_upper = calculate_confidence_interval(values)
                    
                    dataset_stats[f'{metric_name}_mean'] = round_to_n_significant_digits(mean_val)
                    dataset_stats[f'{metric_name}_ci_lower'] = round_to_n_significant_digits(ci_lower) if not np.isnan(ci_lower) else np.nan
                    dataset_stats[f'{metric_name}_ci_upper'] = round_to_n_significant_digits(ci_upper) if not np.isnan(ci_upper) else np.nan
                    dataset_stats[f'{metric_name}_std'] = round_to_n_significant_digits(np.std(values))
                    dataset_stats[f'{metric_name}_n_runs'] = len(values)
                    
                    # Add to model aggregation
                    all_model_metrics[metric_name].extend(values)
                else:
                    # No valid values for this metric
                    dataset_stats[f'{metric_name}_mean'] = np.nan
                    dataset_stats[f'{metric_name}_ci_lower'] = np.nan
                    dataset_stats[f'{metric_name}_ci_upper'] = np.nan
                    dataset_stats[f'{metric_name}_std'] = np.nan
                    dataset_stats[f'{metric_name}_n_runs'] = 0
            
            # Add to per-dataset results with metadata
            per_dataset_data.append({
                'model': model_name,
                'original_model_name': original_model_name,
                'archive_source': archive_source,
                'model_used': model_used,
                'task_id': task_id,
                'dataset_name': dataset_name,
                'n_classes': n_classes,
                **dataset_stats
            })
        
        # Calculate aggregated statistics across all datasets for this model
        agg_stats = {
            'model': model_name,
            'original_model_name': original_model_name,
            'archive_source': archive_source,
            'model_used': model_used
        }
        
        for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 
                          'f1_weighted', 'precision_macro', 'recall_macro', 'roc_auc']:
            values = all_model_metrics[metric_name]
            
            if values:
                mean_val = np.mean(values)
                median_val = np.median(values)
                ci_lower, ci_upper = calculate_confidence_interval(values)
                
                agg_stats[f'{metric_name}_mean'] = round_to_n_significant_digits(mean_val)
                agg_stats[f'{metric_name}_median'] = round_to_n_significant_digits(median_val)
                agg_stats[f'{metric_name}_ci_lower'] = round_to_n_significant_digits(ci_lower) if not np.isnan(ci_lower) else np.nan
                agg_stats[f'{metric_name}_ci_upper'] = round_to_n_significant_digits(ci_upper) if not np.isnan(ci_upper) else np.nan
                agg_stats[f'{metric_name}_std'] = round_to_n_significant_digits(np.std(values))
                agg_stats[f'{metric_name}_n_datasets'] = len([task_id for task_id, task_results in model_results.items() 
                                                              if task_results['metrics']])
                agg_stats[f'{metric_name}_n_runs'] = len(values)
            else:
                agg_stats[f'{metric_name}_mean'] = np.nan
                agg_stats[f'{metric_name}_median'] = np.nan
                agg_stats[f'{metric_name}_ci_lower'] = np.nan
                agg_stats[f'{metric_name}_ci_upper'] = np.nan
                agg_stats[f'{metric_name}_std'] = np.nan
                agg_stats[f'{metric_name}_n_datasets'] = 0
                agg_stats[f'{metric_name}_n_runs'] = 0
        
        aggregated_data.append(agg_stats)
    
    # Convert to dataframes
    aggregated_df = pd.DataFrame(aggregated_data)
    per_dataset_df = pd.DataFrame(per_dataset_data)
    
    return aggregated_df, per_dataset_df


def save_results(aggregated_df: pd.DataFrame, per_dataset_df: pd.DataFrame, output_dir: str):
    """Save results to Excel files with proper formatting."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results
    agg_path = os.path.join(output_dir, 'openml_cc18_aggregated_results.xlsx')
    with pd.ExcelWriter(agg_path, engine='openpyxl') as writer:
        aggregated_df.to_excel(writer, sheet_name='Aggregated_Results', index=False)
    
    logger.info(f"Saved aggregated results to {agg_path}")
    
    # Save per-dataset results
    dataset_path = os.path.join(output_dir, 'openml_cc18_per_dataset_results.xlsx')
    with pd.ExcelWriter(dataset_path, engine='openpyxl') as writer:
        per_dataset_df.to_excel(writer, sheet_name='Per_Dataset_Results', index=False)
    
    logger.info(f"Saved per-dataset results to {dataset_path}")
    
    # Also save as CSV for easier analysis
    aggregated_df.to_csv(os.path.join(output_dir, 'openml_cc18_aggregated_results.csv'), index=False)
    per_dataset_df.to_csv(os.path.join(output_dir, 'openml_cc18_per_dataset_results.csv'), index=False)
    
    logger.info("Also saved CSV versions of both files")


def create_critical_difference_plot(algorithm_scores_matrix: Dict[str, Dict[str, float]], 
                                  output_path: str, 
                                  title: str = "Critical Difference Diagram",
                                  alpha: float = 0.05):
    """
    Create a critical difference plot for algorithm comparison.
    
    Args:
        algorithm_scores_matrix: Dict mapping algorithm -> dataset -> score
        output_path: Path to save the plot
        title: Title for the plot
        alpha: Significance level for statistical tests
    """
    logger = logging.getLogger(__name__)
    
    try:
        import scikit_posthocs as sp
    except ImportError:
        logger.warning("scikit-posthocs not installed. Installing...")
        os.system("pip install scikit-posthocs")
        try:
            import scikit_posthocs as sp
        except ImportError:
            logger.error("Failed to install scikit-posthocs. Skipping critical difference plot.")
            return
    
    # Convert to pandas DataFrame
    algorithms = list(algorithm_scores_matrix.keys())
    datasets = set()
    for alg_data in algorithm_scores_matrix.values():
        datasets.update(alg_data.keys())
    datasets = sorted(list(datasets))
    
    # Create matrix with algorithms as columns and datasets as rows
    data_matrix = []
    for dataset in datasets:
        row = []
        for algorithm in algorithms:
            # Use the score if available, otherwise use NaN
            score = algorithm_scores_matrix[algorithm].get(dataset, np.nan)
            row.append(score)
        data_matrix.append(row)
    
    df = pd.DataFrame(data_matrix, columns=algorithms, index=datasets)
    
    # Remove rows with any NaN values for fair comparison
    df_clean = df.dropna()
    
    if len(df_clean) < 3:
        logger.warning(f"Not enough complete datasets ({len(df_clean)}) for statistical testing. Skipping CD plot.")
        return
    
    # Perform Friedman test
    stat, p_value = stats.friedmanchisquare(*[df_clean[col] for col in df_clean.columns])
    
    logger.info(f"Friedman Test Results:")
    logger.info(f"   Statistic: {stat:.4f}")
    logger.info(f"   p-value: {p_value:.4f}")
    
    if p_value < alpha:
        logger.info(f"   Significant differences found (p < {alpha})")
        
        # Perform post-hoc Nemenyi test
        # Pass the dataframe directly so column names are preserved
        nemenyi_results = sp.posthoc_nemenyi_friedman(df_clean)
        
        # Calculate average ranks
        ranks = df_clean.rank(axis=1, ascending=False, method='average')
        avg_ranks = ranks.mean(axis=0).sort_values()
        
        logger.info("Average Ranks:")
        for i, (alg, rank) in enumerate(avg_ranks.items(), 1):
            logger.info(f"   {i}. {alg}: {rank:.3f}")
        
        # Create critical difference plot
        plt.figure(figsize=(10, 6))
        
        # Generate color palette for the number of algorithms
        import matplotlib.cm as cm
        n_algorithms = len(avg_ranks)
        colors = [cm.Set1(i/n_algorithms) for i in range(n_algorithms)]
        
        # Use scikit-posthocs CD diagram
        sp.critical_difference_diagram(
            avg_ranks,
            nemenyi_results,
            label_fmt_left='{label} ({rank:.2f})',
            label_fmt_right='{label} ({rank:.2f})',
            label_props={'size': 12}
        )
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Critical difference plot saved to: {output_path}")
    else:
        logger.info(f"No significant differences found (p >= {alpha})")


def create_performance_matrix_plot(algorithm_dataset_results: Dict[str, Dict[str, List[float]]], 
                                 output_path: str,
                                 metric_name: str = "Accuracy"):
    """
    Create a heatmap showing algorithm performance across datasets with mean/median summary.
    
    Args:
        algorithm_dataset_results: Dict mapping algorithm -> dataset -> [scores]
        output_path: Path to save the plot
        metric_name: Name of the metric being plotted
    """
    logger = logging.getLogger(__name__)
    
    # Get all algorithms and datasets
    algorithms = list(algorithm_dataset_results.keys())
    datasets = set()
    for alg_data in algorithm_dataset_results.values():
        datasets.update(alg_data.keys())
    datasets = sorted(list(datasets))
    
    # Calculate median scores for each algorithm to sort by
    algorithm_median_scores = {}
    algorithm_mean_scores = {}
    
    for algorithm in algorithms:
        scores_across_datasets = []
        for dataset in datasets:
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                if scores:
                    scores_across_datasets.append(statistics.mean(scores))
        
        if scores_across_datasets:
            algorithm_median_scores[algorithm] = statistics.median(scores_across_datasets)
            algorithm_mean_scores[algorithm] = statistics.mean(scores_across_datasets)
        else:
            algorithm_median_scores[algorithm] = 0
            algorithm_mean_scores[algorithm] = 0
    
    # Sort algorithms by mean score (descending)
    algorithms_sorted = sorted(algorithms, key=lambda x: algorithm_mean_scores[x], reverse=True)
    
    # Create matrix with extra columns for mean and median
    matrix = np.zeros((len(algorithms_sorted), len(datasets) + 2))  # +2 for mean and median columns
    
    for i, algorithm in enumerate(algorithms_sorted):
        # Fill dataset scores
        for j, dataset in enumerate(datasets):
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                matrix[i, j] = statistics.mean(scores) if scores else 0
            else:
                matrix[i, j] = np.nan
        
        # Add mean and median to the last two columns
        matrix[i, -2] = algorithm_mean_scores[algorithm]
        matrix[i, -1] = algorithm_median_scores[algorithm]
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(22, 8))
    
    # Create masked arrays for main data and summary columns
    main_matrix = matrix[:, :-2]
    summary_matrix = matrix[:, -2:]
    
    masked_main = np.ma.masked_invalid(main_matrix)
    
    # Plot main heatmap
    im1 = ax.imshow(masked_main, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, 
                    extent=[-0.5, len(datasets)-0.5, len(algorithms_sorted)-0.5, -0.5])
    
    # Add thick black separator line
    separator_x = len(datasets) - 0.5
    ax.axvline(x=separator_x, color='black', linewidth=3, linestyle='-')
    
    # Plot summary columns with same colormap
    summary_extent = [len(datasets)+0.5, len(datasets)+2.5, len(algorithms_sorted)-0.5, -0.5]
    im2 = ax.imshow(summary_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                    extent=summary_extent)
    
    # Set x-axis ticks and labels
    dataset_labels = list(datasets) + ['Mean', 'Median']
    tick_positions = list(range(len(datasets))) + [len(datasets)+1, len(datasets)+2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(dataset_labels, rotation=90, ha='right')
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(len(algorithms_sorted)))
    ax.set_yticklabels(algorithms_sorted)
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=ax)
    cbar.set_label(f'{metric_name}', rotation=270, labelpad=20)
    
    # Add text annotations for all values
    for i in range(len(algorithms_sorted)):
        # Dataset values
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)
        
        # Mean value
        ax.text(len(datasets)+1, i, f'{matrix[i, -2]:.2f}',
                ha="center", va="center", color="black", fontsize=6)
        
        # Median value
        ax.text(len(datasets)+2, i, f'{matrix[i, -1]:.2f}',
                ha="center", va="center", color="black", fontsize=6)
    
    # Style the mean/median columns differently
    ax.axvspan(len(datasets)+0.5, len(datasets)+2.5, facecolor='lightgray', alpha=0.3, zorder=0)
    
    # Add title and labels
    ax.set_title(f'Algorithm Performance Matrix ({metric_name}) - Sorted by Mean Score', 
                 fontsize=14, pad=20)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance matrix plot saved to: {output_path}")


def generate_analysis_report(aggregated_df: pd.DataFrame, per_dataset_df: pd.DataFrame, output_dir: str):
    """Generate additional analysis artifacts including plots and JSON summary."""
    logger = logging.getLogger(__name__)
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Prepare data for analysis
    algorithm_dataset_results = defaultdict(lambda: defaultdict(list))
    algorithm_scores_matrix = defaultdict(dict)
    
    # Extract data from per-dataset results - now using balanced_accuracy for performance matrix
    balanced_accuracy_scores_matrix = defaultdict(dict)
    for _, row in per_dataset_df.iterrows():
        model = row['model']
        dataset = row['dataset_name']
        task_id = str(row['task_id'])
        
        # Use balanced accuracy for both performance matrix and critical difference plot
        if not pd.isna(row['balanced_accuracy_mean']):
            algorithm_dataset_results[model][dataset].append(row['balanced_accuracy_mean'])
            algorithm_scores_matrix[model][dataset] = row['balanced_accuracy_mean']
            balanced_accuracy_scores_matrix[model][dataset] = row['balanced_accuracy_mean']
    
    # Create critical difference plot for balanced accuracy
    if len(balanced_accuracy_scores_matrix) > 1:
        cd_plot_path = plots_dir / "critical_difference_balanced_accuracy.png"
        create_critical_difference_plot(
            dict(balanced_accuracy_scores_matrix),
            str(cd_plot_path),
            title="Critical Difference Diagram - Balanced Accuracy Performance",
            alpha=0.05
        )
        
        # Create performance matrix heatmap with balanced accuracy
        matrix_plot_path = plots_dir / "performance_matrix_heatmap.png"
        create_performance_matrix_plot(
            dict(algorithm_dataset_results),
            str(matrix_plot_path),
            metric_name="Balanced Accuracy"
        )
    
    # Calculate win rates and additional statistics
    win_rates = calculate_win_rates(per_dataset_df)
    performance_distribution = calculate_performance_distribution(aggregated_df)
    
    # Extract all available metrics for comprehensive summary
    all_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
                   'precision_macro', 'recall_macro', 'roc_auc']
    timing_metrics = ['training_time', 'prediction_time', 'total_time']
    
    # Create comprehensive model performance data including all metrics and timing
    comprehensive_model_performance = []
    for _, row in aggregated_df.iterrows():
        model_data = {
            'model': row['model'],
            'original_model_name': row.get('original_model_name', row['model']),
            'archive_source': row.get('archive_source', 'unknown'),
            'model_used': row.get('model_used', None)
        }
        
        # Add all performance metrics
        for metric in all_metrics:
            for stat in ['mean', 'median', 'ci_lower', 'ci_upper', 'std', 'n_datasets', 'n_runs']:
                key = f'{metric}_{stat}'
                if key in row:
                    model_data[key] = row[key]
        
        # Add timing metrics from per-dataset results if available
        model_timing = per_dataset_df[per_dataset_df['model'] == row['model']]
        for timing_metric in timing_metrics:
            if f'{timing_metric}_mean' in model_timing.columns:
                timing_values = model_timing[f'{timing_metric}_mean'].dropna()
                if len(timing_values) > 0:
                    model_data[f'{timing_metric}_mean'] = timing_values.mean()
                    model_data[f'{timing_metric}_median'] = timing_values.median()
                    model_data[f'{timing_metric}_std'] = timing_values.std()
                    model_data[f'{timing_metric}_min'] = timing_values.min()
                    model_data[f'{timing_metric}_max'] = timing_values.max()
        
        comprehensive_model_performance.append(model_data)
    
    # Create comprehensive JSON summary with all metrics
    analysis_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_models": len(aggregated_df),
        "total_datasets": len(per_dataset_df['task_id'].unique()),
        "metrics_included": {
            "performance": all_metrics,
            "timing": timing_metrics
        },
        "model_performance": {
            "comprehensive_results": comprehensive_model_performance,
            "ranking_by_balanced_accuracy": aggregated_df[['model', 'balanced_accuracy_mean', 'balanced_accuracy_median']].sort_values('balanced_accuracy_mean', ascending=False).to_dict('records'),
            "ranking_by_accuracy": aggregated_df[['model', 'accuracy_mean', 'accuracy_median']].sort_values('accuracy_mean', ascending=False).to_dict('records'),
            "win_rates": win_rates,
            "performance_distribution": performance_distribution
        },
        "statistical_tests": {
            "note": "Critical difference plots show statistical significance between models"
        }
    }
    
    # Save analysis summary
    summary_path = Path(output_dir) / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    logger.info(f"Analysis summary saved to: {summary_path}")
    logger.info(f"Plots saved to: {plots_dir}")


def calculate_win_rates(per_dataset_df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate win rates for each model across datasets using balanced accuracy."""
    win_rates = {}
    
    # Group by dataset and find winner for each
    dataset_winners = {}
    for task_id in per_dataset_df['task_id'].unique():
        dataset_data = per_dataset_df[per_dataset_df['task_id'] == task_id]
        if not dataset_data.empty:
            # Filter out rows with NaN balanced_accuracy values
            valid_data = dataset_data.dropna(subset=['balanced_accuracy_mean'])
            if not valid_data.empty:
                winner_row = valid_data.loc[valid_data['balanced_accuracy_mean'].idxmax()]
                dataset_winners[task_id] = winner_row['model']
    
    # Count wins for each model
    total_datasets = len(dataset_winners)
    for model in per_dataset_df['model'].unique():
        wins = sum(1 for winner in dataset_winners.values() if winner == model)
        win_rates[model] = {
            "wins": wins,
            "total_datasets": total_datasets,
            "win_rate": wins / total_datasets if total_datasets > 0 else 0
        }
    
    return win_rates


def calculate_performance_distribution(aggregated_df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate performance distribution categories for each model."""
    distribution = {}
    
    for _, row in aggregated_df.iterrows():
        model = row['model']
        accuracy = row['accuracy_mean']
        
        if pd.isna(accuracy):
            continue
            
        # Categorize performance
        if accuracy >= 0.9:
            category = "excellent"
        elif accuracy >= 0.8:
            category = "good"
        elif accuracy >= 0.7:
            category = "fair"
        else:
            category = "poor"
        
        distribution[model] = {
            "accuracy": accuracy,
            "category": category,
            "f1_macro": row.get('f1_macro_mean', None)
        }
    
    return distribution


def main():
    parser = argparse.ArgumentParser(description="Parse OpenML CC18 results from tar archives")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing tar archives with results (alias for --results_dir)"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        help="Directory containing tar archives with results (deprecated, use --input_dir)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../results/analysis",
        help="Directory to save summary spreadsheets"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--retry_dir", 
        type=str, 
        default=None,
        help="Path to retry results directory (defaults to input_dir/tabular_baselines_retry)"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    # Handle input_dir vs results_dir - prefer input_dir if provided
    if args.input_dir and args.results_dir:
        logger.warning("Both --input_dir and --results_dir provided. Using --input_dir.")
        input_dir = args.input_dir
    elif args.input_dir:
        input_dir = args.input_dir
    elif args.results_dir:
        logger.warning("--results_dir is deprecated. Please use --input_dir instead.")
        input_dir = args.results_dir
    else:
        logger.error("Either --input_dir or --results_dir must be specified")
        parser.print_help()
        return
    
    # Convert paths to absolute
    results_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    logger.info(f"Processing results from: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find tar archives - exclude regression archives
    tar_files = []
    regression_archives = {'jolt_reg.tar', 'clam-reg.tar', 'tabular_baselines_reg.tar'}
    
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.tar'):
            if file_name in regression_archives:
                logger.info(f"Skipping regression archive: {file_name}")
                continue
            tar_path = os.path.join(results_dir, file_name)
            tar_files.append(tar_path)
            logger.info(f"Found classification archive: {file_name}")
    
    if not tar_files:
        logger.error(f"No tar files found in {results_dir}")
        return
    
    # Extract and process results
    all_results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for tar_path in tar_files:
            logger.info(f"Processing {os.path.basename(tar_path)}...")
            try:
                results = extract_results_from_tar(tar_path, temp_dir)
                all_results.extend(results)
                logger.info(f"Extracted {len(results)} results from {os.path.basename(tar_path)}")
            except Exception as e:
                logger.error(f"Error processing {tar_path}: {e}")
                logger.debug(traceback.format_exc())
    
    # Also process retry results if they exist
    retry_dir = args.retry_dir if args.retry_dir else os.path.join(results_dir, "tabular_baselines_retry")
    if os.path.exists(retry_dir):
        logger.info(f"Processing retry results from {retry_dir}")
        try:
            retry_results = extract_results_from_retry_dir(retry_dir)
            all_results.extend(retry_results)
            logger.info(f"Extracted {len(retry_results)} retry results")
        except Exception as e:
            logger.error(f"Error processing retry results: {e}")
            logger.debug(traceback.format_exc())
    
    if not all_results:
        logger.error("No results found in any tar archives")
        return
    
    logger.info(f"Total results collected: {len(all_results)}")
    
    # Process results and generate summaries
    try:
        aggregated_df, per_dataset_df = process_results(all_results)
        
        logger.info(f"Generated aggregated summary with {len(aggregated_df)} model entries")
        logger.info(f"Generated per-dataset summary with {len(per_dataset_df)} entries")
        
        # Display summary
        print("\\n" + "="*80)
        print("AGGREGATED RESULTS SUMMARY")
        print("="*80)
        print(aggregated_df[['model', 'accuracy_mean', 'accuracy_ci_lower', 'accuracy_ci_upper', 
                           'f1_macro_mean', 'f1_macro_ci_lower', 'f1_macro_ci_upper']].to_string(index=False))
        
        # Save results
        save_results(aggregated_df, per_dataset_df, output_dir)
        
        # Generate additional analysis artifacts
        generate_analysis_report(aggregated_df, per_dataset_df, output_dir)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()