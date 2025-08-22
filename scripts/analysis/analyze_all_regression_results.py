#!/usr/bin/env python3
"""
Analyze regression results from per-dataset results instead of overall aggregated files.

This script reads regression results from:
- marvis-reg.tar (MARVIS results)  
- jolt_reg.tar (JOLT results)
- tabular_baselines_reg.tar (Tabular baseline results)

It loads individual per-dataset results from task_*/split_*/*/aggregated_results.json files
and provides a unified comparison.

Usage:
    python scripts/analyze_all_regression_results.py --input_dir /path/to/results --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict
import statistics
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def extract_tar_and_find_per_dataset_results(tar_path: str, temp_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Extract tar file and find all per-dataset result JSON files.
    
    Args:
        tar_path: Path to tar file
        temp_dir: Temporary directory to extract to
        
    Returns:
        List of tuples (file_path, task_id, split_id, source_type)
    """
    result_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)
    
    temp_path = Path(temp_dir)
    
    # Pattern 1: MARVIS results - look for actual metrics files
    for marvis_file in temp_path.glob('**/task_*/split_*/llm_baselines/dataset_*/marvis_t_sne_tabular_results.json'):
        parts = marvis_file.parts
        task_id = None
        split_id = None
        
        for part in parts:
            if part.startswith('task_'):
                task_id = part
            elif part.startswith('split_'):
                split_id = part
        
        if task_id and split_id:
            result_files.append((str(marvis_file), task_id, split_id, 'marvis'))
    
    # Pattern 2: JOLT results - look for jolt_results.json files  
    for jolt_file in temp_path.glob('**/task_*/split_*/llm_baselines/dataset_*/jolt_results.json'):
        parts = jolt_file.parts
        task_id = None
        split_id = None
        
        for part in parts:
            if part.startswith('task_'):
                task_id = part
            elif part.startswith('split_'):
                split_id = part
        
        if task_id and split_id:
            result_files.append((str(jolt_file), task_id, split_id, 'jolt'))
    
    # Pattern 3: Tabular baselines - task_*/split_*/baselines/all_evaluation_results_*.json
    for eval_file in temp_path.glob('**/task_*/split_*/baselines/all_evaluation_results_*.json'):
        parts = eval_file.parts
        task_id = None
        split_id = None
        
        for part in parts:
            if part.startswith('task_'):
                task_id = part
            elif part.startswith('split_'):
                split_id = part
        
        if task_id and split_id:
            result_files.append((str(eval_file), task_id, split_id, 'baselines'))
    
    return result_files

def parse_marvis_per_dataset_format(data: Dict, task_id: str, split_id: str) -> List[Dict]:
    """Parse MARVIS per-dataset format results."""
    results = []
    
    # MARVIS results format - metrics are directly in the JSON
    if isinstance(data, dict):
        # Use task_id as the consistent dataset identifier
        dataset_name = task_id
        
        # Metrics are at the top level
        r2_score = data.get('r2_score', data.get('r2', None))
        mae = data.get('mae', None)
        rmse = data.get('rmse', None)
        
        if r2_score is not None:
            results.append({
                'algorithm': 'MARVIS',
                'dataset_name': dataset_name,
                'task_id': task_id,
                'split_id': split_id,
                'r2_score': r2_score,
                'mae': mae,
                'rmse': rmse
            })
    
    return results

def parse_jolt_per_dataset_format(data: Dict, task_id: str, split_id: str) -> List[Dict]:
    """Parse JOLT per-dataset format results."""
    results = []
    
    if isinstance(data, dict):
        # Use task_id as the consistent dataset identifier
        dataset_name = task_id
        
        # JOLT metrics are at the top level
        r2_score = data.get('r2_score', data.get('r2', None))
        mae = data.get('mae', None)
        rmse = data.get('rmse', None)
        
        if r2_score is not None:
            results.append({
                'algorithm': 'JOLT',
                'dataset_name': dataset_name,
                'task_id': task_id,
                'split_id': split_id,
                'r2_score': r2_score,
                'mae': mae,
                'rmse': rmse
            })
    
    return results

def parse_baseline_per_dataset_format(data: any, task_id: str, split_id: str) -> List[Dict]:
    """Parse tabular baseline per-dataset format results."""
    results = []
    
    if isinstance(data, list):
        # List of model results
        for entry in data:
            model_name = entry.get('model_name', entry.get('algorithm', entry.get('model', 'baseline')))
            # Use task_id as consistent dataset identifier
            dataset_name = task_id
            
            # Extract R² score from various possible locations
            r2_score = None
            if 'metrics' in entry:
                r2_score = entry['metrics'].get('r2_score', entry['metrics'].get('r2', entry['metrics'].get('test_r2', None)))
            else:
                r2_score = entry.get('r2_score', entry.get('r2', entry.get('test_r2', None)))
            
            results.append({
                'algorithm': model_name,
                'dataset_name': dataset_name,
                'task_id': task_id,
                'split_id': split_id,
                'r2_score': r2_score,
                'mae': entry.get('mae', entry.get('test_mae', None)),
                'rmse': entry.get('rmse', entry.get('test_rmse', None))
            })
    elif isinstance(data, dict):
        # Check if it's an evaluation summary with model_results
        if 'model_results' in data:
            # Use task_id as consistent dataset identifier
            dataset_name = task_id
            for model_name, metrics in data['model_results'].items():
                r2_score = metrics.get('r2_score', metrics.get('r2', metrics.get('test_r2', None)))
                results.append({
                    'algorithm': model_name,
                    'dataset_name': dataset_name,
                    'task_id': task_id,
                    'split_id': split_id,
                    'r2_score': r2_score,
                    'mae': metrics.get('mae', metrics.get('test_mae', None)),
                    'rmse': metrics.get('rmse', metrics.get('test_rmse', None))
                })
        else:
            # Try to handle as single model results
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ['r2', 'r2_score', 'test_r2']):
                    r2_score = value.get('r2_score', value.get('r2', value.get('test_r2', None)))
                    results.append({
                        'algorithm': key,
                        'dataset_name': task_id,  # Use task_id consistently
                        'task_id': task_id,
                        'split_id': split_id,
                        'r2_score': r2_score,
                        'mae': value.get('mae', value.get('test_mae', None)),
                        'rmse': value.get('rmse', value.get('test_rmse', None))
                    })
    
    return results

def process_per_dataset_json_file(file_path: str, task_id: str, split_id: str, source: str) -> List[Dict]:
    """Process a single per-dataset JSON file and extract results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return []
    
    # Determine format and parse accordingly
    if source == 'marvis':
        return parse_marvis_per_dataset_format(data, task_id, split_id)
    elif source == 'jolt':
        return parse_jolt_per_dataset_format(data, task_id, split_id)
    else:  # baselines
        return parse_baseline_per_dataset_format(data, task_id, split_id)

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
    try:
        import scikit_posthocs as sp
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("⚠️  scikit-posthocs not installed. Installing...")
        os.system("pip install scikit-posthocs")
        import scikit_posthocs as sp
        from matplotlib.backends.backend_pdf import PdfPages
    
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
    
    # For critical difference plot, we need datasets with results from most algorithms
    # Let's require all major algorithms (at least 5) to have results for a dataset
    min_algorithms = min(5, len(algorithms) - 1)  # Require most algorithms, but be flexible
    
    # Only keep rows (datasets) that have data for at least min_algorithms
    df_filtered = df.dropna(thresh=min_algorithms)
    
    if len(df_filtered) < 3:
        print(f"⚠️  Not enough datasets with {min_algorithms}+ algorithms ({len(df_filtered)}) for statistical testing. Skipping CD plot.")
        return
    
    # Only keep algorithms (columns) that have reasonable data coverage
    # More permissive to include MARVIS and JOLT which have good coverage
    min_datasets_per_algorithm = max(1, len(df_filtered) // 3)  # Algorithm must have data for at least 1/3 of datasets
    algorithms_to_keep = []
    for alg in df_filtered.columns:
        non_nan_count = df_filtered[alg].count()
        print(f"   Algorithm {alg}: {non_nan_count}/{len(df_filtered)} datasets ({non_nan_count/len(df_filtered)*100:.1f}%)")
        if non_nan_count >= min_datasets_per_algorithm:
            algorithms_to_keep.append(alg)
    
    if len(algorithms_to_keep) < 3:
        print(f"⚠️  Not enough algorithms with sufficient data coverage for statistical testing. Skipping CD plot.")
        return
    
    # Keep only the algorithms with good coverage, but fill NaN values for fair comparison
    df_algorithms_filtered = df_filtered[algorithms_to_keep]
    
    # For missing values, use a conservative approach: fill with the algorithm's mean across other datasets
    df_clean = df_algorithms_filtered.fillna(df_algorithms_filtered.mean())
    
    if len(df_clean) < 3:
        print(f"⚠️  After filtering, only {len(df_clean)} complete datasets remain. Need at least 3 for statistical testing. Skipping CD plot.")
        return
        
    print(f"📊 Using {len(df_clean)} datasets and {len(algorithms_to_keep)} algorithms for critical difference analysis")
    print(f"📊 Datasets: {list(df_clean.index)}")
    print(f"📊 Algorithms: {algorithms_to_keep}")
    
    # Check for issues that might cause Friedman test to fail
    print(f"📊 Data shape: {df_clean.shape}")
    print(f"📊 Data summary:")
    for col in df_clean.columns:
        col_data = df_clean[col]
        print(f"   {col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}")
    
    # Check if any columns have identical values (which would cause issues)
    identical_cols = []
    for col in df_clean.columns:
        if df_clean[col].std() == 0:
            identical_cols.append(col)
    
    if identical_cols:
        print(f"⚠️  Warning: Algorithms with identical scores across all datasets: {identical_cols}")
        # Remove algorithms with no variance
        df_clean = df_clean.drop(columns=identical_cols)
        if len(df_clean.columns) < 3:
            print(f"⚠️  After removing constant algorithms, only {len(df_clean.columns)} remain. Skipping CD plot.")
            return
    
    # Perform Friedman test
    try:
        stat, p_value = stats.friedmanchisquare(*[df_clean[col] for col in df_clean.columns])
    except Exception as e:
        print(f"⚠️  Error in Friedman test: {e}")
        print(f"📊 Skipping statistical analysis due to data issues.")
        return
    
    print(f"\n📊 Friedman Test Results:")
    print(f"   Statistic: {stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"   ✅ Significant differences found (p < {alpha})")
        
        # Perform post-hoc Nemenyi test
        nemenyi_results = sp.posthoc_nemenyi_friedman(df_clean)
        
        # Calculate average ranks
        ranks = df_clean.rank(axis=1, ascending=False, method='average')
        avg_ranks = ranks.mean(axis=0).sort_values()
        
        print(f"\n📊 Average Ranks:")
        for i, (alg, rank) in enumerate(avg_ranks.items(), 1):
            print(f"   {i}. {alg}: {rank:.3f}")
        
        # Create critical difference plot
        plt.figure(figsize=(10, 6))
        
        # Use scikit-posthocs CD diagram
        sp.critical_difference_diagram(
            avg_ranks,
            nemenyi_results,
            label_fmt_left='{label} ({rank:.2f})',
            label_fmt_right='{label} ({rank:.2f})',
            label_props={'size': 12},
            color_palette=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        )
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Critical difference plot saved to: {output_path}")
    else:
        print(f"   ❌ No significant differences found (p >= {alpha})")

def create_performance_matrix_plot(algorithm_dataset_results: Dict[str, Dict[str, List[float]]], 
                                 output_path: str,
                                 metric_name: str = "R²"):
    """
    Create a heatmap showing algorithm performance across datasets with mean/median summary.
    
    Args:
        algorithm_dataset_results: Dict mapping algorithm -> dataset -> [scores]
        output_path: Path to save the plot
        metric_name: Name of the metric being plotted
    """
    # Mapping from task_id to friendly dataset names (from regression_semantic directory)
    task_to_name_mapping = {
        "task_361085": "sulfur",
        "task_361086": "medical_charges",
        "task_361087": "MiamiHousing2016",
        "task_361088": "superconduct",
        "task_361099": "Bike_Sharing_Demand",
        "task_361103": "particulate-matter-ukair-2017",
        "task_361104": "SGEMM_GPU_kernel_performance",
        "task_363370": "google_qa_answer_type_reason_explanation",
        "task_363371": "google_qa_question_type_reason_explanation",
        "task_363372": "bookprice_prediction",
        "task_363373": "jc_penney_products",
        "task_363374": "women_clothing_review",
        "task_363375": "ae_price_prediction",
        "task_363376": "news_popularity2",
        "task_363377": "mercari_price_suggestion100K",
        "task_363387": "convai2_inferred",
        "task_363388": "light_inferred",
        "task_363389": "opensubtitles_inferred",
        "task_363391": "jigsaw-unintended-bias-in-toxicity",
        "task_363394": "UCC (Unhealthy Comments Corpus)",
        "task_363396": "Wheat",
        "task_363397": "Phenotype_202",
        "task_363399": "QSAR_Bioconcentration_regression",
        "task_363417": "heart_failure_clinical_records",
        "task_363418": "infrared_thermography_temperature",
        "task_363426": "biosses",
        "task_363431": "Violent_Crime_by_County_1975_to_2016",
        "task_363432": "Crime_Data_from_2010",
        "task_363434": "climate_change_impact_on_agriculture_2024",
        "task_363435": "all-natural-disasters-19002021-eosdis",
        "task_363436": "climate_change_dataset2020-2024",
        "task_363437": "climate_insights_dataset",
        "task_363438": "reddit_opinion_climate_change",
        "task_363439": "temperature_emissions_environmental_trends_2000_2024",
        "task_363440": "pakistan_hunger_data",
        "task_363442": "world_food_wealth_bank",
        "task_363443": "sustainable_development_report_zero_hunger",
        "task_363444": "methane_emissions_rice_crop",
        "task_363447": "IoT_Agriculture_2024",
        "task_363448": "coffee_distribution_across_94_counties",
        "task_363452": "sleep-deprivation-and-cognitive-performance",
        "task_363453": "social-media-impact-on-suicide-rates",
    }
    
    # Get all algorithms and datasets
    algorithms = list(algorithm_dataset_results.keys())
    datasets = set()
    for alg_data in algorithm_dataset_results.values():
        datasets.update(alg_data.keys())
    
    # Convert task IDs to friendly names for display
    dataset_display_names = []
    datasets_sorted = sorted(list(datasets))
    for dataset in datasets_sorted:
        friendly_name = task_to_name_mapping.get(dataset, dataset)
        dataset_display_names.append(friendly_name)
    
    # Calculate mean scores for each algorithm to sort by
    algorithm_mean_scores = {}
    
    for algorithm in algorithms:
        scores_across_datasets = []
        for dataset in datasets_sorted:
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                if scores:
                    scores_across_datasets.append(statistics.mean(scores))
        
        if scores_across_datasets:
            algorithm_mean_scores[algorithm] = statistics.mean(scores_across_datasets)
        else:
            algorithm_mean_scores[algorithm] = 0
    
    # Sort algorithms by mean score (descending)
    algorithms_sorted = sorted(algorithms, key=lambda x: algorithm_mean_scores[x], reverse=True)
    
    # Create matrix with extra columns for mean and median
    matrix = np.zeros((len(algorithms_sorted), len(datasets_sorted) + 2))  # +2 for mean and median columns
    
    for i, algorithm in enumerate(algorithms_sorted):
        # Fill dataset scores
        for j, dataset in enumerate(datasets_sorted):
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                matrix[i, j] = statistics.mean(scores) if scores else 0
            else:
                matrix[i, j] = np.nan
        
        # Calculate mean and median across datasets for this algorithm
        dataset_scores = []
        for dataset in datasets_sorted:
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                if scores:
                    dataset_scores.append(statistics.mean(scores))
        
        if dataset_scores:
            matrix[i, -2] = statistics.mean(dataset_scores)  # Mean
            matrix[i, -1] = statistics.median(dataset_scores)  # Median
        else:
            matrix[i, -2] = 0
            matrix[i, -1] = 0
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(22, 8))
    
    # Create masked arrays for main data and summary columns
    main_matrix = matrix[:, :-2]
    summary_matrix = matrix[:, -2:]
    
    masked_main = np.ma.masked_invalid(main_matrix)
    
    # Plot main heatmap
    im1 = ax.imshow(masked_main, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, 
                    extent=[-0.5, len(datasets_sorted)-0.5, len(algorithms_sorted)-0.5, -0.5])
    
    # Add thick black separator line
    separator_x = len(datasets_sorted) - 0.5
    ax.axvline(x=separator_x, color='black', linewidth=3, linestyle='-')
    
    # Plot summary columns with same colormap
    summary_extent = [len(datasets_sorted)+0.5, len(datasets_sorted)+2.5, len(algorithms_sorted)-0.5, -0.5]
    im2 = ax.imshow(summary_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                    extent=summary_extent)
    
    # Set x-axis ticks and labels
    dataset_labels = dataset_display_names + ['Mean', 'Median']
    tick_positions = list(range(len(datasets_sorted))) + [len(datasets_sorted)+1, len(datasets_sorted)+2]
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
        for j in range(len(datasets_sorted)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)
        
        # Mean value
        ax.text(len(datasets_sorted)+1, i, f'{matrix[i, -2]:.2f}',
                ha="center", va="center", color="black", fontsize=6)
        
        # Median value
        ax.text(len(datasets_sorted)+2, i, f'{matrix[i, -1]:.2f}',
                ha="center", va="center", color="black", fontsize=6)
    
    # Style the mean/median columns differently
    ax.axvspan(len(datasets_sorted)+0.5, len(datasets_sorted)+2.5, facecolor='lightgray', alpha=0.3, zorder=0)
    
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
    
    print(f"📊 Performance matrix plot saved to: {output_path}")

def analyze_all_results(results: List[Dict], output_dir: str = "./regression_analysis_output"):
    """Analyze combined results from all sources."""
    
    # Group by algorithm
    algorithm_results = defaultdict(list)
    algorithm_dataset_results = defaultdict(lambda: defaultdict(list))
    algorithm_dataset_coverage = defaultdict(set)
    algorithm_task_split_results = defaultdict(list)  # For subset analysis
    
    # First pass: collect all results and count total datasets
    all_dataset_results = defaultdict(lambda: defaultdict(list))
    all_datasets_encountered = set()
    
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        task_id = result.get('task_id', 'unknown')
        
        # Track all datasets encountered
        if dataset != 'unknown':
            all_datasets_encountered.add(dataset)
        
        if dataset != 'unknown' and r2_score is not None:
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            all_dataset_results[dataset][algorithm].append(r2_score)
    
    # Filter out datasets where all algorithms perform poorly
    valid_datasets = set()
    filtered_dataset_count = 0
    
    print("\n" + "="*80)
    print("📊 DATASET FILTERING")
    print("="*80)
    
    for dataset, algorithm_scores in all_dataset_results.items():
        max_avg_score = 0.0
        for algorithm, scores in algorithm_scores.items():
            if scores:
                # Use average across splits, not max of individual splits
                avg_score = statistics.mean(scores)
                max_avg_score = max(max_avg_score, avg_score)
        
        if max_avg_score >= 0.1:
            valid_datasets.add(dataset)
        else:
            filtered_dataset_count += 1
            print(f"  Filtering out '{dataset}' - max average R² = {max_avg_score:.6f}")
    
    print(f"\n  Total datasets encountered: {len(all_datasets_encountered)}")
    print(f"  Total datasets filtered out: {filtered_dataset_count}")
    print(f"  Remaining valid datasets: {len(valid_datasets)}")
    
    # Second pass: collect all results and group by dataset, averaging across splits
    dataset_split_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # dataset -> algorithm -> split -> [scores]
    
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        task_id = result.get('task_id', 'unknown')
        split_id = result.get('split_id', 'split_0')
        
        # Only process if dataset is valid
        if dataset not in valid_datasets:
            continue
            
        if r2_score is not None:
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            
            # Store by dataset/algorithm/split
            dataset_split_results[dataset][algorithm][split_id].append(r2_score)
    
    # Now average across splits for each dataset/algorithm combination
    for dataset in dataset_split_results:
        if dataset != 'unknown':
            for algorithm in dataset_split_results[dataset]:
                algorithm_dataset_coverage[algorithm].add(dataset)
                
                # Average across all splits for this dataset/algorithm
                all_split_scores = []
                for split_id in dataset_split_results[dataset][algorithm]:
                    split_scores = dataset_split_results[dataset][algorithm][split_id]
                    if split_scores:
                        # Average within split (in case there are multiple results per split)
                        avg_split_score = sum(split_scores) / len(split_scores)
                        all_split_scores.append(avg_split_score)
                
                if all_split_scores:
                    # Average across splits for this dataset
                    avg_dataset_score = sum(all_split_scores) / len(all_split_scores)
                    algorithm_results[algorithm].append(avg_dataset_score)
                    algorithm_dataset_results[algorithm][dataset].append(avg_dataset_score)
                    
                    # Store with task/dataset info for subset analysis
                    algorithm_task_split_results[algorithm].append({
                        'r2_score': avg_dataset_score,
                        'dataset': dataset,
                        'task_id': task_id,
                        'num_splits': len(all_split_scores)
                    })
    
    # Print overall comparison
    print("\n" + "="*80)
    print("📊 ALGORITHM COMPARISON - OVERALL R² SCORES")
    print("="*80)
    
    algorithm_stats = []
    for algorithm in sorted(algorithm_results.keys()):
        scores = algorithm_results[algorithm]
        if scores:
            avg_r2 = statistics.mean(scores)
            median_r2 = statistics.median(scores)
            min_r2 = min(scores)
            max_r2 = max(scores)
            n_samples = len(scores)
            
            algorithm_stats.append((avg_r2, algorithm))
            
            print(f"\n{algorithm:20s}")
            print(f"  Samples:     {n_samples}")
            print(f"  Average R²:  {avg_r2:.6f}")
            print(f"  Median R²:   {median_r2:.6f}")
            print(f"  Min R²:      {min_r2:.6f}")
            print(f"  Max R²:      {max_r2:.6f}")
    
    # Rank algorithms
    print("\n" + "="*80)
    print("🏆 ALGORITHM RANKING BY AVERAGE R²")
    print("="*80)
    
    algorithm_stats.sort(reverse=True)
    for rank, (avg_r2, algorithm) in enumerate(algorithm_stats, 1):
        print(f"{rank}. {algorithm:20s}: {avg_r2:.6f}")
    
    # Dataset-level comparison
    print("\n" + "="*80)
    print("📋 DATASET-LEVEL COMPARISON")
    print("="*80)
    
    # Get all unique datasets
    all_datasets = set()
    for algorithm in algorithm_dataset_results:
        all_datasets.update(algorithm_dataset_results[algorithm].keys())
    
    # For each dataset, show algorithm comparison
    for dataset in sorted(all_datasets):
        if dataset == 'unknown':
            continue
            
        print(f"\n{dataset}:")
        dataset_scores = []
        
        for algorithm in sorted(algorithm_dataset_results.keys()):
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                avg_score = statistics.mean(scores)
                dataset_scores.append((avg_score, algorithm))
        
        # Sort by score
        dataset_scores.sort(reverse=True)
        for score, algorithm in dataset_scores:
            print(f"  {algorithm:20s}: {score:.6f}")
    
    # Performance distribution
    print("\n" + "="*80)
    print("📊 PERFORMANCE DISTRIBUTION BY ALGORITHM")
    print("="*80)
    
    for algorithm in sorted(algorithm_results.keys()):
        scores = algorithm_results[algorithm]
        if scores:
            excellent = sum(1 for r2 in scores if r2 >= 0.9)
            good = sum(1 for r2 in scores if 0.7 <= r2 < 0.9)
            fair = sum(1 for r2 in scores if 0.5 <= r2 < 0.7)
            poor = sum(1 for r2 in scores if 0.0 <= r2 < 0.5)
            
            total = len(scores)
            
            print(f"\n{algorithm}:")
            print(f"  Excellent (R² ≥ 0.9):  {excellent:3d} ({excellent/total*100:5.1f}%)")
            print(f"  Good (0.7 ≤ R² < 0.9): {good:3d} ({good/total*100:5.1f}%)")
            print(f"  Fair (0.5 ≤ R² < 0.7): {fair:3d} ({fair/total*100:5.1f}%)")
            print(f"  Poor (0.0 ≤ R² < 0.5): {poor:3d} ({poor/total*100:5.1f}%)")
    
    # Dataset coverage analysis
    print("\n" + "="*80)
    print(f"📊 DATASET COVERAGE ANALYSIS (out of {len(valid_datasets)} valid datasets after filtering)")
    print("="*80)
    
    # Get all unique datasets across all algorithms
    all_covered_datasets = set()
    for algorithm in algorithm_dataset_coverage:
        all_covered_datasets.update(algorithm_dataset_coverage[algorithm])
    
    print(f"\nTotal unique datasets covered across all algorithms: {len(all_covered_datasets)}")
    
    # Sort algorithms by coverage
    coverage_stats = []
    for algorithm in algorithm_dataset_coverage:
        num_datasets = len(algorithm_dataset_coverage[algorithm])
        coverage_stats.append((num_datasets, algorithm))
    
    coverage_stats.sort(reverse=True)
    
    print("\nDataset coverage by algorithm:")
    for num_datasets, algorithm in coverage_stats:
        coverage_percent = (num_datasets / len(valid_datasets)) * 100
        print(f"  {algorithm:25s}: {num_datasets:2d} datasets ({coverage_percent:5.1f}%)")
    
    # Generate plots
    print("\n" + "="*80)
    print("📊 GENERATING STATISTICAL PLOTS")
    print("="*80)
    
    # Create output directory for plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for critical difference plot
    algorithm_scores_matrix = {}
    for algorithm in algorithm_dataset_results:
        algorithm_scores_matrix[algorithm] = {}
        for dataset in algorithm_dataset_results[algorithm]:
            # Only include valid datasets
            if dataset not in valid_datasets:
                continue
            scores = algorithm_dataset_results[algorithm][dataset]
            if scores:
                avg_score = statistics.mean(scores)
                algorithm_scores_matrix[algorithm][dataset] = avg_score
    
    # Create critical difference plot
    cd_plot_path = plots_dir / "critical_difference_r2.png"
    create_critical_difference_plot(
        algorithm_scores_matrix,
        str(cd_plot_path),
        title="Critical Difference Diagram - R² Performance",
        alpha=0.05
    )
    
    # Create performance matrix heatmap
    matrix_plot_path = plots_dir / "performance_matrix_heatmap.png"
    create_performance_matrix_plot(
        algorithm_dataset_results,
        str(matrix_plot_path),
        metric_name="R²"
    )
    
    print(f"\n✅ All plots saved to: {plots_dir}")

def load_tabpfnv2_supplemental_results(supplemental_path: str) -> Dict[str, Dict]:
    """Load TabPFNv2 supplemental results from separate JSON file. Returns dict for replacement."""
    supplemental_results = {}
    
    if not Path(supplemental_path).exists():
        print(f"⚠️  TabPFNv2 supplemental results not found: {supplemental_path}")
        return supplemental_results
    
    # Mapping from friendly names to task IDs (corrected based on semantic files)
    name_to_task_mapping = {
        'Crime_Data_from_2010': 'task_363432',  # Fixed - was task_363442 
        'SGEMM_GPU_kernel_performance': 'task_361104',
        'convai2_inferred': 'task_363387',  # Fixed - was task_363389
        'jigsaw-unintended-bias-in-toxicity': 'task_363391',
        'light_inferred': 'task_363388',
        'medical_charges': 'task_361086',  # Fixed - was task_363375 (ae_price_prediction)
        'mercari_price_suggestion100K': 'task_363377',
        'methane_emissions_rice_crop': 'task_363444',
        'opensubtitles_inferred': 'task_363389',  # Fixed - was task_363387 (convai2_inferred)
        'particulate-matter-ukair-2017': 'task_361103',  # Fixed - was task_363373 (jc_penney_products)
        'world_food_wealth_bank': 'task_363442'  # Fixed - was task_363372 (bookprice_prediction)
    }
    
    try:
        with open(supplemental_path, 'r') as f:
            data = json.load(f)
        
        print(f"📦 Processing TabPFNv2 supplemental results from {supplemental_path}")
        print(f"  Found {len(data)} supplemental TabPFNv2 results")
        
        for entry in data:
            if entry.get('model_name') == 'tabpfn_v2' and entry.get('task_type') == 'regression':
                dataset_name = entry.get('dataset_name', 'unknown')
                r2_score = entry.get('r2', None)
                
                # Map to task_id if available
                task_id = name_to_task_mapping.get(dataset_name)
                if task_id and r2_score is not None:
                    supplemental_results[task_id] = {
                        'algorithm': 'tabpfn_v2',
                        'dataset_name': task_id,  # Use task_id consistently
                        'task_id': task_id,
                        'split_id': 'split_0',  # Supplemental results as single split
                        'r2_score': r2_score,
                        'mae': entry.get('mae', None),
                        'rmse': entry.get('rmse', None)
                    }
                    print(f"    Mapped {dataset_name} -> {task_id}")
        
        print(f"    Loaded {len(supplemental_results)} TabPFNv2 replacement results")
        
    except Exception as e:
        print(f"⚠️  Error loading TabPFNv2 supplemental results: {e}")
    
    return supplemental_results

def save_analysis_summary(results: List[Dict], output_dir: str):
    """Save comprehensive analysis summary to JSON."""
    import json
    
    # Organize results by algorithm for summary
    algorithm_results = defaultdict(list)
    algorithm_dataset_results = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        
        if r2_score is not None and dataset != 'unknown':
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            algorithm_results[algorithm].append(r2_score)
            algorithm_dataset_results[algorithm][dataset].append(r2_score)
    
    # Calculate comprehensive statistics
    comprehensive_model_performance = []
    for algorithm in algorithm_results:
        scores = algorithm_results[algorithm]
        if scores:
            model_data = {
                'algorithm': algorithm,
                'r2_mean': statistics.mean(scores),
                'r2_median': statistics.median(scores),
                'r2_std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'r2_min': min(scores),
                'r2_max': max(scores),
                'n_datasets': len(algorithm_dataset_results[algorithm]),
                'n_results': len(scores)
            }
            
            # Calculate additional metrics if available
            mae_scores = []
            rmse_scores = []
            for result in results:
                if result['algorithm'] == algorithm:
                    if result.get('mae') is not None:
                        mae_scores.append(result['mae'])
                    if result.get('rmse') is not None:
                        rmse_scores.append(result['rmse'])
            
            if mae_scores:
                model_data.update({
                    'mae_mean': statistics.mean(mae_scores),
                    'mae_median': statistics.median(mae_scores),
                    'mae_std': statistics.stdev(mae_scores) if len(mae_scores) > 1 else 0
                })
            
            if rmse_scores:
                model_data.update({
                    'rmse_mean': statistics.mean(rmse_scores),
                    'rmse_median': statistics.median(rmse_scores),
                    'rmse_std': statistics.stdev(rmse_scores) if len(rmse_scores) > 1 else 0
                })
            
            comprehensive_model_performance.append(model_data)
    
    # Create analysis summary
    analysis_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_algorithms": len(algorithm_results),
        "total_datasets": len(set(result['dataset_name'] for result in results if result['dataset_name'] != 'unknown')),
        "metrics_included": {
            "performance": ["r2_score", "mae", "rmse"],
            "timing": []  # No timing metrics in regression analysis currently
        },
        "algorithm_performance": {
            "comprehensive_results": comprehensive_model_performance,
            "ranking_by_r2_mean": sorted(comprehensive_model_performance, key=lambda x: x['r2_mean'], reverse=True),
            "ranking_by_r2_median": sorted(comprehensive_model_performance, key=lambda x: x['r2_median'], reverse=True)
        }
    }
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(output_dir) / "regression_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"📊 Analysis summary saved to: {summary_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze regression results from tar archives")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing tar archives with results"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        help="Directory containing tar archives with results (deprecated, use --input_dir)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./regression_analysis_output",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Handle input_dir vs results_dir - prefer input_dir if provided
    if args.input_dir and args.results_dir:
        print("⚠️  Both --input_dir and --results_dir provided. Using --input_dir.")
        input_dir = args.input_dir
    elif args.input_dir:
        input_dir = args.input_dir
    elif args.results_dir:
        print("⚠️  --results_dir is deprecated. Please use --input_dir instead.")
        input_dir = args.results_dir
    else:
        # Default fallback
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        input_dir = str(project_root / "results")
        print(f"📁 Using default input directory: {input_dir}")
    
    results_dir = Path(input_dir)
    output_dir = args.output_dir
    
    tar_files = {
        'marvis': results_dir / "clam-reg.tar",
        'jolt': results_dir / "jolt_reg.tar", 
        'baselines': results_dir / "tabular_baselines_reg.tar"
    }
    
    all_results = []
    
    # Process each tar file for per-dataset results
    for source, tar_path in tar_files.items():
        if not tar_path.exists():
            print(f"⚠️  Skipping {source}: {tar_path} not found")
            continue
            
        print(f"\n📦 Processing {source} from {tar_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract and find per-dataset result files
            per_dataset_files = extract_tar_and_find_per_dataset_results(str(tar_path), temp_dir)
            print(f"  Found {len(per_dataset_files)} per-dataset result files")
            
            # Process each per-dataset JSON file
            for file_path, task_id, split_id, file_source in per_dataset_files:
                results = process_per_dataset_json_file(file_path, task_id, split_id, file_source)
                all_results.extend(results)
                if results:
                    print(f"    {task_id}/{split_id}: {len(results)} results")
    
    # Load TabPFNv2 supplemental results (these replace existing TabPFNv2 results)
    supplemental_path = results_dir / "all_evaluation_results_20250626_133915.json"
    tabpfnv2_replacements = load_tabpfnv2_supplemental_results(str(supplemental_path))
    
    # Replace TabPFNv2 results with supplemental results where available
    # Remove all existing TabPFNv2 results for tasks that have supplemental data
    tasks_to_replace = set(tabpfnv2_replacements.keys())
    original_count = len(all_results)
    
    # Remove existing TabPFNv2 results for tasks that have replacements
    all_results = [result for result in all_results 
                   if not (result['algorithm'] == 'tabpfn_v2' and result['task_id'] in tasks_to_replace)]
    
    removed_count = original_count - len(all_results)
    
    # Add the supplemental TabPFNv2 results
    for task_id, replacement_result in tabpfnv2_replacements.items():
        # Add the replacement for all 3 splits to match other algorithms
        for split_id in ['split_0', 'split_1', 'split_2']:
            replacement_copy = replacement_result.copy()
            replacement_copy['split_id'] = split_id
            all_results.append(replacement_copy)
    
    added_count = len(tabpfnv2_replacements) * 3  # 3 splits per task
    
    print(f"\n📦 Removed {removed_count} existing TabPFNv2 results for {len(tasks_to_replace)} tasks")
    print(f"📦 Added {added_count} TabPFNv2 supplemental results ({len(tasks_to_replace)} tasks × 3 splits)")
    print(f"📦 Tasks replaced: {sorted(tasks_to_replace)}")
    
    # Analyze all results
    if all_results:
        print(f"\n📊 Total results collected: {len(all_results)}")
        print(f"📊 Loading from per-dataset results - latest updates included")
        
        # Save comprehensive analysis summary
        save_analysis_summary(all_results, output_dir)
        
        # Run analysis with updated output directory
        analyze_all_results(all_results, output_dir)
    else:
        print("\n❌ No results found to analyze!")

if __name__ == "__main__":
    main()