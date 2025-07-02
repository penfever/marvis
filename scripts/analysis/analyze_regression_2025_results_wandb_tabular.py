#!/usr/bin/env python
"""
Script to analyze and visualize results from the New OpenML Suite 2025 regression collection.

This script:
1. Fetches results from Weights & Biases for regression experiments
2. Generates comprehensive analysis including:
   - Model performance comparisons (MSE, RMSE, MAE, R²)
   - Statistical significance tests
   - Visualization of results across datasets
   - Performance correlation analysis
3. Saves results and visualizations for further analysis

Requirements:
- wandb installed (pip install wandb)
- matplotlib, seaborn for visualization
- scipy for statistical tests
- pandas for data analysis

Usage:
    python analyze_regression_2025_results_wandb_tabular.py --project_name marvis-regression-llm-baselines-2025 --output_dir ./analysis
"""

import os
import argparse
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import wandb
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze OpenML regression 2025 results from W&B")
    
    parser.add_argument(
        "--project_name",
        type=str,
        default="marvis-regression-llm-baselines-2025",
        help="W&B project name prefix (will search for projects with this pattern)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="nyu-dice-lab",
        help="W&B entity/team name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./regression_2025_analysis",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--min_runs_per_model",
        type=int,
        default=5,
        help="Minimum number of runs required per model for inclusion in analysis"
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Confidence level for statistical tests"
    )
    parser.add_argument(
        "--save_raw_data",
        action="store_true",
        help="Save raw extracted data as CSV"
    )
    
    return parser.parse_args()

def connect_to_wandb(entity: str) -> wandb.Api:
    """Connect to Weights & Biases API."""
    try:
        api = wandb.Api()
        logger.info(f"Connected to W&B for entity: {entity}")
        return api
    except Exception as e:
        logger.error(f"Failed to connect to W&B: {e}")
        raise

def fetch_regression_runs(api: wandb.Api, entity: str, project_pattern: str) -> List[wandb.Run]:
    """Fetch all runs from regression projects matching the pattern."""
    all_runs = []
    
    try:
        # Get all projects for the entity
        projects = api.projects(entity)
        regression_projects = [p for p in projects if project_pattern in p.name.lower()]
        
        logger.info(f"Found {len(regression_projects)} regression projects")
        
        for project in regression_projects:
            logger.info(f"Fetching runs from project: {project.name}")
            try:
                runs = api.runs(f"{entity}/{project.name}")
                project_runs = [run for run in runs if run.state == "finished"]
                all_runs.extend(project_runs)
                logger.info(f"  Found {len(project_runs)} completed runs")
            except Exception as e:
                logger.warning(f"Error fetching runs from {project.name}: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        return []
    
    logger.info(f"Total completed runs found: {len(all_runs)}")
    return all_runs

def extract_regression_metrics(runs: List[wandb.Run]) -> pd.DataFrame:
    """Extract regression metrics from W&B runs."""
    data = []
    
    regression_metrics = ['mse', 'rmse', 'mae', 'r2', 'r2_score']  # Common regression metrics
    
    for run in runs:
        try:
            # Extract basic run information
            run_data = {
                'run_id': run.id,
                'run_name': run.name,
                'project': run.project,
                'state': run.state,
                'created_at': run.created_at,
                'runtime': run._attrs.get('runtime', 0)
            }
            
            # Extract config
            config = run.config
            run_data.update({
                'model': config.get('model', 'unknown'),
                'task_id': config.get('task_id', None),
                'dataset_name': config.get('dataset_name', 'unknown'),
                'split_idx': config.get('split_idx', 0),
                'seed': config.get('seed', None),
                'nn_k': config.get('nn_k', None),
                'use_3d': config.get('use_3d', False),
                'model_id': config.get('model_id', 'unknown')
            })
            
            # Extract summary metrics (final values)
            summary = run.summary
            for metric in regression_metrics:
                run_data[metric] = summary.get(metric, None)
                run_data[f'{metric}_best'] = summary.get(f'{metric}_best', None)
                run_data[f'{metric}_final'] = summary.get(f'{metric}_final', None)
            
            # Extract additional metrics that might be present
            for key, value in summary.items():
                if isinstance(value, (int, float)) and not np.isnan(float(value)):
                    if 'loss' in key.lower() or 'score' in key.lower() or 'error' in key.lower():
                        run_data[key] = value
            
            data.append(run_data)
            
        except Exception as e:
            logger.warning(f"Error extracting data from run {run.id}: {e}")
            continue
    
    df = pd.DataFrame(data)
    logger.info(f"Extracted data from {len(df)} runs")
    return df

def clean_and_validate_data(df: pd.DataFrame, min_runs_per_model: int) -> pd.DataFrame:
    """Clean and validate the extracted data."""
    logger.info("Cleaning and validating data...")
    
    # Remove runs without task_id
    initial_count = len(df)
    df = df.dropna(subset=['task_id'])
    logger.info(f"Removed {initial_count - len(df)} runs without task_id")
    
    # Convert task_id to integer
    df['task_id'] = df['task_id'].astype(int)
    
    # Remove models with insufficient runs
    model_counts = df['model'].value_counts()
    valid_models = model_counts[model_counts >= min_runs_per_model].index.tolist()
    df = df[df['model'].isin(valid_models)]
    logger.info(f"Keeping models with >= {min_runs_per_model} runs: {valid_models}")
    
    # Create a primary performance metric (prefer MSE, then RMSE, then MAE)
    df['primary_metric'] = np.nan
    for metric in ['mse', 'rmse', 'mae']:
        mask = df['primary_metric'].isna() & df[metric].notna()
        df.loc[mask, 'primary_metric'] = df.loc[mask, metric]
        df.loc[mask, 'primary_metric_name'] = metric
    
    # Remove runs without any regression metrics
    initial_count = len(df)
    df = df.dropna(subset=['primary_metric'])
    logger.info(f"Removed {initial_count - len(df)} runs without regression metrics")
    
    logger.info(f"Final dataset: {len(df)} runs across {df['model'].nunique()} models and {df['task_id'].nunique()} tasks")
    return df

def calculate_performance_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance statistics for each model."""
    stats_data = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Calculate statistics for each metric
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            metric_data = model_data[metric].dropna()
            if len(metric_data) > 0:
                stats_data.append({
                    'model': model,
                    'metric': metric,
                    'count': len(metric_data),
                    'mean': metric_data.mean(),
                    'median': metric_data.median(),
                    'std': metric_data.std(),
                    'min': metric_data.min(),
                    'max': metric_data.max(),
                    'q25': metric_data.quantile(0.25),
                    'q75': metric_data.quantile(0.75)
                })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df

def perform_statistical_tests(df: pd.DataFrame, confidence_level: float) -> Dict:
    """Perform statistical significance tests between models."""
    results = {}
    models = df['model'].unique()
    
    # Pairwise comparisons for each metric
    for metric in ['mse', 'rmse', 'mae', 'r2']:
        metric_data = df.pivot_table(
            values=metric, 
            index=['task_id', 'split_idx'], 
            columns='model', 
            aggfunc='mean'
        ).dropna()
        
        if len(metric_data) < 5:  # Need minimum samples for meaningful tests
            continue
            
        results[metric] = {}
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if model1 in metric_data.columns and model2 in metric_data.columns:
                    data1 = metric_data[model1].dropna()
                    data2 = metric_data[model2].dropna()
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        # Wilcoxon signed-rank test (paired)
                        try:
                            stat, p_value = stats.wilcoxon(data1, data2)
                            results[metric][f"{model1}_vs_{model2}"] = {
                                'statistic': stat,
                                'p_value': p_value,
                                'significant': p_value < (1 - confidence_level),
                                'effect_size': (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2),
                                'n_pairs': len(data1)
                            }
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {model1} vs {model2} on {metric}: {e}")
    
    return results

def create_visualizations(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Model performance comparison (boxplots)
    metrics_to_plot = ['mse', 'rmse', 'mae', 'r2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        metric_data = df[df[metric].notna()]
        if len(metric_data) > 0:
            sns.boxplot(data=metric_data, x='model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Distribution by Model')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance across datasets
    pivot_data = df.pivot_table(
        values='primary_metric', 
        index='task_id', 
        columns='model', 
        aggfunc='mean'
    )
    
    if not pivot_data.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data.T, annot=False, cmap='viridis', cbar_kws={'label': 'Performance Metric'})
        plt.title('Model Performance Across Regression Tasks')
        plt.xlabel('Task ID')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance correlation between models
    if len(pivot_data.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = pivot_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Model Performance Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Summary statistics plot
    if not stats_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            metric_stats = stats_df[stats_df['metric'] == metric]
            if not metric_stats.empty:
                sns.barplot(data=metric_stats, x='model', y='mean', ax=axes[i])
                axes[i].errorbar(range(len(metric_stats)), metric_stats['mean'], 
                               yerr=metric_stats['std'], fmt='none', color='black', capsize=5)
                axes[i].set_title(f'{metric.upper()} - Mean ± Std')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def save_analysis_results(df: pd.DataFrame, stats_df: pd.DataFrame, 
                         statistical_tests: Dict, output_dir: str, save_raw: bool):
    """Save all analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    stats_df.to_csv(os.path.join(output_dir, 'performance_statistics.csv'), index=False)
    
    # Save statistical test results
    with open(os.path.join(output_dir, 'statistical_tests.json'), 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_safe_tests = {}
        for metric, tests in statistical_tests.items():
            json_safe_tests[metric] = {}
            for comparison, result in tests.items():
                json_safe_tests[metric][comparison] = {k: convert_numpy(v) for k, v in result.items()}
        
        json.dump(json_safe_tests, f, indent=2)
    
    # Save raw data if requested
    if save_raw:
        df.to_csv(os.path.join(output_dir, 'raw_experiment_data.csv'), index=False)
    
    # Create summary report
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_runs': len(df),
        'unique_models': df['model'].nunique(),
        'unique_tasks': df['task_id'].nunique(),
        'models_analyzed': df['model'].unique().tolist(),
        'tasks_analyzed': sorted(df['task_id'].unique().tolist()),
        'metrics_available': [col for col in df.columns if col in ['mse', 'rmse', 'mae', 'r2']],
        'best_performing_model': {
            'by_mse': stats_df[stats_df['metric'] == 'mse'].loc[stats_df[stats_df['metric'] == 'mse']['mean'].idxmin(), 'model'] if 'mse' in stats_df['metric'].values else None,
            'by_rmse': stats_df[stats_df['metric'] == 'rmse'].loc[stats_df[stats_df['metric'] == 'rmse']['mean'].idxmin(), 'model'] if 'rmse' in stats_df['metric'].values else None,
            'by_mae': stats_df[stats_df['metric'] == 'mae'].loc[stats_df[stats_df['metric'] == 'mae']['mean'].idxmin(), 'model'] if 'mae' in stats_df['metric'].values else None,
            'by_r2': stats_df[stats_df['metric'] == 'r2'].loc[stats_df[stats_df['metric'] == 'r2']['mean'].idxmax(), 'model'] if 'r2' in stats_df['metric'].values else None
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Analysis results saved to {output_dir}")
    return report

def main():
    args = parse_args()
    
    # Connect to W&B
    api = connect_to_wandb(args.entity)
    
    # Fetch runs
    runs = fetch_regression_runs(api, args.entity, args.project_name)
    if not runs:
        logger.error("No runs found. Exiting.")
        return
    
    # Extract metrics
    df = extract_regression_metrics(runs)
    if df.empty:
        logger.error("No data extracted. Exiting.")
        return
    
    # Clean and validate data
    df = clean_and_validate_data(df, args.min_runs_per_model)
    if df.empty:
        logger.error("No valid data after cleaning. Exiting.")
        return
    
    # Calculate statistics
    stats_df = calculate_performance_statistics(df)
    
    # Perform statistical tests
    statistical_tests = perform_statistical_tests(df, args.confidence_level)
    
    # Create visualizations
    create_visualizations(df, stats_df, args.output_dir)
    
    # Save results
    report = save_analysis_results(df, stats_df, statistical_tests, args.output_dir, args.save_raw_data)
    
    # Print summary
    print("\n" + "="*80)
    print("REGRESSION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total runs analyzed: {report['total_runs']}")
    print(f"Models: {', '.join(report['models_analyzed'])}")
    print(f"Tasks: {len(report['tasks_analyzed'])}")
    print(f"Best performing models:")
    for metric, model in report['best_performing_model'].items():
        if model:
            print(f"  {metric}: {model}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()