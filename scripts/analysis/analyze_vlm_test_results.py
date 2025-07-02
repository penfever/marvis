#!/usr/bin/env python3
"""
Analyze VLM prompting test results and generate comprehensive reports.

This script analyzes the output of test_comprehensive_vlm_prompting.py and produces:
1. A spreadsheet with metrics for each method and 95% confidence intervals
2. Comparative bar charts grouping tests by various factors
3. Summary statistics and insights

The script abstracts utility functions from parse_openml_cc18_results.py for statistical
analysis and from analyze_cc18_results_wandb_tabular.py for visualization.

Usage:
    python scripts/analyze_vlm_test_results.py --input_dir results/test_vlm_outputs_31 --output_dir analysis/vlm_analysis
"""

import argparse
import json
import logging
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


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
    if np.isnan(x) or np.isinf(x):
        return x
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values using t-distribution."""
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


def calculate_metrics_from_predictions(ground_truth: List[int], predictions: List[int]) -> Dict[str, float]:
    """Calculate comprehensive metrics from ground truth and predictions."""
    if not ground_truth or not predictions or len(ground_truth) != len(predictions):
        return {}
    
    try:
        # Calculate various metrics
        accuracy = accuracy_score(ground_truth, predictions)
        balanced_acc = balanced_accuracy_score(ground_truth, predictions)
        
        # Calculate F1 scores (handle different averaging methods)
        f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1_micro = f1_score(ground_truth, predictions, average='micro', zero_division=0)
        
        # Calculate precision and recall
        precision_macro = precision_score(ground_truth, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall_macro = recall_score(ground_truth, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error calculating metrics: {e}")
        return {}


def parse_vlm_response(response_text: str, num_classes: int = 2) -> List[int]:
    """Parse VLM response to extract predictions."""
    if not response_text:
        return []
    
    # This is a simplified parser - you may need to adapt based on your VLM response format
    # Look for patterns like "Prediction: [0, 1, 1, 0, ...]" or similar
    predictions = []
    
    try:
        # Try to find prediction patterns
        import re
        
        # Look for arrays/lists of numbers
        array_pattern = r'\[([0-9,\s]+)\]'
        matches = re.findall(array_pattern, response_text)
        
        if matches:
            # Parse the first match
            pred_str = matches[0]
            predictions = [int(x.strip()) for x in pred_str.split(',') if x.strip().isdigit()]
        else:
            # Look for individual predictions
            number_pattern = r'\b[01]\b'  # For binary classification
            numbers = re.findall(number_pattern, response_text)
            predictions = [int(x) for x in numbers]
        
        # Validate predictions are in valid range
        predictions = [p for p in predictions if 0 <= p < num_classes]
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error parsing VLM response: {e}")
        predictions = []
    
    return predictions


def load_test_results(input_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load test results from the input directory."""
    logger = logging.getLogger(__name__)
    
    # Load summary data
    summary_path = input_dir / "test_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load detailed results
    detailed_path = input_dir / "detailed_results.json"
    if not detailed_path.exists():
        raise FileNotFoundError(f"Detailed results file not found: {detailed_path}")
    
    with open(detailed_path, 'r') as f:
        detailed_results = json.load(f)
    
    logger.info(f"Loaded {len(detailed_results)} test results from {input_dir}")
    
    return summary, detailed_results


def extract_config_features(config_path: Path) -> Dict[str, Any]:
    """Extract features from a config file for categorization."""
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            'enable_multi_viz': config.get('enable_multi_viz', False),
            'use_semantic_names': config.get('use_semantic_names', False),
            'load_semantic_from_cc18': config.get('load_semantic_from_cc18', False),
            'use_3d_tsne': config.get('use_3d_tsne', False),
            'use_3d': config.get('use_3d', False),
            'use_knn_connections': config.get('use_knn_connections', False),
            'tsne_perplexity': config.get('tsne_perplexity', 30),
            'nn_k': config.get('nn_k', 5),
            'enable_semantic_axes': config.get('enable_semantic_axes', False),
            'enable_perturbation_analysis': config.get('enable_perturbation_analysis', False),
            'viz_methods': config.get('viz_methods', []),
            'layout_strategy': config.get('layout_strategy', 'adaptive')
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error loading config {config_path}: {e}")
        return {}


def analyze_test_results(input_dir: Path) -> pd.DataFrame:
    """Analyze test results and return a DataFrame with metrics and features."""
    logger = logging.getLogger(__name__)
    summary, detailed_results = load_test_results(input_dir)
    
    results_data = []
    
    for result in detailed_results:
        if not result.get('success', False):
            logger.warning(f"Skipping failed test: {result.get('config_name', 'unknown')}")
            continue
        
        # Load config features
        config_path = input_dir / result.get('config_path', '')
        config_features = extract_config_features(config_path)
        
        # Load response if available
        response_path = input_dir / result.get('response_path', '')
        predictions = []
        if response_path.exists():
            try:
                with open(response_path, 'r') as f:
                    response_text = f.read()
                predictions = parse_vlm_response(response_text, summary['dataset_info']['n_classes'])
            except Exception as e:
                logger.warning(f"Error loading response from {response_path}: {e}")
        
        # Calculate metrics if we have predictions and ground truth
        metrics = {}
        ground_truth = result.get('ground_truth_labels', [])
        if predictions and ground_truth:
            # Truncate to same length
            min_len = min(len(predictions), len(ground_truth))
            if min_len > 0:
                metrics = calculate_metrics_from_predictions(
                    ground_truth[:min_len], 
                    predictions[:min_len]
                )
        
        # Build result record
        record = {
            'test_idx': result.get('test_idx'),
            'config_name': result.get('config_name'),
            'success': result.get('success', False),
            'num_test_samples': result.get('num_test_samples', 0),
            'is_multi_viz': result.get('is_multi_viz', False),
            'visualization_methods': ','.join(result.get('visualization_methods', [])),
            'avg_prompt_length': result.get('avg_prompt_length', 0),
            
            # Config features
            **config_features,
            
            # Metrics
            **metrics,
            
            # Additional derived features
            'has_semantic': config_features.get('use_semantic_names', False) or config_features.get('load_semantic_from_cc18', False),
            'is_3d': config_features.get('use_3d_tsne', False) or config_features.get('use_3d', False),
            'has_metadata': config_features.get('load_semantic_from_cc18', False),
            'has_knn': config_features.get('use_knn_connections', False),
            'num_viz_methods': len(result.get('visualization_methods', [])),
            'primary_viz_method': result.get('visualization_methods', ['unknown'])[0] if result.get('visualization_methods') else 'unknown'
        }
        
        results_data.append(record)
    
    df = pd.DataFrame(results_data)
    logger.info(f"Analyzed {len(df)} successful test results")
    
    return df


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics with confidence intervals."""
    logger = logging.getLogger(__name__)
    
    # Define metrics to analyze
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'f1_micro',
               'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted']
    
    # Define grouping factors
    grouping_factors = [
        'config_name',
        'is_multi_viz',
        'has_semantic',
        'is_3d', 
        'has_metadata',
        'has_knn',
        'primary_viz_method'
    ]
    
    summary_data = []
    
    for factor in grouping_factors:
        if factor not in df.columns:
            continue
            
        for group_value in df[factor].unique():
            if pd.isna(group_value):
                continue
                
            group_df = df[df[factor] == group_value]
            
            for metric in metrics:
                if metric not in df.columns:
                    continue
                    
                values = group_df[metric].dropna().tolist()
                if not values:
                    continue
                
                mean_val = np.mean(values)
                ci_lower, ci_upper = calculate_confidence_interval(values)
                
                summary_data.append({
                    'grouping_factor': factor,
                    'group_value': group_value,
                    'metric': metric,
                    'count': len(values),
                    'mean': round_to_n_significant_digits(mean_val),
                    'std': round_to_n_significant_digits(np.std(values)),
                    'ci_lower': round_to_n_significant_digits(ci_lower),
                    'ci_upper': round_to_n_significant_digits(ci_upper),
                    'min': round_to_n_significant_digits(np.min(values)),
                    'max': round_to_n_significant_digits(np.max(values))
                })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Generated summary statistics for {len(summary_df)} metric groups")
    
    return summary_df


def create_comparison_charts(df: pd.DataFrame, output_dir: Path):
    """Create comparative bar charts grouping tests by various factors."""
    logger = logging.getLogger(__name__)
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 10
    })
    
    # Define metrics to plot
    key_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    
    # Define comparison factors
    comparison_factors = [
        ('is_multi_viz', 'Single vs Multi-Visualization'),
        ('has_semantic', 'Semantic Names Present/Absent'),
        ('is_3d', '2D vs 3D Visualization'),
        ('has_metadata', 'Metadata Present/Absent'),
        ('has_knn', 'KNN Connections Present/Absent'),
        ('primary_viz_method', 'Primary Visualization Method')
    ]
    
    for factor, title in comparison_factors:
        if factor not in df.columns:
            continue
            
        # Filter out missing values
        plot_df = df[df[factor].notna()].copy()
        if plot_df.empty:
            continue
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if metric not in plot_df.columns:
                continue
                
            ax = axes[i]
            
            # Create bar plot
            sns.barplot(data=plot_df, x=factor, y=metric, ax=ax, 
                       ci=95, capsize=0.1, errwidth=2)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(factor.replace('_', ' ').title())
            ax.set_ylabel('Score')
            
            # Rotate x-axis labels if needed
            if factor == 'primary_viz_method':
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Performance Comparison: {title}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        safe_factor_name = factor.replace('_', '-')
        output_path = output_dir / f"comparison_{safe_factor_name}.png"
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison chart: {output_path}")
        except Exception as e:
            logger.error(f"Error saving chart {output_path}: {e}")
        finally:
            plt.close()
    
    # Create overall performance summary chart
    plt.figure(figsize=(12, 8))
    
    # Calculate mean performance for each config
    config_summary = df.groupby('config_name')[key_metrics].mean().round(3)
    
    # Create grouped bar chart
    config_summary.plot(kind='bar', figsize=(15, 8))
    plt.title('Performance by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "performance_by_config.png"
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved performance summary chart: {output_path}")
    except Exception as e:
        logger.error(f"Error saving chart {output_path}: {e}")
    finally:
        plt.close()


def save_results(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """Save results to Excel and CSV files."""
    logger = logging.getLogger(__name__)
    
    # Save detailed results
    detailed_csv = output_dir / "vlm_detailed_results.csv"
    detailed_excel = output_dir / "vlm_detailed_results.xlsx"
    
    try:
        df.to_csv(detailed_csv, index=False)
        df.to_excel(detailed_excel, index=False)
        logger.info(f"Saved detailed results to {detailed_csv} and {detailed_excel}")
    except Exception as e:
        logger.error(f"Error saving detailed results: {e}")
    
    # Save summary statistics
    summary_csv = output_dir / "vlm_summary_statistics.csv"
    summary_excel = output_dir / "vlm_summary_statistics.xlsx"
    
    try:
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_excel(summary_excel, index=False)
        logger.info(f"Saved summary statistics to {summary_csv} and {summary_excel}")
    except Exception as e:
        logger.error(f"Error saving summary statistics: {e}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze VLM test results")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing VLM test results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Starting VLM test results analysis")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Analyze test results
        df = analyze_test_results(input_dir)
        
        if df.empty:
            logger.error("No valid test results found")
            return 1
        
        # Generate summary statistics
        summary_df = generate_summary_statistics(df)
        
        # Create comparison charts
        create_comparison_charts(df, output_dir)
        
        # Save results
        save_results(df, summary_df, output_dir)
        
        # Print summary to console
        logger.info("\n=== VLM TEST ANALYSIS SUMMARY ===")
        logger.info(f"Total tests analyzed: {len(df)}")
        logger.info(f"Success rate: {df['success'].mean():.1%}")
        
        if 'accuracy' in df.columns:
            accuracy_mean = df['accuracy'].mean()
            accuracy_std = df['accuracy'].std()
            logger.info(f"Overall accuracy: {accuracy_mean:.3f} ± {accuracy_std:.3f}")
        
        # Show best performing configurations
        if 'accuracy' in df.columns:
            best_configs = df.groupby('config_name')['accuracy'].mean().sort_values(ascending=False).head(5)
            logger.info("\nTop 5 configurations by accuracy:")
            for config, acc in best_configs.items():
                logger.info(f"  {config}: {acc:.3f}")
        
        logger.info(f"\nAnalysis complete. Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())