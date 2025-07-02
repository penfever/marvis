#!/usr/bin/env python3
"""
Analyze ablation experiment results and create comprehensive comparison spreadsheet.

This script loads experimental results from all ablation test directories
(except frequent_patterns) and creates a comprehensive analysis spreadsheet.

Usage:
    python scripts/analyze_ablation_experiments.py
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
ABLATIONS_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"
OUTPUT_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"
EXCLUDE_DIRS = {"frequent_patterns", "_METADATA"}

def load_experiment_result(test_dir: str, config_name: str) -> Optional[Dict]:
    """Load experiment result from detailed_vlm_outputs.json file."""
    config_path = Path(test_dir) / config_name / "detailed_vlm_outputs.json"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {config_path}: {e}")
        return None

def calculate_accuracy(ground_truth: List[int], predictions: List[int]) -> float:
    """Calculate accuracy from ground truth and predictions."""
    if len(ground_truth) != len(predictions):
        return 0.0
    
    correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    return correct / len(ground_truth)

def extract_predictions_from_test_results(test_results: List[Dict]) -> Tuple[List[int], List[int]]:
    """Extract ground truth and predictions from test_results."""
    ground_truth = []
    predictions = []
    
    for result in test_results:
        try:
            # Extract true label
            true_label = result.get('true_label')
            if isinstance(true_label, str):
                true_label = int(true_label)
            ground_truth.append(true_label)
            
            # Extract prediction
            parsed_pred = result.get('parsed_prediction')
            if isinstance(parsed_pred, str):
                parsed_pred = int(parsed_pred)
            elif parsed_pred is None:
                parsed_pred = -1  # Invalid prediction
            predictions.append(parsed_pred)
            
        except (ValueError, TypeError):
            # Handle invalid data
            ground_truth.append(-1)
            predictions.append(-1)
    
    return ground_truth, predictions

def extract_predictions_from_responses(responses: List[str]) -> List[int]:
    """Extract predicted class labels from VLM responses (fallback method)."""
    predictions = []
    
    for response in responses:
        if response is None:
            predictions.append(-1)  # Invalid prediction
            continue
            
        response_lower = response.lower().strip()
        
        # Look for class predictions in various formats
        if 'class 0' in response_lower or 'class: 0' in response_lower or response_lower.startswith('0'):
            predictions.append(0)
        elif 'class 1' in response_lower or 'class: 1' in response_lower or response_lower.startswith('1'):
            predictions.append(1)
        elif 'class 2' in response_lower or 'class: 2' in response_lower or response_lower.startswith('2'):
            predictions.append(2)
        else:
            # Try to extract first number
            import re
            numbers = re.findall(r'\b([0-2])\b', response_lower)
            if numbers:
                predictions.append(int(numbers[0]))
            else:
                predictions.append(-1)  # Invalid prediction
    
    return predictions

def analyze_single_experiment(data: Dict, test_id: str, config_name: str) -> Dict:
    """Analyze a single experiment and return metrics."""
    result = {
        'test_id': test_id,
        'config_name': config_name,
        'test_config': data.get('test_config', {}),
        'num_samples': 0,
        'accuracy': 0.0,
        'valid_predictions': 0,
        'invalid_predictions': 0,
        'class_distribution': {},
        'confusion_matrix': None,
        'success': False
    }
    
    try:
        # Check if metrics are already computed
        metrics = data.get('metrics', {})
        if metrics and 'accuracy' in metrics:
            # Use pre-computed metrics
            result['accuracy'] = metrics.get('accuracy', 0.0)
            result['num_samples'] = metrics.get('num_test_samples', 0)
            result['completion_rate'] = metrics.get('completion_rate', 0.0)
            result['balanced_accuracy'] = metrics.get('balanced_accuracy', 0.0)
            result['f1_macro'] = metrics.get('f1_macro', 0.0)
            result['precision_macro'] = metrics.get('precision_macro', 0.0)
            result['recall_macro'] = metrics.get('recall_macro', 0.0)
            result['confusion_matrix'] = metrics.get('confusion_matrix', None)
            result['success'] = result['completion_rate'] > 0 and result['num_samples'] > 0
            result['valid_predictions'] = result['num_samples']
            result['invalid_predictions'] = 0
            
            # Extract class distribution from ground truth
            ground_truth = data.get('ground_truth', [])
            if ground_truth:
                unique_classes = set(ground_truth)
                for cls in unique_classes:
                    result['class_distribution'][f'class_{cls}_count'] = ground_truth.count(cls)
        else:
            # Fallback to manual calculation
            # Try prediction_details format
            prediction_details = data.get('prediction_details', [])
            if prediction_details:
                ground_truth, predictions = extract_predictions_from_test_results(prediction_details)
            else:
                # Try other formats
                ground_truth = data.get('ground_truth', [])
                responses = data.get('responses', [])
                if responses:
                    predictions = extract_predictions_from_responses(responses)
                else:
                    predictions = []
            
            if ground_truth and predictions:
                result['num_samples'] = len(ground_truth)
                
                # Filter out invalid predictions for accuracy calculation
                valid_pairs = [(gt, pred) for gt, pred in zip(ground_truth, predictions) if pred != -1 and gt != -1]
                
                if valid_pairs:
                    valid_gt, valid_pred = zip(*valid_pairs)
                    result['accuracy'] = calculate_accuracy(list(valid_gt), list(valid_pred))
                    result['valid_predictions'] = len(valid_pairs)
                    result['invalid_predictions'] = len(predictions) - len(valid_pairs)
                    result['success'] = True
                    
                    # Class distribution
                    unique_classes = set(gt for gt in ground_truth if gt != -1)
                    for cls in unique_classes:
                        result['class_distribution'][f'class_{cls}_count'] = ground_truth.count(cls)
                    
                    # Confusion matrix (for valid predictions only)
                    if len(set(valid_gt)) > 1 and len(set(valid_pred)) > 1:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(valid_gt, valid_pred)
                        result['confusion_matrix'] = cm.tolist()
        
    except Exception as e:
        print(f"Error analyzing {test_id}/{config_name}: {e}")
    
    return result

def load_all_experiments() -> List[Dict]:
    """Load all experiment results from ablation directories."""
    results = []
    
    # Get all test directories
    test_dirs = [d for d in os.listdir(ABLATIONS_DIR) 
                 if d.startswith('test_vlm_outputs_') and 
                 os.path.isdir(os.path.join(ABLATIONS_DIR, d))]
    
    for test_dir in sorted(test_dirs):
        test_path = os.path.join(ABLATIONS_DIR, test_dir)
        test_id = test_dir.replace('test_vlm_outputs_', '')
        
        print(f"Processing {test_dir}...")
        
        # Get all config directories (except metadata and frequent_patterns)
        config_dirs = [d for d in os.listdir(test_path) 
                      if os.path.isdir(os.path.join(test_path, d)) and 
                      d not in EXCLUDE_DIRS]
        
        for config_name in sorted(config_dirs):
            data = load_experiment_result(test_path, config_name)
            if data:
                result = analyze_single_experiment(data, test_id, config_name)
                results.append(result)
            else:
                print(f"  Skipping {config_name} - no data found")
    
    return results

def create_comprehensive_spreadsheet(results: List[Dict]) -> pd.DataFrame:
    """Create comprehensive spreadsheet from experiment results."""
    rows = []
    
    for result in results:
        row = {
            'test_id': result['test_id'],
            'config_name': result['config_name'],
            'accuracy': result['accuracy'],
            'num_samples': result['num_samples'],
            'valid_predictions': result['valid_predictions'],
            'invalid_predictions': result['invalid_predictions'],
            'success': result['success'],
        }
        
        # Add additional metrics if available
        for metric in ['completion_rate', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            if metric in result:
                row[metric] = result[metric]
        
        # Add test configuration parameters
        test_config = result.get('test_config', {})
        for key, value in test_config.items():
            row[f'config_{key}'] = value
        
        # Add class distribution
        class_dist = result.get('class_distribution', {})
        for key, value in class_dist.items():
            row[key] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Fill NaN values appropriately
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def perform_comparative_analysis(df: pd.DataFrame) -> Dict:
    """Perform comparative analysis on the experimental results."""
    analysis = {}
    
    # Overall statistics
    analysis['overall_stats'] = {
        'total_experiments': len(df),
        'successful_experiments': df['success'].sum(),
        'success_rate': df['success'].mean(),
        'mean_accuracy': df[df['success']]['accuracy'].mean(),
        'std_accuracy': df[df['success']]['accuracy'].std(),
        'max_accuracy': df[df['success']]['accuracy'].max(),
        'min_accuracy': df[df['success']]['accuracy'].min(),
    }
    
    # Performance by configuration
    config_stats = df[df['success']].groupby('config_name').agg({
        'accuracy': ['mean', 'std', 'count', 'min', 'max'],
        'valid_predictions': 'mean',
        'invalid_predictions': 'mean'
    }).round(4)
    
    config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns]
    analysis['config_performance'] = config_stats.to_dict('index')
    
    # Performance by test ID
    test_stats = df[df['success']].groupby('test_id').agg({
        'accuracy': ['mean', 'std', 'count'],
        'success': 'sum'
    }).round(4)
    
    test_stats.columns = ['_'.join(col).strip() for col in test_stats.columns]
    analysis['test_performance'] = test_stats.to_dict('index')
    
    # Top performing configurations
    top_configs = df[df['success']].groupby('config_name')['accuracy'].mean().sort_values(ascending=False).head(10)
    analysis['top_configurations'] = top_configs.to_dict()
    
    # Configuration parameter analysis
    param_analysis = {}
    
    # Analyze boolean parameters
    bool_params = [col for col in df.columns if col.startswith('config_') and df[col].dtype == 'bool']
    for param in bool_params:
        if param in df.columns:
            param_stats = df[df['success']].groupby(param)['accuracy'].agg(['mean', 'std', 'count']).round(4)
            param_analysis[param] = param_stats.to_dict('index')
    
    # Analyze numeric parameters
    numeric_params = [col for col in df.columns if col.startswith('config_') and df[col].dtype in ['int64', 'float64']]
    for param in numeric_params:
        if param in df.columns and df[param].nunique() > 1:
            correlation = df[df['success']]['accuracy'].corr(df[df['success']][param])
            param_analysis[param] = {
                'correlation_with_accuracy': correlation,
                'unique_values': sorted(df[param].unique().tolist()),
                'value_counts': df[param].value_counts().to_dict()
            }
    
    analysis['parameter_analysis'] = param_analysis
    
    return analysis

def create_visualizations(df: pd.DataFrame, analysis: Dict):
    """Create visualization plots for the analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy distribution by configuration
    plt.subplot(2, 3, 1)
    successful_df = df[df['success']]
    if not successful_df.empty:
        config_means = successful_df.groupby('config_name')['accuracy'].mean().sort_values(ascending=True)
        config_means.plot(kind='barh')
        plt.title('Mean Accuracy by Configuration')
        plt.xlabel('Accuracy')
        plt.ylabel('Configuration')
        plt.tight_layout()
    
    # 2. Accuracy distribution histogram
    plt.subplot(2, 3, 2)
    if not successful_df.empty:
        plt.hist(successful_df['accuracy'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Accuracy Scores')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(successful_df['accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {successful_df["accuracy"].mean():.3f}')
        plt.legend()
    
    # 3. Success rate by configuration
    plt.subplot(2, 3, 3)
    success_rates = df.groupby('config_name')['success'].mean().sort_values(ascending=True)
    success_rates.plot(kind='barh')
    plt.title('Success Rate by Configuration')
    plt.xlabel('Success Rate')
    plt.ylabel('Configuration')
    
    # 4. Accuracy by test ID
    plt.subplot(2, 3, 4)
    if not successful_df.empty:
        test_means = successful_df.groupby('test_id')['accuracy'].mean()
        test_means.plot(kind='bar')
        plt.title('Mean Accuracy by Test ID')
        plt.xlabel('Test ID')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
    
    # 5. Invalid predictions by configuration
    plt.subplot(2, 3, 5)
    invalid_means = df.groupby('config_name')['invalid_predictions'].mean().sort_values(ascending=True)
    invalid_means.plot(kind='barh')
    plt.title('Mean Invalid Predictions by Configuration')
    plt.xlabel('Invalid Predictions')
    plt.ylabel('Configuration')
    
    # 6. Accuracy vs Invalid Predictions scatter
    plt.subplot(2, 3, 6)
    if not successful_df.empty:
        plt.scatter(successful_df['invalid_predictions'], successful_df['accuracy'], alpha=0.6)
        plt.title('Accuracy vs Invalid Predictions')
        plt.xlabel('Invalid Predictions')
        plt.ylabel('Accuracy')
        
        # Add correlation coefficient
        corr = successful_df['accuracy'].corr(successful_df['invalid_predictions'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_analysis_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed heatmap of configuration performance
    if not successful_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = successful_df.pivot_table(
            values='accuracy', 
            index='config_name', 
            columns='test_id', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=pivot_data.mean().mean(), ax=ax)
        plt.title('Accuracy Heatmap: Configuration vs Test ID')
        plt.xlabel('Test ID')
        plt.ylabel('Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'configuration_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the analysis."""
    print("Loading ablation experiment results...")
    results = load_all_experiments()
    
    if not results:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create comprehensive spreadsheet
    print("Creating comprehensive spreadsheet...")
    df = create_comprehensive_spreadsheet(results)
    
    # Save the spreadsheet
    output_file = os.path.join(OUTPUT_DIR, "ablation_experiments_comprehensive.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Spreadsheet saved to: {output_file}")
    
    # Perform comparative analysis
    print("Performing comparative analysis...")
    analysis = perform_comparative_analysis(df)
    
    # Save analysis results
    analysis_file = os.path.join(OUTPUT_DIR, "ablation_analysis_results.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis results saved to: {analysis_file}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, analysis)
    print("Visualizations saved to ablation_analysis_plots.png and configuration_heatmap.png")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    overall = analysis['overall_stats']
    print(f"Total experiments: {overall['total_experiments']}")
    print(f"Successful experiments: {overall['successful_experiments']}")
    print(f"Success rate: {overall['success_rate']:.1%}")
    print(f"Mean accuracy: {overall['mean_accuracy']:.3f} ± {overall['std_accuracy']:.3f}")
    print(f"Accuracy range: {overall['min_accuracy']:.3f} - {overall['max_accuracy']:.3f}")
    
    print(f"\nTop 5 configurations by accuracy:")
    for i, (config, acc) in enumerate(list(analysis['top_configurations'].items())[:5], 1):
        print(f"{i}. {config}: {acc:.3f}")
    
    print(f"\nConfiguration performance summary:")
    config_perf = analysis['config_performance']
    for config, stats in list(config_perf.items())[:10]:
        mean_acc = stats.get('accuracy_mean', 0)
        count = stats.get('accuracy_count', 0)
        print(f"  {config}: {mean_acc:.3f} ({count} experiments)")

if __name__ == "__main__":
    main()