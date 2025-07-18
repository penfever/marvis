#!/usr/bin/env python
"""
Time Series Evaluation Script for MARVIS.

This script evaluates MARVIS on time series forecasting using the novel 
distribution classification approach. It:

1. Loads time series datasets from gift-eval benchmark
2. Fits multiple Student's T distributions to training data keypoints
3. Creates visualizations showing forecast paths for each distribution
4. Uses VLM to classify which distribution best matches the pattern
5. Samples from selected distribution for predictions
6. Evaluates against gift-eval metrics (MSE, MASE, etc.)

The approach frames time series forecasting as classification over forecast paths,
leveraging MARVIS's existing VLM reasoning capabilities.

Usage examples:
    # Basic usage on M4 weekly dataset
    python evaluate_time_series.py --datasets m4_weekly --vlm_model_id Qwen/Qwen2.5-VL-3B-Instruct --output_dir ./ts_results
    
    # Multiple datasets with different terms
    python evaluate_time_series.py --datasets m4_weekly,electricity/H --terms short,medium --output_dir ./ts_results
    
    # Custom number of distributions and forecast horizon
    python evaluate_time_series.py --datasets m4_weekly --n_distributions 8 --forecast_horizon 24 --output_dir ./ts_results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

# Add marvis to path if needed
current_dir = Path(__file__).parent
marvis_root = current_dir.parent.parent
sys.path.insert(0, str(marvis_root))

from marvis.data import (
    TimeSeriesDataset,
    load_gift_eval_dataset,
    load_multiple_gift_eval_datasets,
    create_time_series_train_test_split,
    prepare_time_series_for_visualization,
    validate_gift_eval_environment,
    get_available_time_series_datasets
)
from marvis.viz import (
    TimeSeriesDistributionVisualization,
    VisualizationConfig,
    ContextComposer
)
from marvis.utils import setup_logging, set_seed
from marvis.utils.model_loader import ModelLoader

# Try to import gift-eval metrics
try:
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss
    )
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    print("Warning: GluonTS not available. Some metrics will not be computed.")

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MARVIS on time series forecasting using distribution classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--datasets',
        type=str,
        required=True,
        help='Comma-separated list of dataset names (e.g., "m4_weekly,electricity/H")'
    )
    parser.add_argument(
        '--terms',
        type=str,
        default='short',
        help='Comma-separated list of terms for each dataset (e.g., "short,medium")'
    )
    parser.add_argument(
        '--gift_eval_path',
        type=str,
        default=None,
        help='Path to gift-eval installation (if not in PYTHONPATH)'
    )
    
    # Model arguments
    parser.add_argument(
        '--vlm_model_id',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help='Vision Language Model to use for classification'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['transformers', 'vllm', 'auto'],
        default='auto',
        help='Backend for model loading'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device for model (auto, cuda, cpu, mps)'
    )
    
    # Time series parameters
    parser.add_argument(
        '--n_distributions',
        type=int,
        default=5,
        help='Number of Student\'s T distributions to fit'
    )
    parser.add_argument(
        '--forecast_horizon',
        type=int,
        default=None,
        help='Override forecast horizon (uses dataset default if None)'
    )
    parser.add_argument(
        '--n_keypoints',
        type=int,
        default=8,
        help='Number of keypoints for distribution fitting'
    )
    parser.add_argument(
        '--keypoint_strategy',
        type=str,
        choices=['uniform', 'extrema', 'changepoints'],
        default='uniform',
        help='Strategy for selecting keypoints'
    )
    parser.add_argument(
        '--show_confidence_bands',
        action='store_true',
        help='Show confidence bands in visualizations'
    )
    
    # Visualization parameters
    parser.add_argument(
        '--figsize',
        type=str,
        default='16,10',
        help='Figure size as "width,height"'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='Figure DPI'
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save visualization images'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--max_series_per_dataset',
        type=int,
        default=10,
        help='Maximum number of series to evaluate per dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data to use for training vs prediction'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for evaluation."""
    # Set random seed
    set_seed(args.seed, deterministic=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"time_series_evaluation_{timestamp}.log"
    logger = setup_logging(
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=os.path.join(args.output_dir, log_filename)
    )
    
    logger.info(f"Arguments: {args}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate gift-eval environment
    if not validate_gift_eval_environment():
        if args.gift_eval_path:
            logger.info(f"Using gift-eval path: {args.gift_eval_path}")
        else:
            logger.warning("gift-eval environment validation failed")
    
    return logger, timestamp


def load_datasets(args) -> List[TimeSeriesDataset]:
    """Load the specified time series datasets."""
    dataset_names = [name.strip() for name in args.datasets.split(',')]
    terms = [term.strip() for term in args.terms.split(',')]
    
    # Ensure we have terms for all datasets
    if len(terms) == 1 and len(dataset_names) > 1:
        terms = terms * len(dataset_names)
    elif len(terms) != len(dataset_names):
        raise ValueError(f"Number of terms ({len(terms)}) must match number of datasets ({len(dataset_names)})")
    
    logger.info(f"Loading {len(dataset_names)} datasets: {dataset_names}")
    
    datasets = load_multiple_gift_eval_datasets(
        dataset_names=dataset_names,
        terms=terms,
        gift_eval_path=args.gift_eval_path,
        to_univariate=True
    )
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def create_time_series_visualization(args) -> TimeSeriesDistributionVisualization:
    """Create and configure the time series visualization."""
    # Parse figure size
    figsize = tuple(map(int, args.figsize.split(',')))
    
    # Create visualization config
    viz_config = VisualizationConfig(
        figsize=figsize,
        dpi=args.dpi,
        random_state=args.seed,
        task_type='regression',  # Time series forecasting is regression
        extra_params={
            'n_distributions': args.n_distributions,
            'forecast_horizon': args.forecast_horizon,
            'n_keypoints': args.n_keypoints,
            'keypoint_strategy': args.keypoint_strategy,
            'show_confidence_bands': args.show_confidence_bands
        }
    )
    
    # Create visualization
    visualization = TimeSeriesDistributionVisualization(config=viz_config)
    return visualization


def evaluate_time_series(
    dataset: TimeSeriesDataset,
    visualization: TimeSeriesDistributionVisualization,
    vlm_model,
    args
) -> Dict[str, Any]:
    """
    Evaluate MARVIS on a single time series dataset.
    
    Args:
        dataset: Time series dataset to evaluate
        visualization: Configured visualization
        vlm_model: Loaded VLM model
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating dataset: {dataset.name} (term: {dataset.term})")
    
    # Get pre-split training and test data from gift-eval
    train_data_raw = dataset.get_train_data()
    test_data_raw = dataset.get_test_data()
    
    if not train_data_raw:
        logger.warning(f"No training data found for {dataset.name}")
        return {'error': 'No training data'}
    
    if not test_data_raw:
        logger.warning(f"No test data found for {dataset.name}")
        return {'error': 'No test data'}
    
    logger.info(f"Dataset info: {len(train_data_raw)} training series, {len(test_data_raw)} test instances")
    
    # Evaluate on multiple series (up to max_series_per_dataset)
    n_series_to_evaluate = min(args.max_series_per_dataset, len(train_data_raw))
    series_results = []
    
    for series_idx in range(n_series_to_evaluate):
        logger.info(f"Evaluating series {series_idx + 1}/{n_series_to_evaluate}")
        
        try:
            # Get training series for visualization
            train_entry = train_data_raw[series_idx]
            train_series = train_entry['target']
            
            # Debug: Check the data structure
            logger.debug(f"train_entry keys: {list(train_entry.keys())}")
            logger.debug(f"train_series type: {type(train_series)}, shape: {getattr(train_series, 'shape', 'no shape')}")
            
            # Handle different data structures
            if hasattr(train_series, 'shape') and train_series.shape == ():
                # Scalar value - this might be a single point, skip it
                logger.warning(f"Series {series_idx} has scalar target, skipping")
                continue
            elif not hasattr(train_series, 'shape') and np.isscalar(train_series):
                # Single scalar value
                logger.warning(f"Series {series_idx} has single scalar target, skipping")
                continue
            
            # Ensure train_series is numpy array with at least 1D
            if not isinstance(train_series, np.ndarray):
                train_series = np.array(train_series)
            
            if train_series.ndim == 0:
                logger.warning(f"Series {series_idx} has 0-dimensional data, skipping")
                continue
            elif train_series.ndim > 1:
                # Flatten multi-dimensional arrays
                train_series = train_series.flatten()
            
            logger.debug(f"Processed train_series shape: {train_series.shape}")
            
            # Check if series has enough data points
            if len(train_series) < 10:  # Minimum reasonable length
                logger.warning(f"Series {series_idx} too short ({len(train_series)} points), skipping")
                continue
            
            # Find corresponding test entry (gift-eval may have multiple test windows per series)
            test_entry = None
            for test_item in test_data_raw:
                if test_item.get('item_id') == train_entry.get('item_id') or series_idx < len(test_data_raw):
                    test_entry = test_data_raw[series_idx] if series_idx < len(test_data_raw) else test_data_raw[0]
                    break
            
            if test_entry is None:
                logger.warning(f"No test data found for series {series_idx}")
                continue
                
            # Override forecast horizon if specified
            forecast_horizon = args.forecast_horizon or dataset.prediction_length
            visualization.config.extra_params['forecast_horizon'] = forecast_horizon
            
            # Fit distributions and create visualization using the training data
            transformed_data = visualization.fit_transform(train_series)
            viz_result = visualization.generate_plot(transformed_data)
            
            # Save visualization if requested
            if args.save_visualizations:
                viz_path = os.path.join(
                    args.output_dir,
                    f"{dataset.name}_{dataset.term}_series_{series_idx}_visualization.png"
                )
                viz_result.image.save(viz_path)
                logger.info(f"Saved visualization to {viz_path}")
            
            # Generate VLM prompt for classification
            prompt = generate_time_series_classification_prompt(viz_result)
            
            # Get VLM prediction (class selection)
            predicted_class = get_vlm_classification(
                vlm_model, viz_result.image, prompt, args
            )
            
            # Sample from selected distribution
            if predicted_class is not None and 0 <= predicted_class < len(visualization._distributions):
                predictions = visualization.predict_from_class(
                    predicted_class, random_state=args.seed + series_idx
                )
                
                # Get ground truth from test entry
                test_targets = test_entry.get('target', [])
                if not isinstance(test_targets, np.ndarray):
                    test_targets = np.array(test_targets)
                
                # Compute metrics
                metrics = compute_time_series_metrics(
                    predictions, test_targets, train_series
                )
                
                series_result = {
                    'series_index': series_idx,
                    'predicted_class': predicted_class,
                    'selected_distribution': visualization._distributions[predicted_class].name,
                    'predictions': predictions.tolist(),
                    'ground_truth': test_targets.tolist(),
                    'metrics': metrics,
                    'train_data_shape': train_series.shape,
                    'test_data_shape': test_targets.shape,
                    'visualization_metadata': viz_result.metadata
                }
                
                logger.info(f"Series {series_idx}: Class {predicted_class}, MSE {metrics.get('mse', 'N/A'):.4f}")
                
            else:
                logger.warning(f"Invalid class prediction for series {series_idx}: {predicted_class}")
                series_result = {
                    'series_index': series_idx,
                    'error': f'Invalid class prediction: {predicted_class}',
                    'train_data_shape': train_series.shape
                }
            
            series_results.append(series_result)
            
        except Exception as e:
            logger.error(f"Error evaluating series {series_idx}: {e}")
            series_results.append({
                'series_index': series_idx,
                'error': str(e)
            })
    
    # Aggregate results
    valid_results = [r for r in series_results if 'metrics' in r]
    
    if valid_results:
        # Compute average metrics
        avg_metrics = {}
        for metric_name in valid_results[0]['metrics'].keys():
            values = [r['metrics'][metric_name] for r in valid_results if metric_name in r['metrics']]
            if values:
                avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                avg_metrics[f'std_{metric_name}'] = np.std(values)
        
        dataset_result = {
            'dataset_name': dataset.name,
            'dataset_term': dataset.term,
            'dataset_info': dataset.dataset_info,
            'n_series_evaluated': len(valid_results),
            'series_results': series_results,
            'average_metrics': avg_metrics
        }
        
        logger.info(f"Dataset {dataset.name}: Evaluated {len(valid_results)} series, "
                   f"avg MSE {avg_metrics.get('avg_mse', 'N/A'):.4f}")
    else:
        dataset_result = {
            'dataset_name': dataset.name,
            'dataset_term': dataset.term,
            'dataset_info': dataset.dataset_info,
            'n_series_evaluated': 0,
            'series_results': series_results,
            'error': 'No valid series results'
        }
        logger.warning(f"Dataset {dataset.name}: No valid results")
    
    return dataset_result


def generate_time_series_classification_prompt(viz_result) -> str:
    """Generate prompt for VLM to classify time series distributions."""
    from marvis.utils.vlm_prompting import create_time_series_classification_prompt
    
    class_names = viz_result.metadata.get('class_names', [])
    forecast_horizon = viz_result.metadata.get('forecast_horizon', 12)
    distribution_params = viz_result.metadata.get('distribution_params', [])
    legend_text = viz_result.legend_text
    
    return create_time_series_classification_prompt(
        class_names=class_names,
        forecast_horizon=forecast_horizon,
        distribution_params=distribution_params,
        legend_text=legend_text
    )


def get_vlm_classification(model, image, prompt, args) -> Optional[int]:
    """Get classification from VLM model."""
    try:
        # This is a simplified VLM interface - you might need to adapt based on your VLM setup
        from marvis.utils.vlm_prompting import extract_time_series_classification_response
        from marvis.utils.vlm_prompting import get_vlm_response
        
        response = get_vlm_response(
            model=model,
            image=image,
            prompt=prompt,
            max_new_tokens=200  # Increased to allow for analysis
        )
        
        logger.debug(f"VLM response: {response}")
        
        # Extract class number from response using specialized parser
        class_num = extract_time_series_classification_response(response)
        
        if class_num is not None:
            return class_num
        else:
            logger.warning(f"Could not extract class number from VLM response: {response}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting VLM classification: {e}")
        return None


def compute_time_series_metrics(predictions, ground_truth, full_series) -> Dict[str, float]:
    """Compute time series forecasting metrics."""
    metrics = {}
    
    # Ensure arrays
    pred = np.array(predictions)
    true = np.array(ground_truth)
    
    # Basic metrics
    metrics['mse'] = float(np.mean((pred - true) ** 2))
    metrics['mae'] = float(np.mean(np.abs(pred - true)))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # MAPE (avoid division by zero)
    mask = true != 0
    if np.any(mask):
        metrics['mape'] = float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)
    else:
        metrics['mape'] = float('inf')
    
    # SMAPE
    denominator = (np.abs(true) + np.abs(pred)) / 2
    mask = denominator != 0
    if np.any(mask):
        metrics['smape'] = float(np.mean(np.abs(true[mask] - pred[mask]) / denominator[mask]) * 100)
    else:
        metrics['smape'] = float('inf')
    
    # MASE (Mean Absolute Scaled Error) if we have enough history
    if len(full_series) > len(true):
        naive_errors = np.abs(np.diff(full_series[:-len(true)]))
        if len(naive_errors) > 0:
            mae_naive = np.mean(naive_errors)
            if mae_naive > 0:
                metrics['mase'] = float(metrics['mae'] / mae_naive)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up environment
    logger, timestamp = setup_environment(args)
    
    # Load datasets
    try:
        datasets = load_datasets(args)
        if not datasets:
            raise ValueError("No datasets loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return 1
    
    # Load VLM model
    try:
        logger.info(f"Loading VLM model: {args.vlm_model_id}")
        model_loader = ModelLoader()
        vlm_model = model_loader.load_vlm(
            model_name=args.vlm_model_id,
            backend=args.backend,
            device=args.device
        )
        logger.info("VLM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load VLM model: {e}")
        return 1
    
    # Create visualization
    visualization = create_time_series_visualization(args)
    
    # Evaluate each dataset
    all_results = []
    for dataset in datasets:
        try:
            result = evaluate_time_series(
                dataset, visualization, vlm_model, args
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset.name}: {e}")
            all_results.append({
                'dataset_name': dataset.name,
                'dataset_term': dataset.term,
                'error': str(e)
            })
    
    # Save results
    results_file = os.path.join(args.output_dir, f"time_series_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create summary
    summary = {
        'experiment_name': args.experiment_name or f"time_series_eval_{timestamp}",
        'timestamp': timestamp,
        'args': vars(args),
        'n_datasets': len(datasets),
        'dataset_summaries': []
    }
    
    for result in all_results:
        if 'average_metrics' in result:
            summary['dataset_summaries'].append({
                'dataset': f"{result['dataset_name']}_{result['dataset_term']}",
                'n_series': result['n_series_evaluated'],
                'avg_mse': result['average_metrics'].get('avg_mse'),
                'avg_mae': result['average_metrics'].get('avg_mae'),
                'avg_mape': result['average_metrics'].get('avg_mape')
            })
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"time_series_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Print results
    print("\n" + "="*60)
    print("TIME SERIES EVALUATION RESULTS")
    print("="*60)
    
    for result in all_results:
        if 'average_metrics' in result:
            print(f"\nDataset: {result['dataset_name']} ({result['dataset_term']})")
            print(f"Series evaluated: {result['n_series_evaluated']}")
            avg_metrics = result['average_metrics']
            print(f"Average MSE: {avg_metrics.get('avg_mse', 'N/A'):.4f}")
            print(f"Average MAE: {avg_metrics.get('avg_mae', 'N/A'):.4f}")
            print(f"Average MAPE: {avg_metrics.get('avg_mape', 'N/A'):.2f}%")
        else:
            print(f"\nDataset: {result['dataset_name']} ({result['dataset_term']})")
            print(f"Status: Failed - {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    logger.info("Evaluation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())