#!/usr/bin/env python3
"""
Comprehensive test script for audio classification datasets.

Runs few-shot audio classification on ESC-50, RAVDESS, and UrbanSound8K
using MARVIS t-SNE with configurable audio embeddings (Whisper or CLAP).
Can test individual datasets or run comprehensive comparisons across multiple datasets.
"""

import argparse
import logging
import os
import sys
import time
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import centralized argument parser
from marvis.utils.evaluation_args import create_audio_evaluation_parser

from marvis.utils.json_utils import convert_for_json_serialization, safe_json_dump
from marvis.utils.device_utils import log_platform_info
from marvis.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring,
    MetricsLogger,
    set_seed_with_args
)

from examples.audio.marvis_tsne_audio_baseline import MarvisAudioTsneClassifier
from examples.audio.audio_datasets import ESC50Dataset, RAVDESSDataset, UrbanSound8KDataset
from examples.audio.audio_baselines import WhisperKNNClassifier, CLAPZeroShotClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_results_to_wandb(model_name: str, eval_results: dict, args, class_names: list, dataset_name: str = None):
    """Log evaluation results to Weights & Biases (adapted for audio)."""
    if 'error' in eval_results:
        # Log failed runs
        wandb.log({
            f"{model_name}/status": "failed",
            f"{model_name}/error": eval_results['error'],
            "model_name": model_name,
            "dataset": (dataset_name or getattr(args, 'datasets', ['unknown'])[0]).upper(),
            "k_shot": getattr(args, 'k_shot', 5),
            "embedding_model": getattr(args, 'embedding_model', 'whisper'),
            "whisper_model": getattr(args, 'whisper_model', 'large-v2'),
            "clap_version": getattr(args, 'clap_version', '2023'),
            "audio_duration": getattr(args, 'audio_duration', None)
        })
        return
    
    # Create base metrics
    metrics = {
        f"{model_name}/accuracy": eval_results['accuracy'],
        f"{model_name}/training_time": eval_results.get('training_time', 0),
        f"{model_name}/prediction_time": eval_results.get('prediction_time', 0),
        f"{model_name}/num_test_samples": eval_results.get('num_test_samples', 0),
        "model_name": model_name,
        "dataset": (dataset_name or getattr(args, 'datasets', ['unknown'])[0]).upper(),
        "num_classes": len(class_names),
        "k_shot": getattr(args, 'k_shot', 5),
        "embedding_model": getattr(args, 'embedding_model', 'whisper'),
        "whisper_model": getattr(args, 'whisper_model', 'large-v2'),
        "clap_version": getattr(args, 'clap_version', '2023'),
        "audio_duration": getattr(args, 'audio_duration', None)
    }
    
    # Add MARVIS t-SNE specific metrics
    if model_name == 'marvis_tsne':
        config = eval_results.get('config', {})
        metrics.update({
            f"{model_name}/use_3d": config.get('use_3d', False),
            f"{model_name}/use_knn_connections": config.get('use_knn_connections', False),
            f"{model_name}/nn_k": config.get('nn_k', 0),
            f"{model_name}/use_pca_backend": config.get('use_pca_backend', False),
            f"{model_name}/zoom_factor": config.get('zoom_factor', 1.0),
            f"{model_name}/include_spectrogram": config.get('include_spectrogram', True),
            f"{model_name}/vlm_model": config.get('vlm_model_id', 'unknown'),
            f"{model_name}/embedding_model": config.get('embedding_model', 'whisper'),
            f"{model_name}/embedding_size": config.get('embedding_size', 0),
        })
        
        # Add visualization info if available
        if eval_results.get('visualizations_saved', False):
            metrics[f"{model_name}/visualizations_saved"] = True
            metrics[f"{model_name}/output_directory"] = eval_results.get('output_directory', 'unknown')
    
    # Add additional audio-specific metrics
    if 'detailed_results' in eval_results:
        detailed = eval_results['detailed_results']
        if isinstance(detailed, dict):
            # Add balanced accuracy if available
            if 'balanced_accuracy' in detailed:
                metrics[f"{model_name}/balanced_accuracy"] = detailed['balanced_accuracy']
            # Add F1 scores if available
            if 'f1_macro' in detailed:
                metrics[f"{model_name}/f1_macro"] = detailed['f1_macro']
            if 'f1_weighted' in detailed:
                metrics[f"{model_name}/f1_weighted"] = detailed['f1_weighted']
    
    # Add class names for reference
    metrics["class_names"] = class_names[:10]  # Limit to first 10 for wandb
    metrics["num_classes"] = len(class_names)
    
    wandb.log(metrics)


def load_esc50_dataset(data_dir: str, args) -> tuple:
    """
    Load ESC-50 dataset.
    
    Args:
        data_dir: Directory to store/find the dataset
        args: Arguments containing configuration
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    try:
        dataset = ESC50Dataset(data_dir, download=not args.no_download)
        
        # Create few-shot splits
        splits = dataset.create_few_shot_split(
            k_shot=args.k_shot,
            val_size=0.2,
            test_size=0.3,
            random_state=42
        )
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']  # Currently not used but available
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"ESC-50 loaded: {len(train_paths)} train, {len(test_paths)} test samples")
        logger.info(f"Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return train_paths, train_labels, test_paths, test_labels, class_names
        
    except Exception as e:
        raise ValueError(f"Failed to load ESC-50 dataset: {e}")


def load_ravdess_dataset(data_dir: str, args) -> tuple:
    """
    Load RAVDESS dataset.
    
    Args:
        data_dir: Directory to store/find the dataset
        args: Arguments containing configuration
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    try:
        dataset = RAVDESSDataset(data_dir, download=not args.no_download)
        
        # Create few-shot splits
        splits = dataset.create_few_shot_split(
            k_shot=args.k_shot,
            val_size=0.2,
            test_size=0.3,
            random_state=42
        )
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']  # Currently not used but available
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"RAVDESS loaded: {len(train_paths)} train, {len(test_paths)} test samples")
        logger.info(f"Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return train_paths, train_labels, test_paths, test_labels, class_names
        
    except Exception as e:
        raise ValueError(f"Failed to load RAVDESS dataset: {e}")


def load_urbansound8k_dataset(data_dir: str, args) -> tuple:
    """
    Load UrbanSound8K dataset.
    
    Args:
        data_dir: Directory to store/find the dataset
        args: Arguments containing configuration
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    try:
        dataset = UrbanSound8KDataset(data_dir, download=not args.no_download)
        
        # Create few-shot splits
        splits = dataset.create_few_shot_split(
            k_shot=args.k_shot,
            val_size=0.2,
            test_size=0.3,
            random_state=42
        )
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']  # Currently not used but available
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"UrbanSound8K loaded: {len(train_paths)} train, {len(test_paths)} test samples")
        logger.info(f"Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return train_paths, train_labels, test_paths, test_labels, class_names
        
    except Exception as e:
        raise ValueError(f"Failed to load UrbanSound8K dataset: {e}")


def test_single_dataset(dataset_name: str, args):
    """Test a single dataset (reusing vision script pattern)."""
    logger.info(f"\\n{'='*60}")
    logger.info(f"TESTING {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        # Load dataset based on name
        data_dir = os.path.join(args.data_dir, dataset_name)
        
        if dataset_name == 'esc50':
            train_paths, train_labels, test_paths, test_labels, class_names = load_esc50_dataset(data_dir, args)
        elif dataset_name == 'ravdess':
            train_paths, train_labels, test_paths, test_labels, class_names = load_ravdess_dataset(data_dir, args)
        elif dataset_name == 'urbansound8k':
            train_paths, train_labels, test_paths, test_labels, class_names = load_urbansound8k_dataset(data_dir, args)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"  Train: {len(train_paths)} samples ({args.k_shot} per class)")
        logger.info(f"  Test: {len(test_paths)} samples")
        logger.info(f"  Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return run_models_on_dataset(dataset_name, train_paths, train_labels, test_paths, test_labels, class_names, args)
        
    except Exception as e:
        logger.error(f"{dataset_name} dataset loading failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def run_models_on_dataset(dataset_name: str, train_paths, train_labels, test_paths, test_labels, class_names, args):
    """Run all models on a single dataset (reusing vision script pattern)."""
    
    # Debug: Check models argument
    logger.info(f"Models to evaluate: {getattr(args, 'models', 'MISSING')}")
    if not hasattr(args, 'models') or not args.models:
        logger.error("No models specified for evaluation! Using default models.")
        args.models = ["marvis_tsne"]
    
    # Store wandb availability for logging
    use_wandb_logging = args.use_wandb and WANDB_AVAILABLE
    
    # Apply quick test mode if requested
    if args.quick_test:
        logger.info("Running quick test with subset of data")
        test_paths = test_paths[:min(20, len(test_paths))]
        test_labels = test_labels[:min(20, len(test_labels))]
    
    # Configure audio duration based on dataset
    audio_duration = args.audio_duration
    if dataset_name.lower() == 'ravdess' and audio_duration is None:
        audio_duration = 3.0  # RAVDESS clips are ~3 seconds
    elif audio_duration is None:
        audio_duration = 5.0  # Default for ESC-50 and others
    
    results = {}
    
    # Test MARVIS t-SNE (audio → embeddings → t-SNE/PCA → VLM)
    if 'marvis_tsne' in args.models:
        backend_name = "PCA" if args.use_pca_backend else "t-SNE"
        features = []
        if args.use_3d:
            features.append("3D")
        if args.use_knn_connections:
            features.append(f"KNN-{args.nn_k}")
        feature_str = f" ({', '.join(features)})" if features else ""
        
        # Validate feature combinations (reusing vision script pattern)
        if args.use_knn_connections and args.use_pca_backend:
            logger.warning("KNN connections are only supported with t-SNE, not PCA. KNN will be disabled for PCA backend.")
        
        logger.info(f"Testing MARVIS {backend_name}{feature_str} ({args.embedding_model.upper()} → {backend_name} → VLM)...")
        try:
            classifier = MarvisAudioTsneClassifier(
                embedding_model=args.embedding_model,
                whisper_model=args.whisper_model,
                embedding_layer="encoder_last",
                clap_version=args.clap_version,
                tsne_perplexity=min(30.0, len(train_paths) / 4),
                tsne_max_iter=1000,
                vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                use_3d=args.use_3d,
                use_knn_connections=args.use_knn_connections,
                nn_k=args.nn_k,
                max_vlm_image_size=1024,
                zoom_factor=args.zoom_factor,
                use_pca_backend=args.use_pca_backend,
                include_spectrogram=args.include_spectrogram,
                audio_duration=audio_duration,
                cache_dir=args.cache_dir,
                use_semantic_names=args.use_semantic_names,
                num_few_shot_examples=args.num_few_shot_examples,
                balanced_few_shot=args.balanced_few_shot,
                device='cpu' if sys.platform == "darwin" else args.device,
                seed=42
            )
            
            # Pass save_every_n parameter
            classifier.save_every_n = args.save_every_n
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, test_paths[:5], class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels, 
                return_detailed=True,
                save_outputs=args.save_outputs,
                output_dir=os.path.join(args.output_dir, dataset_name.lower()) if args.save_outputs else None
            )
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            eval_results['dataset_info'] = {
                'name': dataset_name,
                'num_classes': len(class_names),
                'class_names': class_names,
                'train_samples': len(train_paths),
                'test_samples': len(test_paths),
                'k_shot': args.k_shot
            }
            
            results['marvis_tsne'] = eval_results
            logger.info(f"{dataset_name} MARVIS t-SNE completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('marvis_tsne', eval_results, args, class_names, dataset_name)
                
        except Exception as e:
            import traceback
            error_msg = f"MARVIS t-SNE failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['marvis_tsne'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test Whisper KNN baseline
    if 'whisper_knn' in args.models:
        logger.info(f"Testing Whisper KNN baseline...")
        try:
            classifier = WhisperKNNClassifier(
                model_name=args.whisper_model,
                k=5,  # KNN neighbors
                audio_duration=audio_duration,
                use_cuda=False if sys.platform == "darwin" else None
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                return_detailed=True
            )
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            eval_results['dataset_info'] = {
                'name': dataset_name,
                'num_classes': len(class_names),
                'class_names': class_names,
                'train_samples': len(train_paths),
                'test_samples': len(test_paths),
                'k_shot': args.k_shot
            }
            
            results['whisper_knn'] = eval_results
            logger.info(f"{dataset_name} Whisper KNN completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('whisper_knn', eval_results, args, class_names, dataset_name)
                
        except Exception as e:
            import traceback
            error_msg = f"Whisper KNN failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['whisper_knn'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test CLAP zero-shot baseline
    if 'clap_zero_shot' in args.models:
        logger.info(f"Testing CLAP zero-shot baseline...")
        try:
            classifier = CLAPZeroShotClassifier(
                version=args.clap_version,  # "2022", "2023", or "clapcap"
                use_cuda=False if sys.platform == "darwin" else None,  # Auto-detect if not Mac
                batch_size=4  # Smaller batch for CPU
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, class_names)  # Only for class names
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                return_detailed=True
            )
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            eval_results['dataset_info'] = {
                'name': dataset_name,
                'num_classes': len(class_names),
                'class_names': class_names,
                'train_samples': len(train_paths),
                'test_samples': len(test_paths),
                'k_shot': args.k_shot
            }
            
            results['clap_zero_shot'] = eval_results
            logger.info(f"{dataset_name} CLAP zero-shot completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('clap_zero_shot', eval_results, args, class_names, dataset_name)
                
        except Exception as e:
            import traceback
            error_msg = f"CLAP zero-shot failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['clap_zero_shot'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Return results in the expected format for compatibility with save_results
    if results:
        # Check if any models succeeded  
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        if successful_results:
            # Return first successful result as primary, but include all results
            first_success = next(iter(successful_results.values()))
            return {
                'status': 'success',
                'results': first_success,
                'all_results': results  # Include all model results
            }
        else:
            # All models failed
            return {
                'status': 'error', 
                'error': "All models failed",
                'all_results': results
            }
    else:
        return {
            'status': 'error',
            'error': "No models specified for testing"
        }


# Old test_dataset function removed - functionality moved to test_single_dataset and run_models_on_dataset


def run_all_audio_tests(args):
    """Run tests on selected audio datasets (modernized to match vision script pattern)."""
    all_results = {}
    
    # Set seed before any dataset operations
    set_seed_with_args(args)
    
    # Validate available datasets
    available_datasets = ["esc50", "ravdess", "urbansound8k"]
    
    # Select datasets to test based on arguments
    datasets_to_test = []
    for dataset_key in args.datasets:
        if dataset_key in available_datasets:
            datasets_to_test.append(dataset_key)
        else:
            logger.warning(f"Unknown dataset: {dataset_key}. Available: {available_datasets}")
    
    if not datasets_to_test:
        logger.error("No valid datasets specified for testing")
        return {}
    
    logger.info(f"Testing {len(datasets_to_test)} audio datasets: {', '.join(datasets_to_test)}")
    
    # Test each selected dataset using the new structure
    for dataset_name in datasets_to_test:
        try:
            result = test_single_dataset(dataset_name, args)
            all_results[dataset_name.lower()] = result
            
            # Log success/failure for this dataset
            if result.get('status') == 'success':
                logger.info(f"✓ {dataset_name.upper()} completed successfully")
            else:
                logger.error(f"✗ {dataset_name.upper()} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Unexpected error testing {dataset_name}: {e}")
            all_results[dataset_name.lower()] = {
                'status': 'error',
                'error': str(e)
            }
    
    return all_results


# Old log_results_to_wandb function removed - using improved version at top of file


def save_results(results: dict, output_dir: str, k_shot: int):
    """Save comprehensive test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"all_audio_k{k_shot}_test_results.json")
    with open(results_file, 'w') as f:
        json_results = convert_for_json_serialization(results)
        json.dump(json_results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for dataset_name, result in results.items():
        if result['status'] == 'success':
            # Handle multiple models if available
            if 'all_results' in result:
                # Multiple models tested
                for model_name, model_result in result['all_results'].items():
                    if 'error' not in model_result:
                        dataset_info = model_result.get('dataset_info', {})
                        summary_data.append({
                            'dataset': dataset_name.upper(),
                            'model': model_name.upper().replace('_', ' '),
                            'status': 'SUCCESS',
                            'accuracy': model_result['accuracy'],
                            'num_classes': dataset_info.get('num_classes', 'N/A'),
                            'train_samples': dataset_info.get('train_samples', 'N/A'),
                            'test_samples': dataset_info.get('test_samples', 'N/A'),
                            'training_time': model_result.get('training_time', 0),
                            'prediction_time': model_result.get('prediction_time', 0),
                            'error': None
                        })
                    else:
                        summary_data.append({
                            'dataset': dataset_name.upper(),
                            'model': model_name.upper().replace('_', ' '),
                            'status': 'ERROR',
                            'accuracy': None,
                            'num_classes': None,
                            'train_samples': None,
                            'test_samples': None,
                            'training_time': None,
                            'prediction_time': None,
                            'error': model_result['error']
                        })
            else:
                # Single model result (backward compatibility)
                eval_results = result['results']
                dataset_info = eval_results.get('dataset_info', {})
                summary_data.append({
                    'dataset': dataset_name.upper(),
                    'model': 'MARVIS TSNE',  # Default model name
                    'status': 'SUCCESS',
                    'accuracy': eval_results['accuracy'],
                    'num_classes': dataset_info.get('num_classes', 'N/A'),
                    'train_samples': dataset_info.get('train_samples', 'N/A'),
                    'test_samples': dataset_info.get('test_samples', 'N/A'),
                    'training_time': eval_results['training_time'],
                    'prediction_time': eval_results['prediction_time'],
                    'error': None
                })
        else:
            summary_data.append({
                'dataset': dataset_name.upper(),
                'model': 'ALL',
                'status': 'ERROR',
                'accuracy': None,
                'num_classes': None,
                'train_samples': None,
                'test_samples': None,
                'training_time': None,
                'prediction_time': None,
                'error': result['error']
            })
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"all_audio_k{k_shot}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print comprehensive summary
    logger.info("\\n" + "="*80)
    logger.info(f"COMPREHENSIVE AUDIO CLASSIFICATION RESULTS (k={k_shot})")
    logger.info("="*80)
    
    success_count = 0
    total_datasets = len(results)
    
    for _, row in summary_df.iterrows():
        if row['status'] == 'SUCCESS':
            success_count += 1
            logger.info(f"{row['dataset']:12s} {row['model']:15s}: ✓ {row['accuracy']:.4f} accuracy "
                       f"({row['num_classes']} classes, {row['train_samples']} train, {row['test_samples']} test) "
                       f"[Train: {row['training_time']:.1f}s, Test: {row['prediction_time']:.1f}s]")
        else:
            logger.info(f"{row['dataset']:12s} {row['model']:15s}: ✗ ERROR - {row['error']}")
    
    logger.info("\\n" + "-"*80)
    logger.info(f"SUMMARY: {success_count}/{len(summary_data)} experiments successful across {total_datasets} datasets")
    
    if success_count > 0:
        successful_results = summary_df[summary_df['status'] == 'SUCCESS']
        mean_accuracy = successful_results['accuracy'].mean()
        logger.info(f"Mean accuracy across successful experiments: {mean_accuracy:.4f}")
        
        # Show per-model averages if multiple models tested
        if 'model' in summary_df.columns and len(summary_df['model'].unique()) > 1:
            logger.info("\\nPer-model averages:")
            for model in summary_df['model'].unique():
                model_results = successful_results[successful_results['model'] == model]
                if len(model_results) > 0:
                    model_mean = model_results['accuracy'].mean()
                    logger.info(f"  {model:15s}: {model_mean:.4f} (across {len(model_results)} datasets)")
    
    logger.info(f"\\nDetailed results saved to: {output_dir}")
    logger.info("="*80)


def parse_args_old():
    """Legacy argument parser - replaced by centralized parser."""
    parser = argparse.ArgumentParser(description="Test audio datasets with MARVIS t-SNE and baselines")


def parse_args():
    """Parse command line arguments using centralized audio evaluation parser."""
    parser = create_audio_evaluation_parser("Test audio datasets with MARVIS t-SNE and baselines")
    
    # Add audio-specific arguments that aren't in the centralized parser
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./audio_data",
        help="Base directory for all audio datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    
    # Set audio-specific defaults
    parser.set_defaults(
        output_dir="./all_audio_test_results",
        datasets="esc50,ravdess"
    )
    
    args = parser.parse_args()
    
    # Use audio_datasets if provided, otherwise fall back to datasets
    if hasattr(args, 'audio_datasets') and args.audio_datasets:
        args.datasets = args.audio_datasets
    elif hasattr(args, 'datasets') and isinstance(args.datasets, str):
        args.datasets = args.datasets.split(',')
    
    # models is already a list from nargs="+" so no conversion needed
    
    return args


def parse_args_old_implementation():
    """Old implementation preserved for reference - not used."""
    parser = argparse.ArgumentParser(description="Test audio datasets with MARVIS t-SNE and baselines")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./audio_data",
        help="Base directory for all audio datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./all_audio_test_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=5,
        help="Number of training examples per class for few-shot learning (e.g., k_shot=5 means 5 samples per class for training)"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=32,
        help="Number of examples to use for in-context learning in LLM prompts (for MARVIS baseline)"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use class-balanced few-shot examples in LLM prompts instead of random selection"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="whisper",
        choices=["whisper", "clap"],
        help="Audio embedding model to use (whisper or clap)"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
        help="Whisper model size (used if embedding_model is whisper)"
    )
    parser.add_argument(
        "--clap_version",
        type=str,
        default="2023",
        choices=["2022", "2023", "clapcap"],
        help="CLAP model version (used if embedding_model is clap)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with subset of data"
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Skip automatic dataset downloads"
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations"
    )
    parser.add_argument(
        "--use_pca_backend",
        action="store_true",
        help="Use PCA instead of t-SNE for dimensionality reduction"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and VLM responses (default: True)"
    )
    parser.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and VLM responses"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors"
    )
    parser.add_argument(
        "--nn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show when using KNN connections"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles instead of 2D"
    )
    parser.add_argument(
        "--include_spectrogram",
        action="store_true",
        default=True,
        help="Include spectrogram in visualization"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["marvis_tsne"],
        choices=["marvis_tsne", "whisper_knn", "clap_zero_shot"],
        help="Models to test (default: marvis_tsne)"
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save visualizations every N predictions"
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=None,
        help="Maximum audio duration to process (seconds, auto-detected per dataset if None)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esc50", "ravdess"],
        choices=["esc50", "ravdess", "urbansound8k"],
        help="Datasets to test (default: esc50, ravdess). Use 'urbansound8k' to include UrbanSound8K"
    )
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="audio-marvis-all",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting audio classification test...")
    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {', '.join(args.datasets)}")
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  k-shot: {args.k_shot}")
    logger.info(f"  Embedding model: {args.embedding_model}")
    if args.embedding_model == "whisper":
        logger.info(f"  Whisper model: {args.whisper_model}")
    elif args.embedding_model == "clap":
        logger.info(f"  CLAP version: {args.clap_version}")
    logger.info(f"  Use PCA: {args.use_pca_backend}")
    logger.info(f"  3D t-SNE: {args.use_3d}")
    logger.info(f"  Include spectrogram: {args.include_spectrogram}")
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.wandb_name is None:
                datasets_str = "_".join(args.datasets)
                feature_suffix = f"_k{args.k_shot}_{datasets_str}"
                if args.use_3d:
                    feature_suffix += "_3d"
                if args.use_knn_connections:
                    feature_suffix += f"_knn{args.nn_k}"
                if args.use_pca_backend:
                    feature_suffix += "_pca"
                args.wandb_name = f"audio_marvis_{timestamp}{feature_suffix}"
            
            gpu_monitor = init_wandb_with_gpu_monitoring(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                output_dir=args.output_dir,
                enable_system_monitoring=True,
                gpu_log_interval=30.0,
                enable_detailed_gpu_logging=True
            )
            logger.info(f"Initialized Weights & Biases run: {args.wandb_name}")
    
    # Log platform information
    platform_info = log_platform_info(logger)
    
    # Run tests
    start_time = time.time()
    results = run_all_audio_tests(args)
    total_time = time.time() - start_time
    
    # Save results
    save_results(results, args.output_dir, args.k_shot)
    
    # Log summary to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        # Count successful experiments
        success_count = 0
        for r in results.values():
            if r.get('status') == 'success':
                if 'all_results' in r:
                    success_count += sum(1 for model_result in r['all_results'].values() if 'error' not in model_result)
                else:
                    success_count += 1
        
        total_datasets = len(results)
        total_experiments = sum(len(r.get('all_results', ['default'])) for r in results.values())
        
        wandb.log({
            "summary/datasets_tested": total_datasets,
            "summary/experiments_successful": success_count,
            "summary/total_experiments": total_experiments,
            "summary/total_test_time": total_time,
            "summary/k_shot": args.k_shot,
        })
        
        # Log mean accuracy across successful experiments
        accuracies = []
        for r in results.values():
            if r.get('status') == 'success':
                if 'results' in r:
                    accuracies.append(r['results']['accuracy'])
                if 'all_results' in r:
                    for model_result in r['all_results'].values():
                        if 'error' not in model_result:
                            accuracies.append(model_result['accuracy'])
        if accuracies:
            wandb.log({"summary/mean_accuracy": np.mean(accuracies)})
    
    # Clean up wandb
    if gpu_monitor is not None:
        cleanup_gpu_monitoring(gpu_monitor)
    
    logger.info(f"\\nTotal test time: {total_time:.1f} seconds")
    logger.info(f"Audio tests completed for datasets: {', '.join(args.datasets)}")


if __name__ == "__main__":
    main()