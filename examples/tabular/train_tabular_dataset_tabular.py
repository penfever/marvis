#!/usr/bin/env python
"""
Example script demonstrating how to use MARVIS to train a model on tabular datasets.
Originally designed for the HAR dataset, but now supports any tabular dataset
that can be loaded with OpenML.

Usage examples:
    # Basic usage with default 100 few-shot examples
    python train_tabular_dataset.py --dataset_name har --output_dir ./models/har_model

    # Training with 50 few-shot examples instead of the default 100
    python train_tabular_dataset.py --dataset_name har --num_few_shot_examples 50

    # Training on a different dataset with more few-shot examples
    python train_tabular_dataset.py --dataset_name airlines --num_few_shot_examples 200

    # Using example order permutation to discourage memorization
    python train_tabular_dataset.py --dataset_name har --permute_examples

    # Using class-to-label mapping permutation to discourage memorization
    python train_tabular_dataset.py --dataset_name har --permute_labels

    # Using variable few-shot examples to improve generalization
    python train_tabular_dataset.py --dataset_name har --variable_few_shot --few_shot_min 10 --few_shot_max 150

    # Combining all generalization strategies
    python train_tabular_dataset.py --dataset_name har --permute_examples --permute_labels --variable_few_shot --few_shot_min 20 --few_shot_max 200
"""

import os
import argparse
import numpy as np
import torch
import random
import datetime
from sklearn.model_selection import train_test_split

from marvis.data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
from marvis.models import prepare_qwen_with_prefix_embedding, prepare_qwen_with_vq_prefix_embedding
from marvis.train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
from marvis.utils import setup_logging, create_single_dataset_parser, MetricsLogger

# Import the evaluate_with_explanations function from evaluate_with_explanations script
# We use a try-except to handle the case where the user might be running an older version
try:
    from .evaluate_with_explanations_tabular import evaluate_with_explanations, get_explanation_prompt
except ImportError:
    # If import fails, we'll provide a dummy implementation that logs the error
    import logging
    def evaluate_with_explanations(*args, **kwargs):
        logging.getLogger(__name__).error("Failed to import evaluate_with_explanations. Please make sure the file is in the same directory.")
        return evaluate_llm_on_test_set(*args, **kwargs)
    def get_explanation_prompt(explanation_type, predicted_class):
        return f"\nExplanation: The data belongs to class {predicted_class} because"

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring utilities
from marvis.utils import init_wandb_with_gpu_monitoring, cleanup_gpu_monitoring, GPUMonitor

def parse_args():
    """Parse command line arguments using the shared argument parser."""
    parser = create_single_dataset_parser()
    return parser.parse_args()


# Old argument parsing function removed - now using shared parser

def main():
    args = parse_args()
    
    # Initialize GPU monitor variable
    gpu_monitor = None
    
    # Set random seed for reproducibility
    from marvis.utils import set_seed
    set_seed(args.seed, deterministic=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_file=os.path.join(args.output_dir, "training.log"))
    logger.info(f"Arguments: {args}")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            # Set up wandb run name if not provided
            if args.wandb_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                args.wandb_name = f"{args.dataset_name}_{timestamp}"
            
            # Check if we're resuming from a checkpoint to set up W&B properly
            resume_wandb = "never"
            wandb_id = None
            
            if args.resume:
                # Check if there's a W&B ID stored in the checkpoint directory
                wandb_id_file = os.path.join(args.resume, "wandb_id.txt")
                if os.path.exists(wandb_id_file):
                    with open(wandb_id_file, "r") as f:
                        wandb_id = f.read().strip()
                        if wandb_id:
                            resume_wandb = "must"
                            logger.info(f"Resuming W&B run with ID: {wandb_id}")
                        else:
                            logger.warning("Empty W&B ID found, starting new run")
            
            # Initialize wandb with GPU monitoring
            # Note: For training scripts, we need to handle resume functionality manually
            if resume_wandb == "must" and wandb_id:
                # For resume, use regular wandb.init to maintain compatibility
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    config=vars(args),
                    dir=args.output_dir,
                    resume=resume_wandb,
                    id=wandb_id,
                    settings=wandb.Settings(
                        _disable_stats=False,  # Enable system stats
                        _disable_meta=False    # Enable metadata collection
                    )
                )
                # Create GPU monitor separately for resume case
                gpu_monitor = GPUMonitor(log_interval=30.0, enable_detailed_logging=True)
                if torch.cuda.is_available():
                    gpu_monitor.start_monitoring()
                else:
                    gpu_monitor = None
            else:
                # For new runs, use our GPU monitoring wrapper
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
            
            # Save the W&B run ID to the checkpoint directory for future resuming
            wandb_id_file = os.path.join(args.output_dir, "wandb_id.txt")
            with open(wandb_id_file, "w") as f:
                f.write(wandb.run.id)
            
            logger.info(f"Initialized Weights & Biases run: {args.wandb_name} (ID: {wandb.run.id})")
    
    # 1. Load and prepare HAR dataset
    logger.info("Loading HAR dataset")
    
    # Clear the _FAILED_DATASET_CACHE to always try loading the dataset
    from marvis.data.dataset import clear_failed_dataset_cache
    clear_failed_dataset_cache()
    logger.info("Cleared _FAILED_DATASET_CACHE to ensure dataset loading is attempted")
    
    # Load the dataset and try to get dataset metadata including ID
    try:
        dataset_info = load_dataset(args.dataset_name, return_metadata=True)
        if len(dataset_info) == 6:
            X, y, categorical_indicator, attribute_names, full_name, metadata = dataset_info
            dataset_id = metadata.get('dataset_id') if metadata else None
            task_id = metadata.get('task_id') if metadata else None
        else:
            # Fallback for older version without metadata
            X, y, categorical_indicator, attribute_names, full_name = dataset_info
            dataset_id = None
            task_id = None
    except TypeError:
        # If return_metadata parameter is not supported, use standard loading
        X, y, categorical_indicator, attribute_names, full_name = load_dataset(args.dataset_name)
        dataset_id = None
        task_id = None
    
    logger.info(f"Loaded dataset: {full_name}")
    if dataset_id:
        logger.info(f"Dataset ID: {dataset_id}, Task ID: {task_id}")
    else:
        logger.warning("Could not extract dataset_id and task_id from dataset metadata")

    # Create a dataset-specific random state for reproducible splits
    dataset_specific_seed = args.seed
    
    # Split into train, validation, and test with fixed random state for reproducibility
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=dataset_specific_seed, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=dataset_specific_seed, shuffle=True
    )
    logger.info(f"Dataset shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Get TabPFN embeddings
    logger.info(f"Getting TabPFN embeddings with size {args.embedding_size}")

    # Handle embedding cache directory
    cache_dir = None
    if args.embedding_cache_dir.lower() != 'none':
        cache_dir = args.embedding_cache_dir
        # Create the directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using embedding cache directory: {cache_dir}")

    train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
        X_train, y_train, X_val, X_test,
        embedding_size=args.embedding_size,
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        force_recompute=args.force_recompute_embeddings,
        seed=args.seed
    )

    # 3. Prepare Qwen model with prefix embedding
    # 3. Model setup - choose between regular and VQ model
    if args.use_vector_quantization:
        logger.info(f"Preparing model {args.model_id} with VQ prefix embedding")
        logger.info(f"VQ Settings: Codebook size: {args.vq_num_embeddings}, Commitment cost: {args.vq_commitment_cost}, Decay: {args.vq_decay}")
        
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_vq_prefix_embedding(
            embedding_size=args.embedding_size,
            model_id=args.model_id,
            vq_num_embeddings=args.vq_num_embeddings,
            vq_commitment_cost=args.vq_commitment_cost,
            vq_decay=args.vq_decay
        )
        
        # Set VQ-specific settings
        if hasattr(model, 'vector_quantizer'):
            if args.vq_debug:
                model.vector_quantizer.set_debug_mode(args.vq_debug)
                logger.info("Enabled VQ debugging mode")
            
            if args.vq_warmup_steps > 0:
                model.vector_quantizer.set_warmup_steps(args.vq_warmup_steps)
                logger.info(f"Set VQ warmup steps to {args.vq_warmup_steps}")
    else:
        logger.info(f"Preparing model {args.model_id} with regular prefix embedding")
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
            embedding_size=args.embedding_size,
            model_id=args.model_id
        )

    # 4. Create LLM dataset
    logger.info("Creating LLM dataset")
    train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        X_train, y_train_sample, X_val, y_val, X_test, y_test,
        train_embeddings, val_embeddings, test_embeddings,
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir=args.output_dir,
        num_few_shot_examples=args.num_few_shot_examples,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )

    # 5. Train LLM
    if args.total_steps is not None:
        logger.info(f"Training model for {args.total_steps} total steps")
    else:
        logger.info(f"Training model for {args.num_epochs} epochs")

    # If few_shot_max is not specified, use the num_few_shot_examples value
    if args.few_shot_max is None:
        few_shot_max = args.num_few_shot_examples
    else:
        few_shot_max = args.few_shot_max

    # Setup wandb callback function if enabled
    wandb_callback = None
    if args.use_wandb and WANDB_AVAILABLE:
        def wandb_log_callback(metrics):
            wandb.log(metrics)
    
    # Track training time
    import time
    training_start_time = time.time()
    
    trained_model, tokenizer, final_class_token_ids = train_llm_with_tabpfn_embeddings(
        model, tokenizer, train_dataset, eval_dataset,
        prefix_start_id, prefix_end_id, class_token_ids, prefix_data_file,
        output_dir=args.output_dir,
        num_train_epochs=None if args.total_steps is not None else args.num_epochs,
        max_steps=args.total_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_samples=args.max_train_samples,
        lr_initial=args.learning_rate,
        lr_final=args.learning_rate * 0.16,  # Match the ratio from old defaults: 8e-6 / 5e-5 = 0.16
        mixup_alpha=args.mixup_alpha,
        min_freq_weight=args.min_freq_weight,
        min_freq_target=args.min_freq_target,
        save_best_model=args.save_best_model,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        permute_examples=args.permute_examples,
        permute_labels=args.permute_labels,
        variable_few_shot=args.variable_few_shot,
        few_shot_min=args.few_shot_min,
        few_shot_max=few_shot_max,
        # New parameters from shared parser
        temperature_initial=args.temperature_initial,
        temperature_final=args.temperature_final,
        temperature_anneal_steps=args.temperature_anneal_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
        gradient_penalty_threshold=args.gradient_penalty_threshold,
        unfreeze_last_k_layers=args.unfreeze_last_k_layers,
        # Component-specific learning rates
        vq_lr_initial=args.vq_learning_rate,
        vq_lr_final=(args.vq_learning_rate * 0.16) if args.vq_learning_rate else None,
        vq_lr_scheduler_type=args.vq_lr_scheduler,
        class_token_lr_initial=args.class_token_learning_rate,
        class_token_lr_final=(args.class_token_learning_rate * 0.16) if args.class_token_learning_rate else None,
        class_token_lr_scheduler_type=args.class_token_lr_scheduler,
        llm_lr_initial=args.llm_learning_rate,
        llm_lr_final=(args.llm_learning_rate * 0.16) if args.llm_learning_rate else None,
        llm_lr_scheduler_type=args.llm_lr_scheduler,
        wandb_callback=wandb_log_callback if (args.use_wandb and WANDB_AVAILABLE) else None,
        resume_from_checkpoint=args.resume,
        resume_optimizer=args.resume_optimizer,
    )

    # Calculate training time
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # 5.5 Save the final model explicitly, even if early stopping occurred
    from marvis.train.save_utils import save_final_model
    logger.info("Saving final model checkpoint")
    save_final_model(
        model=trained_model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        final_class_token_ids=final_class_token_ids,
        original_class_token_ids=class_token_ids,
        restore_original_mapping=args.permute_labels  # Restore original mapping if we used label permutation
    )
    logger.info(f"Final model saved to {os.path.join(args.output_dir, 'final_model')}")
    
    # 6. Evaluate on test set (unless bypassed)
    if args.bypass_eval:
        logger.info("Skipping evaluation as requested with --bypass_eval")
        return
    
    logger.info("Evaluating model on test set")
    # Use the final class token mapping from the last training epoch for evaluation
    logger.info(f"Using final epoch class token mapping for evaluation")
    if args.permute_labels:
        logger.info(f"Original class token IDs: {class_token_ids}")
        logger.info(f"Final epoch class token IDs: {final_class_token_ids}")

    # Track prediction time
    prediction_start_time = time.time()

    # Choose evaluation method based on user request
    if args.evaluate_with_explanations:
        logger.info(f"Using explanation-based evaluation with {args.explanation_type} explanations")
        try:
            results = evaluate_with_explanations(
                trained_model, tokenizer, test_dataset,
                label_encoder, prefix_start_id, prefix_end_id,
                final_class_token_ids, prefix_data_file, 
                explanation_type=args.explanation_type,
                max_explanation_tokens=args.max_explanation_tokens,
                max_test_samples=args.max_test_samples
            )
            logger.info(f"Explanation-based evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error during explanation-based evaluation: {e}")
            logger.warning("Falling back to standard evaluation method")
            # Fall back to standard evaluation if explanation-based evaluation fails
            results = evaluate_llm_on_test_set(
                trained_model, tokenizer, test_dataset,
                label_encoder, prefix_start_id, prefix_end_id,
                final_class_token_ids, prefix_data_file, max_test_samples=args.max_test_samples
            )
    else:
        # Use standard evaluation
        results = evaluate_llm_on_test_set(
            trained_model, tokenizer, test_dataset,
            label_encoder, prefix_start_id, prefix_end_id,
            final_class_token_ids, prefix_data_file, max_test_samples=args.max_test_samples
        )

    # Calculate prediction time
    prediction_end_time = time.time()
    prediction_time = prediction_end_time - prediction_start_time
    logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
    
    # Save results
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'accuracy': float(results['accuracy']),
        'dataset_name': args.dataset_name,
        'evaluation_method': 'explanation_based' if args.evaluate_with_explanations else 'standard',
    }
    
    # Add classification report if available
    if 'classification_report' in results:
        serializable_results['classification_report'] = results['classification_report']
    
    # Add confusion matrix if available
    if 'confusion_matrix' in results:
        # Ensure it's a list for JSON serialization
        serializable_results['confusion_matrix'] = results['confusion_matrix'].tolist() \
            if not isinstance(results['confusion_matrix'], list) else results['confusion_matrix']
    
    # Save to JSON file
    results_filename = "evaluation_results"
    if args.evaluate_with_explanations:
        results_filename += f"_{args.explanation_type}"
    
    with open(os.path.join(args.output_dir, f"{results_filename}.json"), "w") as f:
        # For explanation-based evaluation, remove the explanations from main results file to keep it small
        results_to_save = serializable_results.copy()
        json.dump(results_to_save, f, indent=2)
    
    # Save explanations separately if requested and available
    if args.evaluate_with_explanations and args.save_explanations and 'explanations' in results:
        # Create a detailed results file with explanations for each example
        detailed_results = []
        for i in range(len(results['predictions'])):
            example_result = {
                'example_id': i,
                'prediction': int(results['predictions'][i]),
                'ground_truth': int(results['ground_truth'][i]),
                'probability': float(results['probabilities'][i]) if 'probabilities' in results else 0.0,
                'explanation': results['explanations'][i],
                'correct': results['predictions'][i] == results['ground_truth'][i]
            }
            detailed_results.append(example_result)
        
        # Save all explanations
        explanations_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.explanation_type}_explanations.json")
        with open(explanations_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Saved explanations to {explanations_file}")
    
    # Log to Weights & Biases using unified metrics
    if args.use_wandb and WANDB_AVAILABLE:
        # Initialize unified metrics logger
        metrics_logger = MetricsLogger(
            model_name="marvis",  # This is the MARVIS model being trained
            dataset_name=args.dataset_name,
            use_wandb=True,
            logger=logger
        )
        
        # Calculate total time and estimate from GPU power if available
        total_time = training_time + prediction_time
        
        # Try to get GPU power usage statistics for total time estimation
        gpu_power_estimate = None
        try:
            # Check if we have GPU power logs in wandb
            if hasattr(wandb.run, 'history') and wandb.run.history is not None:
                # Try to get GPU power data from wandb logs
                history_df = wandb.run.history()
                if 'gpu.0.powerWatts' in history_df.columns:
                    # Calculate average power usage during the run
                    power_data = history_df['gpu.0.powerWatts'].dropna()
                    if len(power_data) > 0:
                        avg_power_watts = power_data.mean()
                        # Estimate total energy consumed (Watt-hours)
                        total_energy_wh = (avg_power_watts * total_time) / 3600  # Convert seconds to hours
                        gpu_power_estimate = {
                            'avg_power_watts': float(avg_power_watts),
                            'total_energy_wh': float(total_energy_wh),
                            'total_time_seconds': float(total_time)
                        }
                        logger.info(f"GPU power estimation: {avg_power_watts:.1f}W avg, {total_energy_wh:.4f}Wh total energy")
        except Exception as e:
            logger.debug(f"Could not estimate GPU power usage: {e}")

        # Prepare metrics for unified logging
        metrics_dict = {
            'accuracy': float(results['accuracy']),
            'training_time': float(training_time),
            'prediction_time': float(prediction_time),
            'total_time': float(total_time),
            'dataset_name': args.dataset_name,
        }
        
        # Add dataset_id and task_id if available
        if dataset_id is not None:
            metrics_dict['dataset_id'] = dataset_id
        if task_id is not None:
            metrics_dict['task_id'] = task_id
        
        # Add GPU power estimation if available
        if gpu_power_estimate:
            metrics_dict.update({
                'gpu_avg_power_watts': gpu_power_estimate['avg_power_watts'],
                'gpu_total_energy_wh': gpu_power_estimate['total_energy_wh']
            })
        
        # Add F1 score, precision, and recall if available
        if 'classification_report' in results:
            # Get metrics from the 'weighted avg' section, which provides overall metrics
            weighted_avg = results['classification_report'].get('weighted avg', {})
            if weighted_avg:
                metrics_dict.update({
                    'f1_weighted': weighted_avg.get('f1-score', 0),
                    'precision': weighted_avg.get('precision', 0),
                    'recall': weighted_avg.get('recall', 0)
                })
        
        # Add confusion matrix if available
        if 'confusion_matrix' in results:
            metrics_dict['confusion_matrix'] = results['confusion_matrix']
        
        # Add dataset info if available
        if hasattr(args, 'num_classes') and args.num_classes:
            metrics_dict['num_classes'] = args.num_classes
        
        # Log all metrics using unified system
        metrics_logger.log_all_metrics(metrics_dict)
        
        # Log explanation-specific metrics using unified system
        if args.evaluate_with_explanations:
            # Calculate mean probability if available
            mean_prob = None
            if 'probabilities' in results:
                mean_prob = sum(results['probabilities']) / len(results['probabilities'])
            
            # Prepare explanation samples if available
            explanation_samples = None
            if 'explanations' in results:
                try:
                    # Get a few samples (both correct and incorrect predictions if possible)
                    sample_indices = []
                    correct_indices = [i for i, (p, g) in enumerate(zip(results['predictions'], results['ground_truth'])) if p == g]
                    incorrect_indices = [i for i, (p, g) in enumerate(zip(results['predictions'], results['ground_truth'])) if p != g]
                    
                    # Take up to 5 correct and 5 incorrect examples
                    sample_indices.extend(correct_indices[:5])
                    sample_indices.extend(incorrect_indices[:5])
                    
                    if sample_indices:
                        explanation_samples = {
                            'id': [i for i in sample_indices],
                            'predicted': [results['predictions'][i] for i in sample_indices],
                            'actual': [results['ground_truth'][i] for i in sample_indices],
                            'correct': [results['predictions'][i] == results['ground_truth'][i] for i in sample_indices],
                            'explanation': [results['explanations'][i] for i in sample_indices]
                        }
                except Exception as e:
                    logger.warning(f"Failed to prepare explanation samples: {e}")
            
            # Log all explanation metrics using unified system
            metrics_logger.log_explanation_metrics(
                explanation_type=args.explanation_type,
                mean_probability=mean_prob,
                explanation_samples=explanation_samples
            )
        
        # Finalize wandb run
        wandb.finish()
    
    logger.info(f"Finished! Results saved to {args.output_dir}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)

if __name__ == "__main__":
    main()