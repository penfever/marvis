#!/usr/bin/env python
"""
Script for training a model on multiple tabular datasets, switching between them periodically.

This script enables multi-dataset training to create more robust in-context learners.
It can accept multiple sources of datasets:
1. A list of OpenML dataset IDs
2. One or more directories containing CSV files 
3. A number specifying how many random datasets to sample from OpenML

The script divides datasets into train/test splits and evaluates performance across datasets.
It also supports Vector Quantization (VQ) for more efficient embedding representation.

Usage examples:
    # Train on 5 randomly selected OpenML datasets
    python train_tabular_mix.py --num_datasets 5 --output_dir ./models/multi_dataset_model
    
    # Train on specific OpenML datasets
    python train_tabular_mix.py --dataset_ids 1590,1478,40975 --output_dir ./models/multi_dataset_model
    
    # Train on CSV files in a single directory
    python train_tabular_mix.py --data_dir ./my_datasets --output_dir ./models/multi_dataset_model
    
    # Train on CSV files from multiple directories
    python train_tabular_mix.py --data_dir ./datasets_dir1 ./datasets_dir2 ./datasets_dir3 --output_dir ./models/multi_dataset_model
    
    # Set specific training parameters
    python train_tabular_mix.py --num_datasets 3 --total_steps 10000 --output_dir ./models/multi_model
    
    # Train with Vector Quantization enabled
    python train_tabular_mix.py --num_datasets 3 --use_vector_quantization --vq_num_embeddings 128 --vq_commitment_cost 1.0 --output_dir ./models/vq_multi_model
    
    # Use different learning rate schedulers
    python train_tabular_mix.py --num_datasets 3 --lr_scheduler cosine --lr_warmup_ratio 0.1 --lr_eta_min 1e-6 --output_dir ./models/cosine_lr_model
    python train_tabular_mix.py --num_datasets 3 --lr_scheduler linear --lr_warmup_steps 500 --output_dir ./models/linear_lr_model
    python train_tabular_mix.py --num_datasets 3 --lr_scheduler exponential --lr_gamma 0.95 --output_dir ./models/exp_lr_model
"""

import os
import argparse
import numpy as np
import torch
import datetime
import random
import glob
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import csv
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from pandas.api.types import is_numeric_dtype

from marvis.data import (
    load_dataset, 
    get_tabpfn_embeddings, 
    create_llm_dataset, 
    list_available_datasets,
    is_csv_dataset,
    load_csv_dataset,
    load_dataset_with_metadata,
    find_csv_with_fallbacks
)
from marvis.models import prepare_qwen_with_prefix_embedding, prepare_qwen_with_vq_prefix_embedding
from marvis.train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
from marvis.utils import setup_logging, create_multi_dataset_parser

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring utilities
from marvis.utils import init_wandb_with_gpu_monitoring, cleanup_gpu_monitoring, GPUMonitor

# Constants for data validation
MIN_SAMPLES_PER_CLASS = 2
MIN_CLASSES = 2

def create_scheduler(optimizer, scheduler_type, num_training_steps, args):
    """
    Create a learning rate scheduler based on the specified type.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler to create
        num_training_steps: Total number of training steps
        args: Command line arguments containing scheduler parameters
        
    Returns:
        A tuple of (scheduler, num_warmup_steps) where scheduler is a PyTorch scheduler
        and num_warmup_steps is the number of warmup steps (for logging)
    """
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, PolynomialLR, CosineAnnealingWarmRestarts
    
    # Calculate warmup steps
    if args.lr_warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * args.lr_warmup_ratio)
    else:
        num_warmup_steps = args.lr_warmup_steps
    
    if scheduler_type == "constant":
        # Constant learning rate with optional warmup
        def lr_lambda(current_step):
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "linear":
        # Linear decay to 0 with optional warmup
        def lr_lambda(current_step):
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "cosine":
        # Cosine annealing with optional warmup
        if num_warmup_steps > 0:
            # Create a custom scheduler that combines warmup with cosine annealing
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(args.lr_eta_min / args.learning_rate, 0.5 * (1.0 + math.cos(math.pi * args.lr_num_cycles * 2.0 * progress)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            # Use standard cosine annealing
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.lr_eta_min)
            
    elif scheduler_type == "exponential":
        # Exponential decay with optional warmup
        if num_warmup_steps > 0:
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return args.lr_gamma ** (float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
            
    elif scheduler_type == "polynomial":
        # Polynomial decay with optional warmup
        def lr_lambda(current_step):
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step >= num_training_steps:
                return args.lr_eta_min / args.learning_rate
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(args.lr_eta_min / args.learning_rate, (1.0 - progress) ** args.lr_power)
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "cosine_with_restarts":
        # Cosine annealing with warm restarts
        if args.lr_cycle_length is None:
            # Default: restart every 10% of total steps
            T_0 = max(1, num_training_steps // 10)
        else:
            T_0 = args.lr_cycle_length
            
        if num_warmup_steps > 0:
            # Combine warmup with restarts
            base_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=args.lr_eta_min)
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return 1.0  # Let the base scheduler handle the rest
            scheduler = LambdaLR(optimizer, lr_lambda)
            # Note: This is a simplified version. In practice, you might want to chain schedulers
        else:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=args.lr_eta_min)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler, num_warmup_steps


def parse_args():
    """Parse command line arguments using the shared argument parser."""
    parser = create_multi_dataset_parser()
    return parser.parse_args()


def extract_vq_class(model):
    """Extracts the VectorQuantizer class from model if available."""
    if hasattr(model, 'vector_quantizer'):
        return model.vector_quantizer.__class__
    return None

def patch_vector_quantizer_if_needed(model):
    """Patch the vector quantizer to ensure it's tracking usage properly."""
    import types
    import torch
    import logging
    
    if not hasattr(model, 'vector_quantizer'):
        return False
    
    vq = model.vector_quantizer
    logger = logging.getLogger(__name__)
    
    # Define the improved forward method
    def patched_forward(self, inputs, return_indices=False):
        # inputs shape: [batch_size, embedding_dim]
        # Ensure input is 2D
        orig_shape = inputs.shape
        if len(orig_shape) > 2:
            flat_inputs = inputs.reshape(-1, self.embedding_dim)
        else:
            flat_inputs = inputs
        
        # Calculate distances between inputs and codebook vectors
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook**2, dim=1) - \
                    2 * torch.matmul(flat_inputs, self.codebook.t())
        
        # Find the nearest codebook vector for each input
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Update codebook usage tracking - IMPROVED VERSION
        unique_indices = torch.unique(encoding_indices)
        self._codebook_usage[unique_indices] += 1
        
        # Calculate the perplexity of the quantized distribution
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        # Get the quantized vectors
        quantized = self._get_codebook_entries(encoding_indices)
        
        # Update the codebook using Exponential Moving Average when using decay
        if self.training and self.decay > 0:
            self._update_codebook_ema(flat_inputs, encodings)
        
        # CRITICAL FIX: Use higher weight for the commitment loss
        commitment_loss = torch.mean((quantized.detach() - flat_inputs)**2)
        codebook_loss = torch.mean((quantized - flat_inputs.detach())**2)
        
        # Use a much higher commitment cost to ensure the model uses the codebook
        effective_commitment_cost = max(self.commitment_cost, 0.5)
        loss = codebook_loss + effective_commitment_cost * commitment_loss
        
        # Straight-through estimator
        # Pass gradients from quantized vectors back to inputs
        quantized = flat_inputs + (quantized - flat_inputs).detach()
        
        # Reshape to original shape if needed
        if len(orig_shape) > 2:
            quantized = quantized.reshape(orig_shape)
        
        if return_indices:
            return quantized, loss, perplexity, encoding_indices
        else:
            return quantized, loss, perplexity
    
    # Check if we should patch
    try:
        # Only patch once
        if not hasattr(vq, '_patched_by_fix'):
            # Save the original method
            vq._original_forward = vq.forward
            
            # Replace with our patched version
            vq.forward = types.MethodType(patched_forward, vq)
            vq._patched_by_fix = True
            
            logger.info("Vector Quantizer forward method patched to improve codebook utilization")
            return True
    except Exception as e:
        logger.warning(f"Failed to patch Vector Quantizer: {e}")
    
    return False

def diagnose_vq_issues(model, output_dir, log_to_wandb=False):
    """Diagnose Vector Quantization issues and save detailed report.
    
    Args:
        model: The model with vector quantizer
        output_dir: Directory to save the report
        log_to_wandb: Whether to log the report to Weights & Biases
    """
    import logging
    import os
    import torch
    
    logger = logging.getLogger(__name__)
    report_lines = ["========== VQ DIAGNOSTIC REPORT =========="]
    
    # Dictionary to store metrics for W&B logging
    vq_metrics = {}
    
    if not hasattr(model, 'vector_quantizer'):
        report_lines.append("ERROR: Model does not have a vector_quantizer attribute")
        with open(os.path.join(output_dir, "vq_diagnostic_report.txt"), "w") as f:
            f.write("\n".join(report_lines))
        return
    
    vq = model.vector_quantizer
    
    # Check codebook properties
    try:
        codebook_shape = vq.codebook.shape
        codebook_mean = vq.codebook.mean().item()
        codebook_std = vq.codebook.std().item()
        codebook_min = vq.codebook.min().item()
        codebook_max = vq.codebook.max().item()
        
        report_lines.append(f"Codebook shape: {codebook_shape}")
        report_lines.append(f"Codebook mean: {codebook_mean:.6f}")
        report_lines.append(f"Codebook std: {codebook_std:.6f}")
        report_lines.append(f"Codebook min: {codebook_min:.6f}")
        report_lines.append(f"Codebook max: {codebook_max:.6f}")
        
        # Store metrics for W&B
        vq_metrics["vq/codebook_size"] = codebook_shape[0]
        vq_metrics["vq/codebook_dim"] = codebook_shape[1]
        vq_metrics["vq/codebook_mean"] = codebook_mean
        vq_metrics["vq/codebook_std"] = codebook_std
        vq_metrics["vq/codebook_min"] = codebook_min
        vq_metrics["vq/codebook_max"] = codebook_max
        
        # Check for NaNs or infinities
        has_nan = torch.isnan(vq.codebook).any().item()
        has_inf = torch.isinf(vq.codebook).any().item()
        report_lines.append(f"Codebook has NaNs: {has_nan}")
        report_lines.append(f"Codebook has Infs: {has_inf}")
        
        vq_metrics["vq/has_nan"] = has_nan
        vq_metrics["vq/has_inf"] = has_inf
        
        # Check for zero vectors
        zero_vectors = (torch.abs(vq.codebook).sum(dim=1) < 1e-6).sum().item()
        report_lines.append(f"Codebook zero vectors: {zero_vectors}")
        
        vq_metrics["vq/zero_vectors"] = zero_vectors
        
        # Check codebook usage
        usage = vq._codebook_usage.cpu().numpy()
        active_codes = (usage > 0).sum()
        total_codes = len(usage) 
        utilization_rate = active_codes/total_codes
        report_lines.append(f"Active codes: {active_codes}/{total_codes} ({utilization_rate:.2%})")
        
        vq_metrics["vq/active_codes"] = active_codes
        vq_metrics["vq/total_codes"] = total_codes
        vq_metrics["vq/utilization_rate"] = utilization_rate
        
        # Check if any codes are used heavily vs. others
        if active_codes > 0:
            nonzero_usage = usage[usage > 0]
            min_usage = nonzero_usage.min()
            max_usage = usage.max()
            mean_usage = nonzero_usage.mean()
            
            report_lines.append(f"Min non-zero usage: {min_usage:.1f}")
            report_lines.append(f"Max usage: {max_usage:.1f}")
            report_lines.append(f"Mean usage: {mean_usage:.1f}")
            
            vq_metrics["vq/min_nonzero_usage"] = float(min_usage)
            vq_metrics["vq/max_usage"] = float(max_usage)
            vq_metrics["vq/mean_usage"] = float(mean_usage)
        
        # Check model parameters and hyperparameters
        report_lines.append(f"Commitment cost: {vq.commitment_cost}")
        report_lines.append(f"Decay factor: {vq.decay}")
        report_lines.append(f"Embedding dimension: {vq.embedding_dim}")
        
        vq_metrics["vq/commitment_cost"] = vq.commitment_cost
        vq_metrics["vq/decay_factor"] = vq.decay
        vq_metrics["vq/embedding_dim"] = vq.embedding_dim
        
    except Exception as e:
        report_lines.append(f"Error during diagnosis: {e}")
        vq_metrics["vq/diagnostic_error"] = str(e)
    
    # Add recommendations
    report_lines.append("\nRECOMMENDATIONS:")
    if 'active_codes' in locals() and 'total_codes' in locals():
        if active_codes == 0:
            report_lines.append("- Increase commitment cost to at least 1.0")
            report_lines.append("- Increase training epochs to at least a few thousand steps")
            report_lines.append("- Reduce codebook size to 128 or 64")
            report_lines.append("- Check if the VQ layer is correctly integrated in the forward pass")
            report_lines.append("- Consider using a non-VQ model if the issue persists")
        elif active_codes < total_codes * 0.1:  # Less than 10% utilization
            report_lines.append("- Increase commitment cost")
            report_lines.append("- Reduce codebook size to match observed utilization")
            report_lines.append("- Train for more epochs to increase utilization")
    else:
        report_lines.append("- Unable to generate recommendations due to diagnostic errors")
    
    # Save the report
    report_path = os.path.join(output_dir, "vq_diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"VQ diagnostic report saved to {report_path}")
    
    # Log to W&B if requested and available
    if log_to_wandb and WANDB_AVAILABLE:
        try:
            # Log all VQ metrics
            wandb.log(vq_metrics)
            
            # Also log the full report as a W&B artifact
            artifact = wandb.Artifact(
                name="vq_diagnostic_report",
                type="diagnostic_report",
                description="Vector Quantization diagnostic report"
            )
            artifact.add_file(report_path)
            wandb.log_artifact(artifact)
            
            # Log the report content as a text panel
            wandb.log({"vq/diagnostic_report": wandb.Html(
                f"<pre>{chr(10).join(report_lines)}</pre>"
            )})
            
            logger.info("VQ diagnostic report logged to Weights & Biases")
        except Exception as e:
            logger.warning(f"Failed to log VQ report to W&B: {e}")

def load_datasets(args) -> List[Dict[str, Any]]:
    """
    Load multiple datasets based on the provided arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of dictionaries with dataset information
    """
    logger = logging.getLogger(__name__)
    datasets = []
    
    # Case 1: Load from dataset IDs
    if args.dataset_ids:
        dataset_ids = [id.strip() for id in args.dataset_ids.split(",")]
        logger.info(f"Loading {len(dataset_ids)} datasets from provided IDs: {dataset_ids}")
        
        for dataset_id in dataset_ids:
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(dataset_id)
                datasets.append({
                    "id": dataset_id,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id})")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
    
    # Case 2: Load from directory or directories of CSV files
    elif args.data_dir:
        # Require --force_recompute_embeddings when using --data_dir
        if not args.force_recompute_embeddings:
            raise ValueError("When using --data_dir, you must also set --force_recompute_embeddings to ensure fresh embeddings are computed for the shuffled dataset order.")
        
        # With nargs='+', args.data_dir will always be a list
        data_dirs = args.data_dir
            
        logger.info(f"Loading datasets from {len(data_dirs)} directories: {data_dirs}")
        
        all_csv_files = []
        for data_dir in data_dirs:
            logger.info(f"Searching for CSV files in directory: {data_dir}")
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in directory {data_dir}")
            else:
                logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
                all_csv_files.extend(csv_files)
        
        if not all_csv_files:
            raise ValueError(f"No CSV files found in any of the directories: {data_dirs}")
            
        logger.info(f"Found {len(all_csv_files)} total CSV files across all directories")
        
        # Filter to avoid double loading when we have separate X and y files
        # If we have both X and y files, only process the X files
        x_files = [f for f in all_csv_files if '_X.csv' in f]
        non_xy_files = [f for f in all_csv_files if '_X.csv' not in f and '_y.csv' not in f]
        
        # Use X files if available, otherwise use regular CSV files
        files_to_process = x_files if x_files else non_xy_files if non_xy_files else all_csv_files
        
        # Create a seeded random generator for reproducible shuffling
        shuffle_rng = random.Random(args.seed)
        
        # Shuffle the files to randomize the order
        shuffle_rng.shuffle(files_to_process)
        
        logger.info(f"Processing {len(files_to_process)} dataset files (shuffled with seed {args.seed})")
        
        for file_path in files_to_process:
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_csv_dataset(file_path)
                datasets.append({
                    "id": dataset_name,  # Use dataset_name instead of basename to avoid _X suffix
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names,
                    "is_csv": True  # Mark this as a CSV dataset
                })
                logger.info(f"Successfully loaded dataset {dataset_name} from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset from {file_path}: {e}")
    
    # Case 3: Sample random datasets from OpenML
    elif args.num_datasets:
        logger.info(f"Sampling {args.num_datasets} random datasets from OpenML")
        
        # Try to use OpenML API to get additional datasets
        try:
            import openml
            
            # First, try to get datasets through OpenML API
            logger.info("Fetching available classification datasets from OpenML")
            
            # Use a simpler approach to get datasets to avoid server errors
            # First try to get datasets from a curated list
            try:
                # Use hardcoded list of known working datasets that we've verified work
                # This is a subset of datasets known to work reliably with ≤ 10 classes
                benchmark_datasets = [
                    1590,   # adult (binary)
                    1461,   # bank-marketing (binary)
                    40975,  # car (4 classes)
                    31,     # credit-g (binary)
                    37,     # diabetes (binary)
                    54,     # vehicle (4 classes)
                    1489,   # phoneme (binary)
                    40498,  # wine-quality-white (7 classes)
                    40701,  # australian (binary)
                    40981,  # kr-vs-kp (binary)
                ]
                datasets_info = {str(id): {} for id in benchmark_datasets}
                logger.info(f"Using {len(datasets_info)} curated benchmark datasets")
            except Exception as e:
                logger.warning(f"Error initializing benchmark datasets: {e}")
                # Fall back to empty dict
                datasets_info = {}
                
            # If we didn't get enough datasets, use the list_datasets function with minimal filters
            if len(datasets_info) < args.num_datasets:
                logger.info(f"Only found {len(datasets_info)} datasets, trying broader search")
                try:
                    # Use dataframe output format to avoid FutureWarning
                    more_datasets_df = openml.datasets.list_datasets(output_format="dataframe", limit=100)
                    # Convert dataframe to dict format for compatibility
                    more_datasets = {str(row['did']): {} for _, row in more_datasets_df.iterrows()}
                    datasets_info.update(more_datasets)
                    logger.info(f"Got a total of {len(datasets_info)} datasets after broader search")
                except Exception as e:
                    logger.warning(f"Error in broader dataset search: {e}")
            
            # If we still don't have enough, use predefined ones too
            available_datasets = list_available_datasets()
            predefined_ids = list(available_datasets.values())
            
            # Combine API datasets with predefined ones
            dataset_choices = list(datasets_info.keys()) + predefined_ids
            # Remove duplicates
            dataset_choices = list(set(dataset_choices))
            
            logger.info(f"Found {len(dataset_choices)} potential datasets to sample from")
            
            # If we still don't have enough, use random IDs as a last resort
            if len(dataset_choices) < args.num_datasets:
                logger.info(f"Not enough datasets found ({len(dataset_choices)}), adding random IDs to reach {args.num_datasets}")
                max_id_to_try = 100000
                random_ids = set()
                while len(dataset_choices) + len(random_ids) < args.num_datasets:
                    # Add random IDs between 1 and max_id_to_try
                    random_id = random.randint(1, max_id_to_try)
                    if random_id not in dataset_choices and random_id not in random_ids:
                        random_ids.add(random_id)
                
                # Add the random IDs to our choices
                dataset_choices.extend(list(random_ids))
                logger.info(f"Added {len(random_ids)} random dataset IDs to reach {len(dataset_choices)} total")
            
            # Prioritize the known good datasets 
            # Create a dedicated random generator with the specified seed for dataset selection
            rng = random.Random(args.seed)
            
            # If we have enough known working datasets, use only those
            known_working_datasets = [
                1590,   # adult (binary)
                1461,   # bank-marketing (binary)
                40975,  # car (4 classes)
                31,     # credit-g (binary)
                37,     # diabetes (binary)
                54,     # vehicle (4 classes)
                1489,   # phoneme (binary)
                40498,  # wine-quality-white (7 classes)
                40701,  # australian (binary)
                40981,  # kr-vs-kp (binary)
            ]
            if len(known_working_datasets) >= args.num_datasets:
                dataset_ids = rng.sample(known_working_datasets, args.num_datasets)
                logger.info(f"Using {len(dataset_ids)} known working datasets")
            else:
                # Use all known working datasets plus some random ones
                random_choices = [id for id in dataset_choices if id not in known_working_datasets]
                random_ids = rng.sample(random_choices, args.num_datasets - len(known_working_datasets))
                dataset_ids = known_working_datasets + random_ids
                logger.info(f"Using {len(known_working_datasets)} known working datasets plus {len(random_ids)} random datasets")
            
        except ImportError:
            logger.warning("OpenML package not available for advanced dataset search")
            # Fall back to predefined datasets and random sampling
            available_datasets = list_available_datasets()
            
            # Create a dedicated random generator with the specified seed
            rng = random.Random(args.seed)
            
            if args.num_datasets > len(available_datasets):
                logger.warning(f"Requested {args.num_datasets} datasets, but only {len(available_datasets)} are predefined")
                # Use all predefined ones
                predefined_ids = list(available_datasets.values())
                # Add random IDs to reach the requested number
                random_ids = []
                max_id_to_try = 100000
                while len(predefined_ids) + len(random_ids) < args.num_datasets:
                    random_id = rng.randint(1, max_id_to_try)
                    if random_id not in predefined_ids and random_id not in random_ids:
                        random_ids.append(random_id)
                
                dataset_ids = predefined_ids + random_ids
                logger.info(f"Using {len(predefined_ids)} predefined datasets plus {len(random_ids)} random IDs")
            else:
                # Sample from predefined datasets
                sampled_datasets = rng.sample(list(available_datasets.items()), args.num_datasets)
                dataset_ids = [id for _, id in sampled_datasets]
        
        logger.info(f"Selected {len(dataset_ids)} dataset IDs to try")
        
        # Keep track of how many datasets we successfully loaded
        successful_loads = 0
        attempted_ids = set()
        max_attempts = max(args.num_datasets * 8, 100)  # Try at least 100 datasets or 8x the requested number
        
        # Try loading datasets until we get enough or run out of options
        while successful_loads < args.num_datasets and len(attempted_ids) < max_attempts:
            # If we've exhausted our initial list, generate more random IDs
            if len(attempted_ids) >= len(dataset_ids):
                max_id_to_try = 100000
                new_random_ids = []
                new_count = min(50, args.num_datasets - successful_loads)  # Generate up to 50 new IDs at a time
                
                logger.info(f"Exhausted initial dataset list. Generating {new_count} new random IDs to try.")
                
                # Create a dedicated random generator with the specified seed plus the current attempt count
                # This ensures reproducibility but different seeds at different stages
                current_rng = random.Random(args.seed + len(attempted_ids))
                
                # Generate new random IDs that we haven't tried yet
                attempts = 0
                while len(new_random_ids) < new_count and attempts < 1000:
                    random_id = current_rng.randint(1, max_id_to_try)
                    if random_id not in attempted_ids:
                        new_random_ids.append(random_id)
                    attempts += 1
                
                dataset_ids.extend(new_random_ids)
                logger.info(f"Added {len(new_random_ids)} new random dataset IDs to try")
            
            # Get an ID we haven't tried yet
            remaining_ids = [id for id in dataset_ids if id not in attempted_ids]
            if not remaining_ids:
                logger.warning("No more dataset IDs to try")
                break
                
            dataset_id = remaining_ids[0]
            attempted_ids.add(dataset_id)
            
            try:
                logger.info(f"Attempting to load dataset ID: {dataset_id}")
                try:
                    X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(dataset_id))
                except Exception as e:
                    # Print the full error information
                    import traceback
                    logger.error(f"Failed to load dataset {dataset_id}: {e}")
                    logger.debug(f"Detailed error: {traceback.format_exc()}")
                    continue
                
                # Dataset quality checks
                if X is None or y is None or len(X) == 0 or len(y) == 0:
                    logger.warning(f"Skipping dataset {dataset_id} - empty dataset")
                    continue
                
                # Skip datasets that don't meet our criteria
                if X.shape[0] < 50:  # Skip if fewer than 50 samples
                    logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too few samples: {X.shape[0]}")
                    continue
                    
                unique_classes = len(np.unique(y))
                if unique_classes < 2:  # Skip if fewer than 2 classes
                    logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too few classes: {unique_classes}")
                    continue
                
                if unique_classes > 10:  # Skip if more than 10 classes
                    logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too many classes: {unique_classes}")
                    continue
                
                # Skip datasets with too many features
                if X.shape[1] > 5000:  # Arbitrary limit to avoid memory issues
                    logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too many features: {X.shape[1]}")
                    continue
                
                # Check for extreme class imbalance (any class with fewer than 10 examples)
                # Count examples per class
                unique_vals, counts = np.unique(y, return_counts=True)
                min_class_count = np.min(counts)
                if min_class_count < 10:
                    logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - extreme class imbalance detected: class with only {min_class_count} examples")
                    continue
                
                # Log class distribution for reference
                class_distribution = {int(cls): int(count) for cls, count in zip(unique_vals, counts)}
                class_imbalance_ratio = np.max(counts) / np.min(counts)
                logger.info(f"Class distribution for {dataset_name} (ID: {dataset_id}): {class_distribution}, imbalance ratio: {class_imbalance_ratio:.2f}x")
                
                # Handle missing values by removing affected rows
                if isinstance(X, np.ndarray) and X.size > 0:
                    # Convert to pandas for easier handling of NaN values
                    df_X = pd.DataFrame(X)
                    df_y = pd.Series(y)
                    
                    # Check for NaN values
                    nan_mask_X = df_X.isna().any(axis=1)
                    nan_mask_y = df_y.isna()
                    combined_mask = nan_mask_X | nan_mask_y
                    
                    # Count missing values
                    num_rows_with_nan = combined_mask.sum()
                    if num_rows_with_nan > 0:
                        # If more than 50% of rows have NaN, skip the dataset
                        if num_rows_with_nan > 0.5 * len(df_X):
                            logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too many missing values: {num_rows_with_nan}/{len(df_X)} rows")
                            continue
                        
                        # Otherwise, drop rows with NaN values
                        clean_indices = ~combined_mask
                        X = df_X.iloc[clean_indices].values
                        y = df_y.iloc[clean_indices].values
                        logger.info(f"Removed {num_rows_with_nan} rows with missing values from dataset {dataset_name} (ID: {dataset_id})")
                        
                        # If we have too few samples after cleaning, skip the dataset
                        if len(X) < 50:
                            logger.warning(f"Skipping dataset {dataset_name} (ID: {dataset_id}) - too few samples after removing NaN values: {len(X)}")
                            continue
                
                datasets.append({
                    "id": dataset_id,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id}) - {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
                successful_loads += 1
                
                # Print progress
                if successful_loads % 5 == 0 or successful_loads == args.num_datasets:
                    logger.info(f"Progress: Loaded {successful_loads}/{args.num_datasets} datasets ({len(attempted_ids)} attempts)")
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
        
        # If we still don't have enough datasets, warn the user
        if successful_loads < args.num_datasets:
            logger.warning(f"Only able to load {successful_loads} datasets out of {args.num_datasets} requested ({len(attempted_ids)} attempts)")
            
        logger.info(f"Successfully loaded {len(datasets)} datasets for training/testing")
    
    if not datasets:
        raise ValueError("No datasets could be loaded. Please check your input parameters.")
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets

def split_datasets_train_test(datasets: List[Dict[str, Any]], test_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split datasets into training and test sets.
    
    Args:
        datasets: List of dataset dictionaries
        test_ratio: Ratio of datasets to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_datasets: List of training datasets
        test_datasets: List of test datasets
    """
    # Create a separate random generator with the seed to avoid affecting global state
    rng = random.Random(seed)
    
    # Determine number of test datasets
    num_test = max(1, int(len(datasets) * test_ratio))
    
    # Create a copy to avoid modifying the original list
    datasets_copy = datasets.copy()
    
    # Use the rng to shuffle the datasets
    rng.shuffle(datasets_copy)
    
    # Split into train and test
    train_datasets = datasets_copy[num_test:]
    test_datasets = datasets_copy[:num_test]
    
    return train_datasets, test_datasets

def preprocess_features(X: np.ndarray, categorical_indicator: List[bool]) -> np.ndarray:
    """
    Preprocess features, converting string features to numeric values 
    and handling missing values.
    
    Args:
        X: Feature matrix
        categorical_indicator: Boolean list indicating categorical features
        
    Returns:
        Processed feature matrix
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X)
    
    # Process each column
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx]
        is_categorical = categorical_indicator[col_idx] if col_idx < len(categorical_indicator) else False
        
        # Check if column has object/string data
        if col.dtype == 'object' or is_categorical:
            logger.info(f"Converting feature at column {col_idx} to numeric")
            
            # For categorical features, use label encoding
            try:
                from sklearn.preprocessing import LabelEncoder
                # Explicitly call infer_objects to handle the FutureWarning about silent downcasting
                col_filled = col.infer_objects(copy=False)
                
                # Use label encoder
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(col_filled)
                
                # Explicitly cast to the same dtype as the destination column before assignment
                # Get the dtype of the destination column
                dest_dtype = df.iloc[:, col_idx].dtype

                # Convert encoded values to the same dtype and then assign
                df.iloc[:, col_idx] = pd.Series(encoded_values, index=df.index, dtype=dest_dtype)
                logger.info(f"  Encoded {len(encoder.classes_)} unique categories for column {col_idx}")
            except Exception as e:
                logger.warning(f"  Error encoding column {col_idx}: {e}")
                # If encoding fails, replace with zeros - use Series to avoid dtype warning
                df.iloc[:, col_idx] = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        else:
            # For numeric features, just fill NaN values with the mean or median
            if col.isna().any():
                # If more than 75% of the values are NaN, fill with zeros
                if col.isna().mean() > 0.75:
                    fill_value = 0
                # Otherwise, use the median for small number of NaNs
                else:
                    fill_value = col.median() if not np.isnan(col.median()) else 0
                
                # Use Series constructor to ensure type compatibility
                df.iloc[:, col_idx] = pd.Series(col.fillna(fill_value), index=df.index)
                logger.info(f"  Filled {col.isna().sum()} missing values in column {col_idx}")
    
    # Convert back to numpy array
    X_processed = df.values
    
    return X_processed

def process_dataset(dataset: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Process a single dataset: split into train/val/test, compute embeddings, and create LLM datasets.
    
    Args:
        dataset: Dictionary with dataset information
        args: Command line arguments
        
    Returns:
        Processed dataset with additional fields
    """
    logger = logging.getLogger(__name__)
    
    # Get the dataset attributes
    X = dataset["X"]
    y = dataset["y"]
    categorical_indicator = dataset.get("categorical_indicator", [False] * X.shape[1])
    
    # Initial data validation
    if len(X) != len(y):
        raise ValueError(f"Dataset {dataset['name']}: X and y lengths do not match: {len(X)} vs {len(y)}")
    
    # Check initial class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    logger.info(f"Dataset {dataset['name']} initial class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Filter out classes with too few samples
    valid_classes = unique_classes[class_counts >= MIN_SAMPLES_PER_CLASS]
    
    if len(valid_classes) < MIN_CLASSES:
        logger.warning(f"Dataset {dataset['name']} has only {len(valid_classes)} classes with at least {MIN_SAMPLES_PER_CLASS} samples. Skipping.")
        # Return a flag indicating this dataset should be skipped
        dataset["skip"] = True
        dataset["skip_reason"] = f"Insufficient classes: only {len(valid_classes)} classes with >= {MIN_SAMPLES_PER_CLASS} samples"
        return dataset
    
    # Filter data to only include valid classes
    if len(valid_classes) < len(unique_classes):
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]
        logger.info(f"Filtered out {len(mask) - mask.sum()} samples from classes with < {MIN_SAMPLES_PER_CLASS} members")
        
        # Re-encode labels to ensure consecutive integers starting from 0
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        dataset["label_encoder"] = encoder
    
    # Preprocess features to convert strings to numeric
    X = preprocess_features(X, categorical_indicator)
    
    # Determine if this is a classification or regression task
    is_classification = True
    
    # Check if labels are continuous (regression) or discrete (classification)
    if isinstance(y, np.ndarray):
        # If we have string labels, it's definitely classification
        if y.dtype.kind == 'O':
            logger.info(f"Dataset {dataset['name']} has string labels. Encoding to integers.")
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            logger.info(f"Encoded {len(encoder.classes_)} unique classes")
        # If numeric, check if it's continuous or discrete
        elif y.dtype.kind in ('i', 'u'):  # Integer type
            # It's likely classification if few unique values
            unique_vals = np.unique(y)
            logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} unique integer values")
            # Keep as classification with integer labels
        elif y.dtype.kind == 'f':  # Float type
            # Check if it's actually discrete disguised as float (e.g., 1.0, 2.0, 3.0)
            unique_vals = np.unique(y)
            if len(unique_vals) <= 10 and all(float(val).is_integer() for val in unique_vals):
                # Convert to integers for classification
                logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} discrete float values. Converting to integers.")
                y = y.astype(int)
            else:
                # It's truly continuous - must bin for classification tasks
                logger.info(f"Dataset {dataset['name']} has continuous target. Converting to classification by binning.")
                # For TabPFN which expects classification, bin the continuous values into discrete categories
                from sklearn.preprocessing import KBinsDiscretizer
                # Use quantile binning to create balanced classes
                n_bins = min(10, len(np.unique(y)))  # Use at most 10 bins
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                y = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
                logger.info(f"Binned continuous target into {n_bins} classes")
                dataset["target_discretizer"] = discretizer  # Store for future reference
    
    # After all label processing, verify we still have valid class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique_classes, class_counts))
    logger.info(f"Dataset {dataset['name']} processed class distribution: {class_dist}")
    
    # Check minimum requirements again after all processing
    min_class_count = min(class_counts)
    if min_class_count < MIN_SAMPLES_PER_CLASS:
        logger.warning(f"Dataset {dataset['name']}: After processing, minimum class count is {min_class_count}. Skipping.")
        dataset["skip"] = True
        dataset["skip_reason"] = f"After processing, minimum class count is {min_class_count} < {MIN_SAMPLES_PER_CLASS}"
        return dataset
    
    if len(unique_classes) < MIN_CLASSES:
        logger.warning(f"Dataset {dataset['name']}: After processing, only {len(unique_classes)} classes remain. Skipping.")
        dataset["skip"] = True
        dataset["skip_reason"] = f"After processing, only {len(unique_classes)} classes remain < {MIN_CLASSES}"
        return dataset
    
    # Use args.seed and dataset ID to create a dataset-specific but reproducible random state
    # This ensures different datasets get different splits, but the same dataset always gets the same split
    dataset_id_hash = hash(str(dataset["id"])) % 10000
    dataset_specific_seed = args.seed + dataset_id_hash
    
    # Split into train, validation, and test
    # Use stratify parameter to ensure each split has representation from all classes
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=dataset_specific_seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=dataset_specific_seed, stratify=y_temp
        )
    except ValueError as e:
        # If stratified split fails, it means some classes are too small
        logger.warning(f"Dataset {dataset['name']}: Stratified split failed: {e}. Skipping.")
        dataset["skip"] = True
        dataset["skip_reason"] = f"Stratified split failed: {e}"
        return dataset
    
    logger.info(f"Dataset {dataset['name']} shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Check if this dataset already has precomputed embeddings in the model directory
    # Skip this check if --force_recompute_embeddings is set
    has_precomputed_embeddings = False
    precomputed_prefix_data = None
    precomputed_dir = dataset.get("precomputed_dir")
    
    if dataset.get("has_precomputed_embeddings", False) and precomputed_dir and not args.force_recompute_embeddings:
        # Check if prefix_data.npz exists in the precomputed directory
        prefix_data_path = os.path.join(precomputed_dir, "prefix_data.npz")
        
        # First try to locate either prefix_data.npz or any embedding cache file in the directory
        if not os.path.exists(prefix_data_path):
            # Look for embedding cache files which contain "tabpfn_embeddings" in their name
            cache_files = glob.glob(os.path.join(precomputed_dir, "*tabpfn_embeddings*.npz"))
            if cache_files:
                # Use the most recent cache file
                prefix_data_path = sorted(cache_files, key=os.path.getmtime)[-1]
                logger.info(f"Found embedding cache file: {os.path.basename(prefix_data_path)}")
        
        if os.path.exists(prefix_data_path):
            try:
                logger.info(f"Loading precomputed embeddings for {dataset['name']} from {prefix_data_path}")
                precomputed_prefix_data = np.load(prefix_data_path, allow_pickle=True)
                
                # Check if the file has prefix embeddings (prefix_data.npz format)
                if 'embeddings' in precomputed_prefix_data and 'class_labels' in precomputed_prefix_data:
                    # This is a prefix_data.npz file
                    
                    # We can't directly use this file - we need the raw embeddings 
                    # not the few-shot subset, so we'll fall back to recalculating
                    logger.info(f"Found prefix data file, but we need the full raw embeddings. Will recalculate.")
                    has_precomputed_embeddings = False
                    
                # Check if the file has the raw embeddings
                elif all(key in precomputed_prefix_data for key in ['train_embeddings', 'val_embeddings', 'test_embeddings']):
                    # This is a cache file with raw embeddings
                    
                    # Extract embeddings
                    train_embeddings = precomputed_prefix_data['train_embeddings']
                    val_embeddings = precomputed_prefix_data['val_embeddings']
                    test_embeddings = precomputed_prefix_data['test_embeddings']
                    
                    # Check if y_train_sample is available
                    if 'y_train_sample' in precomputed_prefix_data:
                        y_train_sample = precomputed_prefix_data['y_train_sample']
                    else:
                        # If not available, we'll have to use the full y_train
                        # This is not ideal but better than recalculating everything
                        logger.warning("y_train_sample not found in cache, using y_train instead")
                        y_train_sample = y_train
                    
                    # Verify embedding size matches what we need
                    if train_embeddings.shape[1] != args.embedding_size:
                        logger.warning(f"Precomputed embeddings have size {train_embeddings.shape[1]}, but requested size is {args.embedding_size}")
                        # Attempt to resize the embeddings
                        from marvis.data.embeddings import resize_embeddings
                        try:
                            train_embeddings, val_embeddings, test_embeddings = resize_embeddings(
                                train_embeddings, val_embeddings, test_embeddings, args.embedding_size
                            )
                            logger.info(f"Successfully resized embeddings to {args.embedding_size}")
                            has_precomputed_embeddings = True
                        except Exception as resize_err:
                            logger.warning(f"Error resizing embeddings: {resize_err}. Will recalculate.")
                            has_precomputed_embeddings = False
                    else:
                        logger.info(f"Successfully loaded precomputed embeddings with shapes - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
                        has_precomputed_embeddings = True
                    
                    if has_precomputed_embeddings:
                        # Instantiate a dummy TabPFN for compatibility with downstream code
                        try:
                            from tabpfn import TabPFNClassifier
                            tabpfn = TabPFNClassifier(
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                n_estimators=8,
                                ignore_pretraining_limits=True
                            )
                        except ImportError:
                            logger.warning("TabPFN not available. Using None as a placeholder.")
                            tabpfn = None
                else:
                    # Unknown format
                    logger.warning(f"Unrecognized embedding file format in {prefix_data_path}")
                    has_precomputed_embeddings = False
            except Exception as e:
                logger.warning(f"Error loading precomputed embeddings: {e}")
                logger.info("Will recalculate embeddings")
                has_precomputed_embeddings = False
    
    # If we don't have valid precomputed embeddings, generate them
    if not has_precomputed_embeddings:
        # Get TabPFN embeddings
        logger.info(f"Generating TabPFN embeddings for {dataset['name']} with size {args.embedding_size}")
        
        # Handle embedding cache directory
        cache_dir = None
        if args.embedding_cache_dir.lower() != 'none':
            cache_dir = args.embedding_cache_dir
            # Create the directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Before recomputing embeddings, check if we already have a cache file for this dataset
            if not args.force_recompute_embeddings:
                # Use dataset ID or name as cache identifier
                dataset_identifier = str(dataset["id"])
                
                # Search for existing cache files in the cache directory
                try:
                    # Look for cache files matching this dataset in the cache directory
                    cache_pattern = os.path.join(cache_dir, f"*{dataset_identifier}*tabpfn_embeddings*.npz")
                    cache_files = glob.glob(cache_pattern)
                    
                    if cache_files:
                        # Sort by modification time to get the most recent
                        cache_file = sorted(cache_files, key=os.path.getmtime)[-1]
                        logger.info(f"Found existing embedding cache: {os.path.basename(cache_file)}")
                        
                        try:
                            # Load the cache file
                            cache_data = np.load(cache_file, allow_pickle=True)
                            if all(key in cache_data for key in ['train_embeddings', 'val_embeddings', 'test_embeddings']):
                                # Extract embeddings and y_train_sample
                                train_embeddings = cache_data['train_embeddings']
                                val_embeddings = cache_data['val_embeddings']
                                test_embeddings = cache_data['test_embeddings']
                                
                                if 'y_train_sample' in cache_data:
                                    y_train_sample = cache_data['y_train_sample']
                                else:
                                    # Fall back to using full y_train
                                    y_train_sample = y_train
                                
                                # Verify embedding size
                                if train_embeddings.shape[1] != args.embedding_size:
                                    logger.warning(f"Cached embeddings size ({train_embeddings.shape[1]}) doesn't match requested size ({args.embedding_size}). Resizing...")
                                    # Resize them
                                    from marvis.data.embeddings import resize_embeddings
                                    train_embeddings, val_embeddings, test_embeddings = resize_embeddings(
                                        train_embeddings, val_embeddings, test_embeddings, args.embedding_size
                                    )
                                
                                logger.info(f"Successfully loaded cached embeddings with shapes - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
                                
                                # Create a dummy TabPFN for compatibility
                                try:
                                    from tabpfn import TabPFNClassifier
                                    tabpfn = TabPFNClassifier(
                                        device='cuda' if torch.cuda.is_available() else 'cpu',
                                        n_estimators=8,
                                        ignore_pretraining_limits=True
                                    )
                                except ImportError:
                                    logger.warning("TabPFN not available. Using None as a placeholder")
                                    tabpfn = None
                                
                                # Set flag to indicate we have embeddings
                                has_precomputed_embeddings = True
                        except Exception as e:
                            logger.warning(f"Error loading cached embeddings from {cache_file}: {e}")
                except Exception as e:
                    logger.warning(f"Error searching for cached embeddings: {e}")
        
        # If we still don't have valid embeddings, compute them
        if not has_precomputed_embeddings:
            # Use dataset ID or name as cache identifier
            dataset_identifier = str(dataset["id"])
            
            train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
                X_train, y_train, X_val, X_test,
                embedding_size=args.embedding_size,
                cache_dir=cache_dir,
                dataset_name=dataset_identifier,
                force_recompute=args.force_recompute_embeddings,
                seed=args.seed
            )
    
    # Add processed data to the dataset dictionary
    dataset.update({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_embeddings": train_embeddings,
        "val_embeddings": val_embeddings,
        "test_embeddings": test_embeddings,
        "y_train_sample": y_train_sample,
        "is_classification": is_classification
    })
    
    return dataset

def create_llm_datasets(dataset: Dict[str, Any], model_info: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Create LLM datasets for a processed dataset.
    
    Args:
        dataset: Dictionary with processed dataset information
        model_info: Dictionary with model information
        args: Command line arguments
        
    Returns:
        Dictionary with LLM datasets and related information
    """
    logger = logging.getLogger(__name__)
    
    # Create dataset-specific output directory with hash to prevent collisions
    import hashlib
    
    # Create a unique identifier based on dataset properties
    # Include dataset id, name, and data characteristics to ensure uniqueness
    hash_input = f"{dataset['id']}_{dataset['name']}_{dataset['X_train'].shape}_{dataset['y_train'].shape}"
    dataset_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]  # Use first 8 chars of hash
    
    # Create directory name that includes both ID and hash
    dataset_output_dir = os.path.join(args.output_dir, f"dataset_{dataset['id']}_{dataset_hash}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    logger.info(f"Created dataset directory: {os.path.basename(dataset_output_dir)} for {dataset['name']}")
    
    # Save dataset information for future reference
    dataset_info = {
        "id": dataset["id"],
        "name": dataset["name"],
        "hash": dataset_hash,  # Store the hash for reference
        "directory_name": os.path.basename(dataset_output_dir),  # Store full directory name
        "num_samples": len(dataset["X_train"]),
        "num_features": dataset["X_train"].shape[1],
        "num_classes": len(np.unique(dataset["y_train"])),
        "class_distribution": {
            str(cls): int(count) 
            for cls, count in zip(*np.unique(dataset["y_train"], return_counts=True))
        },
        "is_csv": dataset.get("is_csv", False)  # Track whether this is a CSV dataset
    }
    
    # Save this information
    with open(os.path.join(dataset_output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create LLM dataset
    logger.info(f"Creating LLM dataset for {dataset['name']}")
    train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        dataset["X_train"], dataset["y_train_sample"], 
        dataset["X_val"], dataset["y_val"], 
        dataset["X_test"], dataset["y_test"],
        dataset["train_embeddings"], dataset["val_embeddings"], dataset["test_embeddings"],
        model_info["tokenizer"], model_info["prefix_start_id"], 
        model_info["prefix_end_id"], model_info["class_token_ids"],
        output_dir=dataset_output_dir,
        num_few_shot_examples=args.num_few_shot_examples,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "test_dataset": test_dataset,
        "label_encoder": label_encoder,
        "prefix_data_file": prefix_data_file,
        "output_dir": dataset_output_dir
    }

def main():
    import torch
    
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
                args.wandb_name = f"multi_dataset_{timestamp}"
            
            # Check if we're resuming from a checkpoint to set up W&B properly
            resume_wandb = "never"
            wandb_id = None
            
            # Only attempt to resume wandb if --resume, --use_wandb, and --resume_wandb are all specified
            if args.resume and args.use_wandb and args.resume_wandb:
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
                else:
                    logger.info("No wandb_id.txt found in checkpoint directory, starting new run")
            elif args.resume and args.use_wandb:
                logger.info("Resuming training from checkpoint, but starting a new W&B run (use --resume_wandb to resume the W&B run too)")
            
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
                # Use wandb.run.id if we used the GPU monitoring wrapper, otherwise use wandb_run.id
                if resume_wandb == "must" and wandb_id:
                    f.write(wandb_run.id)
                else:
                    f.write(wandb.run.id)
            
            # Log the run ID appropriately
            run_id = wandb_run.id if (resume_wandb == "must" and wandb_id) else wandb.run.id
            logger.info(f"Initialized Weights & Biases run: {args.wandb_name} (ID: {run_id})")
    
    # Check if we can find precomputed datasets in the output directory
    # Skip this check if --force_recompute_embeddings is set
    model_datasets = []
    
    if not args.force_recompute_embeddings:
        logger.info(f"Checking for precomputed embeddings in model directory: {args.output_dir}")
        
        # Check for dataset_* subdirectories in the output dir that might contain embeddings
        dataset_dirs = glob.glob(os.path.join(args.output_dir, "dataset_*"))
        
        if dataset_dirs:
            logger.info(f"Found {len(dataset_dirs)} potential dataset directories in the model directory")
            
            for dataset_dir in dataset_dirs:
                try:
                    # Load dataset metadata using the new utility
                    dataset_info = load_dataset_with_metadata(dataset_dir)
                    
                    if dataset_info.get('has_embeddings', False):
                        logger.info(f"Found precomputed embeddings for dataset {dataset_info['name']} (ID: {dataset_info['id']})")
                        model_datasets.append(dataset_info)
                        
                except Exception as e:
                    logger.warning(f"Error checking dataset directory {dataset_dir}: {e}")
    else:
        logger.info("Skipping check for precomputed embeddings in model directory due to --force_recompute_embeddings flag")
    
    # If we found enough datasets in the model directory, use them preferentially
    min_datasets_needed = 2  # Need at least 2 (1 for train, 1 for test)
    
    # 1. Load datasets - try to prioritize ones found in the model directory
    logger.info("Loading datasets")
    all_datasets = []
    
    # If we found enough datasets in the model directory, load just those first
    if len(model_datasets) >= min_datasets_needed:
        logger.info(f"Using {len(model_datasets)} datasets with precomputed embeddings found in model directory")
        
        # If we have predefined dataset IDs, filter model_datasets to only include those
        if args.dataset_ids:
            specified_ids = [id.strip() for id in args.dataset_ids.split(",")]
            filtered_model_datasets = [d for d in model_datasets if str(d["id"]) in specified_ids]
            
            if filtered_model_datasets:
                logger.info(f"Filtered to {len(filtered_model_datasets)} datasets that match specified IDs")
                model_datasets = filtered_model_datasets
        
        # If using num_datasets, truncate model_datasets if necessary
        if args.num_datasets and len(model_datasets) > args.num_datasets:
            model_datasets = model_datasets[:args.num_datasets]
            logger.info(f"Using first {args.num_datasets} datasets with precomputed embeddings")
        
        # Now try to load each of these datasets
        for dataset_info in model_datasets:
            try:
                dataset_id = dataset_info["id"]
                
                # Check if this is a CSV dataset
                if dataset_info.get("is_csv", False):
                    # Use the new CSV finding utility
                    csv_file = find_csv_with_fallbacks(
                        dataset_id=dataset_id,
                        primary_dir=dataset_info.get("embed_dir"),
                        data_dir=args.data_dir,
                        embed_dir=dataset_info.get("embed_dir")
                    )
                    
                    if csv_file:
                        X, y, categorical_indicator, attribute_names, dataset_name = load_csv_dataset(csv_file)
                    else:
                        logger.warning(f"Cannot find CSV file for dataset {dataset_id}. Skipping this dataset.")
                        continue
                else:
                    # Load from OpenML
                    X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(dataset_id))
                
                # Add the dataset to our list
                all_datasets.append({
                    "id": dataset_id,
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names,
                    "has_precomputed_embeddings": not args.force_recompute_embeddings,
                    "precomputed_dir": dataset_info.get("embed_dir") if not args.force_recompute_embeddings else None
                })
                logger.info(f"Successfully loaded dataset {dataset_name} (ID: {dataset_id}) with precomputed embeddings")
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_info['id']} despite having precomputed embeddings: {e}")
                # Check if this was likely a CSV dataset that we couldn't find
                if "episode" in str(dataset_info['id']).lower() or not str(dataset_info['id']).isdigit():
                    logger.info(f"Dataset {dataset_info['id']} appears to be a CSV dataset. Skipping since original data is not available.")
                else:
                    logger.info(f"Dataset {dataset_info['id']} appears to be an OpenML dataset that failed to load.")
    
    # If we don't have enough datasets yet, load more using the standard approach
    if len(all_datasets) < min_datasets_needed or (args.num_datasets and len(all_datasets) < args.num_datasets):
        logger.info(f"Need more datasets. Currently have {len(all_datasets)}, loading more...")
        additional_datasets = load_datasets(args)
        
        # Filter out datasets that we already loaded from precomputed embeddings
        existing_ids = {str(d["id"]) for d in all_datasets}
        additional_datasets = [d for d in additional_datasets if str(d["id"]) not in existing_ids]
        
        logger.info(f"Loaded {len(additional_datasets)} additional datasets")
        all_datasets.extend(additional_datasets)
    
    # Make sure we have at least 2 datasets (1 for training, 1 for testing)
    if len(all_datasets) < 2:
        raise ValueError(f"At least 2 datasets are required for training and testing, but only {len(all_datasets)} were loaded.")
    
    # Adjust test_split_ratio if needed to ensure at least 1 test dataset
    min_test_datasets = 1
    if int(len(all_datasets) * args.test_split_ratio) < min_test_datasets:
        effective_test_ratio = min_test_datasets / len(all_datasets)
        logger.warning(f"Adjusting test_split_ratio from {args.test_split_ratio} to {effective_test_ratio} to ensure at least {min_test_datasets} test dataset(s)")
        args.test_split_ratio = effective_test_ratio
    
    # Make sure at least 1 dataset is left for training
    max_test_datasets = len(all_datasets) - 1
    if int(len(all_datasets) * args.test_split_ratio) > max_test_datasets:
        effective_test_ratio = max_test_datasets / len(all_datasets)
        logger.warning(f"Adjusting test_split_ratio from {args.test_split_ratio} to {effective_test_ratio} to ensure at least 1 training dataset")
        args.test_split_ratio = effective_test_ratio
    
    # 2. Split datasets into train and test
    train_datasets, test_datasets = split_datasets_train_test(
        all_datasets, args.test_split_ratio, args.seed
    )
    
    # Cap the number of test datasets at 15
    if len(test_datasets) > 15:
        logger.info(f"Capping test datasets from {len(test_datasets)} to 15")
        test_datasets = test_datasets[:15]
    
    logger.info(f"Split datasets: {len(train_datasets)} for training, {len(test_datasets)} for testing")
    logger.info(f"Training datasets: {[d['name'] for d in train_datasets]}")
    logger.info(f"Testing datasets: {[d['name'] for d in test_datasets]}")
    
    # 3. Prepare Qwen model with prefix embedding (standard or vector-quantized)
    if args.use_vector_quantization:
        logger.info(f"Preparing model {args.model_id} with Vector Quantized prefix embedding")
        logger.info(f"VQ Settings: Codebook size: {args.vq_num_embeddings}, Commitment cost: {args.vq_commitment_cost}, Decay: {args.vq_decay}")
        
        # Use a higher default commitment cost if not explicitly set by user
        effective_commitment_cost = args.vq_commitment_cost
        
        # Check if we're using the default commitment cost and suggest a higher value
        if args.vq_commitment_cost == 0.25:
            logger.info("Using default commitment cost of 0.25. For better VQ utilization, consider using --vq_commitment_cost 0.5 or higher.")
        
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_vq_prefix_embedding(
            embedding_size=args.embedding_size,
            model_id=args.model_id,
            vq_num_embeddings=args.vq_num_embeddings,
            vq_commitment_cost=effective_commitment_cost,
            vq_decay=args.vq_decay
        )
        
        # Apply VQ patches to improve codebook utilization
        patched = patch_vector_quantizer_if_needed(model)
        if patched:
            logger.info("Applied Vector Quantizer patches for improved training")
        else:
            logger.warning("Could not patch Vector Quantizer - codebook utilization might be low")
        
        # Pass VQ debugging flag to the model if supported
        if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'set_debug_mode'):
            model.vector_quantizer.set_debug_mode(args.vq_debug)
            
        # Set VQ warmup steps if supported
        if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'set_warmup_steps'):
            model.vector_quantizer.set_warmup_steps(args.vq_warmup_steps)
        
        # Force VQ codebook reset before training to accurately measure utilization
        if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, '_codebook_usage'):
            try:
                # Zero out the usage counter
                model.vector_quantizer._codebook_usage.zero_()
                logger.info("Reset VQ codebook usage counter for accurate tracking")
                
                # Randomize codebook entries to prevent initialization bias
                # This can help increase utilization by making entries more distinct
                import torch
                with torch.no_grad():
                    embedding_dim = model.vector_quantizer.embedding_dim
                    num_embeddings = model.vector_quantizer.num_embeddings
                    init_bound = 1 / num_embeddings
                    new_codebook = torch.rand(
                        num_embeddings, embedding_dim, 
                        device=model.vector_quantizer.codebook.device
                    ).uniform_(-init_bound, init_bound)
                    model.vector_quantizer.codebook.copy_(new_codebook)
                    logger.info("Initialized VQ codebook with new random values")
            except Exception as e:
                logger.warning(f"Error resetting VQ codebook: {e}")
        
        # CRITICAL: Force update reset settings for more frequent feedback
        if hasattr(model, 'vector_quantizer'):
            try:
                model.vector_quantizer.reset_usage_every = 100  # Check every 100 steps instead of 1000
                logger.info("Set VQ usage tracking to check every 100 steps")
            except:
                pass
    else:
        logger.info(f"Preparing model {args.model_id} with standard prefix embedding")
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
            embedding_size=args.embedding_size,
            model_id=args.model_id
        )
    
    model_info = {
        "model": model,
        "tokenizer": tokenizer,
        "prefix_start_id": prefix_start_id,
        "prefix_end_id": prefix_end_id,
        "class_token_ids": class_token_ids
    }
    
    # 4. Process each dataset (train and test)
    logger.info("Processing datasets")
    
    processed_train_datasets = []
    for dataset in train_datasets:
        processed_dataset = process_dataset(dataset, args)
        # Only add datasets that weren't skipped
        if not processed_dataset.get("skip", False):
            processed_train_datasets.append(processed_dataset)
        else:
            logger.warning(f"Skipping dataset {dataset['name']}: {processed_dataset.get('skip_reason', 'Unknown reason')}")
    
    processed_test_datasets = []
    for dataset in test_datasets:
        processed_dataset = process_dataset(dataset, args)
        # Only add datasets that weren't skipped
        if not processed_dataset.get("skip", False):
            processed_test_datasets.append(processed_dataset)
        else:
            logger.warning(f"Skipping dataset {dataset['name']}: {processed_dataset.get('skip_reason', 'Unknown reason')}")
    
    # Check if we have any valid datasets left
    if not processed_train_datasets:
        raise ValueError("No valid training datasets remain after processing")
    if not processed_test_datasets:
        logger.warning("No valid test datasets remain after processing")
    
    # 5. Create LLM datasets for each processed dataset
    logger.info("Creating LLM datasets")
    
    llm_train_datasets = []
    for dataset in processed_train_datasets:
        llm_dataset = create_llm_datasets(dataset, model_info, args)
        llm_train_datasets.append({
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            **llm_dataset
        })
    
    llm_test_datasets = []
    for dataset in processed_test_datasets:
        llm_dataset = create_llm_datasets(dataset, model_info, args)
        llm_test_datasets.append({
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            **llm_dataset
        })
    
    # 6. Combine all training datasets and train
    logger.info("Combining all training datasets")
    
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
        wandb_callback = wandb_log_callback

    import json
    # Save dataset information for later reference
    datasets_info = {
        "train_datasets": [{"id": d["dataset_id"], "name": d["dataset_name"]} for d in llm_train_datasets],
        "test_datasets": [{"id": d["dataset_id"], "name": d["dataset_name"]} for d in llm_test_datasets]
    }
    with open(os.path.join(args.output_dir, "datasets_info.json"), "w") as f:
        json.dump(datasets_info, f, indent=2)
    
    # Initialize model
    trained_model = model_info["model"]
    final_class_token_ids = model_info["class_token_ids"]
    
    # Combine all training datasets
    from torch.utils.data import ConcatDataset
    
    # Extract all train datasets
    all_train_datasets = [d["train_dataset"] for d in llm_train_datasets]
    
    # Combine training datasets
    combined_train_dataset = ConcatDataset(all_train_datasets)
    
    # For eval, we'll use the first dataset's eval dataset
    # (or you could create a combined eval dataset if preferred)
    eval_dataset = llm_train_datasets[0]["eval_dataset"]
    
    logger.info(f"Combined {len(all_train_datasets)} training datasets")
    logger.info(f"Total training samples: {len(combined_train_dataset)}")
    logger.info(f"Using eval dataset from first dataset with {len(eval_dataset)} samples")
    
    # Use the prefix data from the first dataset (they should all be the same)
    prefix_data_file = llm_train_datasets[0]["prefix_data_file"]
    
    # Determine if we should disable label permutation
    effective_permute_labels = args.permute_labels
    
    # Log permutation behavior
    if args.permute_labels:
        if args.permute_labels_every_k_steps is not None:
            logger.info(f"Label permutation enabled: will permute every {args.permute_labels_every_k_steps} steps")
        else:
            logger.info("Label permutation enabled: will permute every epoch")
        
        if args.no_permute_last_k is not None:
            logger.info(f"Label permutation will be disabled for the last {args.no_permute_last_k} steps")
    
    # For backward compatibility with the current training function,
    # we need to convert the new scheduler parameters to lr_initial and lr_final
    # TODO: Update train_llm_with_tabpfn_embeddings to use PyTorch schedulers directly
    
    # Set lr_initial to the base learning rate
    lr_initial = args.learning_rate
    
    # Calculate lr_final based on scheduler type and total steps
    if args.lr_scheduler == "constant":
        lr_final = args.learning_rate
    elif args.lr_scheduler == "linear":
        lr_final = 0.0  # Linear decay goes to 0
    elif args.lr_scheduler == "cosine":
        lr_final = args.lr_eta_min
    elif args.lr_scheduler == "exponential":
        # Calculate final LR after ALL training steps
        decay_steps = args.total_steps - int(args.total_steps * args.lr_warmup_ratio)
        lr_final = args.learning_rate * (args.lr_gamma ** decay_steps)
    elif args.lr_scheduler == "polynomial":
        lr_final = args.lr_eta_min
    else:  # cosine_with_restarts
        lr_final = args.lr_eta_min
    
    # Log the learning rate schedule being used
    logger.info(f"Learning rate schedule: {args.lr_scheduler} (initial: {lr_initial:.2e}, final: {lr_final:.2e})")
    if args.lr_warmup_steps > 0 or args.lr_warmup_ratio > 0:
        warmup_steps = args.lr_warmup_steps if args.lr_warmup_ratio == 0 else int(args.total_steps * args.lr_warmup_ratio)
        logger.info(f"  Warmup steps: {warmup_steps} (out of {args.total_steps} total steps)")
    
    # Prepare component-specific learning rates
    # For VQ
    vq_lr_initial = args.vq_learning_rate if args.vq_learning_rate is not None else args.learning_rate
    vq_lr_scheduler = args.vq_lr_scheduler if args.vq_lr_scheduler is not None else args.lr_scheduler
    # Calculate VQ lr_final based on scheduler type
    if vq_lr_scheduler == "constant":
        vq_lr_final = vq_lr_initial
    elif vq_lr_scheduler == "linear":
        vq_lr_final = 0.0
    elif vq_lr_scheduler == "cosine":
        vq_lr_final = args.lr_eta_min
    elif vq_lr_scheduler == "exponential":
        decay_steps = args.total_steps - int(args.total_steps * args.lr_warmup_ratio)
        vq_lr_final = vq_lr_initial * (args.lr_gamma ** decay_steps)
    else:
        vq_lr_final = args.lr_eta_min
        
    # For class tokens
    class_token_lr_initial = args.class_token_learning_rate if args.class_token_learning_rate is not None else args.learning_rate
    class_token_lr_scheduler = args.class_token_lr_scheduler if args.class_token_lr_scheduler is not None else args.lr_scheduler
    # Calculate class token lr_final
    if class_token_lr_scheduler == "constant":
        class_token_lr_final = class_token_lr_initial
    elif class_token_lr_scheduler == "linear":
        class_token_lr_final = 0.0
    elif class_token_lr_scheduler == "cosine":
        class_token_lr_final = args.lr_eta_min
    elif class_token_lr_scheduler == "exponential":
        decay_steps = args.total_steps - int(args.total_steps * args.lr_warmup_ratio)
        class_token_lr_final = class_token_lr_initial * (args.lr_gamma ** decay_steps)
    else:
        class_token_lr_final = args.lr_eta_min
        
    # For LLM
    llm_lr_initial = args.llm_learning_rate if args.llm_learning_rate is not None else args.learning_rate
    llm_lr_scheduler = args.llm_lr_scheduler if args.llm_lr_scheduler is not None else args.lr_scheduler
    # Calculate LLM lr_final
    if llm_lr_scheduler == "constant":
        llm_lr_final = llm_lr_initial
    elif llm_lr_scheduler == "linear":
        llm_lr_final = 0.0
    elif llm_lr_scheduler == "cosine":
        llm_lr_final = args.lr_eta_min
    elif llm_lr_scheduler == "exponential":
        decay_steps = args.total_steps - int(args.total_steps * args.lr_warmup_ratio)
        llm_lr_final = llm_lr_initial * (args.lr_gamma ** decay_steps)
    else:
        llm_lr_final = args.lr_eta_min
    
    # Log component-specific learning rates if they differ from defaults
    if args.vq_learning_rate is not None or args.vq_lr_scheduler is not None:
        logger.info(f"VQ learning rate: {vq_lr_scheduler} (initial: {vq_lr_initial:.2e}, final: {vq_lr_final:.2e})")
    if args.class_token_learning_rate is not None or args.class_token_lr_scheduler is not None:
        logger.info(f"Class token learning rate: {class_token_lr_scheduler} (initial: {class_token_lr_initial:.2e}, final: {class_token_lr_final:.2e})")
    if args.llm_learning_rate is not None or args.llm_lr_scheduler is not None:
        logger.info(f"LLM learning rate: {llm_lr_scheduler} (initial: {llm_lr_initial:.2e}, final: {llm_lr_final:.2e})")
    
    # Train on the combined dataset
    logger.info("Starting training on combined dataset")
    trained_model, _, final_class_token_ids = train_llm_with_tabpfn_embeddings(
        trained_model, model_info["tokenizer"], 
        combined_train_dataset, eval_dataset,
        model_info["prefix_start_id"], model_info["prefix_end_id"], 
        model_info["class_token_ids"], prefix_data_file,
        output_dir=args.output_dir,
        num_train_epochs=None,  # Use max_steps instead of epochs
        max_steps=args.total_steps,  # Use total steps for proper scheduler behavior
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_samples=args.max_train_samples,
        lr_initial=lr_initial,
        lr_final=lr_final,
        mixup_alpha=args.mixup_alpha,
        min_freq_weight=args.min_freq_weight,
        min_freq_target=args.min_freq_target,
        save_best_model=args.save_best_model,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        permute_examples=args.permute_examples,
        permute_labels=effective_permute_labels,
        permute_labels_every_k_steps=args.permute_labels_every_k_steps,
        no_permute_last_k=args.no_permute_last_k,
        variable_few_shot=args.variable_few_shot,
        few_shot_min=args.few_shot_min,
        few_shot_max=few_shot_max,
        wandb_callback=wandb_callback,
        resume_from_checkpoint=args.resume,
        resume_optimizer=args.resume_optimizer,
        temperature_initial=args.temperature_initial,
        temperature_final=args.temperature_final,
        temperature_anneal_steps=args.temperature_anneal_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
        gradient_penalty_threshold=args.gradient_penalty_threshold,
        unfreeze_last_k_layers=args.unfreeze_last_k_layers,
        # Component-specific learning rates
        vq_lr_initial=vq_lr_initial,
        vq_lr_final=vq_lr_final,
        vq_lr_scheduler_type=vq_lr_scheduler,
        class_token_lr_initial=class_token_lr_initial,
        class_token_lr_final=class_token_lr_final,
        class_token_lr_scheduler_type=class_token_lr_scheduler,
        llm_lr_initial=llm_lr_initial,
        llm_lr_final=llm_lr_final,
        llm_lr_scheduler_type=llm_lr_scheduler,
        # Scheduler config (for warmup, etc.)
        lr_scheduler_config={
            'warmup_steps': args.lr_warmup_steps,
            'warmup_ratio': args.lr_warmup_ratio,
            'eta_min': args.lr_eta_min,
            'gamma': args.lr_gamma,
            'power': args.lr_power,
            'num_cycles': args.lr_num_cycles,
            'cycle_length': args.lr_cycle_length,
        }
    )
    
    logger.info("Training completed")
    
    # 7. Evaluate the final model on all test datasets
    logger.info("Evaluating final model on all test datasets")
    
    all_results = {}
    for dataset in llm_test_datasets:
        logger.info(f"Evaluating on dataset: {dataset['dataset_name']} (ID: {dataset['dataset_id']})")
        
        results = evaluate_llm_on_test_set(
            trained_model, model_info["tokenizer"], dataset["test_dataset"],
            dataset["label_encoder"], model_info["prefix_start_id"], 
            model_info["prefix_end_id"], final_class_token_ids, 
            dataset["prefix_data_file"], max_test_samples=args.max_test_samples
        )
        
        logger.info(f"Test accuracy on {dataset['dataset_name']}: {results['accuracy']:.4f}")
        
        # Store results
        all_results[dataset['dataset_id']] = {
            'dataset_name': dataset['dataset_name'],
            'accuracy': float(results['accuracy']),
            'num_samples': len(dataset["test_dataset"]),
            'classification_report': results.get('classification_report'),
            'confusion_matrix': results.get('confusion_matrix', []).tolist() 
                if 'confusion_matrix' in results else None
        }
    
    # Calculate average accuracy across all test datasets
    accuracies = [result['accuracy'] for result in all_results.values()]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    logger.info(f"Average test accuracy across all datasets: {avg_accuracy:.4f}")
    
    # Save all evaluation results
    with open(os.path.join(args.output_dir, "all_evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Log final results to W&B
    if args.use_wandb and WANDB_AVAILABLE:
        # Log overall metrics
        wandb.log({
            "test/average_accuracy": avg_accuracy,
            "test/num_datasets": len(llm_test_datasets)
        })
        
        # Log per-dataset metrics
        for dataset_id, results in all_results.items():
            wandb.log({
                f"test/{results['dataset_name']}_accuracy": results['accuracy']
            })
        
        # Create a summary table
        dataset_table = wandb.Table(columns=["Dataset ID", "Dataset Name", "Accuracy", "Samples"])
        for dataset_id, results in all_results.items():
            dataset_table.add_data(
                str(dataset_id),  # Convert to string for wandb compatibility
                results['dataset_name'], 
                results['accuracy'], 
                results['num_samples']
            )
        
        wandb.log({"test/datasets_summary": dataset_table})
        
        # Finalize wandb run
        wandb.finish()
    
    # Check VQ codebook utilization after training (if VQ was used)
    if args.use_vector_quantization and hasattr(trained_model, 'vector_quantizer') and hasattr(trained_model.vector_quantizer, '_codebook_usage'):
        try:
            # Check utilization statistics
            usage = trained_model.vector_quantizer._codebook_usage.cpu().numpy()
            active_codes = (usage > 0).sum()
            total_codes = len(usage)
            utilization_pct = 100 * active_codes / total_codes if total_codes > 0 else 0
            
            logger.info(f"Final VQ codebook utilization: {active_codes}/{total_codes} vectors used ({utilization_pct:.2f}%)")
            
            # Save VQ utilization stats
            final_model_path = os.path.join(args.output_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            
            with open(os.path.join(final_model_path, "vq_usage_stats.txt"), "w") as f:
                f.write(f"Codebook size: {total_codes}\n")
                f.write(f"Active entries: {active_codes} ({active_codes/total_codes:.2%})\n")
                f.write("--\n")
                for i, count in enumerate(usage):
                    if count > 0:
                        f.write(f"Entry {i}: {count} uses\n")
                
                # Add recommendations if no vectors were used
                if active_codes == 0:
                    f.write("\nRECOMMENDATIONS:\n")
                    f.write("- Increase the VQ commitment cost (try 1.0 or 2.0)\n")
                    f.write("- Train for more epochs (at least 5-10)\n")
                    f.write("- Reduce the codebook size (try 64 or 128 instead of 256/512)\n")
                    f.write("- Use the patched version with 'autopatch' enabled\n")
            
            # Run extended diagnostics
            diagnose_vq_issues(trained_model, final_model_path, log_to_wandb=args.use_wandb)
        except Exception as e:
            logger.warning(f"Error checking VQ codebook utilization: {e}")
    
    logger.info(f"Training and evaluation complete! Results saved to {args.output_dir}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)

if __name__ == "__main__":
    main()