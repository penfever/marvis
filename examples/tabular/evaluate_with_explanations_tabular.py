#!/usr/bin/env python
"""
Script for evaluating a pretrained MARVIS model on one or more tabular datasets with explanations.
This script enhances standard evaluation by leveraging the language modeling capabilities:
1. First predicts the class token like the standard evaluator
2. Then generates an explanation of why that class was chosen
3. Optionally can generate feature insights, counterfactual explanations, or decision rules

Datasets can be specified in multiple ways:
- Single dataset by name or ID
- List of dataset IDs
- Directory of CSV files
- Random sampling from OpenML

Usage examples:
    # Basic usage with a single dataset
    python evaluate_with_explanations.py --model_path ./models/marvis_output --dataset_name har
    
    # Generating longer explanations
    python evaluate_with_explanations.py --model_path ./models/marvis_output --dataset_name har --max_explanation_tokens 100
    
    # Adding counterfactual explanations
    python evaluate_with_explanations.py --model_path ./models/marvis_output --dataset_name har --explanation_type counterfactual
    
    # Including feature importance analysis
    python evaluate_with_explanations.py --model_path ./models/marvis_output --dataset_name har --explanation_type feature_importance
    
    # Evaluating on multiple datasets with W&B tracking
    python evaluate_with_explanations.py --model_path ./models/marvis_output --dataset_ids 1590,40975,37,54 --use_wandb
    
    # Evaluating on 5 randomly sampled datasets with feature importance analysis
    python evaluate_with_explanations.py --model_path ./models/marvis_output --num_datasets 5 --explanation_type feature_importance
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import glob
import datetime
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

from marvis.data import load_dataset, get_tabpfn_embeddings, create_llm_dataset, list_available_datasets
from marvis.models import prepare_qwen_with_prefix_embedding, QwenWithPrefixEmbedding
from marvis.train import evaluate_llm_on_test_set
from marvis.utils import setup_logging

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained MARVIS model with explanations")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory"
    )
    
    # Dataset source options (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the OpenML dataset to evaluate on (e.g., 'har', 'airlines', 'albert', 'volkert', 'higgs')"
    )
    dataset_group.add_argument(
        "--dataset_ids",
        type=str,
        help="Comma-separated list of OpenML dataset IDs to evaluate on"
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing CSV files to use as datasets"
    )
    dataset_group.add_argument(
        "--num_datasets",
        type=int,
        help="Number of random datasets to sample from OpenML"
    )
    
    # General parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./explanation_results",
        help="Directory to save evaluation results and explanations"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1000,
        help="Size of the embeddings (must match the pretrained model)"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use for evaluation"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=100,
        help="Number of few-shot examples to include in the prefix during inference"
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default="./data",
        help="Directory to store cached embeddings. Set to 'none' to disable caching."
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation (auto, cuda, cpu)"
    )
    
    # Explanation parameters
    parser.add_argument(
        "--explanation_type",
        type=str,
        default="standard",
        choices=["standard", "counterfactual", "feature_importance", "decision_rules"],
        help="Type of explanation to generate"
    )
    parser.add_argument(
        "--max_explanation_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate for explanations"
    )
    parser.add_argument(
        "--save_explanations",
        action="store_true",
        help="Save all explanations to a separate file"
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
        default="marvis-explanations",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team) name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name (defaults to 'explanations_' + timestamp)"
    )

    return parser.parse_args()

def load_csv_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[bool], List[str], str]:
    """
    Load a dataset from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        X: Features
        y: Labels
        categorical_indicator: Boolean list indicating categorical features
        attribute_names: List of feature names
        dataset_name: Name of the dataset (derived from file name)
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    
    # Extract dataset name from file path
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    
    logger.info(f"Loading dataset from CSV file: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Assume the last column is the target
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        
        # Get feature names
        attribute_names = df.columns[:-1].tolist()
        
        # Determine categorical features (simple heuristic)
        categorical_indicator = []
        for col in df.iloc[:, :-1].columns:
            # Consider a column categorical if it has fewer than 10 unique values
            # or if it contains string values
            unique_vals = df[col].nunique()
            is_categorical = unique_vals < 10 or df[col].dtype == 'object'
            categorical_indicator.append(is_categorical)
            
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        logger.info(f"Dataset info: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        return X, y, categorical_indicator, attribute_names, dataset_name
    
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise ValueError(f"Failed to load dataset from {file_path}: {e}")

def load_datasets(args) -> List[Dict[str, Any]]:
    """
    Load multiple datasets based on the provided arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of dictionaries with dataset information
    """
    import logging
    logger = logging.getLogger(__name__)
    from marvis.data.dataset import clear_failed_dataset_cache
    clear_failed_dataset_cache()
    logger.info("Cleared _FAILED_DATASET_CACHE to ensure dataset loading is attempted")
    datasets = []
    
    # Case 1: Single dataset name provided
    if args.dataset_name:
        logger.info(f"Loading single dataset: {args.dataset_name}")
        try:
            X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(args.dataset_name)
            datasets.append({
                "id": args.dataset_name,
                "name": dataset_name,
                "X": X,
                "y": y,
                "categorical_indicator": categorical_indicator,
                "attribute_names": attribute_names
            })
            logger.info(f"Successfully loaded dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {args.dataset_name}: {e}")
            raise ValueError(f"Failed to load dataset {args.dataset_name}: {e}")
    
    # Case 2: Load from dataset IDs
    elif args.dataset_ids:
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
    
    # Case 3: Load from directory of CSV files
    elif args.data_dir:
        logger.info(f"Loading datasets from directory: {args.data_dir}")
        csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory {args.data_dir}")
            
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for file_path in csv_files:
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_csv_dataset(file_path)
                datasets.append({
                    "id": os.path.basename(file_path),
                    "name": dataset_name,
                    "X": X,
                    "y": y,
                    "categorical_indicator": categorical_indicator,
                    "attribute_names": attribute_names
                })
                logger.info(f"Successfully loaded dataset {dataset_name} from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset from {file_path}: {e}")
    
    # Case 4: Sample random datasets from OpenML
    elif args.num_datasets:
        logger.info(f"Sampling {args.num_datasets} random datasets from OpenML")
        
        # Similar to previous implementation
        try:
            import openml
            
            # Use a curated list of known working datasets
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
            
            # Create a dedicated random generator with the specified seed
            rng = random.Random(args.seed)
            
            # Sample from benchmark datasets if we have enough
            if len(benchmark_datasets) >= args.num_datasets:
                dataset_ids = rng.sample(benchmark_datasets, args.num_datasets)
                logger.info(f"Sampled {len(dataset_ids)} datasets from benchmark set")
            else:
                # Use all benchmark datasets and fill with random ones
                dataset_ids = benchmark_datasets.copy()
                
                # Get more datasets from OpenML API
                try:
                    more_datasets_df = openml.datasets.list_datasets(output_format="dataframe", limit=100)
                    more_dataset_ids = [str(row['did']) for _, row in more_datasets_df.iterrows()]
                    
                    # Filter out datasets we already have
                    more_dataset_ids = [id for id in more_dataset_ids if int(id) not in benchmark_datasets]
                    
                    # Sample random datasets to reach the requested number
                    if len(more_dataset_ids) >= args.num_datasets - len(dataset_ids):
                        random_ids = rng.sample(more_dataset_ids, args.num_datasets - len(dataset_ids))
                        dataset_ids.extend([int(id) for id in random_ids])
                    else:
                        dataset_ids.extend([int(id) for id in more_dataset_ids])
                        logger.warning(f"Could only find {len(dataset_ids)} datasets, less than requested {args.num_datasets}")
                except Exception as e:
                    logger.warning(f"Error getting datasets from OpenML API: {e}")
            
            # Now try to load each dataset
            for dataset_id in dataset_ids:
                try:
                    X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(dataset_id))
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
        
        except ImportError:
            logger.warning("OpenML package not available for advanced dataset sampling")
            # Fall back to predefined datasets
            available_datasets = list_available_datasets()
            
            if args.num_datasets > len(available_datasets):
                logger.warning(f"Requested {args.num_datasets} datasets, but only {len(available_datasets)} are predefined")
                # Use all available ones
                for name, dataset_id in available_datasets.items():
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
            else:
                # Create a dedicated random generator with the specified seed
                rng = random.Random(args.seed)
                
                # Sample from predefined datasets
                sampled_datasets = rng.sample(list(available_datasets.items()), args.num_datasets)
                for name, dataset_id in sampled_datasets:
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
    
    if not datasets:
        raise ValueError("No datasets could be loaded. Please check your input parameters.")
    
    logger.info(f"Successfully loaded {len(datasets)} datasets for evaluation")
    return datasets

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
    import pandas as pd
    import logging
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
                # Handle warnings with infer_objects
                col_filled = col.infer_objects(copy=False)
                
                # Use label encoder
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(col_filled)
                
                # Get destination dtype and assign properly
                dest_dtype = df.iloc[:, col_idx].dtype
                df.iloc[:, col_idx] = pd.Series(encoded_values, index=df.index, dtype=dest_dtype)
                logger.info(f"  Encoded {len(encoder.classes_)} unique categories for column {col_idx}")
            except Exception as e:
                logger.warning(f"  Error encoding column {col_idx}: {e}")
                # If encoding fails, replace with zeros
                df.iloc[:, col_idx] = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        else:
            # For numeric features, fill NaN values
            if col.isna().any():
                # If more than 75% of the values are NaN, fill with zeros
                if col.isna().mean() > 0.75:
                    fill_value = 0
                else:
                    # Otherwise use median
                    fill_value = col.median() if not np.isnan(col.median()) else 0
                
                # Use Series to ensure type compatibility
                df.iloc[:, col_idx] = pd.Series(col.fillna(fill_value), index=df.index)
                logger.info(f"  Filled {col.isna().sum()} missing values in column {col_idx}")
    
    # Convert back to numpy array
    X_processed = df.values
    
    return X_processed

def get_token_ids_from_model(tokenizer):
    """Extract special token IDs from tokenizer."""
    prefix_start_id = tokenizer.convert_tokens_to_ids("<PREFIX_START>")
    prefix_end_id = tokenizer.convert_tokens_to_ids("<PREFIX_END>")
    class_token_ids = [tokenizer.convert_tokens_to_ids(f"<CLASS_{i}>") for i in range(10)]
    
    return prefix_start_id, prefix_end_id, class_token_ids

def load_pretrained_model(model_path, device_map="auto", embedding_size=1000):
    """Load the pretrained model and tokenizer.
    
    This function is a simple wrapper around the load_pretrained_model function from marvis.models,
    which handles various model loading scenarios including custom model formats.
    """
    from marvis.models import load_pretrained_model as core_load_pretrained_model
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading pretrained model from {model_path} using core utility function")
    
    return core_load_pretrained_model(
        model_path=model_path,
        device_map=device_map,
        embedding_size=embedding_size
    )

def get_explanation_prompt(explanation_type, predicted_class):
    """Generate the explanation prompt based on the type of explanation requested."""
    if explanation_type == "standard":
        return f"\nExplanation: The above data belongs to class {predicted_class} because"
    elif explanation_type == "counterfactual":
        return f"\nExplanation: The data belongs to class {predicted_class}. If it were to be classified differently, we would need to see changes like"
    elif explanation_type == "feature_importance":
        return f"\nFeature Analysis: The data belongs to class {predicted_class}. The most important features for this classification are"
    elif explanation_type == "decision_rules":
        return f"\nDecision Rules: The data belongs to class {predicted_class}. The rules that led to this classification are"
    else:
        return f"\nExplanation: The data belongs to class {predicted_class} because"

def evaluate_with_explanations(
    model: torch.nn.Module,
    tokenizer: Any,
    test_dataset: Any,
    label_encoder: Any,
    prefix_start_id: int,
    prefix_end_id: int,
    class_token_ids: List[int],
    prefix_data_file: str,
    explanation_type: str = "standard",
    max_explanation_tokens: int = 50,
    max_test_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate the trained LLM on the test set with explanations.

    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        test_dataset: Test dataset
        label_encoder: Label encoder for class labels
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
        prefix_data_file: Path to the saved prefix data
        explanation_type: Type of explanation to generate
        max_explanation_tokens: Maximum number of tokens to generate for explanations
        max_test_samples: Maximum number of test samples to use

    Returns:
        Dictionary with evaluation results including explanations
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model on test set with {explanation_type} explanations")
    
    # Get the primary device
    if hasattr(model, "_hf_hook") and hasattr(model._hf_hook, "execution_device"):
        param_device = model._hf_hook.execution_device
    else:
        param_device = next(model.parameters()).device
    logger.info(f"Primary model device: {param_device}")
    
    # Load prefix data
    prefix_data = np.load(prefix_data_file)
    prefix_embeddings = prefix_data['embeddings']
    prefix_class_labels = prefix_data['class_labels']
    
    # Convert prefix data to tensors and move to primary device
    prefix_embeddings_tensor = torch.tensor(prefix_embeddings, dtype=torch.float32).to(param_device)
    prefix_class_labels_tensor = torch.tensor(prefix_class_labels, dtype=torch.long).to(param_device)
    
    # Truncate test dataset if requested
    if max_test_samples and max_test_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(max_test_samples))
        logger.info(f"TRUNCATED test dataset to {len(test_dataset)} examples for faster evaluation")
    
    # Ensure embedding_projector is on the right device
    if hasattr(model, 'embedding_projector'):
        model.embedding_projector = model.embedding_projector.to(param_device)
    
    # For results
    predictions = []
    ground_truth = []
    explanations = []
    probabilities = []
    
    # Switch to evaluation mode
    model.eval()
    
    # Test loop
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating with explanations"):
            # Get example
            example = test_dataset[i]
            label_id = example["label_id"]
            
            # Create prompt with placeholder tokens
            placeholder_tokens = " ".join(["_"] * 100)
            prompt = f"Predict the correct class for the given data.\n\n<PREFIX_START>{placeholder_tokens}<PREFIX_END>\n\nLook at the data patterns and predict the class.\n\nThe class is:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(param_device) for k, v in inputs.items()}
            
            try:
                # Get the current example's query embedding
                query_embedding = torch.tensor(np.array(example["query_embedding"]), dtype=torch.float32).to(param_device)
                
                # First, get the initial embedding matrix from the tokens
                input_embeds = model.base_model.get_input_embeddings()(inputs["input_ids"]).to(param_device)
                
                # Find positions of PREFIX_START and PREFIX_END tokens
                start_positions = (inputs["input_ids"] == prefix_start_id).nonzero(as_tuple=True)
                end_positions = (inputs["input_ids"] == prefix_end_id).nonzero(as_tuple=True)
                
                # Process each sequence in batch
                for seq_idx in range(inputs["input_ids"].shape[0]):
                    batch_start_positions = torch.where(start_positions[0] == seq_idx)[0]
                    batch_end_positions = torch.where(end_positions[0] == seq_idx)[0]
                    
                    # Process each PREFIX_START/END pair
                    for start_idx_pos, end_idx_pos in zip(batch_start_positions, batch_end_positions):
                        start_pos = start_positions[1][start_idx_pos]
                        end_pos = end_positions[1][end_idx_pos]
                        
                        if start_pos >= end_pos - 1:  # Need at least 1 token between markers
                            continue
                        
                        # Calculate how many tokens we have between markers
                        num_tokens = end_pos - start_pos - 1
                        
                        # Reserve space for the query embedding (10 tokens)
                        query_space = min(10, num_tokens // 3)  # Reserve up to 1/3 of available space, max 10 tokens
                        example_space = num_tokens - query_space
                        
                        # Project query embedding to model hidden size
                        projected_query = model.embedding_projector(query_embedding).to(param_device)
                        
                        # Get the embeddings and class labels for examples
                        embeddings = prefix_embeddings_tensor
                        class_labels = prefix_class_labels_tensor
                        
                        # Ensure we have a good mix of classes
                        # Get the unique classes in the prefix examples
                        unique_prefix_classes = torch.unique(class_labels).cpu().numpy()
                        logger.info(f"Unique classes in prefix data: {unique_prefix_classes}")
                        
                        # Safety check - if we don't have at least 2 classes, we need to log a warning
                        if len(unique_prefix_classes) < 2:
                            logger.warning(f"WARNING: Only {len(unique_prefix_classes)} classes found in prefix data! This may lead to poor performance.")
                            # Continue anyway - we'll still try with the limited classes
                        
                        # Determine how many example embeddings we can use based on remaining space
                        # Each example pair takes 2 tokens (embedding + class token)
                        num_examples = min(example_space // 2, embeddings.shape[0])
                        
                        # Make sure we have example indices that ensure class diversity
                        # By shuffling indices, we ensure different random examples are selected each time
                        indices = torch.randperm(len(class_labels))[:num_examples].to(param_device)
                        
                        # Project example embeddings to model hidden size
                        selected_embeddings = embeddings[indices]
                        selected_labels = class_labels[indices]
                        projected_examples = model.embedding_projector(selected_embeddings).to(param_device)
                        
                        # Double-check we have class diversity in selected examples
                        selected_classes = torch.unique(selected_labels).cpu().numpy()
                        logger.info(f"Selected {len(selected_classes)} unique classes for prefix examples: {selected_classes}")
                        
                        # Create a tensor to hold all our embeddings (query + examples)
                        all_embeddings = torch.zeros(
                            num_tokens,  # Total space between markers
                            model.config.hidden_size,
                            device=param_device
                        )
                        
                        # First, add the query embedding with repetition for emphasis
                        query_separator_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")  # Use EOS as separator if no special token
                        if "<QUERY>" in tokenizer.get_vocab():
                            query_separator_id = tokenizer.convert_tokens_to_ids("<QUERY>")
                            
                        # Make sure separator is on same device
                        query_separator = model.base_model.get_input_embeddings()(
                            torch.tensor([query_separator_id], device=param_device)
                        ).squeeze(0)
                        
                        # Add query separator and repeated query embedding
                        all_embeddings[0] = query_separator
                        for j in range(1, query_space - 1):
                            all_embeddings[j] = projected_query
                        all_embeddings[query_space - 1] = query_separator
                        
                        # Next, add the interleaved example embeddings and class tokens
                        example_offset = query_space
                        for j in range(len(indices)):
                            # Example embedding
                            all_embeddings[example_offset + j*2] = projected_examples[j]
                            
                            # Class token - make sure on same device
                            current_class_label = int(selected_labels[j])
                            class_token_id = class_token_ids[current_class_label]
                            class_token_embedding = model.base_model.get_input_embeddings()(
                                torch.tensor([class_token_id], device=param_device)
                            ).squeeze(0)
                            all_embeddings[example_offset + j*2 + 1] = class_token_embedding
                        
                        # Replace token embeddings with our custom embeddings
                        # +1 to skip the PREFIX_START token
                        input_embeds[seq_idx, start_pos+1:end_pos, :] = all_embeddings
                
                # Instead of generating tokens, get the logits for the next token directly
                outputs = model(
                    inputs_embeds=input_embeds,
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                
                # Get logits for the last token position
                logits = outputs.logits[:, -1, :]
                
                # Convert logits to probabilities using softmax
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Extract probabilities for the specified class tokens
                class_probs = torch.zeros(len(class_token_ids), device=param_device)
                for idx, token_id in enumerate(class_token_ids):
                    class_probs[idx] = probs[0, token_id]
                
                # Get the class with highest probability
                max_prob_idx = torch.argmax(class_probs).item()
                predicted_label_id = max_prob_idx
                
                # For logging purposes, get the token and probability
                max_prob_token_id = class_token_ids[max_prob_idx]
                max_prob_token = tokenizer.convert_ids_to_tokens(max_prob_token_id)
                max_prob = class_probs[max_prob_idx].item()
                
                # First, generate the class token only
                # Instead of using forced_decoder_ids (which is deprecated),
                # we'll directly use the predicted token as input
                
                # Generate the initial token text using the predicted token directly
                class_token_text = tokenizer.decode([max_prob_token_id], skip_special_tokens=False)
                
                # Create dummy token_output to bypass the generation step
                # This will just be the original input with our predicted class token appended
                token_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False) + class_token_text
                token_output = inputs["input_ids"].clone()  # Just use this as a placeholder since we only need token_text
                    
                # Now append the explanation prompt and generate the explanation
                # Create prompt for explanation generation
                explanation_prompt = get_explanation_prompt(explanation_type, class_token_text)
                
                # Create a new prompt with the generated class token and explanation request
                full_prompt = token_text + explanation_prompt
                
                # Tokenize the new prompt
                explanation_inputs = tokenizer(full_prompt, return_tensors="pt")
                explanation_inputs = {k: v.to(param_device) for k, v in explanation_inputs.items()}
                
                # We need to handle the embeddings for this new input too
                explanation_embeds = model.base_model.get_input_embeddings()(explanation_inputs["input_ids"]).to(param_device)
                
                # Find positions of PREFIX_START and PREFIX_END in the new input
                explanation_start_positions = (explanation_inputs["input_ids"] == prefix_start_id).nonzero(as_tuple=True)
                explanation_end_positions = (explanation_inputs["input_ids"] == prefix_end_id).nonzero(as_tuple=True)
                
                # If the token contains prefix placeholders, fill them the same way we did before
                if len(explanation_start_positions[0]) > 0 and len(explanation_end_positions[0]) > 0:
                    # The process is the same as before - using the same code for consistency
                    for seq_idx in range(explanation_inputs["input_ids"].shape[0]):
                        batch_start_positions = torch.where(explanation_start_positions[0] == seq_idx)[0]
                        batch_end_positions = torch.where(explanation_end_positions[0] == seq_idx)[0]
                        
                        for start_idx_pos, end_idx_pos in zip(batch_start_positions, batch_end_positions):
                            start_pos = explanation_start_positions[1][start_idx_pos]
                            end_pos = explanation_end_positions[1][end_idx_pos]
                            
                            if start_pos >= end_pos - 1:
                                continue
                            
                            # For explanation generation, we want to keep using the same prefix embeddings
                            # to maintain consistency with the prediction step
                            explanation_embeds[seq_idx, start_pos+1:end_pos, :] = all_embeddings
                            logger.info("Using same prefix embeddings for explanation generation")
                
                # Generate explanations
                explanation_config = {
                    "max_new_tokens": max_explanation_tokens,
                    "num_beams": 3,  # Use beam search for better quality explanations
                    "temperature": 0.7,  # Add some temperature for creativity
                    "do_sample": True,
                    "top_p": 0.9,  # Nucleus sampling for more coherent outputs
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                # Generate the explanation
                with torch.no_grad():
                    explanation_output = model.generate(
                        inputs_embeds=explanation_embeds,
                        attention_mask=explanation_inputs["attention_mask"],
                        **explanation_config
                    )
                    full_output_text = tokenizer.decode(explanation_output[0], skip_special_tokens=False)
                
                # Extract the explanation text - everything after "Explanation: "
                explanation_marker = explanation_prompt.strip()
                if explanation_marker in full_output_text:
                    explanation_start = full_output_text.find(explanation_marker) + len(explanation_marker)
                    explanation_text = full_output_text[explanation_start:].strip()
                else:
                    explanation_text = "Explanation: " + full_output_text.strip()
                
                # Store the results
                predictions.append(predicted_label_id)
                ground_truth.append(label_id)
                explanations.append(explanation_text)
                probabilities.append(max_prob)
                
                if i % 5 == 0:
                    logger.info(f"Example {i}:")
                    logger.info(f"Prediction: {predicted_label_id}, True: {label_id}, Probability: {max_prob:.4f}")
                    logger.info(f"Explanation: {explanation_text}")
                    logger.info("-" * 50)
            
            except Exception as e:
                logger.error(f"Error in prediction for example {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                predictions.append(-1)  # Unknown
                ground_truth.append(label_id)
                explanations.append("Error generating explanation")
                probabilities.append(0.0)
    
    # Calculate accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions)
    
    logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # Calculate per-class metrics if possible
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Filter out "unknown" predictions (-1)
        valid_indices = [i for i, p in enumerate(predictions) if p != -1]
        valid_pred = [predictions[i] for i in valid_indices]
        valid_truth = [ground_truth[i] for i in valid_indices]
        
        if valid_pred:
            # Generate classification report with correctly sized target_names
            # Get the unique classes in the ground truth to determine actual number of classes
            unique_classes = sorted(set(valid_truth))
            class_names = [f"Class {i}" for i in unique_classes]

            # Log the number of classes to help with debugging
            logger.info(f"Dataset has {len(unique_classes)} unique classes: {unique_classes}")
            logger.info(f"Using {len(class_names)} target names: {class_names}")

            # Use only target_names that match the dataset's actual classes
            report = classification_report(valid_truth, valid_pred, target_names=class_names, output_dict=True)

            # Calculate confusion matrix
            cm = confusion_matrix(valid_truth, valid_pred)
            
            logger.info("Per-class metrics computed")
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'ground_truth': ground_truth,
                'probabilities': probabilities,
                'explanations': explanations,
                'classification_report': report,
                'confusion_matrix': cm
            }
        
    except (ImportError, ValueError) as e:
        logger.warning(f"Could not compute detailed metrics: {e}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'probabilities': probabilities,
        'explanations': explanations
    }

def process_dataset(args, model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, dataset_info):
    """
    Process and evaluate a single dataset with explanations.
    
    Args:
        args: Command line arguments
        model: Pretrained model
        tokenizer: Tokenizer for the model
        prefix_start_id, prefix_end_id, class_token_ids: Token IDs
        dataset_info: Dictionary containing dataset information and data
        
    Returns:
        Dictionary containing the evaluation results
    """
    import logging
    logger = logging.getLogger(__name__)
    dataset_name = dataset_info['name']
    dataset_id = dataset_info['id']
    X = dataset_info['X']
    y = dataset_info['y']
    categorical_indicator = dataset_info['categorical_indicator']
    
    # Preprocess features if needed
    X = preprocess_features(X, categorical_indicator)
    
    # Split into train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    logger.info(f"Dataset {dataset_name} shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Handle embedding cache directory
    cache_dir = None
    if args.embedding_cache_dir.lower() != 'none':
        cache_dir = args.embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using embedding cache directory: {cache_dir}")
    
    # Get TabPFN embeddings
    logger.info(f"Computing TabPFN embeddings for dataset {dataset_name}")
    train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
        X_train, y_train, X_val, X_test,
        embedding_size=args.embedding_size,
        cache_dir=cache_dir,
        dataset_name=str(dataset_id),
        force_recompute=args.force_recompute_embeddings,
        seed=args.seed
    )
    
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(args.output_dir, f"{dataset_name}_{dataset_id}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Create the dataset for evaluation
    logger.info(f"Creating evaluation dataset for {dataset_name}")
    _, _, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        X_train, y_train_sample, X_val, y_val, X_test, y_test,
        train_embeddings, val_embeddings, test_embeddings,
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir=dataset_output_dir,
        num_few_shot_examples=args.num_few_shot_examples
    )
    
    # Evaluate the model on test set with explanations
    logger.info(f"Evaluating model on {dataset_name} test set with {args.explanation_type} explanations")
    results = evaluate_with_explanations(
        model, tokenizer, test_dataset, 
        label_encoder, prefix_start_id, prefix_end_id,
        class_token_ids, prefix_data_file, 
        explanation_type=args.explanation_type,
        max_explanation_tokens=args.max_explanation_tokens,
        max_test_samples=args.max_test_samples
    )
    
    logger.info(f"Evaluation complete. {dataset_name} test accuracy: {results['accuracy']:.4f}")
    
    # Create result summary
    result_summary = {
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'model_path': args.model_path,
        'accuracy': float(results['accuracy']),
        'explanation_type': args.explanation_type,
        'num_classes': len(np.unique(y)),
        'num_features': X.shape[1],
        'num_samples': X.shape[0],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    }
    
    # Add classification report if available
    if 'classification_report' in results:
        result_summary['classification_report'] = results['classification_report']
    
    # Add confusion matrix if available
    if 'confusion_matrix' in results:
        result_summary['confusion_matrix'] = results['confusion_matrix'].tolist()
    
    # Save as JSON
    results_file = os.path.join(dataset_output_dir, f"{dataset_name}_{args.explanation_type}_evaluation_results.json")
    with open(results_file, "w") as f:
        # Remove explanations from main results file to keep it small
        result_summary_without_explanations = result_summary.copy()
        json.dump(result_summary_without_explanations, f, indent=2)
    
    logger.info(f"Saved detailed results to {results_file}")
    
    # Save explanations separately if requested
    if args.save_explanations:
        # Create a detailed results file with explanations for each example
        detailed_results = []
        for i in range(len(results['predictions'])):
            example_result = {
                'example_id': i,
                'prediction': int(results['predictions'][i]),
                'ground_truth': int(results['ground_truth'][i]),
                'probability': float(results['probabilities'][i]),
                'explanation': results['explanations'][i],
                'correct': results['predictions'][i] == results['ground_truth'][i]
            }
            detailed_results.append(example_result)
        
        # Save all explanations
        explanations_file = os.path.join(dataset_output_dir, f"{dataset_name}_{args.explanation_type}_explanations.json")
        with open(explanations_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Saved explanations to {explanations_file}")
    
    # Log to W&B if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        # Log dataset results
        wandb.log({
            f"{dataset_name}/accuracy": result_summary['accuracy'],
            f"{dataset_name}/num_classes": result_summary['num_classes'],
            f"{dataset_name}/num_features": result_summary['num_features'],
            f"{dataset_name}/num_samples": result_summary['num_samples'],
            f"{dataset_name}/explanation_type": args.explanation_type
        })
        
        # Log per-class metrics if available
        if 'classification_report' in result_summary:
            report = result_summary['classification_report']
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):  # Skip avg/total sections
                    wandb.log({
                        f"{dataset_name}/class_{class_name}/precision": metrics['precision'],
                        f"{dataset_name}/class_{class_name}/recall": metrics['recall'],
                        f"{dataset_name}/class_{class_name}/f1": metrics['f1-score']
                    })
        
        # Log confusion matrix if available
        if 'confusion_matrix' in result_summary:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import ConfusionMatrixDisplay
            
            # Create a figure for the confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = np.array(result_summary['confusion_matrix'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format=".0f", ax=ax)
            plt.title(f"Confusion Matrix - {dataset_name}")
            
            # Log to W&B
            wandb.log({f"{dataset_name}/confusion_matrix": wandb.Image(fig)})
            plt.close(fig)
            
            # Log sample explanations
            if 'explanations' in results:
                # Sample up to 10 explanations (5 correct, 5 incorrect if available)
                correct_indices = [i for i, (p, g) in enumerate(zip(results['predictions'], results['ground_truth'])) if p == g]
                incorrect_indices = [i for i, (p, g) in enumerate(zip(results['predictions'], results['ground_truth'])) if p != g]
                
                # Take up to 5 from each category
                sampled_correct = correct_indices[:5] if correct_indices else []
                sampled_incorrect = incorrect_indices[:5] if incorrect_indices else []
                sampled_indices = sampled_correct + sampled_incorrect
                
                # Create a table for W&B
                explanation_data = []
                for i in sampled_indices:
                    explanation_data.append([
                        int(i),
                        int(results['predictions'][i]),
                        int(results['ground_truth'][i]),
                        "✓" if results['predictions'][i] == results['ground_truth'][i] else "✗",
                        float(results['probabilities'][i]),
                        results['explanations'][i]
                    ])
                
                # Log as a table
                explanation_table = wandb.Table(
                    columns=["Example ID", "Prediction", "Ground Truth", "Correct", "Probability", "Explanation"],
                    data=explanation_data
                )
                wandb.log({f"{dataset_name}/sample_explanations": explanation_table})
    
    # Print class-specific metrics if available
    if 'classification_report' in results:
        report = results['classification_report']
        logger.info(f"\nPer-class metrics for {dataset_name}:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):  # Skip avg/total sections
                logger.info(f"{class_name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
    
    # Print some example explanations
    logger.info(f"\nExample explanations for {dataset_name}:")
    num_examples = min(5, len(results['explanations']))
    for i in range(num_examples):
        correct = "✓" if results['predictions'][i] == results['ground_truth'][i] else "✗"
        logger.info(f"Example {i} ({correct}):")
        logger.info(f"Pred: {results['predictions'][i]}, True: {results['ground_truth'][i]}")
        logger.info(f"Explanation: {results['explanations'][i]}")
        logger.info("-" * 50)
    
    return result_summary

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    from marvis.utils import set_seed_with_args
    set_seed_with_args(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"eval_explanations_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    logger.info(f"Arguments: {args}")
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_datasets(args)
    logger.info(f"Loaded {len(datasets)} datasets for evaluation")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) package not available. Will not log to W&B.")
        else:
            logger.info("Initializing Weights & Biases...")
            wandb_run_name = args.wandb_name or f"explanations_{timestamp}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                config=vars(args)
            )
            # Log system info
            wandb.log({
                "system/device": args.device,
                "system/cuda_available": torch.cuda.is_available(),
                "system/cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "evaluation/num_datasets": len(datasets),
                "evaluation/explanation_type": args.explanation_type
            })
    
    # Load the model
    logger.info(f"Loading pretrained model from {args.model_path}")
    model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = load_pretrained_model(
        args.model_path, 
        device_map=args.device,
        embedding_size=args.embedding_size
    )
    
    # Process each dataset
    all_results = []
    for i, dataset in enumerate(datasets):
        logger.info(f"Processing dataset {i+1}/{len(datasets)}: {dataset['name']} (ID: {dataset['id']})")
        
        try:
            result = process_dataset(
                args, model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
                dataset
            )
            all_results.append(result)
            logger.info(f"Successfully processed dataset {dataset['name']}")
        except Exception as e:
            logger.error(f"Error processing dataset {dataset['name']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Add a placeholder result to maintain ordering
            all_results.append({
                'dataset_name': dataset['name'],
                'error': str(e)
            })
    
    # Compute and save overall metrics
    logger.info("Computing overall metrics...")
    
    # Calculate average accuracy
    valid_results = [r for r in all_results if 'accuracy' in r]
    
    if valid_results:
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        logger.info(f"Average accuracy across {len(valid_results)} datasets: {avg_accuracy:.4f}")
        
        # Save summary JSON
        summary = {
            'model_path': args.model_path,
            'explanation_type': args.explanation_type,
            'num_datasets': len(valid_results),
            'average_accuracy': float(avg_accuracy),
            'dataset_accuracies': {r['dataset_name']: r['accuracy'] for r in valid_results},
            'timestamp': timestamp
        }
        
        summary_file = os.path.join(args.output_dir, f"summary_{args.explanation_type}_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Log to W&B if enabled
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "overall/num_datasets": len(valid_results),
                "overall/average_accuracy": avg_accuracy
            })
            
            # Create accuracy comparison plot
            import matplotlib.pyplot as plt
            
            # Dataset comparison plot
            fig, ax = plt.subplots(figsize=(12, 6))
            dataset_names = [r['dataset_name'] for r in valid_results]
            accuracies = [r['accuracy'] for r in valid_results]
            
            # Sort by accuracy for better visualization
            sorted_indices = np.argsort(accuracies)[::-1]  # Sort in descending order
            sorted_names = [dataset_names[i] for i in sorted_indices]
            sorted_accs = [accuracies[i] for i in sorted_indices]
            
            ax.bar(range(len(sorted_names)), sorted_accs)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Accuracy across datasets ({args.explanation_type} explanations)')
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            ax.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'Average: {avg_accuracy:.4f}')
            ax.legend()
            
            plt.tight_layout()
            wandb.log({"overall/accuracy_plot": wandb.Image(fig)})
            plt.close(fig)
    else:
        logger.warning("No valid results found. Cannot compute average metrics.")
    
    # Finalize W&B run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info(f"Evaluation with explanations complete! Results saved to {args.output_dir}")
    
    # Return the average accuracy for potential scripting use
    return avg_accuracy if valid_results else None

if __name__ == "__main__":
    main()