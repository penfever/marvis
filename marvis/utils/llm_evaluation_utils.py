"""
Utility functions for LLM baseline evaluations.
Extracted from evaluate_llm_baselines.py to reduce code duplication.
"""

import logging
import numpy as np
import pandas as pd
from typing import Set, List, Dict, Any, Optional, Tuple
import json
import os
import math
import torch


def drop_feature_for_oom(dropped_features: Set[int], num_features: int, logger: logging.Logger, min_features: int = 1) -> bool:
    """
    Drop one more feature when encountering OOM. Drops features from the end.
    
    Args:
        dropped_features: Set of already dropped feature indices
        num_features: Total number of features
        logger: Logger instance
        min_features: Minimum number of features to keep (default: 1)
        
    Returns:
        bool: True if a feature was dropped, False if no more features to drop
    """
    # Check if we've reached the minimum number of features
    remaining_features = num_features - len(dropped_features)
    if remaining_features <= min_features:
        logger.warning(f"Cannot drop more features: only {remaining_features} features remain (minimum: {min_features})")
        return False
    
    # Find the next feature to drop (from the end, working backwards)
    for idx in range(num_features - 1, -1, -1):
        if idx not in dropped_features:
            dropped_features.add(idx)
            logger.warning(f"Dropping feature at index {idx} due to OOM. Total dropped: {len(dropped_features)}, remaining: {num_features - len(dropped_features)}")
            return True
    return False


def is_oom_error(error: Exception) -> bool:
    """
    Check if an exception is an out-of-memory error.
    
    Args:
        error: The exception to check
        
    Returns:
        bool: True if this is an OOM error
    """
    error_str = str(error).lower()
    return ("out of memory" in error_str or 
            "oom" in error_str or 
            ("cuda" in error_str and "memory" in error_str))


def create_tabllm_note(X_row, attribute_names: List[str], 
                      dataset_name: str = "unknown", 
                      semantic_info: Optional[Dict] = None, 
                      dropped_features: Optional[Set[int]] = None) -> str:
    """
    Create TabLLM-style textual note from tabular data row using semantic information when available.
    
    Args:
        X_row: Data row
        attribute_names: List of attribute names
        dataset_name: Name of the dataset
        semantic_info: Semantic information dictionary
        dropped_features: Set of feature indices to drop (for OOM handling)
        
    Returns:
        str: TabLLM-style note
    """
    # Convert row to simple format that matches our generated examples
    note_parts = []
    
    # Convert row to dict for easier access
    if hasattr(X_row, 'iloc'):
        row_values = X_row.values if hasattr(X_row, 'values') else list(X_row)
    else:
        row_values = list(X_row)
    
    # Create TabLLM-style notes using semantic descriptions when available
    for idx, (attr_name, value) in enumerate(zip(attribute_names, row_values)):
        # Skip dropped features
        if dropped_features and idx in dropped_features:
            continue
            
        if semantic_info and 'columns' in semantic_info:
            # Look for semantic description for this attribute
            semantic_desc = None
            for col in semantic_info['columns']:
                if col['name'] == attr_name:
                    semantic_desc = col['semantic_description']
                    break
            
            if semantic_desc:
                # Use semantic description with value
                note_parts.append(f"{semantic_desc}: {value}")
            else:
                # Fall back to attribute name
                note_parts.append(f"{attr_name}: {value}")
        else:
            # No semantic info available
            note_parts.append(f"{attr_name}: {value}")
    
    # Join with semicolons to match TabLLM format
    return "; ".join(note_parts)


def regenerate_few_shot_examples(X_train, y_train, example_indices: List[int],
                               attribute_names: List[str], dataset_name: str,
                               semantic_info: Optional[Dict], dropped_features: Set[int],
                               class_to_name: Dict) -> List[Tuple[str, str]]:
    """
    Regenerate few-shot examples with dropped features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        example_indices: Indices of examples to use
        attribute_names: Feature names
        dataset_name: Dataset name
        semantic_info: Semantic information
        dropped_features: Set of dropped feature indices
        class_to_name: Mapping from class values to names
        
    Returns:
        List of (note, label) tuples
    """
    few_shot_examples = []
    for idx in example_indices:
        x_example = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_example = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        note = create_tabllm_note(x_example, attribute_names, dataset_name, semantic_info, dropped_features)
        class_label = class_to_name.get(y_example, str(y_example))
        few_shot_examples.append((note, class_label))
    return few_shot_examples


def safe_json_dump(data: Any, filepath: str, logger: logging.Logger, 
                   minimal_fallback: bool = False) -> bool:
    """
    Safely save data to JSON with multiple fallback options.
    
    Args:
        data: Data to save
        filepath: Path to save to
        logger: Logger instance
        minimal_fallback: Whether to use minimal serialization on failure
        
    Returns:
        bool: True if successful
    """
    def convert_to_serializable(obj):
        """Convert non-JSON-serializable objects."""
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    try:
        # First attempt: direct JSON dump
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except (TypeError, ValueError) as e:
        logger.warning(f"Initial JSON serialization failed: {e}")
        
        # Second attempt: convert numpy types
        try:
            converted_data = json.loads(
                json.dumps(data, default=convert_to_serializable)
            )
            with open(filepath, 'w') as f:
                json.dump(converted_data, f, indent=2)
            logger.info("Successfully saved with numpy conversion")
            return True
        except Exception as e2:
            logger.warning(f"Numpy conversion failed: {e2}")
            
            if minimal_fallback:
                # Third attempt: minimal data only
                try:
                    minimal_data = {
                        'model_name': data.get('model_name', 'unknown'),
                        'dataset_name': data.get('dataset_name', 'unknown'),
                        'dataset_id': data.get('dataset_id', 'unknown'),
                        'accuracy': float(data.get('accuracy', 0.0)),
                        'balanced_accuracy': float(data.get('balanced_accuracy', 0.0)),
                        'error': data.get('error', None),
                        'timeout': data.get('timeout', False),
                        'completed_samples': int(data.get('completed_samples', 0)),
                        'num_test_samples': int(data.get('num_test_samples', 0))
                    }
                    with open(filepath, 'w') as f:
                        json.dump(minimal_data, f, indent=2)
                    logger.warning("Saved minimal results only due to serialization issues")
                    return True
                except Exception as e3:
                    logger.error(f"Even minimal save failed: {e3}")
                    return False
            else:
                return False


def load_template_data(dataset_name: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Load template data for a dataset from YAML file.
    
    Args:
        dataset_name: Name of the dataset
        logger: Logger instance
        
    Returns:
        Template data dictionary or None
    """
    template_file = f"templates/templates_{dataset_name}.yaml"
    if os.path.exists(template_file):
        try:
            import yaml
            with open(template_file, 'r') as f:
                template_data = yaml.safe_load(f)
            logger.info(f"Loaded template for {dataset_name}")
            return template_data
        except Exception as e:
            logger.warning(f"Could not load template for {dataset_name}: {e}")
    return None


def load_semantic_info(dataset_name: str, data_dir: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Load semantic information for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing semantic data
        logger: Logger instance
        
    Returns:
        Semantic info dictionary or None
    """
    semantic_file = os.path.join(data_dir, f"{dataset_name}_semantic.json")
    if os.path.exists(semantic_file):
        try:
            with open(semantic_file, 'r') as f:
                semantic_info = json.load(f)
            logger.info(f"Loaded semantic information for {dataset_name}")
            return semantic_info
        except Exception as e:
            logger.warning(f"Could not load semantic info for {dataset_name}: {e}")
    return None


def apply_feature_reduction(X_train, y_train, X_test, dataset: Dict, args, logger: logging.Logger) -> Tuple:
    """
    Apply feature selection for large feature spaces.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        dataset: Dataset dictionary
        args: Arguments namespace
        logger: Logger instance
        
    Returns:
        Tuple of (X_train_reduced, X_test_reduced, dataset_reduced, selected_indices)
    """
    original_num_features = X_train.shape[1]
    feature_threshold = getattr(args, 'feature_selection_threshold', 20)
    
    if original_num_features <= feature_threshold:
        return X_train, X_test, dataset, None
    
    logger.info(f"Dataset has {original_num_features} features (> {feature_threshold}) - applying feature selection")
    
    try:
        from .feature_selection_utils import (
            select_features_for_token_limit, 
            create_reduced_dataset,
            test_feature_selection
        )
        
        # Use a simple tokenizer for estimation
        from transformers import AutoTokenizer
        try:
            tokenizer_temp = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")
        except:
            tokenizer_temp = AutoTokenizer.from_pretrained("gpt2")
        
        # Test different token limits
        test_results = test_feature_selection(
            X_train, y_train, 
            dataset["attribute_names"],
            tokenizer_temp,
            getattr(args, 'num_few_shot_examples', 16),
            categorical_indicator=dataset.get("categorical_indicator", None)
        )
        
        # Log test results
        for result in test_results:
            logger.info(f"Token limit {result['token_limit']}: "
                       f"{result['num_features_selected']} features selected, "
                       f"~{result['estimated_tokens']} tokens ({result['utilization']:.1%} utilization)")
        
        # Select features for our target token limit
        selected_indices, estimated_tokens = select_features_for_token_limit(
            X_train, y_train,
            dataset["attribute_names"],
            tokenizer_temp,
            num_few_shot_examples=getattr(args, 'num_few_shot_examples', 16),
            max_tokens=getattr(args, 'max_context_length', 8192),
            categorical_indicator=dataset.get("categorical_indicator", None),
            prioritize_semantic=True
        )
        
        # Debug information
        logger.info(f"Dataset has {X_train.shape[1]} features, {len(dataset['attribute_names'])} attribute names")
        logger.info(f"Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        # Ensure selected indices are within bounds for the data
        max_feature_index = X_train.shape[1] - 1
        max_name_index = len(dataset["attribute_names"]) - 1
        
        # Check bounds for both data and feature names
        valid_indices = [i for i in selected_indices if i <= max_feature_index and i <= max_name_index]
        if len(valid_indices) != len(selected_indices):
            invalid_indices = [i for i in selected_indices if i > max_feature_index or i > max_name_index]
            logger.warning(f"Some selected indices are out of bounds: {invalid_indices}")
            logger.warning(f"Max data index: {max_feature_index}, max name index: {max_name_index}")
            selected_indices = valid_indices
        
        # Create reduced datasets
        X_train_reduced, reduced_feature_names = create_reduced_dataset(
            X_train, selected_indices, dataset["attribute_names"]
        )
        X_test_reduced, _ = create_reduced_dataset(
            X_test, selected_indices, dataset["attribute_names"]
        )
        
        # Update dataset info for reduced features
        dataset_reduced = dataset.copy()
        dataset_reduced["attribute_names"] = reduced_feature_names
        if "categorical_indicator" in dataset:
            # Ensure selected indices are within bounds
            categorical_indicator = dataset["categorical_indicator"]
            max_valid_index = len(categorical_indicator) - 1
            valid_indices = [i for i in selected_indices if i <= max_valid_index]
            if len(valid_indices) != len(selected_indices):
                logger.warning(f"Some selected indices ({len(selected_indices) - len(valid_indices)}) are out of bounds for categorical_indicator (max index: {max_valid_index})")
            dataset_reduced["categorical_indicator"] = [
                categorical_indicator[i] for i in valid_indices
            ]
        
        logger.info(f"Reduced from {original_num_features} to {len(selected_indices)} features")
        logger.info(f"Selected features (first 10): {reduced_feature_names[:10]}")
        
        return X_train_reduced, X_test_reduced, dataset_reduced, selected_indices
        
    except Exception as e:
        logger.warning(f"Feature selection failed: {e}. Using all features.")
        return X_train, X_test, dataset, None


def predict_with_jolt_logprobs(full_prompt: str, answer_choices: List[str], tokenizer, model, 
                              args, logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    Predict using JOLT's log probability extraction method.
    
    Args:
        full_prompt: The complete prompt without answer
        answer_choices: List of possible answer choices
        tokenizer: Model tokenizer
        model: Model instance
        args: Arguments object with max_context_length
        logger: Logger instance
        
    Returns:
        Tuple of (predicted_class, class_log_probs)
    """
    try:
        # Import JOLT compute_nll functionality
        import sys
        import os
        
        # Try multiple possible paths for the JOLT official_jolt directory
        possible_jolt_paths = [
            # Standard path from utils directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "examples", "tabular", "llm_baselines", "jolt", "official_jolt"),
            # Alternative path for different project structures
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        "examples", "tabular", "llm_baselines", "jolt", "official_jolt"),
            # Check if we're in a different working directory
            os.path.join(os.getcwd(), "examples", "tabular", "llm_baselines", "jolt", "official_jolt"),
            os.path.join(os.getcwd(), "marvis", "examples", "tabular", "llm_baselines", "jolt", "official_jolt"),
            # Direct path if available in environment
            os.path.join(os.path.dirname(__file__), "..", "..", "examples", "tabular", "llm_baselines", "jolt", "official_jolt"),
            # Legacy paths for backward compatibility
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "examples", "llm_baselines", "jolt", "official_jolt"),
            os.path.join(os.getcwd(), "examples", "llm_baselines", "jolt", "official_jolt")
        ]
        
        jolt_path = None
        for path in possible_jolt_paths:
            if os.path.exists(path):
                jolt_path = path
                logger.debug(f"Found JOLT path at: {jolt_path}")
                break
        
        if jolt_path is None:
            logger.warning(f"JOLT official_jolt directory not found. Tried paths: {possible_jolt_paths}")
            raise ImportError(f"JOLT official_jolt directory not found. Searched: {possible_jolt_paths}")
        
        if jolt_path not in sys.path:
            sys.path.insert(0, jolt_path)
            logger.debug(f"Added JOLT path to sys.path: {jolt_path}")
        
        # Check if compute_nll.py exists
        compute_nll_path = os.path.join(jolt_path, "compute_nll.py")
        if not os.path.exists(compute_nll_path):
            logger.warning(f"compute_nll.py not found at: {compute_nll_path}")
            raise ImportError(f"compute_nll.py not found at {compute_nll_path}")
        
        from compute_nll import _get_y_logprobs_categorical
        
        # Prepare inputs for each answer choice using JOLT's format
        choice_inputs = []
        choice_ranges = []
        
        for answer_choice in answer_choices:
            # Create the full prompt with answer
            prompt_with_answer = full_prompt + f" {answer_choice}"
            
            # Tokenize the full sequence
            input_tokens = tokenizer.encode(prompt_with_answer)
            choice_inputs.append(input_tokens)
            
            # Calculate the range where the answer appears
            prompt_tokens = tokenizer.encode(full_prompt)
            answer_start = len(prompt_tokens)
            answer_end = len(input_tokens)
            
            # JOLT expects ranges in a specific format: [[start, end]]
            choice_ranges.append([[answer_start, answer_end]])
        
        # Check if the inputs are too large for JOLT processing
        max_input_length = max(len(tokens) for tokens in choice_inputs)
        estimated_memory_gb = (len(choice_inputs) * max_input_length * 8) / (1024**3)
        
        if estimated_memory_gb > 5.0:  # If > 5GB, too large for JOLT
            logger.warning(f"JOLT inputs too large ({estimated_memory_gb:.2f}GB), falling back")
            raise RuntimeError("Tensor too large for JOLT processing")
        
        # Clear cache before JOLT processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create a mock args object with required attributes for the function
        class MockArgs:
            pass
        mock_args = MockArgs()
        
        y_logprobs = _get_y_logprobs_categorical(
            mock_args, tokenizer, model, choice_inputs, choice_ranges
        )
        
        # Extract log probabilities for each choice
        class_log_probs = {}
        for j, answer_choice in enumerate(answer_choices):
            if j < len(y_logprobs) and len(y_logprobs[j]) > 0:
                # Sum the log probabilities for all tokens in the answer
                answer_log_prob = y_logprobs[j][0].sum().item()
                
                # Check for invalid log probability (NaN or inf)
                if math.isnan(answer_log_prob):
                    logger.warning(f"Invalid log probability (nan) for answer '{answer_choice}'. Raising exception to trigger fallback.")
                    raise ValueError("Invalid log probability (NaN)")
                elif math.isinf(answer_log_prob):
                    logger.warning(f"Invalid log probability (inf) for answer '{answer_choice}'. Using default value.")
                    answer_log_prob = -1e6  # Very low probability as fallback
                
                class_log_probs[answer_choice] = answer_log_prob
            else:
                logger.warning(f"Could not get log probabilities for choice '{answer_choice}'")
                class_log_probs[answer_choice] = -1e6
        
        # Select the class with highest log probability
        predicted_class = max(class_log_probs, key=class_log_probs.get)
        
        return predicted_class, class_log_probs
        
    except Exception as e:
        logger.warning(f"JOLT logprob method failed: {e}")
        raise


def predict_with_simple_logprobs(full_prompt: str, answer_choices: List[str], tokenizer, model,
                                args, logger: logging.Logger) -> Tuple[str, Dict[str, float]]:
    """
    Predict using simple log probability extraction method.
    
    Args:
        full_prompt: The complete prompt without answer
        answer_choices: List of possible answer choices
        tokenizer: Model tokenizer
        model: Model instance
        args: Arguments object with max_context_length
        logger: Logger instance
        
    Returns:
        Tuple of (predicted_class, class_log_probs)
    """
    try:
        class_log_probs = {}
        
        for answer_choice in answer_choices:
            # Tokenize the prompt
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_context_length
            )
            
            # Tokenize the answer separately to get its tokens
            answer_tokens = tokenizer(
                f" {answer_choice}",
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Move to device
            if next(model.parameters()).device != torch.device("cpu"):
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
                answer_tokens = {k: v.to(next(model.parameters()).device) for k, v in answer_tokens.items()}
            
            with torch.no_grad():
                # Get model outputs
                outputs = model(**inputs)
            
            # Get logits for the last token position
            last_token_logits = outputs.logits[0, -1, :]
            
            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
            
            # Calculate the log probability of the first token of the answer
            first_answer_token = answer_tokens['input_ids'][0, 0]
            answer_log_prob = log_probs[first_answer_token].item()
            
            # Check for invalid log probability (NaN or inf)
            if math.isnan(answer_log_prob):
                logger.warning(f"Invalid log probability (nan) for answer '{answer_choice}'. Raising exception to trigger fallback.")
                raise ValueError("Invalid log probability (NaN)")
            elif math.isinf(answer_log_prob):
                logger.warning(f"Invalid log probability (inf) for answer '{answer_choice}'. Using default value.")
                answer_log_prob = -1e6  # Very low probability as fallback
            
            class_log_probs[answer_choice] = answer_log_prob
        
        # Select the class with highest log probability
        predicted_class = max(class_log_probs, key=class_log_probs.get)
        
        return predicted_class, class_log_probs
        
    except Exception as e:
        logger.warning(f"Simple logprob method failed: {e}")
        raise


def predict_with_generation(full_prompt: str, answer_choices: List[str], tokenizer, model, 
                          args, logger: logging.Logger, selected_examples: Optional[List] = None,
                          question: Optional[str] = None) -> Tuple[str, str]:
    """
    Predict using generative method with answer matching.
    
    Args:
        full_prompt: The complete prompt without answer
        answer_choices: List of possible answer choices
        tokenizer: Model tokenizer
        model: Model instance
        args: Arguments object with max_context_length
        logger: Logger instance
        selected_examples: Optional list of few-shot examples for fallback reduction
        question: Optional question string for fallback prompt reconstruction
        
    Returns:
        Tuple of (predicted_class, generated_text)
    """
    
    def try_generation(prompt):
        """Try generation with a given prompt, return generated text or None if failed."""
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_context_length - 20  # Reserve space for answer
            )
            
            # Move to device
            if next(model.parameters()).device != torch.device("cpu"):
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            
            # Generate response with simple, fast parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,  # Deterministic and fast
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            if len(outputs[0]) > input_length:
                generated_tokens = outputs[0][input_length:]
                return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return None
    
    # First attempt: Try with the full prompt
    generation_prompt = full_prompt + " Answer:"
    generated_text = try_generation(generation_prompt)
    
    # Second attempt: If empty or failed, try with reduced few-shot examples
    if not generated_text or len(generated_text.strip()) == 0:
        if selected_examples and len(selected_examples) > 2 and question:
            # Remove 50% of examples from the middle
            num_to_keep = len(selected_examples) // 2
            reduced_examples = selected_examples[:num_to_keep//2] + selected_examples[-num_to_keep//2:]
            
            # Reconstruct prompt with fewer examples - need task description
            task_description = f"Task: Given tabular data examples, classify each instance into one of the following categories: {', '.join(answer_choices)}.\\n\\nExamples:\\n\\n"
            
            reduced_example_parts = []
            for note, label in reduced_examples:
                example_prompt = f"{note}\\n\\n{question}\\nAnswer: {label}"
                reduced_example_parts.append(example_prompt)
            
            if reduced_example_parts:
                examples_section = "\\n\\n".join(reduced_example_parts)
                # Extract test prompt from full_prompt
                test_prompt_start = full_prompt.rfind("New Instance:\\n")
                if test_prompt_start != -1:
                    test_prompt = full_prompt[test_prompt_start:]
                    reduced_full_prompt = f"{task_description}{examples_section}\\n\\n{test_prompt}"
                    reduced_generation_prompt = reduced_full_prompt + " Answer:"
                    
                    logger.debug(f"Retrying generation with {len(reduced_examples)} examples (reduced from {len(selected_examples)})")
                    generated_text = try_generation(reduced_generation_prompt)
        
        # If still empty, use minimal fallback
        if not generated_text or len(generated_text.strip()) == 0:
            generated_text = "[generation_empty]"
    
    # Find best match among answer choices
    best_match = None
    if generated_text and generated_text != "[generation_empty]":
        # Try exact substring matching first
        for choice in answer_choices:
            if str(choice).lower() in generated_text.lower():
                best_match = choice
                break
        
        # Try first word matching
        if best_match is None:
            generated_words = generated_text.split()
            if generated_words:
                first_word = generated_words[0].strip('.,!?')
                for choice in answer_choices:
                    if first_word.lower() == str(choice).lower():
                        best_match = choice
                        break
    
    # Final fallback: use first answer choice
    if best_match is None:
        best_match = answer_choices[0]
        logger.debug(f"No match found for '{generated_text}', using fallback: {best_match}")
    
    return best_match, generated_text


def unified_llm_predict(full_prompt: str, answer_choices: List[str], tokenizer, model, args, 
                       logger: logging.Logger, selected_examples: Optional[List] = None,
                       question: Optional[str] = None, test_first_sample: bool = True) -> Dict[str, Any]:
    """
    Unified prediction function with fallback chain: JOLT logprobs -> Simple logprobs -> Generation.
    
    Args:
        full_prompt: The complete prompt without answer
        answer_choices: List of possible answer choices
        tokenizer: Model tokenizer
        model: Model instance
        args: Arguments object with max_context_length
        logger: Logger instance
        selected_examples: Optional list of few-shot examples for generation fallback
        question: Optional question string for generation fallback
        test_first_sample: Whether to test methods on first sample (default: True)
        
    Returns:
        Dictionary with prediction results and method used
    """
    
    # Cache to store which methods are viable for this dataset/model combination
    if not hasattr(unified_llm_predict, '_method_cache'):
        unified_llm_predict._method_cache = {}
    
    # Create a cache key based on model and dataset characteristics
    cache_key = f"{id(model)}_{len(answer_choices)}_{args.max_context_length}_{len(full_prompt)//100}"
    
    # Test methods on first sample if not already cached and test_first_sample is True
    if test_first_sample and cache_key not in unified_llm_predict._method_cache:
        logger.info("Testing prediction methods for viability...")
        viable_methods = _test_prediction_methods(
            full_prompt, answer_choices, tokenizer, model, args, logger, 
            selected_examples, question
        )
        unified_llm_predict._method_cache[cache_key] = viable_methods
        logger.info(f"Viable methods for this dataset: {viable_methods}")
    elif cache_key in unified_llm_predict._method_cache:
        viable_methods = unified_llm_predict._method_cache[cache_key]
    else:
        # If not testing, assume all methods are viable
        viable_methods = ['jolt_logprobs', 'simple_logprobs', 'generation']
    
    # Try methods in order of preference, but only if they're viable
    if 'jolt_logprobs' in viable_methods:
        try:
            predicted_class, class_log_probs = predict_with_jolt_logprobs(
                full_prompt, answer_choices, tokenizer, model, args, logger
            )
            return {
                'predicted_class': predicted_class,
                'method': 'jolt_logprobs',
                'class_log_probs': class_log_probs,
                'generated_text': None
            }
        except Exception as e:
            logger.debug(f"JOLT logprob method failed: {e}. Trying next viable method.")
    
    if 'simple_logprobs' in viable_methods:
        try:
            predicted_class, class_log_probs = predict_with_simple_logprobs(
                full_prompt, answer_choices, tokenizer, model, args, logger
            )
            return {
                'predicted_class': predicted_class,
                'method': 'simple_logprobs',
                'class_log_probs': class_log_probs,
                'generated_text': None
            }
        except Exception as e:
            logger.debug(f"Simple logprob method failed: {e}. Trying next viable method.")
    
    if 'generation' in viable_methods:
        try:
            predicted_class, generated_text = predict_with_generation(
                full_prompt, answer_choices, tokenizer, model, args, logger, 
                selected_examples, question
            )
            return {
                'predicted_class': predicted_class,
                'method': 'generation',
                'class_log_probs': None,
                'generated_text': generated_text
            }
        except Exception as e:
            logger.debug(f"Generation method failed: {e}.")
    
    # If we get here, all viable methods failed
    logger.error("All viable prediction methods failed. Using fallback prediction.")
    return {
        'predicted_class': answer_choices[0],
        'method': 'fallback',
        'class_log_probs': None,
        'generated_text': None
    }


def _test_prediction_methods(full_prompt: str, answer_choices: List[str], tokenizer, model, args,
                           logger: logging.Logger, selected_examples: Optional[List] = None,
                           question: Optional[str] = None) -> List[str]:
    """
    Test each prediction method with a timeout to determine which are viable.
    
    Args:
        full_prompt: The complete prompt without answer
        answer_choices: List of possible answer choices
        tokenizer: Model tokenizer
        model: Model instance
        args: Arguments object with max_context_length
        logger: Logger instance
        selected_examples: Optional list of few-shot examples for generation fallback
        question: Optional question string for generation fallback
        
    Returns:
        List of viable method names
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timeout_context(seconds):
        """Context manager for timeouts using signal where available"""
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                yield
            finally:
                # Clean up
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except (ImportError, AttributeError, OSError, ValueError):
            # Signal not available (Windows, Jupyter, etc.) - use time-based approach
            start_time = time.time()
            
            try:
                yield
            finally:
                elapsed = time.time() - start_time
                if elapsed > seconds:
                    logger.warning(f"Method took {elapsed:.2f}s (longer than {seconds}s timeout)")
                    # Don't raise TimeoutError here as the method already completed
    
    viable_methods = []
    timeout_seconds = 60  # 1 minute timeout per method
    
    # Test 1: JOLT logprob extraction
    logger.info("Testing JOLT logprob method...")
    try:
        with timeout_context(timeout_seconds):
            start_time = time.time()
            predicted_class, class_log_probs = predict_with_jolt_logprobs(
                full_prompt, answer_choices, tokenizer, model, args, logger
            )
            elapsed = time.time() - start_time
            viable_methods.append('jolt_logprobs')
            logger.info(f"JOLT logprob method successful ({elapsed:.2f}s)")
    except TimeoutError:
        logger.warning(f"JOLT logprob method timed out after {timeout_seconds}s - excluding from future predictions")
    except Exception as e:
        logger.info(f"JOLT logprob method failed: {e}")
    
    # Test 2: Simple logprob extraction
    logger.info("Testing simple logprob method...")
    try:
        with timeout_context(timeout_seconds):
            start_time = time.time()
            predicted_class, class_log_probs = predict_with_simple_logprobs(
                full_prompt, answer_choices, tokenizer, model, args, logger
            )
            elapsed = time.time() - start_time
            viable_methods.append('simple_logprobs')
            logger.info(f"Simple logprob method successful ({elapsed:.2f}s)")
    except TimeoutError:
        logger.warning(f"Simple logprob method timed out after {timeout_seconds}s - excluding from future predictions")
    except Exception as e:
        logger.info(f"Simple logprob method failed: {e}")
    
    # Test 3: Generation method
    logger.info("Testing generation method...")
    try:
        with timeout_context(timeout_seconds):
            start_time = time.time()
            predicted_class, generated_text = predict_with_generation(
                full_prompt, answer_choices, tokenizer, model, args, logger,
                selected_examples, question
            )
            elapsed = time.time() - start_time
            viable_methods.append('generation')
            logger.info(f"Generation method successful ({elapsed:.2f}s)")
    except TimeoutError:
        logger.warning(f"Generation method timed out after {timeout_seconds}s - excluding from future predictions")
    except Exception as e:
        logger.info(f"Generation method failed: {e}")
    
    # Check if any methods are viable
    if not viable_methods:
        raise RuntimeError("All prediction methods failed or timed out during testing. Cannot proceed with predictions.")
    
    return viable_methods


def parse_regression_prediction(prediction_text: str, target_stats: Optional[Dict] = None) -> float:
    """
    Parse a numerical prediction from LLM text response.
    
    Args:
        prediction_text: Text response from LLM
        target_stats: Statistics about target variable for validation
        
    Returns:
        Parsed numerical value
    """
    import re
    
    if prediction_text is None:
        return 0.0
    
    # Convert to string if not already
    text = str(prediction_text).strip()
    
    # Try direct float conversion first
    try:
        return float(text)
    except ValueError:
        pass
    
    # Look for numeric patterns in the text
    # Match integers, floats, scientific notation
    numeric_patterns = [
        r'[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?',  # General numeric pattern
        r'(\d+\.?\d*|\d*\.?\d+)',  # Simple decimal numbers
        r'[-+]?\d+',  # Integers
    ]
    
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Take the first match
                value = float(matches[0])
                
                # Validate against target statistics if available
                if target_stats and 'min' in target_stats and 'max' in target_stats:
                    min_val, max_val = target_stats['min'], target_stats['max']
                    # Allow some extrapolation beyond the range
                    range_extension = (max_val - min_val) * 0.5
                    extended_min = min_val - range_extension
                    extended_max = max_val + range_extension
                    
                    # Marvisp to extended range
                    value = max(extended_min, min(extended_max, value))
                
                return value
            except ValueError:
                continue
    
    # If no numeric value found, return 0 or target mean if available
    if target_stats and 'mean' in target_stats:
        return float(target_stats['mean'])
    
    return 0.0


def calculate_llm_metrics(y_test_partial, predictions, unique_classes, all_class_log_probs=None, logger=None, task_type=None, task_id=None, dataset=None):
    """
    Calculate comprehensive metrics for LLM baselines for both classification and regression tasks.
    
    Args:
        y_test_partial: True labels for completed samples
        predictions: Predicted values (class labels for classification, continuous values for regression)
        unique_classes: Array of unique class labels (for classification) or None (for regression)
        all_class_log_probs: List of class log probability dictionaries (optional, classification only)
        logger: Logger instance
        task_type: 'classification' or 'regression' (auto-detected if None)
        task_id: OpenML task ID for task type detection (if task_type is None)
        dataset: Dataset dictionary containing task_id (if task_type is None)
        
    Returns:
        Dictionary containing all calculated metrics
    """
    import math
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score, 
        f1_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    metrics = {}
    completed_samples = len(predictions)
    
    # Auto-detect task type if not provided
    if task_type is None:
        try:
            from .task_detection import detect_task_type
            task_type, detection_method = detect_task_type(
                y=y_test_partial,
                task_id=task_id,
                dataset=dataset
            )
            logger.debug(f"Auto-detected task type: {task_type} (method: {detection_method})")
        except Exception as e:
            logger.error(f"Task type detection failed: {e}")
            raise ValueError(f"Cannot determine task type. Please provide task_type, task_id, or dataset with task_id. Error: {e}")
    
    # Convert predictions and targets to numpy arrays for consistent handling
    y_true = np.array(y_test_partial)
    y_pred = np.array(predictions)
    
    if task_type == 'classification':
        # Classification metrics
        try:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Could not calculate accuracy metrics: {e}")
            metrics['accuracy'] = 0.0
            metrics['balanced_accuracy'] = 0.0
    else:
        # Regression: accuracy and balanced_accuracy don't apply
        metrics['accuracy'] = None
        metrics['balanced_accuracy'] = None
        
        # Calculate R² as the primary regression metric
        try:
            r2 = r2_score(y_true, y_pred)
            metrics['r2_score'] = float(r2)
        except Exception as e:
            logger.warning(f"Could not calculate R² score: {e}")
            metrics['r2_score'] = None
    
    # ROC AUC - only for classification
    if task_type == 'classification' and unique_classes is not None:
        try:
            if len(unique_classes) == 2:
                # Binary classification
                if all_class_log_probs and len(all_class_log_probs) == completed_samples and all(class_log_probs for class_log_probs in all_class_log_probs):
                    # Convert log probabilities to probabilities for positive class
                    positive_class = unique_classes[1]  # Assume second class is positive
                    y_scores = []
                    for class_log_probs in all_class_log_probs:
                        if str(positive_class) in class_log_probs or positive_class in class_log_probs:
                            # Use the log prob for positive class
                            log_prob = class_log_probs.get(str(positive_class), class_log_probs.get(positive_class, -1e6))
                            y_scores.append(math.exp(log_prob))
                        else:
                            # Fallback if positive class not found in log probs
                            y_scores.append(0.5)
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
                else:
                    # Fallback: use discrete predictions (less accurate but works)
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
            else:
                # Multiclass classification
                if all_class_log_probs and len(all_class_log_probs) == completed_samples and all(class_log_probs for class_log_probs in all_class_log_probs):
                    # Create probability matrix from log probabilities
                    prob_matrix = []
                    for class_log_probs in all_class_log_probs:
                        # Get log probs for all classes in order
                        log_probs = []
                        for cls in unique_classes:
                            # Try both string and original class representations
                            log_prob = class_log_probs.get(str(cls), class_log_probs.get(cls, -1e6))
                            log_probs.append(log_prob)
                        
                        # Convert log probabilities to probabilities using softmax
                        log_probs = np.array(log_probs)
                        # Subtract max for numerical stability
                        log_probs = log_probs - np.max(log_probs)
                        probs = np.exp(log_probs)
                        probs = probs / np.sum(probs)  # Normalize to sum to 1
                        prob_matrix.append(probs)
                    
                    prob_matrix = np.array(prob_matrix)
                    
                    # Calculate multiclass ROC AUC
                    metrics['roc_auc'] = float(roc_auc_score(y_true, prob_matrix, multi_class='ovr', average='weighted'))
                else:
                    # Skip ROC AUC if no log probabilities available for multiclass
                    logger.debug(f"No log probabilities available for multiclass ROC AUC calculation")
                    metrics['roc_auc'] = None
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
    else:
        # ROC AUC not applicable for regression
        metrics['roc_auc'] = None
    
    # F1 scores, precision, and recall - only for classification
    if task_type == 'classification':
        try:
            metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
            metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        except Exception as e:
            logger.warning(f"Could not calculate F1 scores: {e}")
            metrics['f1_macro'] = metrics['f1_micro'] = metrics['f1_weighted'] = None
        
        # Precision and recall
        try:
            metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        except Exception as e:
            logger.warning(f"Could not calculate precision/recall: {e}")
            metrics['precision_macro'] = metrics['recall_macro'] = None
    else:
        # F1 scores and precision/recall not applicable for regression
        metrics['f1_macro'] = metrics['f1_micro'] = metrics['f1_weighted'] = None
        metrics['precision_macro'] = metrics['recall_macro'] = None
    
    # Regression-specific metrics
    if task_type == 'regression':
        try:
            # Convert predictions to numeric values
            y_pred_numeric = []
            for pred in y_pred:
                try:
                    # Try to extract numeric value from prediction
                    if isinstance(pred, str):
                        # Parse potential numeric strings
                        import re
                        # Look for numeric patterns in the string
                        numeric_match = re.search(r'[-+]?(?:\d*\.?\d+)', pred)
                        if numeric_match:
                            y_pred_numeric.append(float(numeric_match.group()))
                        else:
                            # If no number found, use 0 as fallback
                            y_pred_numeric.append(0.0)
                    else:
                        y_pred_numeric.append(float(pred))
                except (ValueError, TypeError):
                    # If conversion fails, use 0 as fallback
                    y_pred_numeric.append(0.0)
            
            y_pred_numeric = np.array(y_pred_numeric)
            
            # Calculate regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred_numeric))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred_numeric))
            
            # R² score (already calculated above if not done yet)
            if 'r2_score' not in metrics or metrics['r2_score'] is None:
                metrics['r2_score'] = float(r2_score(y_true, y_pred_numeric))
            
            # Additional regression metrics
            metrics['mean_absolute_percentage_error'] = float(np.mean(np.abs((y_true - y_pred_numeric) / np.maximum(np.abs(y_true), 1e-8))) * 100)
            
        except Exception as e:
            logger.warning(f"Could not calculate regression metrics: {e}")
            metrics['mse'] = metrics['rmse'] = metrics['mae'] = None
            metrics['mean_absolute_percentage_error'] = None
    else:
        # Regression metrics not applicable for classification
        metrics['mse'] = metrics['rmse'] = metrics['mae'] = None
        metrics['mean_absolute_percentage_error'] = None
    
    # Add task type to metrics for reference
    metrics['task_type'] = task_type
    
    return metrics