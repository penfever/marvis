"""
Evaluation functions for MARVIS models.
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

def check_tensor_devices(model: torch.nn.Module) -> Dict[str, List[str]]:
    """
    Debug function to check devices of all tensors in model.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary mapping devices to parameter names
    """
    devices = {}
    for name, param in model.named_parameters():
        device = param.device
        if device not in devices:
            devices[device] = []
        devices[device].append(name)
    
    for device, params in devices.items():
        logger.info(f"Device {device} has {len(params)} parameters")
        if len(params) < 10:  # Only print if small number
            logger.info(f"Parameters on {device}: {params}")
        else:
            logger.info(f"Sample parameters on {device}: {params[:5]}")
    
    return devices


def evaluate_llm_on_test_set(
    model: torch.nn.Module,
    tokenizer: Any,
    test_dataset: Any,
    label_encoder: Any,
    prefix_start_id: int,
    prefix_end_id: int,
    class_token_ids: List[int],
    prefix_data_file: str,
    max_test_samples: Optional[int] = None,
    return_raw_probabilities: bool = False,
    baseline_probabilities: Optional[np.ndarray] = None,
    score_normalization: str = "none",
    normalization_temperature: float = 2.0,
    class_weights: Optional[np.ndarray] = None,
    allowed_classes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluate the trained LLM on the test set - Accelerate-compatible version.

    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        test_dataset: Test dataset
        label_encoder: Label encoder for class labels
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
            Note: This can contain more tokens than classes in the dataset
            (e.g., 10 class tokens for a dataset with only 6 classes).
            The evaluator will handle this mismatch.
        prefix_data_file: Path to the saved prefix data
        max_test_samples: Maximum number of test samples to use
        allowed_classes: If provided, only consider these class IDs for prediction.
            If None, automatically detect classes from test dataset.

    Returns:
        Dictionary with evaluation results including:
        - accuracy: Overall accuracy
        - predictions: List of predicted class IDs
        - ground_truth: List of true class IDs
        - probabilities: List of probabilities for each prediction
        - classification_report: Detailed per-class metrics (if sklearn is available)
        - confusion_matrix: Confusion matrix (if sklearn is available)
    """
    # Reset static variables for this run
    if hasattr(evaluate_llm_on_test_set, 'probabilities'):
        evaluate_llm_on_test_set.probabilities = []
    else:
        evaluate_llm_on_test_set.probabilities = []
        
    logger.info("Evaluating model on test set")
    
    # Detect allowed classes from ground truth test data if not provided
    if allowed_classes is None:
        # Extract all ground truth labels from test dataset
        ground_truth_labels = []
        for i in range(len(test_dataset)):
            ground_truth_labels.append(test_dataset[i]["label_id"])
        
        # Get unique classes present in test data
        allowed_classes = sorted(list(set(ground_truth_labels)))
        logger.info(f"Detected {len(allowed_classes)} classes in test data: {allowed_classes}")
    
    # Check if model has Accelerate hooks
    has_accelerate = hasattr(model, "_hf_hook") or (hasattr(model, "base_model") and hasattr(model.base_model, "_hf_hook"))
    logger.info(f"Model is {'using' if has_accelerate else 'not using'} Accelerate hooks")
    
    # Get the primary device
    if hasattr(model, "_hf_hook") and hasattr(model._hf_hook, "execution_device"):
        param_device = model._hf_hook.execution_device
    else:
        param_device = next(model.parameters()).device
    logger.info(f"Primary model device: {param_device}")
    
    # Get all unique devices used by the model
    all_devices = {p.device for p in model.parameters()}
    logger.info(f"All devices used by model: {all_devices}")
    
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
    
    # Set up generation config - disable features that might cause device issues
    generation_config = {
        "max_new_tokens": 1,
        "num_beams": 1,
        "do_sample": True,
        "use_cache": False,  # This can help with device issues
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    # For results
    predictions = []
    ground_truth = []
    
    # Switch to evaluation mode
    model.eval()
    
    # Test loop
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
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
                        
                        # Determine how many example embeddings we can use based on remaining space
                        # Each example pair takes 2 tokens (embedding + class token)
                        num_examples = min(example_space // 2, embeddings.shape[0])
                        
                        # Project example embeddings to model hidden size
                        projected_examples = model.embedding_projector(embeddings[:num_examples]).to(param_device)
                        
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
                        for j in range(num_examples):
                            # Example embedding
                            all_embeddings[example_offset + j*2] = projected_examples[j]
                            
                            # Class token - make sure on same device
                            if j < len(class_labels):
                                class_token_id = class_token_ids[int(class_labels[j])]
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
                
                # Extract logits for the specified class tokens
                class_logits = torch.zeros(len(class_token_ids), device=param_device)
                for idx, token_id in enumerate(class_token_ids):
                    class_logits[idx] = logits[0, token_id]
                
                # Apply score normalization
                if score_normalization == "temperature":
                    # Temperature scaling (higher temperature = more uniform distribution)
                    class_logits = class_logits / normalization_temperature
                
                # Filter to only allowed classes
                if allowed_classes is not None:
                    # Create mask for allowed classes
                    allowed_mask = torch.zeros(len(class_logits), dtype=torch.bool, device=param_device)
                    for cls_id in allowed_classes:
                        if cls_id < len(class_logits):
                            allowed_mask[cls_id] = True
                    
                    # Set logits for disallowed classes to very negative value
                    class_logits[~allowed_mask] = -1e10
                
                # Convert logits to probabilities
                class_probs = torch.nn.functional.softmax(class_logits, dim=-1)
                
                # Store raw probabilities if requested
                if return_raw_probabilities:
                    if not hasattr(evaluate_llm_on_test_set, 'raw_probabilities'):
                        evaluate_llm_on_test_set.raw_probabilities = []
                    evaluate_llm_on_test_set.raw_probabilities.append(class_probs.cpu().numpy())
                
                # Apply baseline calibration if provided
                if baseline_probabilities is not None:
                    # Convert baseline to tensor on same device
                    baseline_tensor = torch.tensor(baseline_probabilities, device=param_device)
                    
                    # Compute calibrated probabilities (positive differences)
                    calibrated_probs = torch.clamp(class_probs - baseline_tensor, min=0)
                    
                    # Check if all calibrated probs are zero (edge case)
                    if torch.sum(calibrated_probs) == 0:
                        # Fall back to using raw probabilities minus mean baseline
                        mean_baseline = torch.mean(baseline_tensor)
                        calibrated_probs = torch.clamp(class_probs - mean_baseline, min=0)
                    
                    # Normalize if needed (ensure they sum to positive value)
                    if torch.sum(calibrated_probs) > 0:
                        calibrated_probs = calibrated_probs / torch.sum(calibrated_probs)
                    else:
                        # Ultimate fallback: use original probabilities
                        calibrated_probs = class_probs
                    
                    if i % 20 == 0:
                        logger.info(f"Original probs: {class_probs.cpu().numpy()}")
                        logger.info(f"Baseline probs: {baseline_probabilities}")
                        logger.info(f"Calibrated probs: {calibrated_probs.cpu().numpy()}")
                    
                    # Use calibrated probabilities for prediction
                    class_probs = calibrated_probs
                
                # Apply class weights if provided (minority boosting)
                if class_weights is not None:
                    weights_tensor = torch.tensor(class_weights, device=param_device)
                    weighted_probs = class_probs * weights_tensor
                    
                    # Renormalize
                    if torch.sum(weighted_probs) > 0:
                        weighted_probs = weighted_probs / torch.sum(weighted_probs)
                    else:
                        weighted_probs = class_probs
                    
                    if i % 20 == 0:
                        logger.info(f"Class weights: {class_weights}")
                        logger.info(f"Weighted probs: {weighted_probs.cpu().numpy()}")
                    
                    class_probs = weighted_probs
                
                # Get the class with highest probability
                max_prob_idx = torch.argmax(class_probs).item()
                predicted_label_id = max_prob_idx
                
                # For logging purposes, get the token and probability
                max_prob_token_id = class_token_ids[max_prob_idx]
                max_prob_token = tokenizer.convert_ids_to_tokens(max_prob_token_id)
                max_prob = class_probs[max_prob_idx].item()
                
                # Log detailed info occasionally
                if i % 100 == 0:
                    logger.info(f"Class token probabilities: {class_probs.cpu().numpy()}")
                    logger.info(f"Selected token: {max_prob_token} (id={max_prob_token_id}) with prob={max_prob:.4f}")
                
                # For logging consistency, use the predicted class token as the "text"
                predicted_text = f"<CLASS_{predicted_label_id}>"
                
                predictions.append(predicted_label_id)
                ground_truth.append(label_id)
                
                # Store the probability for later analysis
                if not hasattr(evaluate_llm_on_test_set, 'probabilities'):
                    evaluate_llm_on_test_set.probabilities = []
                evaluate_llm_on_test_set.probabilities.append(max_prob)
                
                if i % 100 == 0:
                    logger.info(f"Example {i}: Pred={predicted_label_id}, True={label_id}, Probability={max_prob:.4f}")
                    logger.info(f"Predicted class token: {predicted_text}")
            
            except Exception as e:
                logger.error(f"Error in prediction for example {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                predictions.append(-1)  # Unknown
                ground_truth.append(label_id)
                # Add a zero probability for error cases
                evaluate_llm_on_test_set.probabilities.append(0.0)
    
    # Calculate accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / max(len(predictions),1)
    
    # Calculate balanced accuracy
    try:
        from sklearn.metrics import balanced_accuracy_score
        balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
        logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{len(predictions)}), Balanced accuracy: {balanced_accuracy:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute balanced accuracy: {e}")
        balanced_accuracy = None
        logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # Calculate per-class metrics if possible
    try:
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
        
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
            
            # Calculate ROC AUC score if probabilities are available
            roc_auc = None
            try:
                raw_probs = getattr(evaluate_llm_on_test_set, 'raw_probabilities', [])
                if raw_probs and len(unique_classes) > 1:
                    # For binary classification
                    if len(unique_classes) == 2:
                        # Get probabilities for the positive class (usually class 1)
                        pos_class_idx = 1 if 1 in unique_classes else unique_classes[1]
                        binary_truth = [1 if y == pos_class_idx else 0 for y in valid_truth]
                        binary_probs = [prob[pos_class_idx] for prob in [raw_probs[i] for i in valid_indices]]
                        roc_auc = roc_auc_score(binary_truth, binary_probs)
                    # For multi-class classification, use one-vs-rest approach
                    else:
                        # Convert to one-hot encoding for multi-class ROC AUC
                        from sklearn.preprocessing import label_binarize
                        # Only use the classes that are present in the dataset
                        y_bin = label_binarize(valid_truth, classes=unique_classes)
                        # Use the relevant probabilities for each class
                        probs_array = np.array([raw_probs[i][unique_classes] for i in valid_indices])
                        # Normalize probabilities if necessary
                        probs_array = probs_array / np.sum(probs_array, axis=1, keepdims=True)
                        roc_auc = roc_auc_score(y_bin, probs_array, multi_class='ovr')
                    
                    logger.info(f"ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                import traceback
                logger.warning(traceback.format_exc())
            
            logger.info("Per-class metrics computed")
            
            # Get probabilities if available
            probabilities = getattr(evaluate_llm_on_test_set, 'probabilities', [])
            
            result_dict = {
                'accuracy': accuracy,
                'predictions': predictions,
                'ground_truth': ground_truth,
                'probabilities': probabilities,
                'classification_report': report,
                'confusion_matrix': cm,
                'roc_auc': roc_auc,
                'balanced_accuracy': balanced_accuracy
            }
            
            # Add raw probabilities if requested
            if return_raw_probabilities:
                raw_probs = getattr(evaluate_llm_on_test_set, 'raw_probabilities', [])
                if raw_probs:
                    result_dict['raw_probabilities'] = raw_probs
            
            return result_dict
        
    except (ImportError, ValueError) as e:
        logger.warning(f"Could not compute detailed metrics: {e}")
    
    # Get probabilities if available
    probabilities = getattr(evaluate_llm_on_test_set, 'probabilities', [])
    
    result_dict = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'probabilities': probabilities
    }
    
    # Add raw probabilities if requested
    if return_raw_probabilities:
        raw_probs = getattr(evaluate_llm_on_test_set, 'raw_probabilities', [])
        if raw_probs:
            result_dict['raw_probabilities'] = raw_probs
    
    return result_dict