"""
Implementation of Qwen model with prefix embedding capability.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

class QwenWithPrefixEmbedding(torch.nn.Module):
    """
    Wrapper for Qwen model with prefix embedding capability.
    Enables the model to accept tabular embeddings as prefix inputs.
    """
    def __init__(self, base_model, embedding_size, prefix_start_id, prefix_end_id, class_token_ids):
        """
        Initialize the QwenWithPrefixEmbedding model.
        
        Args:
            base_model: The base Qwen model
            embedding_size: Size of the input embeddings
            prefix_start_id: Token ID for prefix start marker
            prefix_end_id: Token ID for prefix end marker
            class_token_ids: List of token IDs for class tokens
        """
        super().__init__()
        self.base_model = base_model
        self.embedding_projector = torch.nn.Linear(embedding_size, base_model.config.hidden_size)
        self.prefix_start_id = prefix_start_id
        self.prefix_end_id = prefix_end_id
        self.class_token_ids = class_token_ids
        
        # Copy necessary methods and attributes from base_model
        self.config = base_model.config
        
        # Register the new parameter with accelerate for proper device management
        self._register_with_accelerate()

    def _register_with_accelerate(self):
        """
        Register this module with Accelerate's hooks if base_model has hooks.
        This ensures all parameters are tracked properly during distributed operations.
        """
        # Check if accelerate is being used
        if hasattr(self.base_model, "_hf_hook"):
            try:
                from accelerate.hooks import ModelHook, AlignDevicesHook, CpuOffload
                
                # Get the hook from the base model
                base_hook = self.base_model._hf_hook
                
                # Extract execution_device if available from any hook type
                execution_device = None
                if hasattr(base_hook, "execution_device"):
                    execution_device = base_hook.execution_device
                    logger.info(f"Found execution_device: {execution_device}")
                    
                    # Move embedding_projector to the same device
                    if execution_device is not None:
                        self.embedding_projector.to(execution_device)
                
                # Most reliable way to handle any hook type - grab the device and adapt
                logger.info(f"Using execution device from base model's hook: {execution_device}")
                
                # For more advanced hooks that might not be compatible with our wrapper model,
                # we can at least ensure the embedding_projector is on the correct device
                if execution_device is not None:
                    logger.info(f"Moving embedding_projector to execution_device: {execution_device}")
                    self.embedding_projector = self.embedding_projector.to(execution_device)
                
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not extract info from Accelerate hooks: {e}")
    
    def to(self, device):
        """
        Override to method to ensure all components move to the correct device.
        
        Args:
            device: The target device
            
        Returns:
            The model on the target device
        """
        # Move base_model
        self.base_model = self.base_model.to(device)
        
        # Move embedding_projector
        self.embedding_projector = self.embedding_projector.to(device)
        
        # Move the rest of self
        return super().to(device)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Pass through prepare_inputs_for_generation from base model."""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings from base model."""
        return self.base_model.get_input_embeddings()
    
    def forward(self, input_ids=None, attention_mask=None, prefix_data=None, inputs_embeds=None, **kwargs):
        """
        Forward pass with optional embedding prefix integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            prefix_data: Tabular embedding data to use as prefix
            inputs_embeds: Input embeddings (optional)
            **kwargs: Additional arguments for the base model
            
        Returns:
            Model outputs from the base model
        """
        # For Accelerate compatibility, get device from base_model's hook if present
        current_device = None
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "execution_device"):
            current_device = self._hf_hook.execution_device
        
        # Otherwise fall back to parameter device
        if current_device is None:
            current_device = next(self.parameters()).device
        
        # Either use provided inputs_embeds or create them from input_ids
        if inputs_embeds is None and input_ids is not None:
            # Make sure input_ids is on the right device 
            input_ids = input_ids.to(current_device)
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        elif inputs_embeds is not None:
            # Make sure inputs_embeds is on the right device
            inputs_embeds = inputs_embeds.to(current_device)
            
        # Ensure attention_mask is on the right device too
        if attention_mask is not None:
            attention_mask = attention_mask.to(current_device)
        
        if prefix_data is not None and input_ids is not None:
            # Ensure input_ids is on the right device
            input_ids = input_ids.to(current_device)
            
            # Find positions of PREFIX_START and PREFIX_END tokens
            batch_size = input_ids.shape[0]
            start_positions = (input_ids == self.prefix_start_id).nonzero(as_tuple=True)
            end_positions = (input_ids == self.prefix_end_id).nonzero(as_tuple=True)
            
            # Process each sequence in batch
            for i in range(batch_size):
                batch_start_positions = torch.where(start_positions[0] == i)[0]
                batch_end_positions = torch.where(end_positions[0] == i)[0]
                
                # Process each PREFIX_START/END pair
                for start_idx_pos, end_idx_pos in zip(batch_start_positions, batch_end_positions):
                    start_pos = start_positions[1][start_idx_pos]
                    end_pos = end_positions[1][end_idx_pos]
                    
                    if start_pos >= end_pos - 1:  # Need at least 1 token between markers
                        continue
                    
                    # Calculate how many tokens we have between markers
                    num_tokens = end_pos - start_pos - 1
                    
                    # Get prefix data for this batch item (embeddings with class tokens)
                    cur_prefix = prefix_data[i]  # Should contain both embeddings and class info
                    
                    # Separate embeddings and class info if provided
                    if isinstance(cur_prefix, tuple) and len(cur_prefix) == 2:
                        embeddings, class_labels = cur_prefix
                        
                        # Move embeddings to current device
                        if torch.is_tensor(embeddings):
                            embeddings = embeddings.to(current_device)
                        
                        # Project embeddings to model hidden size
                        projected_embeddings = []
                        for emb in embeddings:
                            # Make sure emb is on the right device
                            emb = emb.to(current_device)
                            projected_emb = self.embedding_projector(emb)
                            projected_embeddings.append(projected_emb)
                        
                        # Interleave embeddings with class tokens
                        interleaved_length = min(num_tokens, len(projected_embeddings) + len(class_labels))
                        interleaved_embeddings = torch.zeros(
                            interleaved_length, 
                            self.base_model.config.hidden_size, 
                            device=current_device  # Use current_device consistently 
                        )
                        
                        # Fill the tensor with embeddings and class token embeddings
                        emb_idx = 0
                        class_idx = 0
                        for i_token in range(interleaved_length):
                            if i_token % 2 == 0 and emb_idx < len(projected_embeddings):
                                # Even positions get embeddings
                                interleaved_embeddings[i_token] = projected_embeddings[emb_idx]
                                emb_idx += 1
                            elif i_token % 2 == 1 and class_idx < len(class_labels):
                                # Odd positions get class token embeddings
                                class_token_id = self.class_token_ids[class_labels[class_idx]]
                                class_token_tensor = torch.tensor([class_token_id], device=current_device)
                                interleaved_embeddings[i_token] = self.base_model.get_input_embeddings()(
                                    class_token_tensor
                                )
                                class_idx += 1
                        
                        # Replace token embeddings with interleaved embeddings
                        # +1 to skip the PREFIX_START token
                        inputs_embeds[i, start_pos+1:start_pos+1+interleaved_length, :] = interleaved_embeddings
                    else:
                        # Handle the case where we just have embeddings without class info
                        # Make sure embeddings are on the right device
                        embeddings_to_use = cur_prefix[:min(num_tokens, len(cur_prefix))].to(current_device)
                        projected_embeddings = self.embedding_projector(embeddings_to_use)
                        
                        # Replace token embeddings with prefix embeddings
                        inputs_embeds[i, start_pos+1:start_pos+1+len(projected_embeddings), :] = projected_embeddings
        
        # Ensure all inputs are on the current device before the forward pass
        kwargs = {k: v.to(current_device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        # Forward pass with modified embeddings
        return self.base_model(
            input_ids=None,  # We're using inputs_embeds
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    # Pass through any other needed methods to the base model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
            
    # Override generate to ensure devices are consistent and to use the hook's device
    def generate(self, *args, **kwargs):
        """
        Generate output sequences using the base model's generate method.
        
        Args:
            *args: Positional arguments for generate
            **kwargs: Keyword arguments for generate
            
        Returns:
            Generated token sequences
        """
        # Get device from hook if present, otherwise from parameters
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "execution_device"):
            current_device = self._hf_hook.execution_device
        else:
            current_device = next(self.parameters()).device
        
        # Move any tensor inputs to the current device
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(current_device)
        
        # For Accelerate compatibility, ensure we're using the base model's generate with proper hooks
        if hasattr(self.base_model, "_hf_hook"):
            # Let the hook handle the generation
            return self.base_model.generate(*args, **kwargs)
        else:
            # Direct call without hooks
            return self.base_model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and embedding projector.

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for save_pretrained
        """
        # Save the base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save the embedding projector
        embedding_projector_path = f"{save_directory}/embedding_projector.pt"
        torch.save(self.embedding_projector.state_dict(), embedding_projector_path)

        # Get final class token IDs from kwargs if provided (for permutation support)
        final_class_token_ids = kwargs.pop('final_class_token_ids', self.class_token_ids)

        # Save the class token IDs and prefix token IDs
        model_info = {
            "prefix_start_id": self.prefix_start_id,
            "prefix_end_id": self.prefix_end_id,
            "class_token_ids": self.class_token_ids,
            "final_class_token_ids": final_class_token_ids,  # Store the permuted mapping
            "embedding_size": self.embedding_projector.in_features,
        }
        torch.save(model_info, f"{save_directory}/model_info.pt")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """
        Load a pretrained model from a directory.

        Args:
            pretrained_model_path: Path to the pretrained model
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded QwenWithPrefixEmbedding model
        """
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            **kwargs
        )

        # Load model info
        model_info_path = f"{pretrained_model_path}/model_info.pt"
        model_info = torch.load(model_info_path)

        # Check if we should use permuted class token IDs
        use_permuted = kwargs.pop('use_permuted_mapping', True)
        use_class_token_ids = model_info.get("final_class_token_ids", model_info["class_token_ids"]) if use_permuted else model_info["class_token_ids"]

        # Log which mapping we're using
        logger.info(f"Loading model with {'permuted' if use_permuted else 'original'} class token mapping")
        if "final_class_token_ids" in model_info and use_permuted:
            logger.info(f"Using final epoch class token IDs: {model_info['final_class_token_ids']}")

        # Create a new instance
        model = cls(
            base_model=base_model,
            embedding_size=model_info["embedding_size"],
            prefix_start_id=model_info["prefix_start_id"],
            prefix_end_id=model_info["prefix_end_id"],
            class_token_ids=use_class_token_ids
        )

        # Load the embedding projector
        embedding_projector_path = f"{pretrained_model_path}/embedding_projector.pt"
        model.embedding_projector.load_state_dict(torch.load(embedding_projector_path))

        return model


def prepare_qwen_with_prefix_embedding(embedding_size=192, model_id="Qwen/Qwen2.5-3B-Instruct"):
    """
    Prepare Qwen model with prefix embedding capability and class tokens.
    
    Args:
        embedding_size: Size of the input embeddings (default: 192)
        model_id: Hugging Face model ID for the Qwen model
        
    Returns:
        model: QwenWithPrefixEmbedding model
        tokenizer: Tokenizer for the model
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
    """
    logger.info(f"Loading {model_id} model and tokenizer")
    
    # Load the Qwen model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Add special tokens for prefix and classes
    special_tokens = {
        "additional_special_tokens": [
            "<PREFIX_START>", "<PREFIX_END>",
            "<CLASS_0>", "<CLASS_1>", "<CLASS_2>", "<CLASS_3>", 
            "<CLASS_4>", "<CLASS_5>", "<CLASS_6>", "<CLASS_7>",
            "<CLASS_8>", "<CLASS_9>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Resize model's embedding matrix to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Define token IDs for later use
    prefix_start_id = tokenizer.convert_tokens_to_ids("<PREFIX_START>")
    prefix_end_id = tokenizer.convert_tokens_to_ids("<PREFIX_END>")
    
    # Get class token IDs
    class_token_ids = [tokenizer.convert_tokens_to_ids(f"<CLASS_{i}>") for i in range(10)]

    logger.info(f"Added special tokens: PREFIX_START_ID={prefix_start_id}, PREFIX_END_ID={prefix_end_id}")
    logger.info(f"Class token IDs: {class_token_ids}")
    
    # Create the model
    qwen_with_prefix = QwenWithPrefixEmbedding(model, embedding_size, prefix_start_id, prefix_end_id, class_token_ids)
    
    return qwen_with_prefix, tokenizer, prefix_start_id, prefix_end_id, class_token_ids


def load_pretrained_model(model_path, device_map="auto", embedding_size=1000, model_id=None):
    """
    Load a pretrained MARVIS model from a checkpoint directory.
    This function handles different model loading scenarios:
    1. Models saved with model_info.pt (saved using QwenWithPrefixEmbedding.save_pretrained)
    2. Custom models requiring initialization using prepare_qwen_with_prefix_embedding
    3. Fallback to standard AutoModelForCausalLM loading when possible
    
    Args:
        model_path: Path to the pretrained model directory
        device_map: Device mapping strategy for model loading
        embedding_size: Size of the embeddings (for initialization if needed)
        
    Returns:
        model: Loaded model
        tokenizer: Tokenizer for the model
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
    """
    logger.info(f"Loading pretrained model from {model_path}")
    
    # First, check if this is a model with best_model directory
    best_model_path = os.path.join(model_path, "best_model")
    if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
        model_path = best_model_path
        logger.info(f"Found best_model directory, using {best_model_path}")

    # Check if we can load the model with our custom from_pretrained method
    model_info_path = os.path.join(model_path, "model_info.pt")
    if os.path.exists(model_info_path):
        # Load using the custom method which will use the final epoch class token IDs
        logger.info(f"Loading model with stored class token mapping from {model_info_path}")
        model = QwenWithPrefixEmbedding.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Extract token IDs
        prefix_start_id = model.prefix_start_id
        prefix_end_id = model.prefix_end_id
        class_token_ids = model.class_token_ids

        logger.info(f"Loaded model with class token IDs: {class_token_ids}")
    else:
        # Since a specific model_path was provided but model_info.pt doesn't exist there,
        # try to load from checkpoint_path directly without using the fallback model
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
                
                # Try to load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Attempt to load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map
                )

                # Extract special token IDs
                prefix_start_id = tokenizer.convert_tokens_to_ids("<PREFIX_START>")
                prefix_end_id = tokenizer.convert_tokens_to_ids("<PREFIX_END>")
                class_token_ids = [tokenizer.convert_tokens_to_ids(f"<CLASS_{i}>") for i in range(10)]

                # Create custom model wrapper
                model = QwenWithPrefixEmbedding(
                    base_model,
                    embedding_size=embedding_size,
                    prefix_start_id=prefix_start_id,
                    prefix_end_id=prefix_end_id,
                    class_token_ids=class_token_ids
                )
                
                # Try to load weights directly
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded model from {model_path}")
                
            except Exception as e:
                error_msg = f"Failed to load model from specified path: {model_path}. Error: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # No model_info.pt and no checkpoint found at the specified path
            error_msg = f"No valid model files found at specified path: {model_path}. " \
                       f"Could not find model_info.pt or pytorch_model.bin."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids