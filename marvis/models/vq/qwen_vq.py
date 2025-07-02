"""
Qwen model with Vector-Quantized prefix embedding capability.

This module implements a modified Qwen model that accepts tabular embeddings as
vector-quantized inputs, providing a more structured and efficient representation.
"""

import os
import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any, Union, Callable

from .vector_quantizer import VectorQuantizer

logger = logging.getLogger(__name__)

class QwenWithVQPrefixEmbedding(torch.nn.Module):
    """
    Wrapper for Qwen model with vector-quantized prefix embedding capability.
    Enables the model to accept tabular embeddings as quantized prefix inputs.
    """
    def __init__(self, 
                base_model,
                embedding_size,
                prefix_start_id,
                prefix_end_id,
                class_token_ids,
                vq_num_embeddings=512,
                vq_commitment_cost=0.25,
                vq_decay=0.99):
        """
        Initialize the QwenWithVQPrefixEmbedding model.
        
        Args:
            base_model: The base Qwen model
            embedding_size: Size of the input embeddings from TabPFN
            prefix_start_id: Token ID for prefix start marker
            prefix_end_id: Token ID for prefix end marker
            class_token_ids: List of token IDs for class tokens
            vq_num_embeddings: Size of the codebook for vector quantization
            vq_commitment_cost: Weight for the commitment loss in VQ
            vq_decay: Decay factor for EMA updates of the codebook (0 for no EMA)
        """
        super().__init__()
        self.base_model = base_model
        
        # Create the vector quantizer
        self.vector_quantizer = VectorQuantizer(
            embedding_dim=embedding_size,
            num_embeddings=vq_num_embeddings,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay
        )
        
        # Linear projection to map quantized vectors to model dimension
        self.embedding_projector = torch.nn.Linear(embedding_size, base_model.config.hidden_size)
        
        # Store token IDs
        self.prefix_start_id = prefix_start_id
        self.prefix_end_id = prefix_end_id
        self.class_token_ids = class_token_ids
        
        # Copy necessary attributes from base_model
        self.config = base_model.config
        self.vq_num_embeddings = vq_num_embeddings
        self.vq_commitment_cost = vq_commitment_cost
        self.vq_decay = vq_decay
        
        # Register with accelerate for proper device management
        self._register_with_accelerate()
        
        # Total VQ loss accumulator for tracking
        self.register_buffer('_total_vq_loss', torch.tensor(0.0))
        self.register_buffer('_num_vq_updates', torch.tensor(0.0))
    
    def _register_with_accelerate(self):
        """
        Register this module with Accelerate's hooks if base_model has hooks.
        This ensures all parameters are tracked properly during distributed operations.
        """
        # Check if accelerate is being used
        if hasattr(self.base_model, "_hf_hook"):
            try:
                # Extract execution_device if available from any hook type
                execution_device = None
                base_hook = self.base_model._hf_hook
                
                if hasattr(base_hook, "execution_device"):
                    execution_device = base_hook.execution_device
                    logger.info(f"Found execution_device: {execution_device}")
                    
                    # Move components to the same device
                    if execution_device is not None:
                        self.vector_quantizer.to(execution_device)
                        self.embedding_projector.to(execution_device)
                
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
        
        # Move vector_quantizer
        self.vector_quantizer = self.vector_quantizer.to(device)
        
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
    
    def forward(self, 
                input_ids=None,
                attention_mask=None,
                prefix_data=None,
                inputs_embeds=None,
                return_vq_loss=False,
                **kwargs):
        """
        Forward pass with vector-quantized embedding prefix integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            prefix_data: Tabular embedding data to use as prefix
            inputs_embeds: Input embeddings (optional)
            return_vq_loss: Whether to return VQ loss in output
            **kwargs: Additional arguments for the base model
            
        Returns:
            Model outputs from the base model with optional VQ loss
        """
        # Get current device
        device = None
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "execution_device"):
            device = self._hf_hook.execution_device
        else:
            device = next(self.parameters()).device
        
        # Either use provided inputs_embeds or create them from input_ids
        if inputs_embeds is None and input_ids is not None:
            input_ids = input_ids.to(device)
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        
        # Move attention_mask to the right device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Variables to track vector quantization
        vq_loss = torch.tensor(0.0, device=device)
        total_perplexity = torch.tensor(0.0, device=device)
        num_quantized = 0
        
        if prefix_data is not None and input_ids is not None:
            input_ids = input_ids.to(device)
            
            # Find positions of PREFIX_START and PREFIX_END tokens
            start_positions = (input_ids == self.prefix_start_id).nonzero(as_tuple=True)
            end_positions = (input_ids == self.prefix_end_id).nonzero(as_tuple=True)
            
            # Process each sequence in batch
            for i in range(input_ids.shape[0]):
                batch_start_positions = torch.where(start_positions[0] == i)[0]
                batch_end_positions = torch.where(end_positions[0] == i)[0]
                
                # Process each PREFIX_START/END pair
                for start_idx_pos, end_idx_pos in zip(batch_start_positions, batch_end_positions):
                    start_pos = start_positions[1][start_idx_pos]
                    end_pos = end_positions[1][end_idx_pos]
                    
                    if start_pos >= end_pos - 1:  # Need at least 1 token between markers
                        continue
                    
                    # Calculate how many tokens we have between markers
                    num_tokens = int((end_pos - start_pos - 1).item()) if isinstance(end_pos, torch.Tensor) else (end_pos - start_pos - 1)
                    
                    # Reserve space for the query embedding (10 tokens)
                    query_space = min(10, num_tokens // 3)  # Reserve up to 1/3 of available space, max 10 tokens
                    example_space = num_tokens - query_space
                    
                    # Convert to Python integers
                    query_space = int(query_space) if isinstance(query_space, torch.Tensor) else query_space
                    example_space = int(example_space) if isinstance(example_space, torch.Tensor) else example_space
                    
                    # Get the current example's query embedding
                    query_embedding = prefix_data[i]
                    
                    # Ensure query_embedding is a tensor on the right device
                    if isinstance(query_embedding, (tuple, list)) and len(query_embedding) == 2:
                        # Handle case where prefix_data contains (embeddings, class_labels)
                        embeddings, class_labels = query_embedding
                        
                        # Move embeddings to current device
                        if torch.is_tensor(embeddings):
                            embeddings = embeddings.to(device)
                        
                        # First, apply vector quantization to each embedding
                        quantized_embeddings = []
                        batch_vq_loss = torch.tensor(0.0, device=device)
                        batch_perplexity = torch.tensor(0.0, device=device)
                        
                        for emb in embeddings:
                            # Ensure emb is on the right device
                            emb = emb.to(device)
                            
                            # Apply vector quantization
                            quantized_emb, emb_vq_loss, emb_perplexity = self.vector_quantizer(emb.unsqueeze(0))
                            quantized_embeddings.append(quantized_emb.squeeze(0))
                            
                            # Accumulate loss and perplexity
                            batch_vq_loss += emb_vq_loss
                            batch_perplexity += emb_perplexity
                            num_quantized += 1
                        
                        # Average the loss and perplexity
                        if len(embeddings) > 0:
                            batch_vq_loss /= len(embeddings)
                            batch_perplexity /= len(embeddings)
                            
                            # Add to total loss and perplexity
                            vq_loss += batch_vq_loss
                            total_perplexity += batch_perplexity
                        
                        # Quantize the query embedding separately
                        query_embedding = query_embedding.to(device)
                        quantized_query, query_vq_loss, query_perplexity = self.vector_quantizer(query_embedding.unsqueeze(0))
                        
                        # Add query VQ loss to total
                        vq_loss += query_vq_loss
                        total_perplexity += query_perplexity
                        num_quantized += 1
                        
                        # Project quantized embeddings to model hidden size
                        projected_quantized_embeddings = []
                        for quantized_emb in quantized_embeddings:
                            projected_emb = self.embedding_projector(quantized_emb)
                            projected_quantized_embeddings.append(projected_emb)
                        
                        # Project quantized query
                        projected_query = self.embedding_projector(quantized_query.squeeze(0))
                        
                        # Determine maximum examples we can use based on space
                        max_examples = min(example_space // 2, len(embeddings))
                        
                        # Create a tensor to hold all our embeddings (query + examples)
                        all_embeddings = torch.zeros(
                            num_tokens,  # Total space between markers
                            self.base_model.config.hidden_size,
                            device=device
                        )
                        
                        # Get separator token embedding
                        query_separator_id = tokenizer.eos_token_id
                        if "<QUERY>" in tokenizer.get_vocab():
                            query_separator_id = tokenizer.convert_tokens_to_ids("<QUERY>")
                        
                        query_separator = self.base_model.get_input_embeddings()(
                            torch.tensor([query_separator_id], device=device)
                        ).squeeze(0)
                        
                        # Add query separator and repeated quantized query embedding
                        all_embeddings[0] = query_separator
                        for j in range(1, query_space - 1):
                            all_embeddings[j] = projected_query
                        all_embeddings[query_space - 1] = query_separator
                        
                        # Next, add the interleaved example embeddings and class tokens
                        example_offset = query_space
                        for j in range(max_examples):
                            # Example embedding
                            all_embeddings[example_offset + j*2] = projected_quantized_embeddings[j]
                            
                            # Class token
                            if j < len(class_labels):
                                # Convert to integer if needed
                                class_idx = int(class_labels[j].item()) if isinstance(class_labels[j], torch.Tensor) else int(class_labels[j])
                                
                                # Get the class token ID for this class
                                class_token_id = self.class_token_ids[class_idx]
                                
                                # Convert to int if it's a tensor
                                if isinstance(class_token_id, torch.Tensor):
                                    class_token_id = int(class_token_id.item())
                                
                                # Get the class token embedding
                                class_token_embedding = self.base_model.get_input_embeddings()(
                                    torch.tensor([class_token_id], device=device)
                                ).squeeze(0)
                                
                                all_embeddings[example_offset + j*2 + 1] = class_token_embedding
                        
                        # Replace token embeddings with our custom embeddings
                        # +1 to skip the PREFIX_START token
                        inputs_embeds[i, start_pos+1:end_pos, :] = all_embeddings
                    else:
                        # Handle the case where we just have a single query embedding without class info
                        # Ensure it's on the right device
                        query_embedding = query_embedding.to(device)
                        
                        # Apply vector quantization
                        quantized_query, query_vq_loss, query_perplexity = self.vector_quantizer(query_embedding.unsqueeze(0))
                        
                        # Update VQ tracking
                        vq_loss += query_vq_loss
                        total_perplexity += query_perplexity
                        num_quantized += 1
                        
                        # Project the quantized query to model hidden size
                        projected_query = self.embedding_projector(quantized_query.squeeze(0))
                        
                        # Replace the prefix placeholder with repeated query embedding
                        for j in range(min(num_tokens, 10)):  # Limit to 10 tokens for single query
                            inputs_embeds[i, start_pos+1+j, :] = projected_query
        
        # Update VQ loss tracking for monitoring
        if num_quantized > 0:
            avg_vq_loss = vq_loss / num_quantized
            avg_perplexity = total_perplexity / num_quantized
            
            # Update running average of VQ loss
            with torch.no_grad():
                self._total_vq_loss += avg_vq_loss.detach()
                self._num_vq_updates += 1
                
                # Periodically log VQ statistics
                if self._num_vq_updates % 100 == 0:
                    avg_total_vq_loss = self._total_vq_loss / self._num_vq_updates
                    logger.info(f"VQ Stats - Avg Loss: {avg_total_vq_loss:.6f}, Avg Perplexity: {avg_perplexity:.2f}")
                    
                    # Reset counters periodically to focus on recent performance
                    if self._num_vq_updates >= 1000:
                        self._total_vq_loss.zero_()
                        self._num_vq_updates.zero_()
        
        # Ensure all inputs are on the current device before the forward pass
        kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        # Forward pass with modified embeddings
        outputs = self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
        # Add VQ loss to the total loss if requested
        if self.training and num_quantized > 0:
            # Only add VQ loss if we actually quantized something
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss + avg_vq_loss
            
            # Return VQ loss separately if requested
            if return_vq_loss:
                outputs.vq_loss = avg_vq_loss
                outputs.vq_perplexity = avg_perplexity
        
        return outputs
    
    # Pass through any other needed methods to the base model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    # Override generate to ensure devices are consistent
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
            device = self._hf_hook.execution_device
        else:
            device = next(self.parameters()).device
        
        # Move any tensor inputs to the current device
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(device)
        
        # For Accelerate compatibility, ensure we're using the base model's generate with proper hooks
        if hasattr(self.base_model, "_hf_hook"):
            # Let the hook handle the generation
            return self.base_model.generate(*args, **kwargs)
        else:
            # Direct call without hooks
            return self.base_model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model including vector quantizer and embedding projector.

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for save_pretrained
        """
        # Save the base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save the vector quantizer
        vector_quantizer_path = f"{save_directory}/vector_quantizer.pt"
        torch.save(self.vector_quantizer.state_dict(), vector_quantizer_path)

        # Save the embedding projector
        embedding_projector_path = f"{save_directory}/embedding_projector.pt"
        torch.save(self.embedding_projector.state_dict(), embedding_projector_path)

        # Get final class token IDs from kwargs if provided (for permutation support)
        final_class_token_ids = kwargs.pop('final_class_token_ids', self.class_token_ids)

        # Save model configuration
        model_info = {
            "prefix_start_id": self.prefix_start_id,
            "prefix_end_id": self.prefix_end_id,
            "class_token_ids": self.class_token_ids,
            "final_class_token_ids": final_class_token_ids,  # Store the permuted mapping
            "embedding_size": self.vector_quantizer.embedding_dim,
            "vq_num_embeddings": self.vq_num_embeddings,
            "vq_commitment_cost": self.vq_commitment_cost,
            "vq_decay": self.vq_decay
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
            Loaded QwenWithVQPrefixEmbedding model
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
        class_token_ids = model_info.get("final_class_token_ids", model_info["class_token_ids"]) if use_permuted else model_info["class_token_ids"]

        # Create a new instance
        model = cls(
            base_model=base_model,
            embedding_size=model_info["embedding_size"],
            prefix_start_id=model_info["prefix_start_id"],
            prefix_end_id=model_info["prefix_end_id"],
            class_token_ids=class_token_ids,
            vq_num_embeddings=model_info.get("vq_num_embeddings", 512),
            vq_commitment_cost=model_info.get("vq_commitment_cost", 0.25),
            vq_decay=model_info.get("vq_decay", 0.99)
        )

        # Load the vector quantizer
        vector_quantizer_path = f"{pretrained_model_path}/vector_quantizer.pt"
        if os.path.exists(vector_quantizer_path):
            model.vector_quantizer.load_state_dict(torch.load(vector_quantizer_path))
        else:
            logger.warning(f"Vector quantizer weights not found at {vector_quantizer_path}")

        # Load the embedding projector
        embedding_projector_path = f"{pretrained_model_path}/embedding_projector.pt"
        if os.path.exists(embedding_projector_path):
            model.embedding_projector.load_state_dict(torch.load(embedding_projector_path))
        else:
            logger.warning(f"Embedding projector weights not found at {embedding_projector_path}")

        return model


def prepare_qwen_with_vq_prefix_embedding(
    embedding_size=192,
    model_id="Qwen/Qwen2.5-3B-Instruct",
    vq_num_embeddings=512,
    vq_commitment_cost=0.25,
    vq_decay=0.99
):
    """
    Prepare Qwen model with vector-quantized prefix embedding capability and class tokens.
    
    Args:
        embedding_size: Size of the input embeddings from TabPFN
        model_id: Hugging Face model ID for the Qwen model
        vq_num_embeddings: Size of the codebook for vector quantization
        vq_commitment_cost: Weight for the commitment loss in VQ
        vq_decay: Decay factor for EMA updates of the codebook (0 for no EMA)
        
    Returns:
        model: QwenWithVQPrefixEmbedding model
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
            "<PREFIX_START>", "<PREFIX_END>", "<QUERY>",
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
    logger.info(f"Using Vector Quantization with codebook size={vq_num_embeddings}")

    # Create the model with vector quantization
    qwen_with_vq_prefix = QwenWithVQPrefixEmbedding(
        base_model=model,
        embedding_size=embedding_size,
        prefix_start_id=prefix_start_id,
        prefix_end_id=prefix_end_id,
        class_token_ids=class_token_ids,
        vq_num_embeddings=vq_num_embeddings,
        vq_commitment_cost=vq_commitment_cost,
        vq_decay=vq_decay
    )
    
    return qwen_with_vq_prefix, tokenizer, prefix_start_id, prefix_end_id, class_token_ids


def load_vq_pretrained_model(model_path: str, device_map: str = "auto", embedding_size: int = 1000, model_id: str = None) -> Tuple[
    torch.nn.Module, Any, int, int, List[int], bool
]:
    """
    Load a pretrained Vector-Quantized MARVIS model from a checkpoint directory.
    
    This function handles different model loading scenarios for VQ models:
    1. Models saved with vector_quantizer.pt (saved using QwenWithVQPrefixEmbedding.save_pretrained)
    2. Custom models requiring initialization using prepare_qwen_with_vq_prefix_embedding
    3. Fallback to standard model loading if this is not a VQ model
    
    Args:
        model_path: Path to the pretrained model directory
        device_map: Device mapping strategy for model loading
        embedding_size: Size of the embeddings (for initialization if needed)
        
    Returns:
        model: Loaded model (either VQ or standard)
        tokenizer: Tokenizer for the model
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
        is_vq: Boolean indicating whether this is a VQ model
    """
    logger.info(f"Loading model from {model_path} (checking for VQ capabilities)")
    
    # First, check if this is a model with best_model directory
    best_model_path = os.path.join(model_path, "best_model")
    if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
        model_path = best_model_path
        logger.info(f"Found best_model directory, using {best_model_path}")
    
    # Look for VQ-specific files
    vector_quantizer_path = os.path.join(model_path, "vector_quantizer.pt")
    is_vq = os.path.exists(vector_quantizer_path)
    
    # Try to load tokenizer first as it's needed for both approaches
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        # Try to create a new tokenizer using the preparation function
        logger.info("Initializing new tokenizer")
        _, tokenizer, _, _, _ = prepare_qwen_with_vq_prefix_embedding(
            embedding_size=embedding_size,
            model_id=model_id if model_id else "Qwen/Qwen2.5-3B-Instruct"
        )
    
    if is_vq:
        logger.info(f"Found vector quantizer at {vector_quantizer_path}, loading as VQ model")
        
        # Check for model_info.pt which contains token IDs and model configuration
        model_info_path = os.path.join(model_path, "model_info.pt")
        if os.path.exists(model_info_path):
            # Load using the VQ model's from_pretrained method
            try:
                model = QwenWithVQPrefixEmbedding.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map
                )
                
                # Extract token IDs
                prefix_start_id = model.prefix_start_id
                prefix_end_id = model.prefix_end_id
                class_token_ids = model.class_token_ids
                
                logger.info(f"Successfully loaded VQ model with codebook size {model.vq_num_embeddings}")
                return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, True
            except Exception as e:
                logger.warning(f"Error loading VQ model with from_pretrained: {e}")
                logger.info("Falling back to manual VQ model initialization")
        
        # If we couldn't load directly, try the manual approach
        try:
            # Extract token IDs manually
            prefix_start_id = tokenizer.convert_tokens_to_ids("<PREFIX_START>")
            prefix_end_id = tokenizer.convert_tokens_to_ids("<PREFIX_END>")
            class_token_ids = [tokenizer.convert_tokens_to_ids(f"<CLASS_{i}>") for i in range(10)]
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
            
            # Create VQ model wrapper
            model = QwenWithVQPrefixEmbedding(
                base_model=base_model,
                embedding_size=embedding_size,
                prefix_start_id=prefix_start_id,
                prefix_end_id=prefix_end_id,
                class_token_ids=class_token_ids,
                vq_num_embeddings=512  # Default values that can be overridden by saved state
            )
            
            # Try to load the vector quantizer state dict
            vq_state = torch.load(vector_quantizer_path, map_location='cpu')
            model.vector_quantizer.load_state_dict(vq_state)
            
            # Try to load the embedding projector state dict
            projector_path = os.path.join(model_path, "embedding_projector.pt")
            if os.path.exists(projector_path):
                projector_state = torch.load(projector_path, map_location='cpu')
                model.embedding_projector.load_state_dict(projector_state)
            
            logger.info("Created VQ model wrapper manually")
            return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, True
        except Exception as e:
            logger.warning(f"Failed to initialize VQ model manually: {e}")
    
    # If we're here, either it's not a VQ model or we failed to load it as one
    # Try using the standard model loading function from the parent module
    try:
        # Import here to avoid circular imports
        from ..qwen_prefix import load_pretrained_model
        logger.info("Falling back to standard model loading")
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = load_pretrained_model(
            model_path=model_path,
            device_map=device_map,
            embedding_size=embedding_size,
            model_id=model_id
        )
        logger.info("Successfully loaded standard (non-VQ) model")
        return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, False
    except Exception as e:
        logger.error(f"All model loading methods failed! Error: {e}")
        raise ValueError(f"Could not load model from {model_path}. Error: {e}")