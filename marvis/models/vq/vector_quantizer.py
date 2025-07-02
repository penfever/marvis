"""
Vector Quantization layer for MARVIS.

This module implements a Vector Quantization (VQ) layer that maps continuous
tabular embeddings to a discrete codebook. The implementation is based on the
VQ-VAE approach described in "Neural Discrete Representation Learning" paper
and has been adapted for use with tabular embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that maps continuous embeddings to a discrete codebook.
    
    Implements a VQ layer with an optional Exponential Moving Average (EMA) update
    rule for the codebook, and a straight-through estimator for backpropagation.
    """
    def __init__(self, 
                embedding_dim: int, 
                num_embeddings: int = 512, 
                commitment_cost: float = 0.25,
                epsilon: float = 1e-5,
                decay: float = 0.99):
        """
        Initialize the Vector Quantizer.
        
        Args:
            embedding_dim: Dimension of the input embeddings and codebook vectors
            num_embeddings: Size of the codebook (number of discrete codes)
            commitment_cost: Weight for the commitment loss
            epsilon: Small constant for numerical stability
            decay: Decay factor for Exponential Moving Average updates (0 for no EMA)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.decay = decay
        
        # Initialize the codebook as a learnable parameter
        # Using Uniform(-1/k, 1/k) initialization for the codebook, where k is num_embeddings
        init_bound = 1 / num_embeddings
        self.codebook = nn.Parameter(
            torch.rand(num_embeddings, embedding_dim).uniform_(-init_bound, init_bound)
        )
        
        # Register buffer for codebook usage tracking
        self.register_buffer('_codebook_usage', torch.zeros(num_embeddings))
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
        
        # Counter for usage resets
        self.usage_count = 0
        self.reset_usage_every = 1000  # Reset usage counter every 1000 forward passes
    
    def forward(self, inputs: torch.Tensor, return_indices: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                                                         Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass that quantizes the input embeddings.
        
        Args:
            inputs: Input embeddings of shape [batch_size, embedding_dim]
            return_indices: Whether to return the indices of the selected codebook vectors
            
        Returns:
            quantized: Quantized embeddings of the same shape as inputs
            loss: The VQ loss value (commitment loss + codebook loss)
            perplexity: Perplexity of the codes (measures codebook usage)
            indices: Indices of the selected codebook vectors (if return_indices=True)
        """
        # inputs shape: [batch_size, embedding_dim]
        # Ensure input is 2D
        orig_shape = inputs.shape
        if len(orig_shape) > 2:
            flat_inputs = inputs.reshape(-1, self.embedding_dim)
        else:
            flat_inputs = inputs
        
        # Calculate distances between inputs and codebook vectors
        # Using expanded form for better numerical stability
        # shape: [batch_size, num_embeddings]
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook**2, dim=1) - \
                    2 * torch.matmul(flat_inputs, self.codebook.t())
        
        # Find the nearest codebook vector for each input
        # shape: [batch_size]
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Update codebook usage tracking
        self._update_usage(encoding_indices)
        
        # Calculate the perplexity of the quantized distribution
        # This measures how many codebook vectors are being used effectively
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        # Get the quantized vectors
        # shape: [batch_size, embedding_dim]
        quantized = self._get_codebook_entries(encoding_indices)
        
        # Update the codebook using Exponential Moving Average when using decay
        if self.training and self.decay > 0:
            self._update_codebook_ema(flat_inputs, encodings)
        
        # Compute the VQ loss
        # Consists of two terms:
        # 1. Commitment loss: encourages the encoder to output vectors close to codebook entries
        # 2. Codebook loss: updates the codebook to better match the encoder outputs
        commitment_loss = torch.mean((quantized.detach() - flat_inputs)**2)
        codebook_loss = torch.mean((quantized - flat_inputs.detach())**2)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
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
    
    def _get_codebook_entries(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve codebook entries for given indices.
        
        Args:
            indices: Tensor of codebook indices
            
        Returns:
            Corresponding codebook vectors
        """
        # shape: [batch_size, embedding_dim]
        return torch.index_select(self.codebook, dim=0, index=indices)
    
    def _update_usage(self, indices: torch.Tensor) -> None:
        """
        Update the codebook usage counter.
        
        Args:
            indices: Tensor of codebook indices used in this batch
        """
        # Count each index once per batch to avoid bias from repeated entries
        unique_indices = torch.unique(indices)
        self._codebook_usage[unique_indices] += 1
        
        # Periodically log and reset usage to monitor codebook utilization
        self.usage_count += 1
        if self.usage_count % self.reset_usage_every == 0:
            total_used = torch.sum(self._codebook_usage > 0).item()
            usage_pct = (total_used / self.num_embeddings) * 100
            
            # Log utilization of codebook
            logger.info(f"Codebook utilization: {total_used}/{self.num_embeddings} vectors used ({usage_pct:.2f}%)")
            
            # Check if usage is too low and log a warning
            if usage_pct < 25 and self.usage_count > self.reset_usage_every * 2:
                logger.warning(f"Low codebook utilization: only {usage_pct:.2f}% in use. Consider reducing codebook size.")
            
            # Reset usage counter
            self._codebook_usage.zero_()
    
    def _update_codebook_ema(self, flat_inputs: torch.Tensor, encodings: torch.Tensor) -> None:
        """
        Update the codebook using Exponential Moving Average (EMA).
        This helps stabilize training by slowly updating the codebook.
        
        Args:
            flat_inputs: Input embeddings 
            encodings: One-hot encodings of selected codebook vectors
        """
        # EMA update for cluster size
        batch_size = flat_inputs.shape[0]
        
        # Sum of encodings is how many times each codebook entry was used
        n = torch.sum(encodings, dim=0)
        
        # Update exponential moving averages
        self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                (1 - self.decay) * n
        
        # Calculate sum of embeddings per codebook entry
        dw = torch.matmul(encodings.t(), flat_inputs)
        
        # Update EMA weights
        self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw
        
        # Normalize cluster size to avoid explosion / decay to zero
        n = torch.clamp(self._ema_cluster_size, min=self.epsilon)
        
        # Update codebook values
        updated_codebook = self._ema_w / n.unsqueeze(1)
        
        # Replace codebook values with updated ones
        with torch.no_grad():
            self.codebook.data.copy_(updated_codebook)
    
    def get_codebook_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get the indices of the nearest codebook entries for the given inputs.
        Useful for encoding inputs with the quantizer.
        
        Args:
            inputs: Input embeddings
            
        Returns:
            indices: Indices of the nearest codebook entries
        """
        # Flatten input if needed
        orig_shape = inputs.shape
        if len(orig_shape) > 2:
            flat_inputs = inputs.reshape(-1, self.embedding_dim)
        else:
            flat_inputs = inputs
        
        # Calculate distances
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook**2, dim=1) - \
                    2 * torch.matmul(flat_inputs, self.codebook.t())
        
        # Return indices
        return torch.argmin(distances, dim=1)