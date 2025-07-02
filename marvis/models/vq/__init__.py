"""
Vector Quantization (VQ) models for MARVIS.

This module contains implementations of the vector quantization approach for
tabular embeddings in language models.
"""

from .vector_quantizer import VectorQuantizer
from .qwen_vq import QwenWithVQPrefixEmbedding, prepare_qwen_with_vq_prefix_embedding, load_vq_pretrained_model

__all__ = [
    "VectorQuantizer",
    "QwenWithVQPrefixEmbedding",
    "prepare_qwen_with_vq_prefix_embedding",
    "load_vq_pretrained_model"
]