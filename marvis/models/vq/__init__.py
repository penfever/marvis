"""
Vector Quantization (VQ) models for MARVIS.

This module contains implementations of the vector quantization approach for
tabular embeddings in language models.
"""

from .qwen_vq import (QwenWithVQPrefixEmbedding, load_vq_pretrained_model,
                      prepare_qwen_with_vq_prefix_embedding)
from .vector_quantizer import VectorQuantizer

__all__ = [
    "VectorQuantizer",
    "QwenWithVQPrefixEmbedding",
    "prepare_qwen_with_vq_prefix_embedding",
    "load_vq_pretrained_model",
]
