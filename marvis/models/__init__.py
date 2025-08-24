"""
Models for MARVIS.

This module includes both standard embedding approaches and vector-quantized
versions for improved efficiency and representation.
"""

from .marvis_tsne import (MarvisAudioTsneClassifier, MarvisImageTsneClassifier,
                          MarvisTsneClassifier)
from .qwen_prefix import (QwenWithPrefixEmbedding, load_pretrained_model,
                          prepare_qwen_with_prefix_embedding)
from .vq import (QwenWithVQPrefixEmbedding, VectorQuantizer,
                 prepare_qwen_with_vq_prefix_embedding)

__all__ = [
    # Standard embedding models
    "QwenWithPrefixEmbedding",
    "prepare_qwen_with_prefix_embedding",
    "load_pretrained_model",
    # Vector-quantized models
    "VectorQuantizer",
    "QwenWithVQPrefixEmbedding",
    "prepare_qwen_with_vq_prefix_embedding",
    # MARVIS t-SNE classifiers
    "MarvisTsneClassifier",
    "MarvisAudioTsneClassifier",
    "MarvisImageTsneClassifier",
]
