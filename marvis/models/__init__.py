"""
Models for MARVIS.

This module includes both standard embedding approaches and vector-quantized
versions for improved efficiency and representation.
"""

from .qwen_prefix import QwenWithPrefixEmbedding, prepare_qwen_with_prefix_embedding, load_pretrained_model
from .vq import VectorQuantizer, QwenWithVQPrefixEmbedding, prepare_qwen_with_vq_prefix_embedding
from .marvis_tsne import MarvisTsneClassifier, MarvisAudioTsneClassifier, MarvisImageTsneClassifier

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
    "MarvisImageTsneClassifier"
]