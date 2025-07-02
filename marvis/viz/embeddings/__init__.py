"""
Embedding-based visualizations for dimensionality reduction.
"""

from .tsne import TSNEVisualization
from .umap import UMAPVisualization
from .pca import PCAVisualization
from .manifold import (
    LocallyLinearEmbeddingVisualization,
    SpectralEmbeddingVisualization,
    IsomapVisualization,
    MDSVisualization
)

__all__ = [
    'TSNEVisualization',
    'UMAPVisualization', 
    'PCAVisualization',
    'LocallyLinearEmbeddingVisualization',
    'SpectralEmbeddingVisualization',
    'IsomapVisualization',
    'MDSVisualization'
]