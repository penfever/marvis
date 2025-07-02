"""
t-SNE visualization classes for classification and regression tasks.

This package provides unified t-SNE visualization classes that replace the
legacy functions in tsne_functions.py while maintaining feature parity and
integrating with the BaseVisualization architecture.
"""

from .base import BaseTSNEVisualization
from .classification import TSNEClassificationVisualization
from .regression import TSNERegressionVisualization

__all__ = [
    'BaseTSNEVisualization',
    'TSNEClassificationVisualization', 
    'TSNERegressionVisualization'
]