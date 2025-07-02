"""
Visualization mixins for adding optional functionality to visualizations.

This package provides mixin classes that can be combined with base visualizations
to add features like KNN analysis, decision boundaries, pattern mining, etc.
"""

from .knn import BaseKNNVisualization

__all__ = [
    'BaseKNNVisualization'
]