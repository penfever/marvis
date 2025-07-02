"""
UMAP (Uniform Manifold Approximation and Projection) visualization implementation.

UMAP is excellent for preserving both local and global structure, often providing
clearer cluster separation than t-SNE while being more computationally efficient.
"""

import numpy as np
from typing import Any, Dict
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from ..base import BaseVisualization, VisualizationResult

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    create_regression_color_map
)

logger = logging.getLogger(__name__)


class UMAPVisualization(BaseVisualization):
    """
    UMAP visualization implementation.
    
    UMAP (Uniform Manifold Approximation and Projection) is a dimensionality
    reduction technique that preserves both local and global structure of data.
    It often provides clearer cluster separation than t-SNE.
    """
    
    @property
    def method_name(self) -> str:
        return "UMAP"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return True
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create UMAP transformer."""
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP not available. Install with: pip install umap-learn"
            )
        
        # Set default parameters optimized for visualization
        umap_params = {
            'n_neighbors': kwargs.get('n_neighbors', 15),
            'n_components': 3 if self.config.use_3d else 2,
            'metric': kwargs.get('metric', 'euclidean'),
            'min_dist': kwargs.get('min_dist', 0.1),
            'spread': kwargs.get('spread', 1.0),
            'random_state': self.config.random_state,
            'n_epochs': kwargs.get('n_epochs', None),  # Auto-determined
            'learning_rate': kwargs.get('learning_rate', 1.0),
            'init': kwargs.get('init', 'spectral'),
            'low_memory': kwargs.get('low_memory', False),
            'verbose': kwargs.get('verbose', False)
        }
        
        # Handle supervised UMAP if target values are available
        if kwargs.get('target_metric'):
            umap_params['target_metric'] = kwargs['target_metric']
        
        # Remove None values
        umap_params = {k: v for k, v in umap_params.items() if v is not None}
        
        self.logger.info(f"Creating UMAP with parameters: {umap_params}")
        
        return umap.UMAP(**umap_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for UMAP."""
        components = "3D" if self.config.use_3d else "2D"
        
        description = (
            f"UMAP {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"UMAP preserves both local neighborhood structure and global topology, "
            f"often revealing clearer cluster boundaries than t-SNE while maintaining "
            f"more of the global structure."
        )
        
        # Add parameter information
        params = self.config.extra_params
        if 'n_neighbors' in params:
            description += f" Using {params['n_neighbors']} neighbors for local structure."
        if 'min_dist' in params:
            description += f" Minimum distance parameter: {params['min_dist']}."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add UMAP-specific quality metrics."""
        if self._transformer is not None and hasattr(self._transformer, 'embedding_'):
            # UMAP doesn't have built-in quality metrics like stress
            # But we can compute some useful information
            
            # Embedding shape
            result.metadata['embedding_shape'] = self._transformer.embedding_.shape
            
            # UMAP parameters used
            result.metadata['n_neighbors'] = self._transformer.n_neighbors
            result.metadata['min_dist'] = self._transformer.min_dist
            result.metadata['metric'] = self._transformer.metric
            
            # If available, add graph information
            if hasattr(self._transformer, 'graph_'):
                n_edges = self._transformer.graph_.nnz
                result.metadata['n_edges'] = n_edges
                result.metadata['avg_degree'] = 2 * n_edges / len(self._transformer.embedding_)
            
            self.logger.debug(f"Added UMAP quality metrics: {result.metadata}")


def create_umap_visualization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    use_3d: bool = False,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create UMAP visualization.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter
        metric: Distance metric to use
        use_3d: Whether to create 3D embedding
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing embedding results and visualization
    """
    from ..base import VisualizationConfig
    
    # Create configuration
    config = VisualizationConfig(
        use_3d=use_3d,
        random_state=random_state,
        extra_params={
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            **kwargs
        }
    )
    
    # Create visualization
    viz = UMAPVisualization(config)
    
    # Fit and transform
    train_embedding = viz.fit_transform(X_train, y_train)
    test_embedding = viz.transform(X_test) if X_test is not None else None
    
    # Generate plot
    result = viz.generate_plot(
        transformed_data=train_embedding,
        y=y_train,
        test_data=test_embedding
    )
    
    return {
        'train_embedding': train_embedding,
        'test_embedding': test_embedding,
        'visualization_result': result,
        'transformer': viz._transformer
    }