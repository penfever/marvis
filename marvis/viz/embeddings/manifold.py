"""
sklearn.manifold visualization implementations.

This module provides wrappers for various manifold learning methods from
scikit-learn, including LocallyLinearEmbedding, SpectralEmbedding, Isomap, and MDS.
"""

import numpy as np
from typing import Any, Dict
import logging

from sklearn.manifold import (
    LocallyLinearEmbedding,
    SpectralEmbedding, 
    Isomap,
    MDS
)
from ..base import BaseVisualization, VisualizationResult

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    create_regression_color_map
)

logger = logging.getLogger(__name__)


class LocallyLinearEmbeddingVisualization(BaseVisualization):
    """
    Locally Linear Embedding (LLE) visualization.
    
    LLE attempts to preserve local linear structure by reconstructing each
    point from its neighbors. Supports standard, modified, Hessian, and LTSA variants.
    """
    
    @property
    def method_name(self) -> str:
        method = self.config.extra_params.get('method', 'standard')
        return f"LLE-{method}"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return False  # LLE doesn't support transform on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create LLE transformer."""
        
        lle_params = {
            'n_neighbors': kwargs.get('n_neighbors', 5),
            'n_components': 3 if self.config.use_3d else 2,
            'reg': kwargs.get('reg', 0.001),
            'eigen_solver': kwargs.get('eigen_solver', 'auto'),
            'tol': kwargs.get('tol', 1e-6),
            'max_iter': kwargs.get('max_iter', 100),
            'method': kwargs.get('method', 'standard'),  # standard, modified, hessian, ltsa
            'hessian_tol': kwargs.get('hessian_tol', 1e-4),
            'modified_tol': kwargs.get('modified_tol', 1e-12),
            'random_state': self.config.random_state,
            'n_jobs': kwargs.get('n_jobs', None)
        }
        
        # Remove None values
        lle_params = {k: v for k, v in lle_params.items() if v is not None}
        
        self.logger.info(f"Creating LLE with parameters: {lle_params}")
        
        return LocallyLinearEmbedding(**lle_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for LLE."""
        components = "3D" if self.config.use_3d else "2D"
        method = self.config.extra_params.get('method', 'standard')
        
        method_descriptions = {
            'standard': "standard LLE preserves local linear structure",
            'modified': "modified LLE adds regularization for stability",
            'hessian': "Hessian LLE uses local Hessian information",
            'ltsa': "LTSA (Local Tangent Space Alignment) aligns local tangent spaces"
        }
        
        description = (
            f"LLE {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"The {method_descriptions.get(method, method)} by reconstructing each point "
            f"from its local neighborhood."
        )
        
        params = self.config.extra_params
        if 'n_neighbors' in params:
            description += f" Using {params['n_neighbors']} neighbors for local reconstruction."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add LLE-specific quality metrics."""
        if self._transformer is not None:
            # Reconstruction error
            if hasattr(self._transformer, 'reconstruction_error_'):
                result.reconstruction_error = float(self._transformer.reconstruction_error_)
                result.metadata['reconstruction_error'] = float(self._transformer.reconstruction_error_)
            
            # LLE parameters
            result.metadata['n_neighbors'] = self._transformer.n_neighbors
            result.metadata['method'] = self._transformer.method
            result.metadata['regularization'] = self._transformer.reg
            
            self.logger.debug(f"Added LLE quality metrics: {result.metadata}")


class SpectralEmbeddingVisualization(BaseVisualization):
    """
    Spectral Embedding visualization.
    
    Spectral embedding uses graph Laplacian eigendecomposition to find
    a low-dimensional representation that preserves the graph structure.
    """
    
    @property
    def method_name(self) -> str:
        return "Spectral"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return False  # Spectral embedding doesn't support transform on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create Spectral Embedding transformer."""
        
        spectral_params = {
            'n_neighbors': kwargs.get('n_neighbors', None),
            'n_components': 3 if self.config.use_3d else 2,
            'affinity': kwargs.get('affinity', 'nearest_neighbors'),
            'gamma': kwargs.get('gamma', None),
            'random_state': self.config.random_state,
            'eigen_solver': kwargs.get('eigen_solver', None),
            'n_jobs': kwargs.get('n_jobs', None)
        }
        
        # Handle affinity-specific parameters
        if spectral_params['affinity'] == 'nearest_neighbors':
            if spectral_params['n_neighbors'] is None:
                spectral_params['n_neighbors'] = max(2, int(np.sqrt(kwargs.get('n_samples', 100))))
        
        # Remove None values
        spectral_params = {k: v for k, v in spectral_params.items() if v is not None}
        
        self.logger.info(f"Creating Spectral Embedding with parameters: {spectral_params}")
        
        return SpectralEmbedding(**spectral_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for Spectral Embedding."""
        components = "3D" if self.config.use_3d else "2D"
        affinity = self.config.extra_params.get('affinity', 'nearest_neighbors')
        
        description = (
            f"Spectral {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"Spectral embedding uses graph Laplacian eigendecomposition to preserve "
            f"the manifold structure using {affinity} affinity."
        )
        
        params = self.config.extra_params
        if 'n_neighbors' in params and affinity == 'nearest_neighbors':
            description += f" Using {params['n_neighbors']} neighbors for graph construction."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add Spectral Embedding-specific quality metrics."""
        if self._transformer is not None:
            # Spectral embedding parameters
            result.metadata['affinity'] = self._transformer.affinity
            result.metadata['n_neighbors'] = getattr(self._transformer, 'n_neighbors', None)
            result.metadata['gamma'] = getattr(self._transformer, 'gamma', None)
            
            self.logger.debug(f"Added Spectral Embedding quality metrics: {result.metadata}")


class IsomapVisualization(BaseVisualization):
    """
    Isomap visualization.
    
    Isomap preserves geodesic distances on the manifold by computing
    shortest paths on the neighborhood graph.
    """
    
    @property
    def method_name(self) -> str:
        return "Isomap"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return True  # Isomap supports transform on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create Isomap transformer."""
        
        isomap_params = {
            'n_neighbors': kwargs.get('n_neighbors', 20),
            'n_components': 3 if self.config.use_3d else 2,
            'eigen_solver': kwargs.get('eigen_solver', 'auto'),
            'tol': kwargs.get('tol', 0),
            'max_iter': kwargs.get('max_iter', None),
            'path_method': kwargs.get('path_method', 'auto'),
            'neighbors_algorithm': kwargs.get('neighbors_algorithm', 'auto'),
            'n_jobs': kwargs.get('n_jobs', None),
            'metric': kwargs.get('metric', 'minkowski'),
            'p': kwargs.get('p', 2)
        }
        
        # Remove None values
        isomap_params = {k: v for k, v in isomap_params.items() if v is not None}
        
        self.logger.info(f"Creating Isomap with parameters: {isomap_params}")
        
        return Isomap(**isomap_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for Isomap."""
        components = "3D" if self.config.use_3d else "2D"
        
        description = (
            f"Isomap {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"Isomap preserves geodesic distances on the manifold by computing shortest "
            f"paths through the neighborhood graph."
        )
        
        params = self.config.extra_params
        if 'n_neighbors' in params:
            description += f" Using {params['n_neighbors']} neighbors for graph construction."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add Isomap-specific quality metrics."""
        if self._transformer is not None:
            # Reconstruction error
            if hasattr(self._transformer, 'reconstruction_error_'):
                result.reconstruction_error = float(self._transformer.reconstruction_error_)
                result.metadata['reconstruction_error'] = float(self._transformer.reconstruction_error_)
            
            # Isomap parameters
            result.metadata['n_neighbors'] = self._transformer.n_neighbors
            result.metadata['path_method'] = self._transformer.path_method
            result.metadata['metric'] = self._transformer.metric
            
            self.logger.debug(f"Added Isomap quality metrics: {result.metadata}")


class MDSVisualization(BaseVisualization):
    """
    Multidimensional Scaling (MDS) visualization.
    
    MDS preserves pairwise distances between points. Supports both
    metric and non-metric variants.
    """
    
    @property
    def method_name(self) -> str:
        metric = self.config.extra_params.get('metric', True)
        return "MDS" if metric else "Non-metric MDS"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return False  # MDS doesn't support transform on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create MDS transformer."""
        
        mds_params = {
            'n_components': 3 if self.config.use_3d else 2,
            'metric': kwargs.get('metric', True),
            'n_init': kwargs.get('n_init', 4),
            'max_iter': kwargs.get('max_iter', 300),
            'verbose': kwargs.get('verbose', 0),
            'eps': kwargs.get('eps', 1e-3),
            'n_jobs': kwargs.get('n_jobs', None),
            'random_state': self.config.random_state,
            'dissimilarity': kwargs.get('dissimilarity', 'euclidean')
        }
        
        # Remove None values
        mds_params = {k: v for k, v in mds_params.items() if v is not None}
        
        self.logger.info(f"Creating MDS with parameters: {mds_params}")
        
        return MDS(**mds_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for MDS."""
        components = "3D" if self.config.use_3d else "2D"
        metric = self.config.extra_params.get('metric', True)
        variant = "metric" if metric else "non-metric"
        
        description = (
            f"{variant.title()} MDS {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"MDS preserves pairwise distances between points"
        )
        
        if metric:
            description += " using exact distance preservation."
        else:
            description += " using rank-order distance preservation."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add MDS-specific quality metrics."""
        if self._transformer is not None:
            # Stress (final value)
            if hasattr(self._transformer, 'stress_'):
                result.stress = float(self._transformer.stress_)
                result.metadata['stress'] = float(self._transformer.stress_)
            
            # Number of iterations
            if hasattr(self._transformer, 'n_iter_'):
                result.metadata['n_iterations'] = int(self._transformer.n_iter_)
            
            # MDS parameters
            result.metadata['metric'] = self._transformer.metric
            result.metadata['dissimilarity'] = self._transformer.dissimilarity
            
            self.logger.debug(f"Added MDS quality metrics: stress = {result.stress:.4f}")


# Convenience functions for creating manifold visualizations
def create_manifold_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    methods: list = None,
    use_3d: bool = False,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a comparison of multiple manifold learning methods.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data (optional)
        methods: List of methods to compare (default: all available)
        use_3d: Whether to use 3D embeddings
        random_state: Random seed
        **kwargs: Additional parameters for methods
        
    Returns:
        Dictionary with results for each method
    """
    from ..base import VisualizationConfig
    
    if methods is None:
        methods = ['lle', 'spectral', 'isomap', 'mds']
    
    method_classes = {
        'lle': LocallyLinearEmbeddingVisualization,
        'spectral': SpectralEmbeddingVisualization,
        'isomap': IsomapVisualization,
        'mds': MDSVisualization
    }
    
    results = {}
    
    for method in methods:
        if method not in method_classes:
            logger.warning(f"Unknown method: {method}, skipping")
            continue
        
        try:
            # Create configuration
            config = VisualizationConfig(
                use_3d=use_3d,
                random_state=random_state,
                extra_params=kwargs.get(f'{method}_params', {})
            )
            
            # Create visualization
            viz = method_classes[method](config)
            
            # Fit and transform
            train_embedding = viz.fit_transform(X_train, y_train)
            
            # Transform test data if supported
            test_embedding = None
            if X_test is not None and viz.supports_new_data:
                test_embedding = viz.transform(X_test)
            
            # Generate plot
            result = viz.generate_plot(
                transformed_data=train_embedding,
                y=y_train,
                test_data=test_embedding
            )
            
            results[method] = {
                'train_embedding': train_embedding,
                'test_embedding': test_embedding,
                'visualization_result': result,
                'transformer': viz._transformer,
                'supports_new_data': viz.supports_new_data
            }
            
            logger.info(f"Successfully created {method} visualization")
            
        except Exception as e:
            logger.error(f"Failed to create {method} visualization: {e}")
            results[method] = {'error': str(e)}
    
    return results