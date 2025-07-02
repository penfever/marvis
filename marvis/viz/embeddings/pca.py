"""
PCA (Principal Component Analysis) visualization implementation.

PCA is a linear dimensionality reduction technique that finds the directions
of maximum variance in the data. It's excellent for understanding linear
structure and comparing with nonlinear methods.
"""

import numpy as np
from typing import Any, Dict, Optional
import logging

from sklearn.decomposition import PCA
from ..base import BaseVisualization, VisualizationResult

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    create_regression_color_map
)

logger = logging.getLogger(__name__)


class PCAVisualization(BaseVisualization):
    """
    PCA visualization implementation.
    
    Principal Component Analysis provides linear dimensionality reduction
    by finding directions of maximum variance. Useful for understanding
    linear structure and as a baseline for comparison with nonlinear methods.
    """
    
    @property
    def method_name(self) -> str:
        return "PCA"
    
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
        """Create PCA transformer."""
        
        # Set parameters
        pca_params = {
            'n_components': 3 if self.config.use_3d else 2,
            'random_state': self.config.random_state,
            'svd_solver': kwargs.get('svd_solver', 'auto'),
            'whiten': kwargs.get('whiten', False),
            'copy': kwargs.get('copy', True)
        }
        
        # Handle different solvers
        if 'n_iter' in kwargs and pca_params['svd_solver'] in ['randomized', 'arpack']:
            pca_params['max_iter'] = kwargs['n_iter']
        
        if 'tol' in kwargs and pca_params['svd_solver'] == 'arpack':
            pca_params['tol'] = kwargs['tol']
        
        # Remove None values
        pca_params = {k: v for k, v in pca_params.items() if v is not None}
        
        self.logger.info(f"Creating PCA with parameters: {pca_params}")
        
        return PCA(**pca_params)
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the PCA method and transform the data with data cleaning.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Optional target values [n_samples]
            **kwargs: Additional parameters for the method
            
        Returns:
            Transformed coordinates [n_samples, n_components]
        """
        # Enhanced data cleaning for PCA numerical stability
        X_clean = self._robust_data_cleaning(X)
        
        # Call parent's fit_transform with cleaned data
        return super().fit_transform(X_clean, y, **kwargs)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted PCA with data cleaning.
        
        Args:
            X: New data to transform [n_samples, n_features]
            
        Returns:
            Transformed coordinates [n_samples, n_components]
        """
        if not self._fitted:
            raise ValueError("Must fit the model before transforming new data")
        
        # Enhanced data cleaning for numerical stability
        X_clean = self._robust_data_cleaning(X)
        
        return self._transformer.transform(X_clean)
    
    def _robust_data_cleaning(self, X: np.ndarray) -> np.ndarray:
        """
        Robust data cleaning to prevent PCA numerical issues.
        
        Args:
            X: Input data array
            
        Returns:
            Cleaned data array safe for PCA
        """
        # Step 1: Handle NaN, inf values with conservative bounds
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Step 2: Clip to conservative range for float16/32 safety
        X_clean = np.clip(X_clean, -1e3, 1e3)
        
        # Step 3: Check for constant or near-constant features that cause issues
        feature_vars = np.var(X_clean, axis=0)
        constant_threshold = 1e-12
        
        # Add small noise to constant features to prevent singular matrices
        constant_mask = feature_vars < constant_threshold
        if np.any(constant_mask):
            self.logger.debug(f"Found {np.sum(constant_mask)} constant/near-constant features, adding noise")
            noise = np.random.normal(0, 1e-6, X_clean[:, constant_mask].shape)
            X_clean[:, constant_mask] += noise
        
        # Step 4: Check for extreme variance ratios that can cause overflow
        feature_vars_updated = np.var(X_clean, axis=0)
        if len(feature_vars_updated) > 1:
            var_ratio = np.max(feature_vars_updated) / np.min(feature_vars_updated[feature_vars_updated > 0])
            if var_ratio > 1e12:
                self.logger.debug(f"Extreme variance ratio {var_ratio:.2e}, applying standardization")
                # Apply StandardScaler to reduce variance differences
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_clean = scaler.fit_transform(X_clean)
        
        return X_clean
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for PCA."""
        components = "3D" if self.config.use_3d else "2D"
        n_comp = 3 if self.config.use_3d else 2
        
        description = (
            f"PCA {components} projection of {n_samples} samples from {n_features} dimensions. "
            f"PCA finds the {n_comp} directions of maximum variance in the data, "
            f"providing a linear view of the data structure."
        )
        
        # Add explained variance if available
        if self._transformer and hasattr(self._transformer, 'explained_variance_ratio_'):
            var_explained = np.sum(self._transformer.explained_variance_ratio_) * 100
            description += f" The first {n_comp} components explain {var_explained:.1f}% of the total variance."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add PCA-specific quality metrics."""
        if self._transformer is not None and hasattr(self._transformer, 'explained_variance_ratio_'):
            # Explained variance ratio
            result.explained_variance = float(np.sum(self._transformer.explained_variance_ratio_))
            
            # Individual component variances
            result.metadata['explained_variance_ratio'] = self._transformer.explained_variance_ratio_.tolist()
            result.metadata['explained_variance'] = self._transformer.explained_variance_.tolist()
            result.metadata['cumulative_variance'] = np.cumsum(self._transformer.explained_variance_ratio_).tolist()
            
            # Singular values if available
            if hasattr(self._transformer, 'singular_values_'):
                result.metadata['singular_values'] = self._transformer.singular_values_.tolist()
            
            # Number of components
            result.metadata['n_components'] = self._transformer.n_components_
            
            self.logger.debug(f"Added PCA quality metrics: explained variance = {result.explained_variance:.3f}")
    
    def get_component_loadings(self) -> np.ndarray:
        """
        Get the component loadings (principal components).
        
        Returns:
            Component loadings matrix [n_features, n_components]
        """
        if not self._fitted or self._transformer is None:
            raise ValueError("Must fit the model before getting component loadings")
        
        return self._transformer.components_.T
    
    def get_explained_variance_plot(self) -> 'VisualizationResult':
        """
        Create a plot showing explained variance by component.
        
        Returns:
            VisualizationResult with explained variance plot
        """
        if not self._fitted or not hasattr(self._transformer, 'explained_variance_ratio_'):
            raise ValueError("Must fit the model before plotting explained variance")
        
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        # Create explained variance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        n_components = len(self._transformer.explained_variance_ratio_)
        components = range(1, n_components + 1)
        
        # Individual explained variance using unified styling
        color_map = create_distinct_color_map(n_components)
        bar_colors = [color_map[i % len(color_map)] for i in range(n_components)]
        
        ax1.bar(components, self._transformer.explained_variance_ratio_, color=bar_colors, alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance using unified styling
        cumulative_var = np.cumsum(self._transformer.explained_variance_ratio_)
        primary_color = list(color_map.values())[0] if color_map else 'blue'
        ax2.plot(components, cumulative_var, 'o-', color=primary_color, linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Apply consistent legend formatting
        apply_consistent_legend_formatting(ax1, use_3d=False)
        apply_consistent_legend_formatting(ax2, use_3d=False)
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=self._transformer.explained_variance_ratio_,
            description=f"Explained variance plot for PCA with {n_components} components",
            method_name="PCA Explained Variance",
            config=self.config,
            metadata={
                'explained_variance_ratio': self._transformer.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumulative_var.tolist(),
                'total_variance_explained': float(cumulative_var[-1])
            }
        )
        
        return result


def create_pca_visualization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_3d: bool = False,
    whiten: bool = False,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create PCA visualization.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        use_3d: Whether to create 3D projection
        whiten: Whether to whiten the components
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing projection results and visualization
    """
    from ..base import VisualizationConfig
    
    # Create configuration
    config = VisualizationConfig(
        use_3d=use_3d,
        random_state=random_state,
        extra_params={
            'whiten': whiten,
            **kwargs
        }
    )
    
    # Create visualization
    viz = PCAVisualization(config)
    
    # Fit and transform
    train_projection = viz.fit_transform(X_train, y_train)
    test_projection = viz.transform(X_test) if X_test is not None else None
    
    # Generate plot
    result = viz.generate_plot(
        transformed_data=train_projection,
        y=y_train,
        test_data=test_projection
    )
    
    # Get explained variance plot
    variance_plot = viz.get_explained_variance_plot()
    
    return {
        'train_projection': train_projection,
        'test_projection': test_projection,
        'visualization_result': result,
        'explained_variance_plot': variance_plot,
        'transformer': viz._transformer,
        'component_loadings': viz.get_component_loadings()
    }