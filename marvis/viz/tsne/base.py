"""
Base t-SNE visualization class that consolidates common functionality.

This module provides the base class for all t-SNE visualizations, eliminating
code duplication between the various t-SNE functions while maintaining
compatibility with the existing BaseVisualization architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from sklearn.manifold import TSNE
from typing import Tuple, Optional, List, Dict, Union, Any
import time

from ..base import BaseVisualization, VisualizationConfig, VisualizationResult
from ..utils.styling import (
    get_distinct_colors,
    create_distinct_color_map,
    create_class_legend,
    extract_visible_classes_from_legend,
    apply_consistent_point_styling
)

logger = logging.getLogger(__name__)


class BaseTSNEVisualization(BaseVisualization):
    """
    Base class for t-SNE visualizations that handles common functionality.
    
    This class provides the core t-SNE fitting and transformation logic
    while allowing subclasses to specialize for classification vs regression
    and different feature combinations (KNN, 3D, etc.).
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None, **kwargs):
        """
        Initialize the base t-SNE visualization.
        
        Args:
            config: Visualization configuration
            **kwargs: Additional parameters for t-SNE (perplexity, learning_rate, etc.)
        """
        super().__init__(config)
        
        # t-SNE specific parameters with sensible defaults
        self.tsne_params = {
            'perplexity': 30,
            'learning_rate': 200,
            'n_iter': 1000,
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'early_exaggeration': 12,
            'min_grad_norm': 1e-7,
            **kwargs
        }
        
        # Determine number of components based on config
        self.n_components = 3 if self.config.use_3d and self.supports_3d else 2
        self.tsne_params['n_components'] = self.n_components
        
        # State for combined data fitting (needed for t-SNE which doesn't support transform)
        self._train_size = None
        self._combined_data = None
        self._combined_labels = None
        self._combined_transformed = None
        
        # Store class information for consistent legend generation
        self._class_names = None
        self._use_semantic_names = False
    
    @property
    def method_name(self) -> str:
        """Return the name of the visualization method."""
        return "t-SNE"
    
    @property
    def supports_3d(self) -> bool:
        """Return whether this method supports 3D visualization."""
        return True
    
    @property
    def supports_regression(self) -> bool:
        """Return whether this method supports regression tasks."""
        return True
    
    @property
    def supports_new_data(self) -> bool:
        """Return whether this method can transform new data after fitting."""
        return False  # t-SNE requires fitting on combined data
    
    def _create_transformer(self, **kwargs) -> TSNE:
        """
        Create the t-SNE transformer object.
        
        Args:
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Configured TSNE transformer
        """
        # Merge parameters
        params = {**self.tsne_params, **kwargs}
        return TSNE(**params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """
        Get a default description for t-SNE visualization.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Description string
        """
        desc = f"t-SNE visualization of {n_samples} samples with {n_features} features"
        if self.config.use_3d:
            desc += " in 3D space"
        else:
            desc += " in 2D space"
        return desc
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit t-SNE on training data, with optional test data for combined fitting.
        
        Since t-SNE doesn't support transform() on new data, we fit on combined
        data when test data is provided, following the pattern from the original
        tsne_functions.py.
        
        Args:
            X: Training data [n_train_samples, n_features]
            y: Training labels [n_train_samples] (optional)
            X_test: Test data [n_test_samples, n_features] (optional)
            **kwargs: Additional parameters for t-SNE
            
        Returns:
            Transformed training data coordinates [n_train_samples, n_components]
        """
        self._train_size = len(X)
        
        if X_test is not None:
            # Combine training and test data for joint t-SNE fitting
            self._combined_data = np.vstack([X, X_test])
            
            if y is not None:
                # Use -1 as placeholder for test labels to distinguish from training
                test_labels = np.full(len(X_test), -1)
                self._combined_labels = np.concatenate([y, test_labels])
            else:
                self._combined_labels = None
            
            # Fit t-SNE on combined data
            self._combined_transformed = super().fit_transform(
                self._combined_data, self._combined_labels, **kwargs
            )
            
            # Return only the training portion
            train_transformed = self._combined_transformed[:self._train_size]
            
        else:
            # Fit only on training data
            self._combined_data = X
            self._combined_labels = y
            train_transformed = super().fit_transform(X, y, **kwargs)
            self._combined_transformed = train_transformed
        
        return train_transformed
    
    def get_test_transformed(self) -> Optional[np.ndarray]:
        """
        Get the transformed coordinates for test data.
        
        Returns:
            Test data coordinates [n_test_samples, n_components] or None
        """
        if (self._combined_transformed is not None and 
            self._train_size is not None and 
            len(self._combined_transformed) > self._train_size):
            return self._combined_transformed[self._train_size:]
        return None
    
    def set_class_info(self, class_names: Optional[List[str]] = None, use_semantic_names: bool = False):
        """
        Set class information for consistent legend generation.
        
        Args:
            class_names: List of class names for semantic labeling
            use_semantic_names: Whether to use semantic names in legends
        """
        self._class_names = class_names
        self._use_semantic_names = use_semantic_names
    
    def create_legend_text(
        self,
        unique_classes: np.ndarray,
        class_color_map: Dict,
        additional_info: str = ""
    ) -> str:
        """
        Create legend text for the visualization.
        
        Args:
            unique_classes: Array of unique class labels present in the data
            class_color_map: Mapping from class labels to colors
            additional_info: Additional information to append to legend
            
        Returns:
            Formatted legend text string
        """
        legend_text = create_class_legend(
            unique_classes,
            class_color_map,
            self._class_names,
            self._use_semantic_names
        )
        
        if additional_info:
            legend_text += f"\n\n{additional_info}"
        
        return legend_text
    
    def get_visible_classes(self, unique_classes: np.ndarray) -> List[Union[int, str]]:
        """
        Get the list of visible classes for metadata.
        
        Args:
            unique_classes: Array of unique class labels present in the data
            
        Returns:
            List of visible class identifiers
        """
        return extract_visible_classes_from_legend(
            unique_classes,
            self._class_names,
            self._use_semantic_names
        )
    
    def apply_zoom_and_viewing_angles(self, ax, use_3d: bool = False):
        """
        Apply zoom factor and viewing angles to the plot.
        
        Args:
            ax: Matplotlib axes object
            use_3d: Whether this is a 3D plot
        """
        # Apply zoom factor
        if self.config.zoom_factor != 1.0:
            if use_3d:
                # For 3D, adjust the viewing distance
                ax.dist = ax.dist * self.config.zoom_factor
            else:
                # For 2D, adjust the axis limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                x_range = (xlim[1] - xlim[0]) / self.config.zoom_factor
                y_range = (ylim[1] - ylim[0]) / self.config.zoom_factor
                
                ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
                ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Apply viewing angles for 3D
        if use_3d and self.config.viewing_angles:
            elev, azim = self.config.viewing_angles[0]  # Use first angle
            ax.view_init(elev=elev, azim=azim)
    
    def create_base_metadata(
        self,
        train_data: np.ndarray,
        test_data: Optional[np.ndarray],
        unique_classes: Optional[np.ndarray],
        highlight_test_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create base metadata common to all t-SNE visualizations.
        
        Args:
            train_data: Transformed training data coordinates
            test_data: Transformed test data coordinates (optional)
            unique_classes: Unique class labels (optional, for classification)
            highlight_test_idx: Index of highlighted test point (optional)
            
        Returns:
            Dictionary with base metadata
        """
        metadata = {
            'n_train_points': len(train_data),
            'n_test_points': len(test_data) if test_data is not None else 0,
            'highlighted_point': highlight_test_idx,
            'is_3d': self.config.use_3d,
            'zoom_factor': self.config.zoom_factor if highlight_test_idx is not None else None,
            'method_name': self.method_name,
            'n_components': self.n_components
        }
        
        # Add class information for classification
        if unique_classes is not None:
            metadata.update({
                'n_classes': len(unique_classes),
                'classes': unique_classes.tolist(),
                'visible_classes': self.get_visible_classes(unique_classes),
                'plot_type': 'classification'
            })
        else:
            metadata.update({
                'visible_classes': [],
                'plot_type': 'regression'
            })
        
        # Add viewing angles for 3D
        if self.config.use_3d and self.config.viewing_angles:
            metadata['viewing_angles'] = self.config.viewing_angles
            metadata['n_views'] = len(self.config.viewing_angles)
        
        return metadata
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add t-SNE specific quality metrics."""
        # t-SNE doesn't provide direct quality metrics like stress or explained variance
        # but we can add the final KL divergence if available from the transformer
        if hasattr(self._transformer, 'kl_divergence_'):
            result.metadata['kl_divergence'] = float(self._transformer.kl_divergence_)
        
        # Add t-SNE parameters to metadata for reproducibility
        result.metadata['tsne_params'] = self.tsne_params.copy()