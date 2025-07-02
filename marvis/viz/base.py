"""
Base classes for the MARVIS visualization system.

This module provides abstract base classes and data structures for implementing
modular visualization components that can be composed together for enhanced
reasoning in VLM backends.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    
    # General parameters
    figsize: Tuple[int, int] = (12, 8)  # Match original high-quality implementation
    dpi: int = 100
    random_state: int = 42
    
    # Color and styling
    colormap: str = 'tab10'
    point_size: float = 50.0
    alpha: float = 0.7
    
    # 3D options
    use_3d: bool = False
    viewing_angles: Optional[List[Tuple[float, float]]] = None
    
    # Zoom and layout
    zoom_factor: float = 2.0
    tight_layout: bool = True
    
    # Text and labels
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    show_legend: bool = True
    
    # Output format
    image_format: str = 'RGB'
    max_image_size: int = 2048
    
    # Task-specific options
    task_type: str = 'classification'  # 'classification' or 'regression'
    
    # KNN connections support
    use_knn_connections: bool = False
    nn_k: int = 5  # Number of nearest neighbors (unified parameter)
    
    # Test point visualization
    show_test_points: bool = False  # Whether to show all test points (gray squares)
    
    # Additional options for specific visualizations
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of a visualization operation."""
    
    # Core outputs
    image: Image.Image
    transformed_data: np.ndarray
    description: str
    
    # Metadata
    method_name: str
    config: VisualizationConfig
    
    # Analysis information
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'visible_classes': [],  # Classes visible in this specific visualization
        'all_classes': [],      # Complete set of classes from the dataset
        'plot_type': 'classification'  # 'classification' or 'regression'
    })
    
    # Performance metrics
    fit_time: float = 0.0
    transform_time: float = 0.0
    plot_time: float = 0.0
    
    # Coordinates for specific points (if highlighting)
    highlighted_indices: Optional[List[int]] = None
    highlighted_coords: Optional[np.ndarray] = None
    
    # Legend information for VLM prompts
    legend_text: str = ""
    
    # Quality metrics (if applicable)
    stress: Optional[float] = None  # For MDS
    reconstruction_error: Optional[float] = None  # For LLE
    explained_variance: Optional[float] = None  # For PCA


class BaseVisualization(ABC):
    """
    Abstract base class for all visualization methods.
    
    This class defines the interface that all visualization methods must implement
    to be compatible with the context composer system.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization method.
        
        Args:
            config: Configuration for the visualization
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State variables
        self._fitted = False
        self._transformer = None
        self._last_result = None
        
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the visualization method."""
        pass
    
    @property
    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this method supports 3D visualization."""
        pass
    
    @property
    @abstractmethod
    def supports_regression(self) -> bool:
        """Return whether this method supports regression tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_new_data(self) -> bool:
        """Return whether this method can transform new data after fitting."""
        pass
    
    @abstractmethod
    def _create_transformer(self, **kwargs) -> Any:
        """
        Create the underlying transformer object.
        
        Returns:
            Transformer object (e.g., TSNE, UMAP, etc.)
        """
        pass
    
    @abstractmethod
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """
        Get a default description for this visualization method.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Description string
        """
        pass
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the visualization method and transform the data.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Optional target values [n_samples]
            **kwargs: Additional parameters for the method
            
        Returns:
            Transformed coordinates [n_samples, n_components]
        """
        import time
        
        start_time = time.time()
        
        # Merge config extra_params with kwargs
        merged_kwargs = {**self.config.extra_params, **kwargs}
        
        # Create transformer
        self._transformer = self._create_transformer(**merged_kwargs)
        
        fit_start = time.time()
        
        # Fit and transform
        transformed = self._transformer.fit_transform(X)
        
        fit_time = time.time() - fit_start
        
        self._fitted = True
        
        # Store embeddings for KNN analysis if enabled
        if self.config.use_knn_connections:
            self._training_embeddings = X.copy()
            if y is not None:
                self._training_labels = y.copy()
        
        # Store timing information
        self._last_fit_time = fit_time
        self._last_transform_time = 0.0  # Included in fit_transform
        
        self.logger.info(f"Fitted {self.method_name} in {fit_time:.2f}s")
        
        return transformed
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted method.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Transformed coordinates [n_samples, n_components]
        """
        if not self._fitted:
            raise ValueError("Must call fit_transform before transform")
        
        if not self.supports_new_data:
            raise NotImplementedError(f"{self.method_name} does not support transforming new data")
        
        import time
        start_time = time.time()
        
        transformed = self._transformer.transform(X)
        
        # Store test embeddings for KNN analysis if enabled
        if self.config.use_knn_connections:
            self._test_embeddings = X.copy()
        
        self._last_transform_time = time.time() - start_time
        
        return transformed
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[List[int]] = None,
        test_data: Optional[np.ndarray] = None,
        highlight_test_indices: Optional[List[int]] = None,
        **kwargs
    ) -> VisualizationResult:
        """
        Generate a plot from transformed data.
        
        Args:
            transformed_data: Transformed coordinates
            y: Optional target values for coloring
            highlight_indices: Indices of points to highlight in training data
            test_data: Optional test data coordinates
            highlight_test_indices: Indices of test points to highlight with red X
            **kwargs: Additional plotting parameters
            
        Returns:
            VisualizationResult object
        """
        import time
        import io
        
        plot_start = time.time()
        
        # Determine number of components
        n_components = transformed_data.shape[1]
        use_3d = self.config.use_3d and n_components >= 3 and self.supports_3d
        
        # Create figure
        if use_3d:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Extract class naming parameters from kwargs
        class_names = kwargs.get('class_names', None)
        use_semantic_names = kwargs.get('use_semantic_names', False)
        
        # Remove class naming parameters from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['class_names', 'use_semantic_names']}
        
        # Plot based on task type
        if self.config.task_type == 'regression' and y is not None:
            plot_result = self._plot_regression(ax, transformed_data, y, highlight_indices, test_data, highlight_test_indices, use_3d, **filtered_kwargs)
        else:
            plot_result = self._plot_classification(ax, transformed_data, y, highlight_indices, test_data, highlight_test_indices, use_3d, class_names, use_semantic_names, **filtered_kwargs)
        
        # Apply styling
        self._apply_plot_styling(ax, use_3d)
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Convert to desired format
        if self.config.image_format == 'RGB' and image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        # Resize if needed
        if image.width > self.config.max_image_size or image.height > self.config.max_image_size:
            ratio = min(self.config.max_image_size / image.width, self.config.max_image_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        plot_time = time.time() - plot_start
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=transformed_data,
            description=self._get_default_description(len(transformed_data), transformed_data.shape[1]),
            method_name=self.method_name,
            config=self.config,
            fit_time=getattr(self, '_last_fit_time', 0.0),
            transform_time=getattr(self, '_last_transform_time', 0.0),
            plot_time=plot_time,
            highlighted_indices=highlight_indices,
            highlighted_coords=transformed_data[highlight_indices] if highlight_indices else None,
            legend_text=plot_result.get('legend_text', ''),
            metadata=plot_result.get('metadata', {})
        )
        
        # Add method-specific quality metrics
        self._add_quality_metrics(result)
        
        # IMPORTANT: Update the instance metadata so it's available to context composer
        self.metadata = result.metadata.copy()
        
        self._last_result = result
        return result
    
    def _plot_classification(
        self,
        ax,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray],
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        highlight_test_indices: Optional[List[int]],
        use_3d: bool,
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Plot for classification tasks using consistent styling."""
        from .utils.styling import apply_consistent_point_styling
        
        # Use passed parameters, fallback to instance attributes if they exist
        if class_names is None:
            class_names = getattr(self, '_class_names', None)
        if not use_semantic_names:
            use_semantic_names = getattr(self, '_use_semantic_names', False)
        
        # Use the shared styling utilities for consistent appearance
        plot_result = apply_consistent_point_styling(
            ax=ax,
            transformed_data=transformed_data,
            y=y,
            highlight_indices=highlight_indices,
            test_data=test_data,
            highlight_test_indices=highlight_test_indices,
            use_3d=use_3d,
            class_names=class_names,
            use_semantic_names=use_semantic_names
        )
        
        # Add KNN connections if enabled and we have test data
        if (self.config.use_knn_connections and 
            test_data is not None and 
            highlight_test_indices and 
            y is not None and
            hasattr(self, '_training_embeddings')):
            
            # For each highlighted test point, add KNN connections
            for test_idx in highlight_test_indices:
                if test_idx < len(test_data):
                    query_coord = test_data[test_idx]
                    query_embedding = getattr(self, '_test_embeddings', test_data)[test_idx]
                    
                    # Compute KNN analysis in embedding space
                    knn_info = self._compute_knn_analysis(
                        query_point=query_embedding,
                        training_data=self._training_embeddings,
                        training_labels=y,
                        k=self.config.nn_k
                    )
                    
                    # Get coordinates of neighbors in visualization space
                    neighbor_coords = transformed_data[knn_info['neighbor_indices']]
                    
                    # Add connection lines
                    self._add_knn_connections_to_plot(
                        ax=ax,
                        query_coord=query_coord,
                        neighbor_coords=neighbor_coords,
                        neighbor_distances=knn_info['neighbor_distances'],
                        use_3d=use_3d,
                        max_connections=self.config.nn_k
                    )
                    
                    # Add KNN information to metadata
                    if 'knn_info' not in plot_result.get('metadata', {}):
                        plot_result.setdefault('metadata', {})['knn_info'] = []
                    plot_result['metadata']['knn_info'].append({
                        'test_idx': test_idx,
                        'neighbor_classes': knn_info['neighbor_labels'].tolist(),
                        'class_distribution': knn_info['class_distribution'],
                        'neighbor_distances': knn_info['neighbor_distances'].tolist()
                    })
        
        return plot_result
    
    def _plot_regression(
        self,
        ax,
        transformed_data: np.ndarray,
        y: np.ndarray,
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        highlight_test_indices: Optional[List[int]],
        use_3d: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Plot for regression tasks."""
        
        # Use continuous colormap for regression
        if use_3d:
            scatter = ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                transformed_data[:, 2],
                c=y, s=self.config.point_size, alpha=self.config.alpha,
                cmap='viridis'
            )
        else:
            scatter = ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                c=y, s=self.config.point_size, alpha=self.config.alpha,
                cmap='viridis'
            )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Target Value')
        
        metadata = {
            'plot_type': 'regression',
            'target_min': float(np.min(y)),
            'target_max': float(np.max(y)),
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y))
        }
        
        # Highlight specific points
        if highlight_indices:
            if use_3d:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    transformed_data[highlight_indices, 2],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            else:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            
            metadata['highlighted_indices'] = highlight_indices
        
        # Plot test data
        if test_data is not None:
            if use_3d:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    test_data[:, 2],
                    c='lightgray', s=self.config.point_size * 1.2, alpha=0.8,
                    marker='s', edgecolors='black', linewidth=0.8,
                    label='Test Points (Light Gray)'
                )
            else:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    c='lightgray', s=self.config.point_size * 1.2, alpha=0.8,
                    marker='s', edgecolors='black', linewidth=0.8,
                    label='Test Points (Light Gray)'
                )
        
        # Highlight specific test points with red X markers
        legend_text_parts = [f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]"]
        if test_data is not None and highlight_test_indices:
            highlighted_test_data = test_data[highlight_test_indices]
            if use_3d:
                ax.scatter(
                    highlighted_test_data[:, 0],
                    highlighted_test_data[:, 1],
                    highlighted_test_data[:, 2],
                    c='red', s=self.config.point_size * 3, alpha=1.0,
                    marker='x', linewidths=4, label='Query point'
                )
            else:
                ax.scatter(
                    highlighted_test_data[:, 0],
                    highlighted_test_data[:, 1],
                    c='red', s=self.config.point_size * 3, alpha=1.0,
                    marker='x', linewidths=4, label='Query point'
                )
            legend_text_parts.append('Query point (red X)')
        
        legend_text = '; '.join(legend_text_parts)
        
        return {
            'legend_text': legend_text,
            'metadata': metadata
        }
    
    def _apply_plot_styling(self, ax, use_3d: bool):
        """Apply styling to the plot."""
        
        if self.config.title:
            ax.set_title(self.config.title)
        
        if self.config.xlabel:
            ax.set_xlabel(self.config.xlabel)
        else:
            ax.set_xlabel(f'{self.method_name} Component 1')
        
        if self.config.ylabel:
            ax.set_ylabel(self.config.ylabel)
        else:
            ax.set_ylabel(f'{self.method_name} Component 2')
        
        if use_3d and self.config.zlabel:
            ax.set_zlabel(self.config.zlabel)
        elif use_3d:
            ax.set_zlabel(f'{self.method_name} Component 3')
        
        if self.config.show_legend:
            # Use consistent legend formatting from shared styling
            from .utils.styling import apply_consistent_legend_formatting
            apply_consistent_legend_formatting(ax, use_3d)
        
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
        
        # Apply custom viewing angles for 3D
        if use_3d and self.config.viewing_angles:
            for elev, azim in self.config.viewing_angles:
                ax.view_init(elev=elev, azim=azim)
                break  # Use first angle by default
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add method-specific quality metrics to the result."""
        # Default implementation - subclasses can override
        pass
    
    def get_description(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
        """
        Get a description of this visualization method for the given data.
        
        Args:
            X: Input data
            y: Optional target values
            
        Returns:
            Description string
        """
        base_desc = self._get_default_description(len(X), X.shape[1])
        
        if y is not None:
            if self.config.task_type == 'classification':
                n_classes = len(np.unique(y))
                base_desc += f" The data contains {n_classes} classes."
            else:
                target_range = np.max(y) - np.min(y)
                base_desc += f" The target values range from {np.min(y):.2f} to {np.max(y):.2f} (range: {target_range:.2f})."
        
        return base_desc
    
    def _compute_knn_analysis(
        self,
        query_point: np.ndarray,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute KNN analysis for a query point.
        
        Args:
            query_point: Query point in embedding space [n_features]
            training_data: Training data in embedding space [n_samples, n_features]
            training_labels: Training labels [n_samples]
            k: Number of nearest neighbors (uses config.nn_k if None)
            
        Returns:
            Dictionary with neighbor information
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        if k is None:
            k = self.config.nn_k
        
        # Fit KNN on training data
        knn = NearestNeighbors(n_neighbors=min(k, len(training_data)), metric='euclidean')
        knn.fit(training_data)
        
        # Find neighbors for query point
        distances, indices = knn.kneighbors([query_point])
        
        # Get neighbor information
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        neighbor_labels = training_labels[neighbor_indices]
        
        # Compute class distribution
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        
        return {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_labels': neighbor_labels,
            'class_distribution': class_distribution,
            'k': len(neighbor_indices)
        }
    
    def _add_knn_connections_to_plot(
        self,
        ax,
        query_coord: np.ndarray,
        neighbor_coords: np.ndarray,
        neighbor_distances: np.ndarray,
        use_3d: bool = False,
        max_connections: int = 5
    ):
        """
        Add KNN connection lines to a plot.
        
        Args:
            ax: Matplotlib axis
            query_coord: Query point coordinates [n_dims]
            neighbor_coords: Neighbor coordinates [n_neighbors, n_dims]
            neighbor_distances: Distances to neighbors [n_neighbors]
            use_3d: Whether this is a 3D plot
            max_connections: Maximum number of connections to draw
        """
        # Limit number of connections for visual clarity
        n_connections = min(max_connections, len(neighbor_coords))
        
        for i in range(n_connections):
            neighbor_coord = neighbor_coords[i]
            distance = neighbor_distances[i]
            
            # Line alpha based on distance (closer = more opaque)
            max_distance = np.max(neighbor_distances) if len(neighbor_distances) > 1 else 1.0
            alpha = max(0.1, 1.0 - (distance / max_distance)) if max_distance > 0 else 0.5
            
            if use_3d:
                ax.plot3D(
                    [query_coord[0], neighbor_coord[0]],
                    [query_coord[1], neighbor_coord[1]],
                    [query_coord[2], neighbor_coord[2]],
                    color='gray',
                    alpha=alpha,
                    linewidth=1.0,
                    linestyle='--'
                )
            else:
                ax.plot(
                    [query_coord[0], neighbor_coord[0]],
                    [query_coord[1], neighbor_coord[1]],
                    color='gray',
                    alpha=alpha,
                    linewidth=1.0,
                    linestyle='--'
                )