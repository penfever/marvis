"""
Refactored t-SNE visualization classes to eliminate code duplication.

This module provides a clean class-based architecture to replace the 
duplicated functions in tsne_functions.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
from typing import Tuple, Optional, List, Dict, Union, Any

# Import shared styling utilities
from .utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    get_standard_test_point_style,
    get_standard_target_point_style,
    create_distinct_color_map,
    create_regression_color_map
)

# Import semantic axes utilities
try:
    from ..utils.semantic_axes import create_compact_axis_labels, create_bottom_legend_text
except ImportError:
    # Fallback for cases where semantic_axes is not available
    def create_compact_axis_labels(semantic_axes, **kwargs):
        return {}
    def create_bottom_legend_text(semantic_axes, **kwargs):
        return ""

# Import KNN regression analysis function
from .mixins.knn import create_knn_regression_analysis

logger = logging.getLogger(__name__)


class TSNEGenerator:
    """
    Core t-SNE computation and coordinate generation.
    
    Eliminates duplication of t-SNE fitting logic across all the original functions.
    Handles embedding combination, perplexity adjustment, and coordinate generation.
    """
    
    def __init__(self, perplexity: int = 30, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize t-SNE generator.
        
        Args:
            perplexity: t-SNE perplexity parameter
            max_iter: Number of t-SNE iterations  
            random_state: Random seed for reproducibility
        """
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def fit_transform(
        self, 
        train_embeddings: np.ndarray, 
        test_embeddings: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate t-SNE coordinates for train and test data.
        
        Args:
            train_embeddings: Training embeddings [n_train, embedding_dim]
            test_embeddings: Test embeddings [n_test, embedding_dim]  
            n_components: Number of t-SNE components (2 for 2D, 3 for 3D)
            
        Returns:
            train_coords: t-SNE coordinates for training data [n_train, n_components]
            test_coords: t-SNE coordinates for test data [n_test, n_components]
        """
        self.logger.info(f"Generating {n_components}D t-SNE coordinates for {len(train_embeddings)} train and {len(test_embeddings)} test samples")
        
        # Step 1: Combine embeddings for joint t-SNE (shared across all original functions)
        combined_embeddings = np.vstack([train_embeddings, test_embeddings])
        n_train = len(train_embeddings)
        
        # Step 2: Enhanced perplexity auto-adjustment based on dataset characteristics
        effective_perplexity = self._calculate_optimal_perplexity(
            combined_embeddings, train_embeddings, test_embeddings
        )
        if effective_perplexity != self.perplexity:
            self.logger.warning(
                f"Auto-adjusting perplexity from {self.perplexity} to {effective_perplexity} "
                f"based on dataset size and characteristics"
            )
        
        # Step 3: Apply t-SNE (shared initialization logic)
        self.logger.info(f"Running t-SNE with perplexity={effective_perplexity}, max_iter={self.max_iter}")
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1
        )
        
        # Step 4: Fit and transform
        combined_coords = tsne.fit_transform(combined_embeddings)
        
        # Step 5: Split back into train and test (shared splitting logic) 
        train_coords = combined_coords[:n_train]
        test_coords = combined_coords[n_train:]
        
        self.logger.info(f"t-SNE completed successfully")
        return train_coords, test_coords
    
    def _calculate_optimal_perplexity(
        self, 
        combined_embeddings: np.ndarray, 
        train_embeddings: np.ndarray, 
        test_embeddings: np.ndarray
    ) -> int:
        """
        Calculate optimal perplexity based on dataset characteristics.
        
        Enhanced logic that considers:
        - Total dataset size (original constraint)
        - Training vs test split balance
        - Sparsity characteristics for few-shot scenarios
        
        Args:
            combined_embeddings: Combined train+test embeddings
            train_embeddings: Training embeddings only
            test_embeddings: Test embeddings only
            
        Returns:
            Optimal perplexity value
        """
        total_samples = len(combined_embeddings)
        train_samples = len(train_embeddings)
        test_samples = len(test_embeddings)
        
        # Original constraint: perplexity must be < n_samples/3
        max_perplexity_by_size = (total_samples - 1) // 3
        
        # For very small datasets (few-shot learning scenarios)
        if total_samples <= 20:
            # Very aggressive reduction for tiny datasets
            suggested_perplexity = max(5, total_samples // 4)
            self.logger.info(f"Few-shot scenario detected ({total_samples} samples) - using conservative perplexity")
        elif total_samples <= 50:
            # Moderate reduction for small datasets  
            suggested_perplexity = max(10, total_samples // 3)
            self.logger.info(f"Small dataset detected ({total_samples} samples) - reducing perplexity for better local structure")
        elif train_samples < 10:
            # Very few training samples - prioritize local structure
            suggested_perplexity = max(5, min(15, self.perplexity))
            self.logger.info(f"Very few training samples ({train_samples}) - using low perplexity for local detail")
        else:
            # Use original perplexity for larger datasets
            suggested_perplexity = self.perplexity
            
        # Apply the size constraint
        effective_perplexity = min(suggested_perplexity, max_perplexity_by_size)
        
        # Ensure minimum perplexity
        effective_perplexity = max(1, effective_perplexity)
        
        # Log detailed adjustment reasoning if changed
        if effective_perplexity != self.perplexity:
            self.logger.info(
                f"Perplexity adjustment details: total={total_samples}, train={train_samples}, "
                f"test={test_samples}, max_by_size={max_perplexity_by_size}, "
                f"suggested={suggested_perplexity}, final={effective_perplexity}"
            )
        
        return effective_perplexity


class BaseTSNEPlotter(ABC):
    """
    Abstract base class for t-SNE plotting with shared infrastructure.
    
    Eliminates duplication of figure creation, zoom logic, and legend handling
    across classification and regression plotters.
    """
    
    def __init__(
        self, 
        figsize: Tuple[int, int] = (10, 8),
        zoom_factor: float = 2.0,
        use_3d: bool = False
    ):
        """
        Initialize base plotter.
        
        Args:
            figsize: Figure size (width, height)
            zoom_factor: Zoom level for highlighted points
            use_3d: Whether to create 3D plots
        """
        self.figsize = figsize
        self.zoom_factor = zoom_factor
        self.use_3d = use_3d
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _create_figure(self) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
        """
        Create figure and axis (2D or 3D based on configuration).
        
        For 3D plots, creates a 2x2 grid with 4 different viewing angles.
        For 2D plots, creates a single axis.
        """
        if self.use_3d:
            # Create 3D multi-view layout with 4 different perspectives
            fig = plt.figure(figsize=(15, 12) if self.figsize == (10, 8) else self.figsize)
            
            # Create 2x2 subplot grid
            axes = []
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                axes.append(ax)
            
            # Set viewing angles for each subplot
            viewing_angles = [
                (20, -60, "Isometric View"),
                (0, -90, "Front View (XZ)"), 
                (0, 0, "Side View (YZ)"),
                (90, 0, "Top View (XY)")
            ]
            
            for ax, (elev, azim, title) in zip(axes, viewing_angles):
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(title, fontsize=10)
            
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            return fig, ax
        
    def _apply_zoom(
        self,
        ax: plt.Axes,
        target_point: np.ndarray,
        train_coords: np.ndarray,
        test_coords: np.ndarray
    ) -> None:
        """
        Apply zoom around target point while ensuring all data points remain visible.
        
        Improved zoom logic that prevents points from appearing outside plot boundaries.
        """
        if self.zoom_factor <= 1.0:
            return
            
        # Calculate the visible range based on zoom factor
        # zoom_factor = 2.0 means we show 1/2 of the original range
        all_coords = np.vstack([train_coords, test_coords])
        
        if self.use_3d:
            # 3D zoom logic - only apply if target_point has 3 dimensions
            if len(target_point) >= 3:
                for i, coord_name in enumerate(['X', 'Y', 'Z']):
                    coord_min_data = all_coords[:, i].min()
                    coord_max_data = all_coords[:, i].max()
                    coord_range = coord_max_data - coord_min_data
                    
                    if coord_range > 0:  # Avoid division by zero
                        visible_range = coord_range / self.zoom_factor
                        center = target_point[i]
                        
                        # Calculate initial zoom bounds
                        coord_min = center - visible_range / 2
                        coord_max = center + visible_range / 2
                        
                        # Ensure zoom bounds don't exceed data bounds by too much
                        # Allow some padding but prevent excessive empty space
                        padding = coord_range * 0.1  # 10% padding
                        coord_min = max(coord_min, coord_min_data - padding)
                        coord_max = min(coord_max, coord_max_data + padding)
                        
                        # Ensure minimum zoom window size
                        if coord_max - coord_min < visible_range * 0.5:
                            # If bounds got too constrained, expand symmetrically
                            mid_point = (coord_min + coord_max) / 2
                            half_range = visible_range * 0.25
                            coord_min = mid_point - half_range
                            coord_max = mid_point + half_range
                        
                        if i == 0:
                            ax.set_xlim(coord_min, coord_max)
                        elif i == 1:
                            ax.set_ylim(coord_min, coord_max)
                        else:
                            ax.set_zlim(coord_min, coord_max)
        else:
            # 2D zoom logic
            for i, coord_name in enumerate(['X', 'Y']):
                coord_min_data = all_coords[:, i].min()
                coord_max_data = all_coords[:, i].max()
                coord_range = coord_max_data - coord_min_data
                
                if coord_range > 0:  # Avoid division by zero
                    visible_range = coord_range / self.zoom_factor
                    center = target_point[i]
                    
                    # Calculate initial zoom bounds
                    coord_min = center - visible_range / 2
                    coord_max = center + visible_range / 2
                    
                    # Ensure zoom bounds don't exceed data bounds by too much
                    # Allow some padding but prevent excessive empty space
                    padding = coord_range * 0.1  # 10% padding
                    coord_min = max(coord_min, coord_min_data - padding)
                    coord_max = min(coord_max, coord_max_data + padding)
                    
                    # Ensure minimum zoom window size
                    if coord_max - coord_min < visible_range * 0.5:
                        # If bounds got too constrained, expand symmetrically
                        mid_point = (coord_min + coord_max) / 2
                        half_range = visible_range * 0.25
                        coord_min = mid_point - half_range
                        coord_max = mid_point + half_range
                    
                    if i == 0:
                        ax.set_xlim(coord_min, coord_max)
                    else:
                        ax.set_ylim(coord_min, coord_max)
                    
    def _add_semantic_legend(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        semantic_axes_labels: Optional[Dict[str, str]]
    ) -> None:
        """
        Add semantic axes legend to the plot.
        
        Shared semantic legend logic.
        """
        if not semantic_axes_labels:
            return
            
        # Create bottom legend text
        legend_text = create_bottom_legend_text(
            semantic_axes_labels,
            max_chars_per_line=80,
            max_lines=2
        )
        
        if legend_text:
            # Add text at bottom of figure, outside the plot area
            fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', 
                    fontsize=8, wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightgray', alpha=0.8))
                    
    @abstractmethod
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_data: np.ndarray,
        test_coords: np.ndarray,
        test_data: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot points on the axis (task-specific implementation).
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_data: Training labels/targets [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]  
            test_data: Test labels/targets [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        pass
        
    def create_plot(
        self,
        train_coords: np.ndarray,
        train_data: np.ndarray,
        test_coords: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        highlight_test_idx: Optional[int] = None,
        semantic_axes_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, str, Dict[str, Any]]:
        """
        Create complete t-SNE plot.
        
        Main plotting method that coordinates all the shared logic.
        
        Args:
            train_coords: Training coordinates [n_train, 2 or 3]
            train_data: Training labels/targets [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_data: Test labels/targets [n_test] (optional)  
            highlight_test_idx: Index of test point to highlight
            semantic_axes_labels: Optional semantic axes labels
            **kwargs: Additional plotting parameters
            
        Returns:
            fig: Matplotlib figure
            legend_text: Legend description
            metadata: Plot metadata dictionary
        """
        # Step 1: Create figure
        fig, axes = self._create_figure()
        
        if self.use_3d:
            # For 3D: axes is a list of 4 subplots
            plot_results = []
            for i, ax in enumerate(axes):
                # Step 2: Apply zoom if highlighting a test point
                if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
                    target_point = test_coords[highlight_test_idx]
                    self._apply_zoom(ax, target_point, train_coords, test_coords)
                    
                # Step 3: Task-specific point plotting
                plot_result = self._plot_points(
                    ax, train_coords, train_data, test_coords, test_data, 
                    highlight_test_idx, 
                    show_legend=(i == 0),  # Only show legend on first subplot
                    **kwargs
                )
                plot_results.append(plot_result)
                
                # Step 4: Apply consistent legend formatting
                apply_consistent_legend_formatting(ax, use_3d=self.use_3d)
                
                # Step 6: Set axis labels with view-specific positioning to prevent overlap
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                
                # Get viewing angles to adjust z-label positioning
                elev, azim = ax.elev, ax.azim
                
                # For frontal views that cause text overlap, adjust z-label positioning
                if abs(elev) < 10:  # Front view (elev=0) or near-front views
                    ax.set_zlabel('t-SNE Component 3', labelpad=20)
                elif abs(azim) < 10 or abs(azim - 360) < 10:  # Side view (azim=0) or similar
                    ax.set_zlabel('t-SNE Component 3', labelpad=15)
                else:
                    ax.set_zlabel('t-SNE Component 3')
            
            # Use the first plot result for overall metadata
            main_plot_result = plot_results[0]
            
            # Add overall figure adjustments
            plt.tight_layout()
            
        else:
            # For 2D: axes is a single axis
            ax = axes
            
            # Step 2: Apply zoom if highlighting a test point
            if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
                target_point = test_coords[highlight_test_idx]
                self._apply_zoom(ax, target_point, train_coords, test_coords)
                
            # Step 3: Task-specific point plotting
            main_plot_result = self._plot_points(
                ax, train_coords, train_data, test_coords, test_data, 
                highlight_test_idx, **kwargs
            )
            
            # Step 4: Apply consistent legend formatting
            apply_consistent_legend_formatting(ax, use_3d=self.use_3d)
            
            # Step 6: Set axis labels
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
        
        # Step 5: Add semantic legend if provided (only for 2D or first 3D subplot)
        first_ax = axes[0] if self.use_3d else axes
        self._add_semantic_legend(fig, first_ax, semantic_axes_labels)
            
        return fig, main_plot_result['legend_text'], main_plot_result['metadata']


class ClassificationTSNEPlotter(BaseTSNEPlotter):
    """
    Handles classification-specific t-SNE plotting.
    
    Manages discrete class visualization with proper color mapping,
    class legends, and semantic name handling.
    """
    
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_labels: np.ndarray,
        test_coords: np.ndarray,
        test_labels: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False,
        show_legend: bool = True,
        all_classes: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot classification points using consistent styling.
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_labels: Training class labels [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_labels: Test class labels [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            class_names: Optional class names for labeling
            use_semantic_names: Whether to use semantic class names
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        # Determine which test indices to highlight
        highlight_test_indices = [highlight_test_idx] if highlight_test_idx is not None else None
        
        # Use the shared styling system for consistent appearance
        plot_result = apply_consistent_point_styling(
            ax=ax,
            transformed_data=train_coords,
            y=train_labels,
            highlight_indices=None,  # No training point highlighting for t-SNE
            test_data=test_coords,
            highlight_test_indices=highlight_test_indices,
            use_3d=self.use_3d,
            class_names=class_names,
            use_semantic_names=use_semantic_names,
            all_classes=all_classes,
            cached_color_mapping=getattr(self, 'cached_color_mapping', None)
        )
        
        return plot_result


class RegressionTSNEPlotter(BaseTSNEPlotter):
    """
    Handles regression-specific t-SNE plotting.
    
    Manages continuous value visualization with colormaps,
    color bars, and regression-specific styling.
    """
    
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_targets: np.ndarray,
        test_coords: np.ndarray,
        test_targets: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        colormap: str = 'viridis',
        show_legend: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot regression points using continuous color mapping.
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_targets: Training target values [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_targets: Test target values [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            colormap: Matplotlib colormap name for target values
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        # Create regression color mapping
        color_map = create_regression_color_map(train_targets, colormap=colormap)
        
        # Plot training points with continuous coloring
        if self.use_3d:
            scatter = ax.scatter(
                train_coords[:, 0],
                train_coords[:, 1], 
                train_coords[:, 2],
                c=train_targets,
                cmap=colormap,
                alpha=0.6,
                s=50,
                label='Training Data'
            )
        else:
            scatter = ax.scatter(
                train_coords[:, 0],
                train_coords[:, 1],
                c=train_targets,
                cmap=colormap,
                alpha=0.6,
                s=50,
                label='Training Data'
            )
            
        # Add colorbar (only for first subplot in 3D multi-view)
        if show_legend:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Target Value', rotation=270, labelpad=15)
        
        # Plot test points with gray styling
        test_style = get_standard_test_point_style()
        if self.use_3d:
            ax.scatter(
                test_coords[:, 0],
                test_coords[:, 1],
                test_coords[:, 2],
                **test_style
            )
        else:
            ax.scatter(
                test_coords[:, 0],
                test_coords[:, 1],
                **test_style
            )
            
        # Highlight specific test point if requested
        if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
            target_style = get_standard_target_point_style()
            if self.use_3d:
                ax.scatter(
                    test_coords[highlight_test_idx, 0],
                    test_coords[highlight_test_idx, 1],
                    test_coords[highlight_test_idx, 2],
                    **target_style
                )
            else:
                ax.scatter(
                    test_coords[highlight_test_idx, 0],
                    test_coords[highlight_test_idx, 1],
                    **target_style
                )
                
        # Create metadata
        metadata = {
            'plot_type': 'regression',
            'visible_classes': [],  # No classes in regression
            'target_range': [float(train_targets.min()), float(train_targets.max())],
            'colormap': colormap
        }
        
        legend_text = f"Regression visualization (target range: {train_targets.min():.2f} - {train_targets.max():.2f})"
        
        return {
            'legend_text': legend_text,
            'metadata': metadata
        }


class KNNMixin:
    """
    Mixin class that adds KNN pie chart functionality to any plotter.
    
    This eliminates the duplication between KNN and non-KNN variants
    of the plotting functions and provides pie chart visualization.
    """
    
    def __init__(self, *args, knn_k: int = 5, task_type: str = 'classification', **kwargs):
        """
        Initialize KNN mixin.
        
        Args:
            knn_k: Number of nearest neighbors to find and visualize
            task_type: 'classification' or 'regression' for appropriate chart type
        """
        super().__init__(*args, **kwargs)
        self.knn_k = knn_k
        self.task_type = task_type
        
    def _compute_knn_analysis(
        self,
        query_point: np.ndarray,
        training_embeddings: np.ndarray,
        training_labels: np.ndarray,
        k: int
    ) -> Dict[str, Any]:
        """
        Compute KNN analysis for a query point.
        
        Args:
            query_point: Query point embedding [embedding_dim]
            training_embeddings: Training embeddings [n_train, embedding_dim] 
            training_labels: Training labels [n_train]
            k: Number of nearest neighbors
            
        Returns:
            Dictionary with KNN analysis results
        """
        # Compute distances to all training points
        distances = np.linalg.norm(training_embeddings - query_point, axis=1)
        
        # Get indices of k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        knn_distances = distances[knn_indices]
        knn_labels = training_labels[knn_indices]
        
        return {
            'indices': knn_indices,
            'distances': knn_distances,
            'labels': knn_labels
        }
        
    def _create_knn_pie_chart(
        self,
        ax: plt.Axes,
        knn_labels: np.ndarray,
        knn_distances: np.ndarray,
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False,
        all_classes: Optional[np.ndarray] = None,
        cached_color_mapping: Optional[Dict] = None
    ) -> str:
        """
        Create KNN pie chart showing class distribution with distance information.
        
        Args:
            ax: Matplotlib axis for pie chart
            knn_labels: Labels of KNN points [k]
            knn_distances: Distances to KNN points [k]
            class_names: Optional class names
            use_semantic_names: Whether to use semantic names
            all_classes: All possible classes for consistent color mapping
            cached_color_mapping: Pre-computed color mapping for consistency
            
        Returns:
            Description text for the KNN analysis
        """
        from collections import Counter
        from .utils.styling import format_class_label, create_distinct_color_map
        
        # Count class occurrences
        label_counts = Counter(knn_labels)
        
        # Create consistent colors using all possible classes for color mapping consistency
        if all_classes is None:
            all_classes = np.unique(knn_labels)
        
        color_map = create_distinct_color_map(all_classes, cached_color_mapping)
        
        # Calculate average distance for each class
        class_avg_distances = {}
        for cls in all_classes:
            mask = knn_labels == cls
            class_avg_distances[cls] = np.mean(knn_distances[mask])
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        for label, count in label_counts.items():
            formatted_label = format_class_label(label, class_names, use_semantic_names)
            avg_dist = class_avg_distances[label]
            labels.append(f"{formatted_label}\n{count}/{len(knn_labels)}\nAvgDist: {avg_dist:.2f}")
            sizes.append(count)
            colors.append(color_map[label])
            
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.0f%%', 
            startangle=90,
            textprops={'fontsize': 8},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
        )
        ax.set_title(f'K-NN Distribution (k={len(knn_labels)})', fontsize=10, weight='bold')
        
        # Generate description text
        k = len(knn_labels)
        description = f"K-NN Analysis (k={k}):\n"
        for label, count in sorted(label_counts.items()):
            percentage = (count / k) * 100
            avg_dist = class_avg_distances[label]
            formatted_label = format_class_label(label, class_names, use_semantic_names)
            description += f"• {formatted_label}: {count} neighbors ({percentage:.0f}%), AvgDist: {avg_dist:.2f}\n"
        
        return description.strip()
    
    def _create_knn_regression_chart(
        self,
        ax: plt.Axes,
        knn_targets: np.ndarray,
        knn_distances: np.ndarray
    ) -> str:
        """
        Create KNN regression bar chart showing target values with distance information.
        
        Args:
            ax: Matplotlib axis for bar chart
            knn_targets: Target values of KNN points [k]
            knn_distances: Distances to KNN points [k]
            
        Returns:
            Description text for the KNN analysis
        """
        k = len(knn_targets)
        
        # Sort neighbors by distance for better visualization
        sorted_indices = np.argsort(knn_distances)
        sorted_targets = knn_targets[sorted_indices]
        sorted_distances = knn_distances[sorted_indices]
        
        # Create bar chart of neighbor target values
        neighbor_indices = np.arange(len(sorted_targets))
        bars = ax.bar(neighbor_indices, sorted_targets, alpha=0.7)
        
        # Color bars by target value (gradient)
        target_min, target_max = np.min(sorted_targets), np.max(sorted_targets)
        if target_max > target_min:
            # Normalize target values for coloring
            normalized_targets = (sorted_targets - target_min) / (target_max - target_min)
            colors = plt.cm.viridis(normalized_targets)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Add distance labels on top of bars
        for i, (target, distance) in enumerate(zip(sorted_targets, sorted_distances)):
            y_offset = (target_max - target_min) * 0.02 if target_max > target_min else 0.1
            ax.text(i, target + y_offset, 
                    f'd={distance:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('Neighbor Rank (by distance)')
        ax.set_ylabel('Target Value')
        ax.set_title(f'K-NN Analysis (k={k})', fontsize=10, weight='bold')
        ax.set_xticks(neighbor_indices)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(sorted_targets))])
        
        # Add statistics as text
        mean_target = np.mean(sorted_targets)
        median_target = np.median(sorted_targets)
        std_target = np.std(sorted_targets)
        
        stats_text = f'Mean: {mean_target:.2f}\nMedian: {median_target:.2f}\nStd: {std_target:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Generate description text
        description = f"K-NN Analysis (k={k}):\n"
        for i, (target, distance) in enumerate(zip(sorted_targets, sorted_distances)):
            description += f"• #{i+1}: target={target:.3f}, distance={distance:.3f}\n"
        
        description += f"\nStatistics: mean={mean_target:.3f}, median={median_target:.3f}, std={std_target:.3f}"
        
        return description.strip()
    
    def create_plot(
        self,
        train_coords: np.ndarray,
        train_data: np.ndarray,
        test_coords: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        highlight_test_idx: Optional[int] = None,
        semantic_axes_labels: Optional[Dict[str, str]] = None,
        train_embeddings: Optional[np.ndarray] = None,
        test_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[plt.Figure, str, Dict[str, Any]]:
        """
        Create KNN plot with pie chart layout.
        
        Overrides the base create_plot to add KNN pie chart functionality.
        """
        # Check if KNN analysis was requested (this is a KNNMixin, so it should always do KNN)
        if highlight_test_idx is not None:
            # Validate that we have the required data for KNN analysis
            if train_embeddings is None:
                raise ValueError("KNN analysis requested but train_embeddings not provided")
            if test_embeddings is None:
                raise ValueError("KNN analysis requested but test_embeddings not provided")
            if not (0 <= highlight_test_idx < len(test_coords)):
                raise ValueError(f"Invalid highlight_test_idx {highlight_test_idx} for {len(test_coords)} test points")
        
        show_knn = highlight_test_idx is not None
        
        if show_knn:
            if not self.use_3d:
                # For 2D with KNN: Create custom GridSpec layout
                fig = plt.figure(figsize=self.figsize)
                gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 2.5, 2, 1.5, 1.5], 
                                    hspace=0.1, wspace=0.15)
                ax = fig.add_subplot(gs[0, :3])  # Main plot spans first 3 columns
                ax_pie = fig.add_subplot(gs[0, 3:])  # Pie chart spans last 2 columns
                
                # Apply zoom if highlighting a test point
                if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
                    target_point = test_coords[highlight_test_idx]
                    self._apply_zoom(ax, target_point, train_coords, test_coords)
                    
                # Task-specific point plotting
                plot_result = self._plot_points(
                    ax, train_coords, train_data, test_coords, test_data, 
                    highlight_test_idx, **kwargs
                )
                
                # Compute KNN analysis and create pie chart
                query_embedding = test_embeddings[highlight_test_idx]
                knn_info = self._compute_knn_analysis(
                    query_embedding, train_embeddings, train_data, self.knn_k
                )
                
                # Use the known task type instead of guessing from data
                if self.task_type == 'regression':
                    knn_description = self._create_knn_regression_chart(
                        ax_pie, knn_info['labels'], knn_info['distances']
                    )
                else:
                    knn_description = self._create_knn_pie_chart(
                        ax_pie, knn_info['labels'], knn_info['distances'],
                        kwargs.get('class_names'), kwargs.get('use_semantic_names', False),
                        all_classes=np.unique(train_data),
                        cached_color_mapping=getattr(self, 'cached_color_mapping', None)
                    )
                
                # Apply consistent legend formatting
                apply_consistent_legend_formatting(ax, use_3d=self.use_3d)
                
                # Add semantic legend if provided
                self._add_semantic_legend(fig, ax, semantic_axes_labels)
                
                # Set axis labels
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                
                # Update legend text with KNN description
                legend_text = plot_result['legend_text']
                if knn_description:
                    legend_text += f"\n\n{knn_description}"
                
                # Update metadata
                metadata = plot_result['metadata']
                metadata['has_knn_analysis'] = True
                metadata['knn_k'] = self.knn_k
                
                return fig, legend_text, metadata
                
            else:
                # For 3D with KNN: Create 4-panel 3D view + pie chart
                fig = plt.figure(figsize=(20, 12))  # Larger figure for 3D + pie chart
                
                # Create GridSpec: 2x3 layout (4 3D plots + 1 pie chart)
                gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.6], 
                                    height_ratios=[1, 1], hspace=0.2, wspace=0.15)
                
                # Create 4 3D subplots
                axes = []
                viewing_angles = [
                    (20, -60, "Isometric View"),
                    (0, -90, "Front View (XZ)"), 
                    (0, 0, "Side View (YZ)"),
                    (90, 0, "Top View (XY)")
                ]
                
                # Add 3D subplots
                for i in range(4):
                    row, col = divmod(i, 2)
                    ax = fig.add_subplot(gs[row, col], projection='3d')
                    elev, azim, title = viewing_angles[i]
                    ax.view_init(elev=elev, azim=azim)
                    ax.set_title(title, fontsize=10)
                    axes.append(ax)
                
                # Add pie chart in the right column
                ax_pie = fig.add_subplot(gs[:, 2])  # Spans both rows in column 2
                
                # Plot on all 3D axes
                plot_results = []
                for i, ax in enumerate(axes):
                    # Apply zoom if highlighting a test point
                    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
                        target_point = test_coords[highlight_test_idx]
                        self._apply_zoom(ax, target_point, train_coords, test_coords)
                        
                    # Task-specific point plotting
                    plot_result = self._plot_points(
                        ax, train_coords, train_data, test_coords, test_data, 
                        highlight_test_idx, 
                        show_legend=(i == 0),  # Only show legend on first subplot
                        **kwargs
                    )
                    plot_results.append(plot_result)
                    
                    # Apply consistent legend formatting
                    apply_consistent_legend_formatting(ax, use_3d=self.use_3d)
                    
                    # Set axis labels with view-specific positioning to prevent overlap
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    
                    # Get viewing angles to adjust z-label positioning
                    elev, azim = ax.elev, ax.azim
                    
                    # For frontal views that cause text overlap, adjust z-label positioning
                    if abs(elev) < 10:  # Front view (elev=0) or near-front views
                        ax.set_zlabel('t-SNE Component 3', labelpad=20)
                    elif abs(azim) < 10 or abs(azim - 360) < 10:  # Side view (azim=0) or similar
                        ax.set_zlabel('t-SNE Component 3', labelpad=15)
                    else:
                        ax.set_zlabel('t-SNE Component 3')
                
                # Compute KNN analysis and create pie chart
                query_embedding = test_embeddings[highlight_test_idx]
                knn_info = self._compute_knn_analysis(
                    query_embedding, train_embeddings, train_data, self.knn_k
                )
                
                # Use the known task type instead of guessing from data
                if self.task_type == 'regression':
                    knn_description = self._create_knn_regression_chart(
                        ax_pie, knn_info['labels'], knn_info['distances']
                    )
                else:
                    knn_description = self._create_knn_pie_chart(
                        ax_pie, knn_info['labels'], knn_info['distances'],
                        kwargs.get('class_names'), kwargs.get('use_semantic_names', False),
                        all_classes=np.unique(train_data),
                        cached_color_mapping=getattr(self, 'cached_color_mapping', None)
                    )
                
                # Use the first plot result for overall metadata
                main_plot_result = plot_results[0]
                
                # Add semantic legend if provided (only to first subplot)
                self._add_semantic_legend(fig, axes[0], semantic_axes_labels)
                
                # Update legend text with KNN description
                legend_text = main_plot_result['legend_text']
                if knn_description:
                    legend_text += f"\n\n{knn_description}"
                
                # Update metadata
                metadata = main_plot_result['metadata']
                metadata['has_knn_analysis'] = True
                metadata['knn_k'] = self.knn_k
                
                return fig, legend_text, metadata
        
        else:
            # For non-KNN: use the parent's create_plot method
            return super().create_plot(
                train_coords=train_coords,
                train_data=train_data,
                test_coords=test_coords,
                test_data=test_data,
                highlight_test_idx=highlight_test_idx,
                semantic_axes_labels=semantic_axes_labels,
                **kwargs
            )


class TSNEVisualizer:
    """
    Main unified interface for t-SNE visualization.
    
    This single class replaces all 14 functions from the original tsne_functions.py,
    eliminating massive code duplication while providing all the same functionality.
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        use_3d: bool = False,
        use_knn: bool = False,
        knn_k: int = 5,
        perplexity: int = 30,
        max_iter: int = 1000,
        random_state: int = 42,
        figsize: Tuple[int, int] = (10, 8),
        zoom_factor: float = 2.0,
        cached_color_mapping: Optional[Dict] = None
    ):
        """
        Initialize unified t-SNE visualizer.
        
        Args:
            task_type: 'classification' or 'regression'
            use_3d: Whether to create 3D visualizations
            use_knn: Whether to include KNN connections and analysis  
            knn_k: Number of nearest neighbors (if use_knn=True)
            perplexity: t-SNE perplexity parameter
            max_iter: Number of t-SNE iterations
            random_state: Random seed for reproducibility
            figsize: Figure size (width, height)
            zoom_factor: Zoom level for highlighted points
            cached_color_mapping: Optional pre-computed color mapping from resource manager
        """
        # Initialize coordinate generator
        self.generator = TSNEGenerator(
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state
        )
        
        # Initialize plotter with appropriate configuration
        self.plotter = self._create_plotter(task_type, use_3d, use_knn, knn_k, figsize, zoom_factor)
        self.task_type = task_type
        self.use_3d = use_3d
        self.use_knn = use_knn
        self.cached_color_mapping = cached_color_mapping
        
    def _create_plotter(
        self,
        task_type: str,
        use_3d: bool,
        use_knn: bool,
        knn_k: int,
        figsize: Tuple[int, int],
        zoom_factor: float
    ):
        """
        Factory method to create appropriate plotter with mixins.
        
        Args:
            task_type: 'classification' or 'regression'
            use_3d: Whether to create 3D plots
            use_knn: Whether to include KNN functionality
            knn_k: Number of nearest neighbors
            figsize: Figure size
            zoom_factor: Zoom level
            
        Returns:
            Configured plotter instance
        """
        # Choose base plotter class
        if task_type == 'classification':
            base_class = ClassificationTSNEPlotter
        elif task_type == 'regression':
            base_class = RegressionTSNEPlotter
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'classification' or 'regression'")
            
        # Apply KNN mixin if requested
        if use_knn:
            # Create a new class that combines KNNMixin with the base plotter
            class_name = f"KNN{base_class.__name__}"
            plotter_class = type(class_name, (KNNMixin, base_class), {})
            return plotter_class(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d, knn_k=knn_k, task_type=task_type)
        else:
            return base_class(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d)
            
    def create_visualization(
        self,
        train_embeddings: np.ndarray,
        train_data: np.ndarray,
        test_embeddings: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        highlight_test_idx: Optional[int] = None,
        semantic_axes_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure, str, Dict[str, Any]]:
        """
        Create complete t-SNE visualization.
        
        This single method replaces all 14 original functions:
        - create_tsne_visualization / create_tsne_3d_visualization
        - create_combined_tsne_plot / create_combined_tsne_3d_plot  
        - create_tsne_plot_with_knn / create_tsne_3d_plot_with_knn
        - create_regression_tsne_visualization / create_regression_tsne_3d_visualization
        - create_combined_regression_tsne_plot / create_combined_regression_tsne_3d_plot
        - create_regression_tsne_plot_with_knn / create_regression_tsne_3d_plot_with_knn
        - And more...
        
        Args:
            train_embeddings: Training embeddings [n_train, embedding_dim]
            train_data: Training labels (classification) or targets (regression) [n_train]
            test_embeddings: Test embeddings [n_test, embedding_dim]
            test_data: Test labels/targets [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            semantic_axes_labels: Optional semantic axes labels
            **kwargs: Additional plotting parameters (class_names, use_semantic_names, colormap, etc.)
            
        Returns:
            train_coords: t-SNE coordinates for training data [n_train, 2 or 3]
            test_coords: t-SNE coordinates for test data [n_test, 2 or 3]
            fig: Matplotlib figure
            legend_text: Legend description
            metadata: Plot metadata dictionary
        """
        # Step 1: Generate t-SNE coordinates
        train_coords, test_coords = self.generator.fit_transform(
            train_embeddings,
            test_embeddings,
            n_components=3 if self.use_3d else 2
        )
        
        # Step 2: Create plot (pass embeddings for KNN analysis if needed)
        fig, legend_text, metadata = self.plotter.create_plot(
            train_coords=train_coords,
            train_data=train_data,
            test_coords=test_coords,
            test_data=test_data,
            highlight_test_idx=highlight_test_idx,
            semantic_axes_labels=semantic_axes_labels,
            train_embeddings=train_embeddings if self.use_knn else None,
            test_embeddings=test_embeddings if self.use_knn else None,
            **kwargs
        )
        
        return train_coords, test_coords, fig, legend_text, metadata
        
    def create_simple_visualization(
        self,
        train_embeddings: np.ndarray,
        train_data: np.ndarray,
        test_embeddings: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
        """
        Create simple t-SNE visualization (backward compatibility).
        
        Matches the original simple function signatures that just returned coordinates and figure.
        """
        train_coords, test_coords, fig, _, _ = self.create_visualization(
            train_embeddings, train_data, test_embeddings, **kwargs
        )
        return train_coords, test_coords, fig


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER FUNCTIONS
# ============================================================================
# 
# These functions provide backward compatibility with the original tsne_functions.py
# by wrapping the new class-based implementation. This allows existing code to
# continue working while benefiting from the reduced duplication.


def create_tsne_visualization(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    use_3d: bool = False,
    zoom_factor: float = 2.0,
    cached_color_mapping: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create t-SNE visualization of train and test embeddings.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    This wrapper is provided for backward compatibility.
    """
    
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=use_3d,
        use_knn=False,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        figsize=figsize,
        zoom_factor=zoom_factor,
        cached_color_mapping=cached_color_mapping
    )
    
    return visualizer.create_simple_visualization(
        train_embeddings, train_labels, test_embeddings,
        class_names=class_names, use_semantic_names=use_semantic_names
    )


def create_tsne_3d_visualization(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (15, 12),
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    zoom_factor: float = 2.0,
    cached_color_mapping: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create 3D t-SNE visualization of train and test embeddings.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_tsne_visualization(
        train_embeddings, train_labels, test_embeddings, test_labels,
        perplexity, max_iter, random_state, figsize, class_names, use_semantic_names, use_3d=True,
        zoom_factor=zoom_factor, cached_color_mapping=cached_color_mapping
    )


def create_combined_tsne_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    use_3d: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None,
    all_classes: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    # Create a dummy visualizer to use the plotter
    plotter = ClassificationTSNEPlotter(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d)
    
    fig, legend_text, metadata = plotter.create_plot(
        train_coords=train_tsne,
        train_data=train_labels,
        test_coords=test_tsne,
        test_data=None,
        highlight_test_idx=highlight_test_idx,
        semantic_axes_labels=semantic_axes_labels,
        class_names=class_names,
        use_semantic_names=use_semantic_names,
        all_classes=all_classes
    )
    
    return fig, legend_text, metadata


def create_combined_tsne_3d_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 12),
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None,
    all_classes: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 3D t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_combined_tsne_plot(
        train_tsne, test_tsne, train_labels, highlight_test_idx,
        figsize, zoom_factor, class_names, use_semantic_names, use_3d=True, semantic_axes_labels=semantic_axes_labels,
        all_classes=all_classes
    )


def create_regression_tsne_visualization(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    colormap: str = 'viridis',
    use_3d: bool = False,
    zoom_factor: float = 2.0,
    cached_color_mapping: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create t-SNE visualization for regression data with continuous color mapping.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    visualizer = TSNEVisualizer(
        task_type='regression',
        use_3d=use_3d,
        use_knn=False,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        figsize=figsize,
        zoom_factor=zoom_factor,
        cached_color_mapping=cached_color_mapping
    )
    
    return visualizer.create_simple_visualization(
        train_embeddings, train_targets, test_embeddings, colormap=colormap
    )


def create_tsne_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10),
    zoom_factor: float = 2.0,
    knn_k: int = 5,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    use_3d: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None,
    all_classes: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a t-SNE plot with KNN connections and analysis.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    # Create KNN-enabled visualizer
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=use_3d,
        use_knn=True,
        knn_k=knn_k,
        figsize=figsize,
        zoom_factor=zoom_factor
    )
    
    # Since we already have coordinates, we need to use the plotter directly
    fig, legend_text, metadata = visualizer.plotter.create_plot(
        train_coords=train_tsne,
        train_data=train_labels,
        test_coords=test_tsne,
        test_data=None,
        highlight_test_idx=highlight_test_idx,
        semantic_axes_labels=semantic_axes_labels,
        class_names=class_names,
        use_semantic_names=use_semantic_names,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        all_classes=all_classes
    )
    
    # Note: KNN visualization is now handled internally by the TSNEVisualizer
    # No need to add connections manually - pie chart is generated automatically
    
    return fig, legend_text, metadata


def create_tsne_3d_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 12),
    zoom_factor: float = 2.0,
    knn_k: int = 5,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None,
    all_classes: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 3D t-SNE plot with KNN connections and analysis.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_tsne_plot_with_knn(
        train_tsne, test_tsne, train_labels, train_embeddings, test_embeddings,
        highlight_test_idx, figsize, zoom_factor, knn_k, class_names, 
        use_semantic_names, use_3d=True, semantic_axes_labels=semantic_axes_labels,
        all_classes=all_classes
    )


def create_regression_tsne_3d_visualization(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (15, 12),
    colormap: str = 'viridis',
    zoom_factor: float = 2.0,
    cached_color_mapping: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create 3D t-SNE visualization for regression data with continuous color mapping.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_regression_tsne_visualization(
        train_embeddings, train_targets, test_embeddings, test_targets,
        perplexity, max_iter, random_state, figsize, colormap, use_3d=True,
        zoom_factor=zoom_factor, cached_color_mapping=cached_color_mapping
    )


def create_combined_regression_tsne_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    zoom_factor: float = 2.0,
    colormap: str = 'viridis',
    use_3d: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a regression t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    # Create a regression plotter
    plotter = RegressionTSNEPlotter(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d)
    
    fig, legend_text, metadata = plotter.create_plot(
        train_coords=train_tsne,
        train_data=train_targets,
        test_coords=test_tsne,
        test_data=None,
        highlight_test_idx=highlight_test_idx,
        semantic_axes_labels=semantic_axes_labels,
        colormap=colormap
    )
    
    return fig, legend_text, metadata


def create_combined_regression_tsne_3d_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 12),
    zoom_factor: float = 2.0,
    colormap: str = 'viridis',
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 3D regression t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_combined_regression_tsne_plot(
        train_tsne, test_tsne, train_targets, highlight_test_idx,
        figsize, zoom_factor, colormap, use_3d=True, semantic_axes_labels=semantic_axes_labels
    )


def create_regression_tsne_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10),
    zoom_factor: float = 2.0,
    knn_k: int = 5,
    colormap: str = 'viridis',
    use_3d: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a regression t-SNE plot with KNN connections and analysis.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    # Create KNN-enabled regression visualizer
    visualizer = TSNEVisualizer(
        task_type='regression',
        use_3d=use_3d,
        use_knn=True,
        knn_k=knn_k,
        figsize=figsize,
        zoom_factor=zoom_factor
    )
    
    # Since we already have coordinates, use the plotter directly
    # IMPORTANT: Must pass train_embeddings and test_embeddings for KNN analysis
    fig, legend_text, metadata = visualizer.plotter.create_plot(
        train_coords=train_tsne,
        train_data=train_targets,
        test_coords=test_tsne,
        test_data=None,
        highlight_test_idx=highlight_test_idx,
        semantic_axes_labels=semantic_axes_labels,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        colormap=colormap
    )
    
    # Note: KNN visualization is now handled internally by the TSNEVisualizer
    # No need to add connections manually - pie chart is generated automatically
    
    return fig, legend_text, metadata


def create_regression_tsne_3d_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 12),
    zoom_factor: float = 2.0,
    knn_k: int = 5,
    colormap: str = 'viridis',
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 3D regression t-SNE plot with KNN connections and analysis.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    
    return create_regression_tsne_plot_with_knn(
        train_tsne, test_tsne, train_targets, train_embeddings, test_embeddings,
        highlight_test_idx, figsize, zoom_factor, knn_k, colormap, use_3d=True, 
        semantic_axes_labels=semantic_axes_labels
    )


__all__ = [
    # New refactored classes (recommended)
    'TSNEGenerator',
    'BaseTSNEPlotter', 
    'ClassificationTSNEPlotter',
    'RegressionTSNEPlotter',
    'KNNMixin',
    'TSNEVisualizer',
    
    # Backward compatibility functions (deprecated)
    'create_tsne_visualization',
    'create_tsne_3d_visualization', 
    'create_combined_tsne_plot',
    'create_combined_tsne_3d_plot',
    'create_tsne_plot_with_knn',
    'create_tsne_3d_plot_with_knn',
    'create_regression_tsne_visualization',
    'create_regression_tsne_3d_visualization',
    'create_combined_regression_tsne_plot',
    'create_combined_regression_tsne_3d_plot',
    'create_regression_tsne_plot_with_knn',
    'create_regression_tsne_3d_plot_with_knn',
]