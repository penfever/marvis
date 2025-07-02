"""
KNN visualization mixin that adds nearest neighbor functionality to visualizations.

This mixin can be combined with any visualization class to add KNN connections,
neighbor analysis, and pie chart generation capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    get_standard_test_point_style,
    get_standard_target_point_style,
    create_distinct_color_map,
    get_class_color_name_map
)

logger = logging.getLogger(__name__)


def create_knn_regression_analysis(
    neighbor_targets: np.ndarray,
    neighbor_distances: np.ndarray,
    ax: plt.Axes,
    k: int
) -> str:
    """
    Create KNN regression analysis visualization (bar chart) with target values and distances.
    
    This function was missing from the original implementation but is called by the 
    deprecated regression KNN functions.
    
    Args:
        neighbor_targets: Target values of KNN neighbors [k]
        neighbor_distances: Distances to KNN neighbors [k]
        ax: Matplotlib axis for the bar chart
        k: Number of neighbors
        
    Returns:
        Description text for the KNN analysis
    """
    # Clear the axis
    ax.clear()
    
    # Sort neighbors by distance for better visualization
    sorted_indices = np.argsort(neighbor_distances)
    sorted_targets = neighbor_targets[sorted_indices]
    sorted_distances = neighbor_distances[sorted_indices]
    
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
        ax.text(i, target + (target_max - target_min) * 0.02, 
                f'd={distance:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Neighbor Rank (by distance)')
    ax.set_ylabel('Target Value')
    ax.set_title(f'KNN Analysis: {k} Nearest Neighbors')
    ax.set_xticks(neighbor_indices)
    ax.set_xticklabels([f'#{i+1}' for i in range(len(sorted_targets))])
    
    # Add statistics as text
    mean_target = np.mean(sorted_targets)
    median_target = np.median(sorted_targets)
    std_target = np.std(sorted_targets)
    
    stats_text = f'Mean: {mean_target:.2f}\nMedian: {median_target:.2f}\nStd: {std_target:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create description text
    description = f"Neighbor Analysis: {k} nearest neighbors found in embedding space:\n"
    for i, (target, distance) in enumerate(zip(sorted_targets, sorted_distances)):
        description += f"  #{i+1}: target={target:.3f}, distance={distance:.3f}\n"
    
    description += f"\nStatistics: mean={mean_target:.3f}, median={median_target:.3f}, std={std_target:.3f}"
    
    return description


class BaseKNNVisualization:
    """
    Mixin class that adds KNN functionality to visualizations.
    
    This class provides methods for:
    - Finding k-nearest neighbors in embedding space
    - Drawing connection lines between query and neighbor points
    - Creating pie charts showing neighbor class distributions
    - Generating KNN analysis metadata
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize KNN functionality."""
        super().__init__(*args, **kwargs)
        
        # KNN state
        self._knn_model = None
        self._train_embeddings = None
        self._train_labels = None
        self._test_embeddings = None
    
    def set_knn_data(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: Optional[np.ndarray] = None
    ):
        """
        Set the embedding data for KNN analysis.
        
        Args:
            train_embeddings: Training data in original embedding space [n_train, n_features]
            train_labels: Training labels [n_train]
            test_embeddings: Test data in original embedding space [n_test, n_features] (optional)
        """
        self._train_embeddings = train_embeddings
        self._train_labels = train_labels
        self._test_embeddings = test_embeddings
        
        # Fit KNN model on training embeddings
        k = getattr(self.config, 'nn_k', 5)
        self._knn_model = NearestNeighbors(
            n_neighbors=min(k, len(train_embeddings)),
            metric='euclidean'
        )
        self._knn_model.fit(train_embeddings)
    
    def find_knn_for_test_point(
        self,
        test_idx: int,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find k-nearest neighbors for a test point.
        
        Args:
            test_idx: Index of the test point
            k: Number of neighbors to find (uses config.nn_k if None)
            
        Returns:
            Dictionary with neighbor information
        """
        if self._knn_model is None or self._test_embeddings is None:
            raise ValueError("Must call set_knn_data() before finding neighbors")
        
        if test_idx >= len(self._test_embeddings):
            raise ValueError(f"test_idx {test_idx} >= number of test points {len(self._test_embeddings)}")
        
        if k is None:
            k = getattr(self.config, 'nn_k', 5)
        
        # Get test point embedding
        test_embedding = self._test_embeddings[test_idx]
        
        # Find neighbors
        distances, indices = self._knn_model.kneighbors([test_embedding], n_neighbors=k)
        
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        neighbor_labels = self._train_labels[neighbor_indices]
        
        # Compute class distribution
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        class_distribution = dict(zip(unique_labels.astype(int), counts.astype(int)))
        
        # Determine predicted class (majority vote)
        predicted_class = unique_labels[np.argmax(counts)]
        confidence = np.max(counts) / len(neighbor_labels)
        
        return {
            'test_idx': test_idx,
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_labels': neighbor_labels,
            'class_distribution': class_distribution,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'k': k
        }
    
    def add_knn_connections_to_plot(
        self,
        ax,
        train_coords: np.ndarray,
        test_coords: np.ndarray,
        test_idx: int,
        use_3d: bool = False,
        k: Optional[int] = None,
        line_alpha: float = 0.6,
        line_color: str = 'gray',
        line_style: str = '--'
    ) -> Dict[str, Any]:
        """
        Add KNN connection lines to an existing plot.
        
        Args:
            ax: Matplotlib axes object
            train_coords: Training data coordinates in visualization space [n_train, n_dims]
            test_coords: Test data coordinates in visualization space [n_test, n_dims]
            test_idx: Index of the test point to analyze
            use_3d: Whether this is a 3D plot
            k: Number of neighbors (uses config.nn_k if None)
            line_alpha: Alpha value for connection lines
            line_color: Color for connection lines
            line_style: Line style for connections
            
        Returns:
            Dictionary with KNN analysis information
        """
        # Get KNN information
        knn_info = self.find_knn_for_test_point(test_idx, k)
        
        # Get coordinates
        test_coord = test_coords[test_idx]
        neighbor_coords = train_coords[knn_info['neighbor_indices']]
        
        # Draw connection lines
        for i, neighbor_coord in enumerate(neighbor_coords):
            # Variable alpha based on distance (closer = more opaque)
            distance = knn_info['neighbor_distances'][i]
            max_distance = np.max(knn_info['neighbor_distances'])
            alpha = max(0.1, line_alpha * (1.0 - distance / max_distance)) if max_distance > 0 else line_alpha
            
            if use_3d:
                ax.plot3D(
                    [test_coord[0], neighbor_coord[0]],
                    [test_coord[1], neighbor_coord[1]],
                    [test_coord[2], neighbor_coord[2]],
                    color=line_color,
                    alpha=alpha,
                    linewidth=1.0,
                    linestyle=line_style
                )
            else:
                ax.plot(
                    [test_coord[0], neighbor_coord[0]],
                    [test_coord[1], neighbor_coord[1]],
                    color=line_color,
                    alpha=alpha,
                    linewidth=1.0,
                    linestyle=line_style
                )
        
        return knn_info
    
    def create_knn_pie_chart(
        self,
        fig,
        knn_info: Dict[str, Any],
        class_color_map: Optional[Dict] = None,
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False,
        position: Tuple[float, float, float, float] = (0.02, 0.02, 0.2, 0.2)
    ) -> plt.Axes:
        """
        Create a pie chart showing the class distribution of neighbors.
        
        Args:
            fig: Matplotlib figure object
            knn_info: KNN analysis information from find_knn_for_test_point()
            class_color_map: Mapping from class labels to colors (optional)
            class_names: List of class names for labeling (optional)
            use_semantic_names: Whether to use semantic class names
            position: Position and size of pie chart (left, bottom, width, height)
            
        Returns:
            Matplotlib axes object for the pie chart
        """
        class_distribution = knn_info['class_distribution']
        
        if not class_distribution:
            return None
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        for class_label, count in class_distribution.items():
            # Format class label
            if use_semantic_names and class_names and class_label < len(class_names):
                label = class_names[class_label]
            else:
                label = f"Class {class_label}"
            
            labels.append(f"{label}\n({count})")
            sizes.append(count)
            
            # Get color for this class
            if class_color_map and class_label in class_color_map:
                colors.append(class_color_map[class_label])
            else:
                # Default color
                colors.append('lightgray')
        
        # Create pie chart in specified position
        ax_pie = fig.add_axes(position)
        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.0f%%',
            startangle=90,
            textprops={'fontsize': 8}
        )
        
        # Style the pie chart
        ax_pie.set_title(
            f"K={knn_info['k']} Neighbors\nPredicted: Class {knn_info['predicted_class']}",
            fontsize=9,
            pad=10
        )
        
        return ax_pie
    
    def generate_knn_description(
        self,
        knn_info: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False
    ) -> str:
        """
        Generate a text description of the KNN analysis.
        
        Args:
            knn_info: KNN analysis information
            class_names: List of class names for labeling (optional)
            use_semantic_names: Whether to use semantic class names
            
        Returns:
            Formatted description string
        """
        k = knn_info['k']
        predicted_class = knn_info['predicted_class']
        confidence = knn_info['confidence']
        class_distribution = knn_info['class_distribution']
        
        # Format predicted class name
        if use_semantic_names and class_names and predicted_class < len(class_names):
            predicted_name = class_names[predicted_class]
        else:
            predicted_name = f"Class {predicted_class}"
        
        description_parts = [
            f"K-Nearest Neighbor Analysis (k={k}):",
            f"Predicted class: {predicted_name} (confidence: {confidence:.1%})"
        ]
        
        # Add class distribution
        if len(class_distribution) > 1:
            dist_parts = []
            for class_label, count in sorted(class_distribution.items()):
                if use_semantic_names and class_names and class_label < len(class_names):
                    class_name = class_names[class_label]
                else:
                    class_name = f"Class {class_label}"
                dist_parts.append(f"{class_name}: {count}")
            
            description_parts.append(f"Neighbor distribution: {', '.join(dist_parts)}")
        
        return "\n".join(description_parts)
    
    def get_knn_metadata(self, knn_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from KNN analysis for inclusion in VisualizationResult.
        
        Args:
            knn_info: KNN analysis information
            
        Returns:
            Dictionary with KNN metadata
        """
        return {
            'knn_k': knn_info['k'],
            'knn_predicted_class': knn_info['predicted_class'],
            'knn_confidence': knn_info['confidence'],
            'knn_class_distribution': knn_info['class_distribution'],
            'knn_neighbor_distances': knn_info['neighbor_distances'].tolist(),
            'knn_description': self.generate_knn_description(
                knn_info,
                getattr(self, '_class_names', None),
                getattr(self, '_use_semantic_names', False)
            )
        }