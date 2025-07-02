"""
t-SNE classification visualization that combines base t-SNE functionality with 
classification-specific plotting and optional KNN analysis.

This class replaces the functions:
- create_combined_tsne_plot
- create_combined_tsne_3d_plot  
- create_tsne_plot_with_knn
- create_tsne_3d_plot_with_knn
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple, Union
import logging

from .base import BaseTSNEVisualization
from ..mixins.knn import BaseKNNVisualization
from ..base import VisualizationConfig, VisualizationResult
from ..utils.styling import (
    create_distinct_color_map,
    apply_consistent_point_styling,
    get_standard_test_point_style,
    get_standard_target_point_style
)

logger = logging.getLogger(__name__)


class TSNEClassificationVisualization(BaseKNNVisualization, BaseTSNEVisualization):
    """
    t-SNE visualization specialized for classification tasks.
    
    This class provides:
    - Standard classification t-SNE plotting (2D and 3D)
    - Optional KNN connections and analysis
    - Consistent legend generation with visible classes
    - Multiple viewing angles for 3D plots
    - Proper integration with BaseVisualization architecture
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None, **kwargs):
        """
        Initialize classification t-SNE visualization.
        
        Args:
            config: Visualization configuration
            **kwargs: Additional parameters for t-SNE
        """
        super().__init__(config, **kwargs)
        
        # Override task type for classification
        self.config.task_type = 'classification'
    
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
        Generate a classification t-SNE plot.
        
        Args:
            transformed_data: Training data coordinates [n_train, n_components]
            y: Training labels [n_train]
            highlight_indices: Indices of training points to highlight
            test_data: Test data coordinates [n_test, n_components] (optional)
            highlight_test_indices: Indices of test points to highlight with red X
            **kwargs: Additional plotting parameters
            
        Returns:
            VisualizationResult with plot image and metadata
        """
        if y is None:
            raise ValueError("Classification visualization requires labels (y)")
        
        use_3d = self.config.use_3d and transformed_data.shape[1] >= 3 and self.supports_3d
        
        # Handle multiple viewing angles for 3D
        if use_3d and self.config.viewing_angles and len(self.config.viewing_angles) > 1:
            return self._generate_multi_view_3d_plot(
                transformed_data, y, highlight_indices, test_data, highlight_test_indices, **kwargs
            )
        else:
            return self._generate_single_view_plot(
                transformed_data, y, highlight_indices, test_data, highlight_test_indices, use_3d, **kwargs
            )
    
    def _generate_single_view_plot(
        self,
        transformed_data: np.ndarray,
        y: np.ndarray,
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        highlight_test_indices: Optional[List[int]],
        use_3d: bool,
        **kwargs
    ) -> VisualizationResult:
        """Generate a single view plot (2D or single 3D view)."""
        import time
        import io
        from PIL import Image
        
        plot_start = time.time()
        
        # Create figure
        if use_3d:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Get unique classes and create color map
        unique_classes = np.unique(y)
        class_color_map = create_distinct_color_map(unique_classes)
        
        # Apply consistent point styling
        show_test_points = getattr(self.config, 'show_test_points', False)
        plot_result = apply_consistent_point_styling(
            ax=ax,
            transformed_data=transformed_data,
            y=y,
            highlight_indices=highlight_indices,
            test_data=test_data,
            highlight_test_indices=highlight_test_indices,
            use_3d=use_3d,
            class_names=self._class_names,
            use_semantic_names=self._use_semantic_names,
            show_test_points=show_test_points
        )
        
        knn_info = None
        ax_pie = None
        
        # Add KNN connections if enabled
        if (self.config.use_knn_connections and 
            test_data is not None and 
            highlight_test_indices and
            self._train_embeddings is not None):
            
            # Process first highlighted test point
            test_idx = highlight_test_indices[0]
            knn_info = self.add_knn_connections_to_plot(
                ax=ax,
                train_coords=transformed_data,
                test_coords=test_data,
                test_idx=test_idx,
                use_3d=use_3d,
                k=self.config.nn_k
            )
            
            # Create pie chart for KNN analysis
            ax_pie = self.create_knn_pie_chart(
                fig=fig,
                knn_info=knn_info,
                class_color_map=class_color_map,
                class_names=self._class_names,
                use_semantic_names=self._use_semantic_names
            )
        
        # Apply zoom and viewing angles
        self.apply_zoom_and_viewing_angles(ax, use_3d)
        
        # Create legend text
        additional_info = ""
        if knn_info:
            additional_info = self.generate_knn_description(
                knn_info, self._class_names, self._use_semantic_names
            )
        
        legend_text = self.create_legend_text(unique_classes, class_color_map, additional_info)
        
        # Apply plot styling
        self._apply_plot_styling(ax, use_3d)
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Process image format
        if self.config.image_format == 'RGB' and image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        # Create metadata
        metadata = self.create_base_metadata(
            transformed_data, test_data, unique_classes, 
            highlight_test_indices[0] if highlight_test_indices else None
        )
        
        # Add KNN metadata if available
        if knn_info:
            metadata.update(self.get_knn_metadata(knn_info))
            metadata['has_knn_pie_chart'] = ax_pie is not None
        
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
            legend_text=legend_text,
            metadata=metadata
        )
        
        self._add_quality_metrics(result)
        return result
    
    def _generate_multi_view_3d_plot(
        self,
        transformed_data: np.ndarray,
        y: np.ndarray,
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        highlight_test_indices: Optional[List[int]],
        **kwargs
    ) -> VisualizationResult:
        """Generate a multi-view 3D plot with multiple viewing angles."""
        import time
        import io
        from PIL import Image
        
        plot_start = time.time()
        
        viewing_angles = self.config.viewing_angles
        n_views = len(viewing_angles)
        
        # Create figure with subplots for multiple views
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Determine subplot layout
        if n_views <= 2:
            rows, cols = 1, n_views
        elif n_views <= 4:
            rows, cols = 2, 2
        else:
            rows = int(np.ceil(np.sqrt(n_views)))
            cols = int(np.ceil(n_views / rows))
        
        # Get unique classes and create color map
        unique_classes = np.unique(y)
        class_color_map = create_distinct_color_map(unique_classes)
        
        view_names = [f'View {i+1}' for i in range(n_views)]
        knn_info = None
        
        # Create each view
        for i, (elev, azim) in enumerate(viewing_angles):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            
            # Apply consistent point styling
            show_test_points = getattr(self.config, 'show_test_points', False)
            apply_consistent_point_styling(
                ax=ax,
                transformed_data=transformed_data,
                y=y,
                highlight_indices=highlight_indices,
                test_data=test_data,
                highlight_test_indices=highlight_test_indices,
                use_3d=True,
                class_names=self._class_names,
                use_semantic_names=self._use_semantic_names,
                show_test_points=show_test_points
            )
            
            # Add KNN connections if enabled (only for first view)
            if (i == 0 and self.config.use_knn_connections and 
                test_data is not None and highlight_test_indices and
                self._train_embeddings is not None):
                
                test_idx = highlight_test_indices[0]
                knn_info = self.add_knn_connections_to_plot(
                    ax=ax,
                    train_coords=transformed_data,
                    test_coords=test_data,
                    test_idx=test_idx,
                    use_3d=True,
                    k=self.config.nn_k
                )
            
            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)
            
            # Apply zoom
            if self.config.zoom_factor != 1.0:
                ax.dist = ax.dist * self.config.zoom_factor
            
            # Set title for this view
            ax.set_title(f'{view_names[i]}: Elevation={elev}°, Azimuth={azim}°', fontsize=10)
            
            # Set axis labels
            ax.set_xlabel(f'{self.method_name} Component 1')
            ax.set_ylabel(f'{self.method_name} Component 2')
            ax.set_zlabel(f'{self.method_name} Component 3')
        
        # Create legend text
        additional_info = f"This visualization shows {n_views} different viewing angles of the same 3D t-SNE space:"
        for i, (elev, azim) in enumerate(viewing_angles):
            additional_info += f"\n- {view_names[i]}: Elevation={elev}°, Azimuth={azim}°"
        
        if knn_info:
            knn_description = self.generate_knn_description(
                knn_info, self._class_names, self._use_semantic_names
            )
            additional_info += f"\n\n{knn_description}"
        
        legend_text = self.create_legend_text(unique_classes, class_color_map, additional_info)
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Process image format
        if self.config.image_format == 'RGB' and image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        # Create metadata
        metadata = self.create_base_metadata(
            transformed_data, test_data, unique_classes,
            highlight_test_indices[0] if highlight_test_indices else None
        )
        
        # Add multi-view specific metadata
        metadata.update({
            'viewing_angles': viewing_angles,
            'n_views': n_views,
            'view_names': view_names
        })
        
        # Add KNN metadata if available
        if knn_info:
            metadata.update(self.get_knn_metadata(knn_info))
        
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
            legend_text=legend_text,
            metadata=metadata
        )
        
        self._add_quality_metrics(result)
        return result