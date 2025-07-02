"""
MARVIS Visualization Module

Enhanced visualization system for MARVIS models with support for multiple
dimensionality reduction techniques, decision boundaries, pattern analysis,
and composable visual reasoning for VLM backends.
"""

from .base import BaseVisualization, VisualizationConfig, VisualizationResult
from .context.composer import ContextComposer

# Embedding visualizations
from .embeddings.tsne import TSNEVisualization
from .embeddings.umap import UMAPVisualization
from .embeddings.manifold import (
    LocallyLinearEmbeddingVisualization,
    SpectralEmbeddingVisualization,
    IsomapVisualization,
    MDSVisualization
)
from .embeddings.pca import PCAVisualization

# Decision and pattern visualizations
from .decision.regions import DecisionRegionsVisualization
from .patterns.frequent import FrequentPatternsVisualization

# t-SNE visualization functions
from .tsne_functions import *

# Utilities
from .utils.common import plot_to_image, save_visualization_with_metadata, create_output_directories, generate_visualization_filename, close_figure_safely
from .utils.styling import (
    get_distinct_colors, create_distinct_color_map, get_class_color_name_map, 
    create_class_legend, format_class_label, create_regression_color_map,
    apply_consistent_point_styling, apply_consistent_legend_formatting
)

__all__ = [
    # Base classes
    'BaseVisualization',
    'VisualizationConfig', 
    'VisualizationResult',
    'ContextComposer',
    
    # Embedding visualizations
    'TSNEVisualization',
    'UMAPVisualization',
    'LocallyLinearEmbeddingVisualization',
    'SpectralEmbeddingVisualization',
    'IsomapVisualization',
    'MDSVisualization',
    'PCAVisualization',
    
    # Decision and pattern visualizations
    'DecisionRegionsVisualization',
    'FrequentPatternsVisualization',
    
    # t-SNE functions (automatically imported from tsne_functions)
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
    
    # Utilities
    'plot_to_image',
    'save_visualization_with_metadata',
    'create_output_directories', 
    'generate_visualization_filename',
    'close_figure_safely',
    
    # Styling utilities
    'get_distinct_colors',
    'create_distinct_color_map', 
    'get_class_color_name_map',
    'create_class_legend',
    'format_class_label',
    'create_regression_color_map',
    'apply_consistent_point_styling',
    'apply_consistent_legend_formatting',
]

# Version
__version__ = "1.0.0"