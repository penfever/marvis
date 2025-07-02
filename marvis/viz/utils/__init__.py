"""
Visualization utilities for MARVIS.

This module provides common functionality for saving and processing
visualizations across different MARVIS implementations.
"""

from .common import plot_to_image, save_visualization_with_metadata, create_output_directories, generate_visualization_filename, close_figure_safely
from .styling import (
    get_distinct_colors,
    create_distinct_color_map,
    get_class_color_name_map,
    create_class_legend,
    format_class_label,
    create_regression_color_map,
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    get_standard_test_point_style,
    get_standard_target_point_style,
    get_standard_training_point_style
)

__all__ = [
    # Common utilities
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
    'get_standard_test_point_style',
    'get_standard_target_point_style',
    'get_standard_training_point_style'
]