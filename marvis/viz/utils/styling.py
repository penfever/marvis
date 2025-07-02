"""
Shared styling utilities for all MARVIS visualizations.

This module provides consistent styling across t-SNE functions, BaseVisualization,
and ContextComposer to ensure uniform appearance and behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from typing import Tuple, Optional, List, Dict, Union, Any

__all__ = [
    'get_distinct_colors',
    'create_distinct_color_map', 
    'get_class_color_name_map',
    'get_color_to_class_map',
    'create_class_legend',
    'format_class_label',
    'create_regression_color_map',
    'apply_consistent_point_styling',
    'apply_consistent_legend_formatting',
    'get_standard_test_point_style',
    'get_standard_target_point_style',
    'get_standard_training_point_style',
    'extract_visible_classes_from_legend'
]


def get_distinct_colors(n_classes: int) -> List[Tuple[np.ndarray, str]]:
    """
    Get distinct colors with semantic names for visualization.
    Generates additional colors programmatically if more than the predefined set are needed.
    
    Args:
        n_classes: Number of distinct colors needed
        
    Returns:
        List of (color_array, color_name) tuples
    """
    # Define a set of highly distinct colors with semantic names
    base_distinct_colors = [
        (np.array([0.12, 0.47, 0.71]), "Blue"),           # Blue
        (np.array([1.00, 0.50, 0.05]), "Orange"),         # Orange  
        (np.array([0.17, 0.63, 0.17]), "Green"),          # Green
        (np.array([0.84, 0.15, 0.16]), "Red"),            # Red
        (np.array([0.58, 0.40, 0.74]), "Purple"),         # Purple
        (np.array([0.55, 0.34, 0.29]), "Brown"),          # Brown
        (np.array([0.89, 0.47, 0.76]), "Pink"),           # Pink
        (np.array([0.50, 0.50, 0.50]), "Gray"),           # Gray
        (np.array([0.74, 0.74, 0.13]), "Olive"),          # Olive
        (np.array([0.09, 0.75, 0.81]), "Cyan"),           # Cyan
        (np.array([1.00, 1.00, 0.20]), "Yellow"),         # Yellow
        (np.array([0.65, 0.33, 0.65]), "Violet"),         # Violet
        (np.array([0.20, 0.20, 0.80]), "Navy"),           # Navy
        (np.array([0.80, 0.20, 0.20]), "Crimson"),        # Crimson
        (np.array([0.00, 0.50, 0.00]), "Dark Green"),     # Dark Green
        (np.array([0.80, 0.60, 0.20]), "Gold"),           # Gold
        (np.array([0.40, 0.20, 0.60]), "Indigo"),         # Indigo
        (np.array([0.90, 0.30, 0.30]), "Coral"),          # Coral
        (np.array([0.30, 0.70, 0.70]), "Teal"),           # Teal
        (np.array([0.70, 0.50, 0.80]), "Lavender"),       # Lavender
        (np.array([0.60, 0.80, 0.20]), "Lime"),           # Lime
        (np.array([0.90, 0.70, 0.50]), "Tan"),            # Tan
        (np.array([0.40, 0.60, 0.90]), "Sky Blue"),       # Sky Blue
        (np.array([0.80, 0.40, 0.60]), "Rose"),           # Rose
        (np.array([0.30, 0.50, 0.30]), "Forest Green"),   # Forest Green
    ]
    
    colors_needed = []
    
    # Use base colors first
    for i in range(min(n_classes, len(base_distinct_colors))):
        colors_needed.append(base_distinct_colors[i])
    
    # Generate additional colors if needed using HSV color space for maximum distinctness
    if n_classes > len(base_distinct_colors):
        additional_needed = n_classes - len(base_distinct_colors)
        
        # Generate colors in HSV space with varied hue, saturation, and value
        for i in range(additional_needed):
            # Vary hue across the spectrum, with some saturation and value variation
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good distribution
            saturation = 0.6 + (i % 3) * 0.15  # Vary between 0.6, 0.75, 0.9
            value = 0.7 + (i % 2) * 0.2  # Vary between 0.7, 0.9
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color_array = np.array(rgb)
            color_name = f"Color_{len(base_distinct_colors) + i + 1}"
            
            colors_needed.append((color_array, color_name))
    
    return colors_needed


def create_distinct_color_map(unique_classes: np.ndarray, cached_color_mapping: Optional[Dict] = None) -> Dict:
    """
    Create a color mapping using distinct, semantically named colors.
    
    Args:
        unique_classes: Array of unique class labels
        cached_color_mapping: Optional pre-computed color mapping from resource manager
        
    Returns:
        Dictionary mapping class labels to colors
    """
    # If cached mapping is provided, use it to ensure consistency
    if cached_color_mapping and 'class_to_color' in cached_color_mapping:
        class_to_color_names = cached_color_mapping.get('class_to_color', {})
        distinct_colors = get_distinct_colors(len(unique_classes))
        class_color_map = {}
        
        # Map color names to RGB values
        color_name_to_rgb = {color_name: color_array for color_array, color_name in distinct_colors}
        
        for class_label in unique_classes:
            # Use cached color name if available
            if class_label in class_to_color_names:
                color_name = class_to_color_names[class_label]
                # Find RGB value for this color name
                for color_array, name in distinct_colors:
                    if name == color_name:
                        class_color_map[class_label] = color_array
                        break
                else:
                    # Fallback if color name not found (shouldn't happen with proper caching)
                    class_color_map[class_label] = distinct_colors[0][0]
            else:
                # Fallback for classes not in cache (shouldn't happen)
                idx = len(class_color_map) % len(distinct_colors)
                class_color_map[class_label] = distinct_colors[idx][0]
        
        return class_color_map
    
    # Original implementation when no cache provided
    distinct_colors = get_distinct_colors(len(unique_classes))
    class_color_map = {}
    
    for i, class_label in enumerate(unique_classes):
        color_array, color_name = distinct_colors[i]
        class_color_map[class_label] = color_array
    
    return class_color_map


def get_class_color_name_map(unique_classes: np.ndarray) -> Dict:
    """
    Create a mapping from class labels to semantic color names.
    
    Args:
        unique_classes: Array of unique class labels
        
    Returns:
        Dictionary mapping class labels to color names
    """
    distinct_colors = get_distinct_colors(len(unique_classes))
    class_name_map = {}
    
    for i, class_label in enumerate(unique_classes):
        color_array, color_name = distinct_colors[i]
        class_name_map[class_label] = color_name
    
    return class_name_map


def get_color_to_class_map(unique_classes: np.ndarray) -> Dict:
    """
    Create a reverse mapping from color names to class labels for VLM response parsing.
    
    Args:
        unique_classes: Array of unique class labels
        
    Returns:
        Dictionary mapping color names to class labels (e.g., {"Blue": 0, "Color_111": 5})
    """
    class_to_color_map = get_class_color_name_map(unique_classes)
    color_to_class_map = {color_name: class_label for class_label, color_name in class_to_color_map.items()}
    return color_to_class_map


def format_class_label(class_label, class_names: Optional[List[str]] = None, use_semantic_names: bool = False, prefix: str = "Class") -> str:
    """
    Format a class label consistently based on semantic names setting.
    
    Args:
        class_label: The class index/label
        class_names: Optional list of semantic class names
        use_semantic_names: Whether to use semantic names
        prefix: Prefix for non-semantic format (e.g., "Class", "Training Class")
        
    Returns:
        Formatted class label string
    """
    if use_semantic_names and class_names and class_label < len(class_names):
        return class_names[class_label]
    else:
        return f"{prefix} {class_label}"


def create_class_legend(unique_classes: np.ndarray, class_color_map: Dict, class_names: Optional[List[str]] = None, use_semantic_names: bool = False, show_test_points: bool = False) -> str:
    """
    Create a text legend describing class colors with both semantic names and RGB values.
    
    Args:
        unique_classes: Array of unique class labels
        class_color_map: Dictionary mapping class labels to colors
        class_names: Optional list of semantic class names
        use_semantic_names: Whether to show semantic names in legend
        show_test_points: Whether test points are shown in the visualization
        
    Returns:
        legend_text: String description of the color legend
    """
    legend_lines = ["Class Legend:"]
    
    # Get all distinct colors that were actually assigned to ensure unique names
    distinct_colors = get_distinct_colors(len(unique_classes))
    color_name_map = {}
    used_names = set()
    
    # Build a mapping from class to unique color name
    for i, class_label in enumerate(unique_classes):
        color_array, original_name = distinct_colors[i]
        
        # Ensure uniqueness by appending a number if name is already used
        unique_name = original_name
        counter = 1
        while unique_name in used_names:
            unique_name = f"{original_name}_{counter}"
            counter += 1
        
        used_names.add(unique_name)
        color_name_map[class_label] = unique_name
    
    for class_label in unique_classes:
        color = class_color_map[class_label]
        
        # Get RGB values
        if hasattr(color, '__len__') and len(color) >= 3:
            rgb = tuple(int(c * 255) for c in color[:3])
        else:
            rgb = (128, 128, 128)  # Default gray
        
        # Get the guaranteed unique color name
        color_name = color_name_map[class_label]
        
        # Format class label consistently
        if use_semantic_names and class_names and class_label < len(class_names):
            # In semantic names mode, show only the semantic name
            class_display = class_names[class_label]
        else:
            class_display = f"Class {class_label}"
            
        legend_lines.append(f"- {class_display}: {color_name} RGB{rgb}")
    
    # Only add test points to legend if they are shown
    if show_test_points:
        legend_lines.append("- Test points: Light Gray RGB(211, 211, 211)")
    
    return "\n".join(legend_lines)


def extract_visible_classes_from_legend(unique_classes: np.ndarray, class_names: Optional[List[str]] = None, use_semantic_names: bool = False) -> List[Union[int, str]]:
    """
    Extract the visible classes that would appear in a legend.
    
    This function returns the class identifiers that are visible in the current
    visualization context, which correspond to the classes shown in the legend.
    
    Args:
        unique_classes: Array of unique class labels that are present in the data
        class_names: Optional list of semantic class names
        use_semantic_names: Whether semantic names are being used
        
    Returns:
        List of visible class identifiers (numeric labels or semantic names)
    """
    if use_semantic_names and class_names:
        # Return semantic names for classes that have them
        visible = []
        for class_label in unique_classes:
            if class_label < len(class_names):
                visible.append(class_names[class_label])
            else:
                visible.append(class_label)  # Fallback to numeric
        return visible
    else:
        # Return numeric class labels
        return list(unique_classes)


def create_regression_color_map(target_values: np.ndarray, colormap: str = 'RdBu_r', n_levels: int = 20) -> Tuple[np.ndarray, mcolors.Colormap, float, float]:
    """
    Create a discrete red-blue colormap for regression target values.
    
    Blue represents minimum values, red represents maximum values.
    Uses 20+ discrete levels for fine-grained VLM estimation.
    
    Args:
        target_values: Array of target values
        colormap: Name of matplotlib colormap to use (default: 'RdBu_r' for red-blue)
        n_levels: Number of discrete color levels (default: 20)
        
    Returns:
        Tuple of (normalized_values, colormap_object, vmin, vmax)
    """
    vmin, vmax = np.min(target_values), np.max(target_values)
    if vmin == vmax:
        # Handle constant values
        vmin -= 0.1
        vmax += 0.1
    
    # Normalize values to [0, 1] range
    normalized_values = (target_values - vmin) / (vmax - vmin)
    
    # Create discrete colormap with specified number of levels
    base_cmap = plt.get_cmap(colormap)
    colors = base_cmap(np.linspace(0, 1, n_levels))
    cmap = mcolors.ListedColormap(colors, name=f'{colormap}_discrete_{n_levels}')
    
    return normalized_values, cmap, vmin, vmax


# Point styling utility functions

def get_standard_test_point_style() -> Dict[str, Any]:
    """
    Get the standard styling for test points (gray squares).
    
    Returns:
        Dictionary of matplotlib scatter parameters
    """
    return {
        'marker': 's',  # square
        'c': 'lightgray',
        's': 60,  # size
        'alpha': 0.8,
        'edgecolors': 'gray',
        'linewidth': 0.8,
        'label': 'Test Points (Light Gray)',
        'zorder': 5  # Above training points but below targets
    }


def get_standard_target_point_style() -> Dict[str, Any]:
    """
    Get the standard styling for target/query points (red stars).
    
    Returns:
        Dictionary of matplotlib scatter parameters
    """
    return {
        'marker': '*',  # star
        'c': 'red',
        's': 120,  # size
        'alpha': 1.0,
        'edgecolors': 'darkred',
        'linewidth': 2,
        'label': 'Query Point (Red Star)',
        'zorder': 10  # On top of everything
    }


def get_standard_training_point_style() -> Dict[str, Any]:
    """
    Get the standard styling for training points (colored circles).
    
    Returns:
        Dictionary of matplotlib scatter parameters
    """
    return {
        's': 50,  # size
        'alpha': 0.7,
        'edgecolors': 'black',
        'linewidth': 0.5,
        'zorder': 1  # Below test and target points
    }


def apply_consistent_point_styling(
    ax, 
    transformed_data: np.ndarray,
    y: Optional[np.ndarray] = None,
    highlight_indices: Optional[List[int]] = None,
    test_data: Optional[np.ndarray] = None,
    highlight_test_indices: Optional[List[int]] = None,
    use_3d: bool = False,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    all_classes: Optional[np.ndarray] = None,
    cached_color_mapping: Optional[Dict] = None,
    show_test_points: bool = False
) -> Dict[str, Any]:
    """
    Apply consistent point styling across all visualization types.
    
    Args:
        ax: Matplotlib axis object
        transformed_data: Training data coordinates
        y: Optional training labels for coloring
        highlight_indices: Indices of training points to highlight 
        test_data: Optional test data coordinates
        highlight_test_indices: Indices of test points to highlight with red stars
        use_3d: Whether this is a 3D plot
        class_names: Optional semantic class names
        use_semantic_names: Whether to use semantic class names
        all_classes: Optional array of all possible class labels in the dataset
        cached_color_mapping: Optional cached color mapping for consistency
        show_test_points: Whether to show all test points (gray squares). If False, only highlighted test points are shown.
        
    Returns:
        Dictionary with metadata and legend information
    """
    legend_text_parts = []
    metadata = {
        'classes': [],  # Legacy field for backward compatibility
        'visible_classes': [],  # Classes visible in this visualization
        'plot_type': 'classification'
    }
    
    # 1. Plot training points
    if y is not None:
        unique_classes = np.unique(y)
        
        # Create color map using all possible classes for consistency, but legend will only show visible classes
        if all_classes is not None:
            class_color_map = create_distinct_color_map(all_classes, cached_color_mapping)
        else:
            # Fallback to unique classes if all_classes not provided (for backward compatibility)
            class_color_map = create_distinct_color_map(unique_classes, cached_color_mapping)
        
        # Get classes that actually have plotted points (addresses legend issue)
        from marvis.utils.class_name_utils import get_actually_plotted_classes
        actually_plotted_classes = get_actually_plotted_classes(y, transformed_data, ax)
        
        # Only create scatter plots for classes that are actually visible
        # This ensures the legend only shows classes that have visible points
        for class_label in actually_plotted_classes:
            mask = y == class_label
            color = class_color_map.get(class_label, (0.5, 0.5, 0.5))  # Default to gray if not in map
            training_style = get_standard_training_point_style()
            
            # Format class label
            formatted_label = format_class_label(class_label, class_names, use_semantic_names)
            
            if use_3d:
                ax.scatter(
                    transformed_data[mask, 0],
                    transformed_data[mask, 1], 
                    transformed_data[mask, 2],
                    c=[color], label=formatted_label, **training_style
                )
            else:
                ax.scatter(
                    transformed_data[mask, 0],
                    transformed_data[mask, 1],
                    c=[color], label=formatted_label, **training_style
                )
            
            metadata['classes'].append(class_label)
        
        # Extract visible classes from the legend using actually plotted classes
        visible_classes = extract_visible_classes_from_legend(
            actually_plotted_classes, class_names, use_semantic_names
        )
        metadata['visible_classes'] = visible_classes
        
        # Create legend text using only actually plotted classes
        # This ensures legend only shows classes that are actually visible in the visualization
        visible_class_color_map = {cls: class_color_map[cls] for cls in actually_plotted_classes if cls in class_color_map}
        legend_text = create_class_legend(
            actually_plotted_classes, visible_class_color_map, class_names, use_semantic_names, show_test_points
        )
        
    else:
        # Regression case or no labels provided
        metadata['plot_type'] = 'regression'
        metadata['visible_classes'] = []  # No classes in regression
        
        # For regression, specialized classes like TSNERegressionVisualization handle colorbar creation
        # This function is primarily for classification legends
        training_style = get_standard_training_point_style()
        if use_3d:
            ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                transformed_data[:, 2],
                **training_style
            )
        else:
            ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                **training_style
            )
        legend_text = "No class labels provided (regression colorbars handled by specialized classes)"
    
    # 2. Plot test points (gray squares) - only if show_test_points is True
    if test_data is not None and show_test_points:
        test_style = get_standard_test_point_style()
        if use_3d:
            ax.scatter(
                test_data[:, 0],
                test_data[:, 1],
                test_data[:, 2],
                **test_style
            )
        else:
            ax.scatter(
                test_data[:, 0],
                test_data[:, 1], 
                **test_style
            )
    
    # 3. Highlight specific test points with red stars
    if highlight_test_indices is not None and test_data is not None:
        target_style = get_standard_target_point_style()
        if use_3d:
            ax.scatter(
                test_data[highlight_test_indices, 0],
                test_data[highlight_test_indices, 1],
                test_data[highlight_test_indices, 2],
                **target_style
            )
        else:
            ax.scatter(
                test_data[highlight_test_indices, 0],
                test_data[highlight_test_indices, 1],
                **target_style
            )
    
    # 4. Highlight specific training points with red stars (if no test highlights)
    if highlight_indices is not None and highlight_test_indices is None:
        target_style = get_standard_target_point_style()
        if use_3d:
            ax.scatter(
                transformed_data[highlight_indices, 0],
                transformed_data[highlight_indices, 1],
                transformed_data[highlight_indices, 2],
                **target_style
            )
        else:
            ax.scatter(
                transformed_data[highlight_indices, 0],
                transformed_data[highlight_indices, 1],
                **target_style
            )
    
    # 5. Create the legend from all labeled artists (this will respect the filtered classes)
    # Only create legend if we have labeled plots (classification case)
    if y is not None and len(actually_plotted_classes) > 0:
        ax.legend()
    elif y is not None and len(actually_plotted_classes) == 0:
        # No visible classes - don't create a legend (matplotlib will warn if we try)
        pass
    
    return {
        'legend_text': legend_text,
        'metadata': metadata
    }


def apply_consistent_legend_formatting(ax, use_3d: bool = False) -> None:
    """
    Apply consistent legend formatting and positioning to an existing legend.
    
    This function enforces separation of concerns by only formatting existing legends,
    not creating new ones. Legend creation should be handled by apply_consistent_point_styling()
    or other content creation functions.
    
    Args:
        ax: Matplotlib axis object
        use_3d: Whether this is a 3D plot
        
    Note:
        If no legend exists (e.g., when no classes are visible in zoomed view), 
        this function silently does nothing rather than raising an error.
    """
    # Check if legend already exists
    existing_legend = ax.get_legend()
    if existing_legend is None:
        # No legend exists (e.g., no visible classes in zoomed view)
        # This is valid - just add grid and return
        ax.grid(True, alpha=0.3)
        return
    
    # Update positioning of existing legend (same for 2D and 3D currently)
    existing_legend.set_bbox_to_anchor((1.05, 1))
    existing_legend.set_loc('upper left')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)