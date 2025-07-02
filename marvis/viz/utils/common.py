"""
Visualization utilities for MARVIS.

This module provides common functionality for saving and processing
visualizations across different MARVIS implementations.
"""

import os
import io
import json
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Import JSON serialization utility
from ...utils.json_utils import convert_for_json_serialization

logger = logging.getLogger(__name__)


@contextmanager
def managed_figure(*args, **kwargs):
    """
    Context manager for matplotlib figures that ensures proper cleanup.
    
    Usage:
        with managed_figure(figsize=(10, 8)) as fig:
            # Use figure
            pass
        # Figure is automatically closed
    """
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


@contextmanager 
def managed_subplots(*args, **kwargs):
    """
    Context manager for matplotlib subplots that ensures proper cleanup.
    
    Usage:
        with managed_subplots(2, 2, figsize=(12, 10)) as (fig, axes):
            # Use figure and axes
            pass
        # Figure is automatically closed
    """
    fig, axes = plt.subplots(*args, **kwargs)
    try:
        yield fig, axes
    finally:
        plt.close(fig)


def ensure_figure_closed(func):
    """
    Decorator that ensures any figures created in a function are properly closed.
    This helps prevent memory leaks from matplotlib figures.
    """
    def wrapper(*args, **kwargs):
        initial_figures = set(plt.get_fignums())
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Close any new figures that were created
            current_figures = set(plt.get_fignums())
            new_figures = current_figures - initial_figures
            for fignum in new_figures:
                plt.close(fignum)
    return wrapper


def cleanup_matplotlib_figures(warn_threshold: int = 10):
    """
    Clean up all current matplotlib figures and warn if many are open.
    
    Args:
        warn_threshold: Number of figures above which to issue a warning
    """
    current_figures = plt.get_fignums()
    if len(current_figures) >= warn_threshold:
        logger.warning(f"Found {len(current_figures)} open matplotlib figures, closing all to prevent memory leaks")
    
    for fignum in current_figures:
        plt.close(fignum)
    
    return len(current_figures)


def force_matplotlib_cleanup():
    """
    Force cleanup of all matplotlib figures and clear internal state.
    Use this as a last resort when memory issues persist.
    """
    # Close all figures
    plt.close('all')
    
    # Clear matplotlib cache if available
    try:
        from matplotlib import font_manager
        font_manager._get_fontconfig_fonts.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Clear any remaining figure manager references
    try:
        import matplotlib._pylab_helpers
        matplotlib._pylab_helpers.Gcf.figs.clear()
    except (ImportError, AttributeError):
        pass


def plot_to_image(
    fig: plt.Figure, 
    dpi: int = 100, 
    force_rgb: bool = True, 
    max_size: Optional[int] = None,
    format: str = 'png',
    close_fig: bool = True
) -> Image.Image:
    """
    Convert matplotlib figure to PIL Image with optional processing.
    
    Args:
        fig: Matplotlib figure
        dpi: Resolution for image conversion
        force_rgb: Convert RGBA to RGB if needed
        max_size: Maximum width/height (resizes if larger)
        format: Image format ('png', 'jpg', etc.)
        close_fig: Whether to close the figure after conversion
        
    Returns:
        PIL Image object
    """
    try:
        # Convert plot to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format=format, dpi=dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        
        # Convert RGBA to RGB if needed
        if force_rgb and image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        
        # Resize image if needed
        if max_size and (image.width > max_size or image.height > max_size):
            ratio = min(max_size / image.width, max_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    finally:
        # Close figure to prevent memory leaks
        if close_fig:
            plt.close(fig)


def save_visualization_with_metadata(
    fig: plt.Figure,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    dpi: int = 100,
    force_rgb: bool = True,
    max_size: Optional[int] = None,
    close_fig: bool = True
) -> Dict[str, Union[str, int]]:
    """
    Save visualization figure with optional metadata.
    
    Args:
        fig: Matplotlib figure to save
        save_path: Path to save the image
        metadata: Optional metadata to save alongside
        dpi: Resolution for saving
        force_rgb: Convert to RGB mode
        max_size: Maximum image size
        close_fig: Whether to close the figure after saving
        
    Returns:
        Dictionary with save information (path, size, etc.)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to image and save (this will also close the figure)
    image = plot_to_image(fig, dpi=dpi, force_rgb=force_rgb, max_size=max_size, close_fig=close_fig)
    image.save(save_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = save_path.replace('.png', '_metadata.json').replace('.jpg', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Ensure metadata is JSON serializable
            json.dump(convert_for_json_serialization(metadata), f, indent=2)
    
    return {
        'path': save_path,
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'dpi': dpi,
        'metadata_saved': metadata is not None
    }


def create_output_directories(base_dir: str, subdirs: list) -> Dict[str, str]:
    """
    Create standardized output directory structure.
    
    Args:
        base_dir: Base output directory
        subdirs: List of subdirectory names to create
        
    Returns:
        Dictionary mapping subdir names to full paths
    """
    dir_paths = {}
    
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
        dir_paths[subdir] = full_path
    
    return dir_paths


def generate_visualization_filename(
    sample_index: int,
    backend: str = "tsne",
    dimensions: str = "2d", 
    use_knn: bool = False,
    nn_k: Optional[int] = None,
    extension: str = "png"
) -> str:
    """
    Generate standardized filename for visualization saves.
    
    Args:
        sample_index: Index of the sample
        backend: Backend used (tsne, pca, etc.)
        dimensions: Dimensionality (2d, 3d)
        use_knn: Whether KNN connections are shown
        nn_k: Number of KNN neighbors (if use_knn=True)
        extension: File extension
        
    Returns:
        Standardized filename
    """
    filename = f"sample_{sample_index:03d}_{backend}_{dimensions}"
    
    if use_knn and nn_k is not None:
        filename += f"_knn{nn_k}"
    
    filename += f".{extension}"
    
    return filename


def close_figure_safely(fig: Optional[plt.Figure]) -> None:
    """
    Safely close matplotlib figure if it exists.
    
    Args:
        fig: Matplotlib figure to close (can be None)
    """
    if fig is not None:
        plt.close(fig)