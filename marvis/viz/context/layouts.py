"""
Layout management for multi-visualization compositions.

This module handles the spatial arrangement of multiple visualizations
in a single composed image optimized for VLM consumption.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from ..base import VisualizationResult

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    get_class_color_name_map
)

logger = logging.getLogger(__name__)


class LayoutStrategy(Enum):
    """Available layout strategies for visualization composition."""
    GRID = "grid"                    # Regular grid layout
    ADAPTIVE_GRID = "adaptive_grid"  # Grid that adapts to content
    SEQUENTIAL = "sequential"        # Sequential horizontal layout
    HIERARCHICAL = "hierarchical"    # Tree-like hierarchical layout
    FOCUS_PLUS_CONTEXT = "focus_plus_context"  # One large + smaller supporting views


@dataclass
class LayoutConfig:
    """Configuration for layout management."""
    strategy: LayoutStrategy = LayoutStrategy.GRID
    max_per_row: int = 2
    spacing: float = 0.1
    padding: float = 0.05
    title_height: float = 0.08
    legend_width: float = 0.15
    background_color: Tuple[int, int, int] = (255, 255, 255)
    border_color: Tuple[int, int, int] = (200, 200, 200)
    border_width: int = 2


class LayoutManager:
    """
    Manages the layout and composition of multiple visualizations.
    
    This class is responsible for arranging multiple visualization results
    into a single, coherent image that can be effectively processed by VLM systems.
    """
    
    def __init__(self, composition_config):
        """
        Initialize the layout manager.
        
        Args:
            composition_config: CompositionConfig object
        """
        self.composition_config = composition_config
        self.logger = logging.getLogger(__name__)
        
        # Create layout config from composition config
        self.layout_config = LayoutConfig(
            strategy=composition_config.layout_strategy,
            max_per_row=composition_config.max_visualizations_per_row,
            spacing=composition_config.subplot_spacing
        )
    
    def compose_layout(
        self,
        results: List[VisualizationResult],
        strategy: Optional[LayoutStrategy] = None
    ) -> Image.Image:
        """
        Compose multiple visualization results into a single image.
        
        Args:
            results: List of VisualizationResult objects
            strategy: Layout strategy to use (overrides config)
            
        Returns:
            Composed PIL Image
        """
        if not results:
            raise ValueError("No visualization results provided")
        
        layout_strategy = strategy or self.layout_config.strategy
                
        # Route to appropriate layout method
        if layout_strategy == LayoutStrategy.GRID:
            return self._compose_grid_layout(results)
        elif layout_strategy == LayoutStrategy.ADAPTIVE_GRID:
            return self._compose_adaptive_grid_layout(results)
        elif layout_strategy == LayoutStrategy.SEQUENTIAL:
            return self._compose_sequential_layout(results)
        elif layout_strategy == LayoutStrategy.HIERARCHICAL:
            return self._compose_hierarchical_layout(results)
        elif layout_strategy == LayoutStrategy.FOCUS_PLUS_CONTEXT:
            return self._compose_focus_plus_context_layout(results)
        else:
            raise ValueError(f"Unsupported layout strategy: {layout_strategy}")
    
    def _compose_grid_layout(self, results: List[VisualizationResult]) -> Image.Image:
        """Compose results in a regular grid layout."""
        n_results = len(results)
        n_cols = min(self.layout_config.max_per_row, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        
        # Calculate grid dimensions
        base_width = max(result.image.width for result in results)
        base_height = max(result.image.height for result in results)
        
        spacing_px = int(self.layout_config.spacing * base_width)
        padding_px = int(self.layout_config.padding * base_width)
        title_height_px = int(self.layout_config.title_height * base_height)
        
        # Calculate total composition size
        total_width = n_cols * base_width + (n_cols - 1) * spacing_px + 2 * padding_px
        total_height = n_rows * (base_height + title_height_px) + (n_rows - 1) * spacing_px + 2 * padding_px
        
        # Create composition image
        composition = Image.new('RGB', (total_width, total_height), self.layout_config.background_color)
        draw = ImageDraw.Draw(composition)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Place images in grid
        for i, result in enumerate(results):
            row = i // n_cols
            col = i % n_cols
            
            # Calculate position
            x = padding_px + col * (base_width + spacing_px)
            y = padding_px + row * (base_height + title_height_px + spacing_px)
            
            # Resize image if necessary
            if result.image.size != (base_width, base_height):
                resized_image = result.image.resize((base_width, base_height), Image.Resampling.LANCZOS)
            else:
                resized_image = result.image
            
            # Paste image
            composition.paste(resized_image, (x, y))
            
            # Add title
            title = f"{result.method_name}"
            if result.config.title:
                title = result.config.title
            
            title_x = x + base_width // 2
            title_y = y + base_height + 5
            
            # Get text bbox to center it
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            
            draw.text(
                (title_x - text_width // 2, title_y),
                title,
                fill=(0, 0, 0),
                font=font
            )
            
            # Add border
            draw.rectangle(
                [x - 1, y - 1, x + base_width + 1, y + base_height + 1],
                outline=self.layout_config.border_color,
                width=self.layout_config.border_width
            )
        
        return composition
    
    def _compose_adaptive_grid_layout(self, results: List[VisualizationResult]) -> Image.Image:
        """Compose results in an adaptive grid that adjusts to content."""
        # Group results by importance/type
        primary_results = []
        secondary_results = []
        
        for result in results:
            # Prioritize results with highlighting or special metadata
            if result.highlighted_indices or result.metadata.get('importance', 0) > 0.5:
                primary_results.append(result)
            else:
                secondary_results.append(result)
        
        # If no prioritization, treat first result as primary
        if not primary_results:
            primary_results = [results[0]]
            secondary_results = results[1:]
        
        # Layout primary results with more space
        primary_size = 400
        secondary_size = 200
        spacing = 20
        padding = 30
        
        # Calculate layout
        n_primary = len(primary_results)
        n_secondary = len(secondary_results)
        
        # Arrange primary results in a row
        primary_width = n_primary * primary_size + (n_primary - 1) * spacing
        
        # Arrange secondary results below
        secondary_cols = min(4, n_secondary) if n_secondary > 0 else 0
        secondary_rows = (n_secondary + secondary_cols - 1) // secondary_cols if n_secondary > 0 else 0
        secondary_width = secondary_cols * secondary_size + (secondary_cols - 1) * spacing if secondary_cols > 0 else 0
        
        # Total dimensions
        total_width = max(primary_width, secondary_width) + 2 * padding
        total_height = primary_size + (secondary_rows * secondary_size + (secondary_rows - 1) * spacing if secondary_rows > 0 else 0) + 2 * padding + spacing
        
        # Create composition
        composition = Image.new('RGB', (total_width, total_height), self.layout_config.background_color)
        
        # Place primary results
        start_x = (total_width - primary_width) // 2
        for i, result in enumerate(primary_results):
            x = start_x + i * (primary_size + spacing)
            y = padding
            
            resized = result.image.resize((primary_size, primary_size), Image.Resampling.LANCZOS)
            composition.paste(resized, (x, y))
            
            # Add method label
            self._add_text_label(composition, result.method_name, x + primary_size // 2, y + primary_size + 5)
        
        # Place secondary results
        if secondary_results:
            start_x = (total_width - secondary_width) // 2
            start_y = padding + primary_size + spacing + 30
            
            for i, result in enumerate(secondary_results):
                row = i // secondary_cols
                col = i % secondary_cols
                
                x = start_x + col * (secondary_size + spacing)
                y = start_y + row * (secondary_size + spacing)
                
                resized = result.image.resize((secondary_size, secondary_size), Image.Resampling.LANCZOS)
                composition.paste(resized, (x, y))
                
                # Add method label
                self._add_text_label(composition, result.method_name, x + secondary_size // 2, y + secondary_size + 5)
        
        return composition
    
    def _compose_sequential_layout(self, results: List[VisualizationResult]) -> Image.Image:
        """Compose results in a horizontal sequence matching tsne_functions style."""
        if not results:
            return Image.new('RGB', (100, 100), (255, 255, 255))
        
        # Use consistent dimensions for all visualizations
        target_width = 600   # Fixed width for consistency
        target_height = 600  # Fixed height for consistency (square format)
        spacing = 40  # Increased spacing for better separation
        padding = 50  # Increased padding for better framing
        
        # Resize all images to same dimensions (square format)
        resized_results = []
        
        for result in results:
            # Resize to exact target dimensions (square format for consistency)
            resized_image = result.image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_results.append((resized_image, result))
        
        # Calculate total composition dimensions
        total_width = len(results) * target_width + (len(results) - 1) * spacing + 2 * padding
        total_height = target_height + 2 * padding + 60  # Extra space for labels
        
        # Create composition with white background matching tsne_functions
        composition = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # Place images side by side
        current_x = padding
        for resized_image, result in resized_results:
            composition.paste(resized_image, (current_x, padding))
            
            # Add method label below image (like tsne_functions legend style)
            self._add_text_label(
                composition, 
                result.method_name,
                current_x + target_width // 2,
                padding + target_height + 15,
                font_size=18  # Larger font for better readability
            )
            
            current_x += target_width + spacing
        
        return composition
    
    def _compose_hierarchical_layout(self, results: List[VisualizationResult]) -> Image.Image:
        """Compose results in a hierarchical tree layout."""
        # For now, implement a simple hierarchical layout
        # This could be expanded to show method relationships
        
        if len(results) == 1:
            return self._add_border_and_title(results[0])
        
        # Split into levels
        level1 = results[:1]  # Main visualization
        level2 = results[1:3] if len(results) > 1 else []  # Supporting visualizations
        level3 = results[3:] if len(results) > 3 else []  # Additional context
        
        level1_size = 500
        level2_size = 250
        level3_size = 150
        spacing = 20
        padding = 30
        
        # Calculate layout
        total_width = max(
            level1_size,
            len(level2) * level2_size + (len(level2) - 1) * spacing if level2 else 0,
            len(level3) * level3_size + (len(level3) - 1) * spacing if level3 else 0
        ) + 2 * padding
        
        total_height = level1_size + (level2_size if level2 else 0) + (level3_size if level3 else 0) + 4 * padding
        
        composition = Image.new('RGB', (total_width, total_height), self.layout_config.background_color)
        
        # Place level 1 (main)
        if level1:
            x = (total_width - level1_size) // 2
            y = padding
            resized = level1[0].image.resize((level1_size, level1_size), Image.Resampling.LANCZOS)
            composition.paste(resized, (x, y))
            self._add_text_label(composition, level1[0].method_name, x + level1_size // 2, y + level1_size + 10)
        
        # Place level 2
        if level2:
            level2_width = len(level2) * level2_size + (len(level2) - 1) * spacing
            start_x = (total_width - level2_width) // 2
            y = padding + level1_size + padding
            
            for i, result in enumerate(level2):
                x = start_x + i * (level2_size + spacing)
                resized = result.image.resize((level2_size, level2_size), Image.Resampling.LANCZOS)
                composition.paste(resized, (x, y))
                self._add_text_label(composition, result.method_name, x + level2_size // 2, y + level2_size + 5)
        
        # Place level 3
        if level3:
            level3_width = len(level3) * level3_size + (len(level3) - 1) * spacing
            start_x = (total_width - level3_width) // 2
            y = padding + level1_size + padding + (level2_size if level2 else 0) + padding
            
            for i, result in enumerate(level3):
                x = start_x + i * (level3_size + spacing)
                resized = result.image.resize((level3_size, level3_size), Image.Resampling.LANCZOS)
                composition.paste(resized, (x, y))
                self._add_text_label(composition, result.method_name, x + level3_size // 2, y + level3_size + 5)
        
        return composition
    
    def _compose_focus_plus_context_layout(self, results: List[VisualizationResult]) -> Image.Image:
        """Compose with one large focal visualization and smaller context views."""
        if not results:
            return Image.new('RGB', (100, 100), (255, 255, 255))
        
        # First result is the focus, others are context
        focus_result = results[0]
        context_results = results[1:]
        
        focus_size = 600
        context_size = 200
        spacing = 20
        padding = 30
        
        # Calculate layout
        context_cols = min(3, len(context_results))
        context_rows = (len(context_results) + context_cols - 1) // context_cols if context_results else 0
        
        context_width = context_cols * context_size + (context_cols - 1) * spacing if context_cols > 0 else 0
        context_height = context_rows * context_size + (context_rows - 1) * spacing if context_rows > 0 else 0
        
        total_width = focus_size + spacing + context_width + 2 * padding if context_results else focus_size + 2 * padding
        total_height = max(focus_size, context_height) + 2 * padding + 50  # Extra for labels
        
        composition = Image.new('RGB', (total_width, total_height), self.layout_config.background_color)
        
        # Place focus visualization
        focus_resized = focus_result.image.resize((focus_size, focus_size), Image.Resampling.LANCZOS)
        composition.paste(focus_resized, (padding, padding))
        self._add_text_label(composition, f"Focus: {focus_result.method_name}", padding + focus_size // 2, padding + focus_size + 10)
        
        # Place context visualizations
        if context_results:
            context_start_x = padding + focus_size + spacing
            context_start_y = padding + (focus_size - context_height) // 2  # Center vertically
            
            for i, result in enumerate(context_results):
                row = i // context_cols
                col = i % context_cols
                
                x = context_start_x + col * (context_size + spacing)
                y = context_start_y + row * (context_size + spacing)
                
                context_resized = result.image.resize((context_size, context_size), Image.Resampling.LANCZOS)
                composition.paste(context_resized, (x, y))
                self._add_text_label(composition, result.method_name, x + context_size // 2, y + context_size + 5)
        
        return composition
    
    def _add_text_label(self, image: Image.Image, text: str, x: int, y: int, font_size: int = 16):
        """Add a text label to an image."""
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        
        draw.text(
            (x - text_width // 2, y),
            text,
            fill=(0, 0, 0),
            font=font
        )
    
    def _add_border_and_title(self, result: VisualizationResult) -> Image.Image:
        """Add border and title to a single visualization result."""
        padding = 30
        title_height = 40
        
        # Create new image with padding and title space
        new_width = result.image.width + 2 * padding
        new_height = result.image.height + 2 * padding + title_height
        
        bordered = Image.new('RGB', (new_width, new_height), self.layout_config.background_color)
        
        # Paste original image
        bordered.paste(result.image, (padding, padding))
        
        # Add title
        self._add_text_label(
            bordered,
            result.method_name,
            new_width // 2,
            padding + result.image.height + 10
        )
        
        # Add border
        draw = ImageDraw.Draw(bordered)
        draw.rectangle(
            [padding - 2, padding - 2, padding + result.image.width + 2, padding + result.image.height + 2],
            outline=self.layout_config.border_color,
            width=self.layout_config.border_width
        )
        
        return bordered