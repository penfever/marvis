#!/usr/bin/env python3
"""
Generate MARVIS Pipeline Figure

This script creates a high-quality figure that visualizes the MARVIS (Multimodal Automatic 
Reasonable VIsual Summaries) method pipeline using an icon-based representation with gradient arrows.

The pipeline consists of four main steps:
1. Data Input (multimodal: tabular, audio, vision)
2. Embedding Generation (TabPFN, Whisper, DINOV2, etc.)
3. Dimensionality Reduction & Visualization (t-SNE, PCA, UMAP)
4. VLM Classification (Vision Language Model reasoning)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

def create_gradient_arrow(ax, start_pos, end_pos, width=0.02, colors=['#3498db', '#e74c3c']):
    """Create a gradient-colored arrow between two positions."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Calculate arrow direction and length
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    # Create gradient arrow using multiple small segments
    n_segments = 50
    for i in range(n_segments):
        t1 = i / n_segments
        t2 = (i + 1) / n_segments
        
        # Interpolate position
        x_start = x1 + t1 * dx
        y_start = y1 + t1 * dy
        x_end = x1 + t2 * dx
        y_end = y1 + t2 * dy
        
        # Interpolate color
        r1, g1, b1 = plt.matplotlib.colors.to_rgb(colors[0])
        r2, g2, b2 = plt.matplotlib.colors.to_rgb(colors[1])
        
        r = r1 + t1 * (r2 - r1)
        g = g1 + t1 * (g2 - g1)
        b = b1 + t1 * (b2 - b1)
        color = (r, g, b)
        
        # Draw segment
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                   arrowprops=dict(arrowstyle='-' if i < n_segments-1 else '->', 
                                 color=color, lw=width*100, alpha=0.8))

def create_icon_box(ax, center, size, title, subtitle, icon_type, color='#34495e'):
    """Create an icon box with title and subtitle."""
    x, y = center
    w, h = size
    
    # Main box
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, 
                        boxstyle="round,pad=0.02", 
                        facecolor=color, 
                        edgecolor='#2c3e50', 
                        linewidth=2,
                        alpha=0.9)
    ax.add_patch(box)
    
    # Icon area (center of box)
    icon_y = y + h * 0.15  # Slightly above center to leave room for text below
    
    # Draw icon based on type (using geometric shapes instead of emojis)
    if icon_type == 'data':
        # Database/data icon using geometric shapes
        # Main data symbol - white circle with icon
        ax.scatter(x, icon_y, s=1500, c='white', marker='o', alpha=0.9, edgecolors='#2c3e50', linewidth=2)
        ax.text(x, icon_y, 'DATA', fontsize=12, weight='bold', ha='center', va='center', color='#2c3e50')
        
        # Add small data type indicators below the main icon
        indicators = ['TAB', 'AUD', 'VIS']  # Tabular, Audio, Vision
        colors = ['#3498db', '#e67e22', '#9b59b6']
        for i, (ind, ind_color) in enumerate(zip(indicators, colors)):
            offset_x = (i - 1) * 0.025
            rect = patches.Rectangle((x + offset_x - 0.01, icon_y - 0.05), 0.02, 0.012, 
                                   facecolor=ind_color, edgecolor='white', linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x + offset_x, icon_y - 0.044, ind, fontsize=5, ha='center', va='center', 
                   color='white', weight='bold')
            
    elif icon_type == 'embedding':
        # Neural network icon using circles and connections
        ax.scatter(x, icon_y, s=1500, c='white', marker='o', alpha=0.9, edgecolors='#2c3e50', linewidth=2)
        ax.text(x, icon_y, 'EMB', fontsize=12, weight='bold', ha='center', va='center', color='#2c3e50')
        
        # Add small neural network visualization
        for i in range(3):
            for j in range(2):
                node_x = x - 0.015 + j * 0.03
                node_y = icon_y + 0.015 - i * 0.015
                ax.scatter(node_x, node_y, s=20, c='#2c3e50', marker='o', alpha=0.4, 
                          edgecolors='#2c3e50', linewidth=0.5)
        
        # Add small model indicators below icon, stacked vertically
        models = ['TabPFN', 'Whisper', 'DINOV2']
        for i, model in enumerate(models):
            ax.text(x, icon_y - 0.045 - i*0.012, model, fontsize=6, ha='center', va='center', 
                   color='white', weight='bold')
        
    elif icon_type == 'visualization':
        # Scatter plot icon using actual scatter points
        ax.scatter(x, icon_y, s=1500, c='white', marker='o', alpha=0.9, edgecolors='#2c3e50', linewidth=2)
        ax.text(x, icon_y, 'VIZ', fontsize=12, weight='bold', ha='center', va='center', color='#2c3e50')
        
        # Add small scatter plot visualization
        np.random.seed(42)  # For reproducible scatter
        scatter_x = x + np.random.normal(0, 0.012, 8)
        scatter_y = icon_y + np.random.normal(0, 0.012, 8)
        colors = ['#e74c3c', '#3498db', '#f39c12'] * 3  # Repeat colors to match 8 points
        ax.scatter(scatter_x, scatter_y, s=10, c=colors[:8], alpha=0.5)
        
        # Add method indicators below icon, stacked vertically
        methods = ['t-SNE', 'PCA', 'UMAP']
        for i, method in enumerate(methods):
            ax.text(x, icon_y - 0.045 - i*0.012, method, fontsize=6, ha='center', va='center', 
                   color='white', weight='bold')
        
    elif icon_type == 'vlm':
        # VLM icon using geometric eye shape
        ax.scatter(x, icon_y, s=1500, c='white', marker='o', alpha=0.9, edgecolors='#2c3e50', linewidth=2)
        ax.text(x, icon_y, 'VLM', fontsize=12, weight='bold', ha='center', va='center', color='#2c3e50')
        
        # Add eye-like visualization
        eye_outer = patches.Ellipse((x, icon_y + 0.02), 0.025, 0.012, facecolor='#2c3e50', alpha=0.4)
        eye_inner = patches.Circle((x, icon_y + 0.02), 0.004, facecolor='#2c3e50')
        ax.add_patch(eye_outer)
        ax.add_patch(eye_inner)
        
        # Add VLM model indicators below icon, stacked vertically
        models = ['GPT-4V', 'Qwen2.5-VL', 'Gemini']
        for i, model in enumerate(models):
            ax.text(x, icon_y - 0.045 - i*0.012, model, fontsize=6, ha='center', va='center', 
                   color='white', weight='bold')
    
    # Title (bold, larger) - positioned near bottom of box
    ax.text(x, y - h*0.25, title, fontsize=12, weight='bold', ha='center', va='center', color='white')
    
    # Subtitle (smaller, wrapped) - positioned at very bottom
    lines = subtitle.split('\n')
    for i, line in enumerate(lines):
        ax.text(x, y - h*0.35 - i*0.02, line, fontsize=8, ha='center', va='center', color='#ecf0f1')

def main():
    """Generate the MARVIS pipeline figure."""
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define positions for the four main components (more spacing)
    positions = [
        (0.125, 0.65),  # Data Input
        (0.375, 0.65),  # Embedding Generation  
        (0.625, 0.65),  # Visualization
        (0.875, 0.65)   # VLM Classification
    ]
    
    # Define colors for each step
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    
    # Component definitions
    components = [
        {
            'title': 'Data',
            'icon_type': 'data'
        },
        {
            'title': 'Embedding', 
            'icon_type': 'embedding'
        },
        {
            'title': 'Plotting',
            'icon_type': 'visualization'
        },
        {
            'title': 'Prediction',
            'icon_type': 'vlm'
        }
    ]
    
    # Draw the main components
    box_size = (0.20, 0.25)
    for i, (pos, comp, color) in enumerate(zip(positions, components, colors)):
        create_icon_box(ax, pos, box_size, comp['title'], "", comp['icon_type'], color)
    
    # Draw gradient arrows between components
    arrow_colors = [
        ['#3498db', '#e74c3c'],  # Blue to Red
        ['#e74c3c', '#f39c12'],  # Red to Orange  
        ['#f39c12', '#27ae60']   # Orange to Green
    ]
    
    for i in range(len(positions) - 1):
        # Adjust arrow positions to avoid overlap with boxes
        start_pos = (positions[i][0] + box_size[0]/2 - 0.01, positions[i][1])
        end_pos = (positions[i+1][0] - box_size[0]/2 + 0.01, positions[i+1][1])
        create_gradient_arrow(ax, start_pos, end_pos, colors=arrow_colors[i])
    
    # Add title
    ax.text(0.5, 0.9, 'MARVIS: Multimodal Automatic Reasonable VIsual Summaries', 
           fontsize=20, weight='bold', ha='center', va='center', color='#2c3e50')
    
    # Add subtitle 
    ax.text(0.5, 0.85, 'A unified multimodal classification pipeline using Vision Language Models', 
           fontsize=14, ha='center', va='center', color='#34495e', style='italic')
    
    # Add step numbers
    for i, pos in enumerate(positions):
        circle = plt.Circle((pos[0], pos[1] + 0.22), 0.025, color='#2c3e50', alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1] + 0.22, str(i+1), fontsize=12, weight='bold', 
               ha='center', va='center', color='white')
    
    # Add detailed pipeline steps at bottom
    pipeline_steps = [
        "1. Input multimodal data (tabular features, audio signals, or images)",
        "2. Generate domain-specific embeddings using specialized encoders", 
        "3. Apply dimensionality reduction and create informative visualizations",
        "4. Use Vision Language Models to reason about patterns and make predictions"
    ]
    
    ax.text(0.5, 0.40, 'Pipeline Overview:', fontsize=12, weight='bold', ha='center', va='top', color='#2c3e50')
    for i, step in enumerate(pipeline_steps):
        ax.text(0.1, 0.36 - i*0.03, step, fontsize=10, ha='left', va='top', color='#34495e')
    
    # Add key advantages box
    advantages_text = """Key Advantages:
• Unified framework for multiple modalities
• Leverages semantic reasoning capabilities
• Interpretable through visualizations  
• No task-specific fine-tuning required"""
    
    # Create advantages box
    adv_box = FancyBboxPatch((0.1, 0.05), 0.35, 0.15, 
                            boxstyle="round,pad=0.01", 
                            facecolor='#ecf0f1', 
                            edgecolor='#bdc3c7', 
                            linewidth=1,
                            alpha=0.9)
    ax.add_patch(adv_box)
    ax.text(0.275, 0.125, advantages_text, fontsize=9, ha='center', va='center', color='#2c3e50')
    
    # Add performance note
    perf_text = """Performance:
• Competitive with specialized models
• Excellent few-shot learning capability
• Robust across diverse datasets
• Scales to new domains efficiently"""
    
    perf_box = FancyBboxPatch((0.55, 0.05), 0.35, 0.15, 
                             boxstyle="round,pad=0.01", 
                             facecolor='#e8f5e8', 
                             edgecolor='#27ae60', 
                             linewidth=1,
                             alpha=0.9)
    ax.add_patch(perf_box)
    ax.text(0.725, 0.125, perf_text, fontsize=9, ha='center', va='center', color='#2c3e50')
    
    # Save the figure
    output_dir = Path(__file__).parent.parent / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "marvis_pipeline_overview.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"✅ MARVIS pipeline figure saved to:")
    print(f"   PNG: {output_path}")
    print(f"   PDF: {output_path.with_suffix('.pdf')}")
    
    # Optionally display the figure
    plt.show()

if __name__ == "__main__":
    main()