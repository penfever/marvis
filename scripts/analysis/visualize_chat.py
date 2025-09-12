#!/usr/bin/env python3
"""
MARVIS Chat Visualization Script

This script converts chat interactions with MARVIS classifiers into beautiful,
compact chat bubble visualizations suitable for publication. It generates 
colorful chat bubbles showing the conversation flow between users and the
MARVIS assistant, with export to PDF and PNG formats.

Author: MARVIS Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import warnings
from pathlib import Path
import textwrap
from datetime import datetime
import matplotlib.patheffects as path_effects

# Add MARVIS to Python path
script_dir = Path(__file__).parent
marvis_root = script_dir.parent.parent
sys.path.insert(0, str(marvis_root))

# Import MARVIS modules
from marvis.models.marvis_tsne import MarvisTsneClassifier
from marvis.utils.audio_utils import create_synthetic_audio
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
import tempfile

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration for chat bubble visualization
CHAT_CONFIG = {
    # Colors
    'user_color': '#4A90E2',      # Blue for user messages
    'assistant_color': '#7ED321',  # Green for MARVIS responses
    'user_text_color': 'white',
    'assistant_text_color': 'black',  # Changed to black for better contrast on green
    'background_color': '#F8F9FA',
    'border_color': '#E1E5E9',
    'title_color': '#2C3E50',
    
    # Layout
    'figure_width': 12,
    'figure_height': 10,  # Increased height for better spacing
    'bubble_margin': 0.025,  # Increased margin
    'vertical_spacing': 0.045,  # Increased spacing between bubbles
    'horizontal_margin': 0.15,
    'max_bubble_width': 0.65,  # Decreased for better wrapping
    'title_spacing': 0.1,  # Spacing for title area
    
    # Typography
    'font_size': 11,
    'font_family': 'Arial',
    'line_spacing': 1.3,  # Increased line spacing
    'chars_per_line': 55,  # Decreased for better wrapping
    
    # Styling
    'bubble_alpha': 0.95,  # Increased for better contrast
    'shadow_offset': 0.0025,
    'border_radius': 0.025,
    'bubble_padding': 0.02,  # Increased padding
    
    # Title Area
    'header_height': 0.15,  # Reserved space for title/subtitle
}

def wrap_text(text, max_chars=60, max_lines=None, is_user=False):
    """Wrap text to fit within chat bubbles with strict limits.
    
    Args:
        text: The text to wrap
        max_chars: Maximum characters per line (default: 60)
        max_lines: Maximum number of lines (None = unlimited)
        is_user: Whether this is a user message (affects formatting)
    
    Returns:
        List of wrapped lines, truncated if needed
    """
    if not text:
        return []
    
    # For user messages, enforce 1 line with exactly 88 characters
    if is_user:
        # Enforce 1 line only
        max_chars = 88
        if len(text) > max_chars:
            return [text[:max_chars-3] + '...']  # Truncate and add ellipsis
        return [text]
    
    # For assistant messages, enforce 3 lines with 88 characters each
    if max_lines is None:
        max_lines = 3  # Default for assistant messages
    max_chars = 88
    
    # Handle newlines properly by splitting and rewrapping each line
    if '\n' in text:
        lines = []
        for line in text.split('\n'):
            if line.strip():
                lines.extend(wrap_text(line, max_chars, max_lines - len(lines), is_user=False))
            else:
                lines.append('')  # Preserve empty lines
            # Check if we've hit the max lines
            if len(lines) >= max_lines:
                break
        return lines[:max_lines]  # Ensure we don't exceed max lines
    
    # Use textwrap for intelligent wrapping
    wrapper = textwrap.TextWrapper(
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=True,
        expand_tabs=False,
        replace_whitespace=False,
        drop_whitespace=True
    )
    wrapped = wrapper.wrap(text)
    
    # Truncate to max_lines if needed
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        # Add ellipsis to last line if truncated
        if wrapped and len(wrapped[-1]) > 3:
            wrapped[-1] = wrapped[-1][:max_chars-3] + '...'
    
    return wrapped if wrapped else ['']

def estimate_text_dimensions(text_lines, font_size):
    """Estimate the dimensions needed for text rendering with improved accuracy."""
    if not text_lines:
        return 0.1, 0.05
    
    # Rough estimation based on font size and character count
    max_line_length = max(len(line) for line in text_lines)
    
    # Improved character width estimation accounting for avg character width
    # Different character widths affect total width, so scale accordingly
    char_width = font_size * 0.00065  # Slightly increased for better fit
    
    # Improved line height estimation
    line_height = font_size * 0.0018  # Increased for better spacing
    
    # Add extra padding for long text
    width = max_line_length * char_width
    height = len(text_lines) * line_height * CHAT_CONFIG['line_spacing']
    
    # Add minimum size constraints
    width = max(width, 0.15)  # Ensure minimum width for short messages
    height = max(height, 0.03 * CHAT_CONFIG['line_spacing'])  # Minimum height
    
    return width, height

def create_chat_bubble(ax, message, is_user, y_position, bubble_index):
    """Create a single chat bubble with uniform dimensions and standardized text."""
    config = CHAT_CONFIG
    
    # Debug info
    print(f"Creating bubble for message: '{message[:30]}...' at y-position: {y_position}")
    
    # Ensure message is a string
    if message is None:
        message = "(No message)"
    elif not isinstance(message, str):
        message = str(message)
    
    # Wrap text with standardized limits
    # - User: 1 line, 88 chars
    # - Assistant: 3 lines, 88 chars per line
    max_lines = 1 if is_user else 3
    wrapped_lines = wrap_text(message, max_chars=88, max_lines=max_lines, is_user=is_user)
    
    # Define standard bubble dimensions (uniform size) with reduced vertical space
    if is_user:
        # User bubbles are fixed size for 1 line
        bubble_width = 0.5  # Fixed width for user bubbles
        bubble_height = 0.05  # Reduced height for user bubbles (was 0.08)
    else:
        # Assistant bubbles are fixed size for 3 lines
        bubble_width = 0.6  # Fixed width for assistant bubbles
        bubble_height = 0.12  # Reduced height for assistant bubbles (was 0.18)
    
    # Convert 0.25 inch to figure coordinates (approximate)
    inch_margin = 0.03  # 0.25 inch margin in figure coordinates
    
    # Position bubble based on sender with standard 0.25 inch margins
    if is_user:
        # User bubbles on the right with 0.25 inch margin
        bubble_x = 1.0 - inch_margin - bubble_width  # 0.25 inch from right edge
        bubble_color = config['user_color']
        text_color = config['user_text_color']
        label_prefix = "👤 User"
        align_h = 'right'  # Align text to right for user messages
        text_x = bubble_x + bubble_width * 0.9  # Adjust text position
    else:
        # Assistant bubbles on the left with 0.25 inch margin
        bubble_x = inch_margin  # 0.25 inch from left edge
        bubble_color = config['assistant_color']
        text_color = config['assistant_text_color']
        label_prefix = "🤖 MARVIS"
        align_h = 'left'  # Align text to left for assistant messages
        text_x = bubble_x + bubble_width * 0.1  # Adjust text position
    
    bubble_y = y_position - bubble_height
    
    # Create bubble with improved shadow effect
    shadow_bubble = FancyBboxPatch(
        (bubble_x + config['shadow_offset'], bubble_y - config['shadow_offset']),
        bubble_width, bubble_height,
        boxstyle=f"round,pad={config['bubble_padding']}",
        facecolor='#333333',  # Darker shadow
        alpha=0.2,
        zorder=1
    )
    ax.add_patch(shadow_bubble)
    
    # Create main bubble with improved styling
    bubble = FancyBboxPatch(
        (bubble_x, bubble_y),
        bubble_width, bubble_height,
        boxstyle=f"round,pad={config['bubble_padding']}",
        facecolor=bubble_color,
        edgecolor='white',
        linewidth=1.5,  # Thicker border
        alpha=config['bubble_alpha'],
        zorder=2
    )
    ax.add_patch(bubble)
    
    # Calculate text position with improved alignment
    text_y = bubble_y + bubble_height / 2
    
    # Join wrapped lines with newlines
    display_text = '\n'.join(wrapped_lines)
    
    # Add text to bubble with improved styling and positioning
    text_obj = ax.text(
        text_x, text_y, display_text,
        ha=align_h, va='center',  # Align to side instead of center for better readability
        fontsize=config['font_size'],
        fontfamily=config['font_family'],
        color=text_color,
        weight='medium',  # Slightly bolder for better readability
        linespacing=config['line_spacing'],
        zorder=3,
        wrap=True  # Enable text wrapping
    )
    
    # Add improved text shadow for better readability
    if is_user or config['assistant_text_color'] == 'white':  # Only add shadow for white text
        text_obj.set_path_effects([
            path_effects.withStroke(linewidth=2.5, foreground=(0, 0, 0, 0.35))
        ])
    
    # Add small label above bubble with improved positioning
    label_y = bubble_y + bubble_height + 0.02
    ax.text(
        bubble_x + (bubble_width / 2), label_y, label_prefix,
        ha='center', va='bottom',
        fontsize=config['font_size'] - 1,  # Slightly larger
        fontfamily=config['font_family'],
        color='#555555',  # Darker gray for better contrast
        weight='bold',
        alpha=0.8,  # Increased opacity
        zorder=3
    )
    
    # Return next position with improved spacing
    return bubble_y - config['vertical_spacing'] - (0.01 * len(wrapped_lines))  # Add extra space for longer messages

def create_chat_visualization(chat_history, title="MARVIS Chat Conversation", output_dir=None):
    """Create a complete chat visualization from chat history with improved layout."""
    config = CHAT_CONFIG
    
    if not chat_history:
        print("⚠️  No chat history available for visualization")
        return None
    
    # Debug info
    print(f"Chat visualization starting with {len(chat_history)} exchanges")
    
    # Adjust figure height based on number of exchanges
    # More exchanges need more height to avoid overcrowding
    dynamic_height = 7 + (len(chat_history) * 2)  # Base height plus 2 units per exchange
    fixed_width = config['figure_width']
    
    # Create figure with calculated dimensions
    fig, ax = plt.subplots(1, 1, figsize=(fixed_width, dynamic_height))
    
    # Set axis limits explicitly for consistent layout
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Set background with improved styling
    fig.patch.set_facecolor(config['background_color'])
    ax.set_facecolor(config['background_color'])
    
    # Add subtle grid for visual separation
    ax.grid(False)
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create title area with background for better separation
    title_bg = patches.Rectangle(
        (0, 0.9), 1, 0.1,
        facecolor='#EFF3F7',  # Light blue-gray background
        edgecolor=None,
        alpha=0.6,
        zorder=0
    )
    ax.add_patch(title_bg)
    
    # Add title with improved styling
    title_text = ax.text(
        0.5, 0.965, title,
        ha='center', va='top',
        fontsize=config['font_size'] + 4,
        fontfamily=config['font_family'],
        weight='bold',
        color=config['title_color'],
        zorder=10  # Ensure title is above background
    )
    
    # Add subtitle with metadata and improved spacing
    subtitle = f"💬 {len(chat_history)} exchanges • Generated by MARVIS"
    ax.text(
        0.5, 0.93, subtitle,  # Moved down from title
        ha='center', va='top',
        fontsize=config['font_size'] - 1,
        fontfamily=config['font_family'],
        color='#5D6D7E',  # Darker gray for better contrast
        style='italic',
        zorder=10  # Ensure subtitle is above background
    )
    
    # Calculate the available space for messages
    available_space = 0.85  # Start from below the title area
    footer_space = 0.05  # Reserve space for footer
    message_space = available_space - footer_space
    
    # Estimate total required space
    estimated_content_height = 0
    for exchange in chat_history:
        user_lines = wrap_text(exchange['user'], config['chars_per_line'])
        assistant_lines = wrap_text(exchange['assistant'], config['chars_per_line'])
        estimated_content_height += len(user_lines) * 0.03 + len(assistant_lines) * 0.03 + 0.05  # Add spacing between messages
    
    # Scale factor to ensure all content fits
    scale_factor = min(1.0, message_space / max(estimated_content_height, 0.1))
    
    # Create bubbles with fixed positions using absolute coordinates
    # rather than relative positioning
    
    # Debug chat history
    print(f"Processing {len(chat_history)} chat exchanges")
    for i, exchange in enumerate(chat_history):
        print(f"Exchange {i+1}:")
        print(f"  User: {exchange['user'][:50]}...")
        print(f"  Assistant: {exchange['assistant'][:50]}...")
    
    # Calculate appropriate vertical positions for each message
    # Start from the top and allocate fixed space for each exchange
    message_positions = []
    
    # Create a more compact vertical spacing based on number of exchanges
    available_height = 0.8  # Available height for bubbles (0.9 - 0.1 for footer)
    exchange_spacing = available_height / (len(chat_history) * 1.2)  # Reduce spacing between exchanges
    
    # Allocate space for each exchange (user + assistant) with dynamic spacing
    for i in range(len(chat_history)):
        # Calculate absolute positions based on exchange index and dynamic spacing
        user_pos = 0.9 - (i * exchange_spacing)
        # Assistant position with reduced gap between user and assistant
        assistant_pos = user_pos - (exchange_spacing / 4)  # Reduced from 1/3 to 1/4
        message_positions.append((user_pos, assistant_pos))
    
    # Render all bubbles at their fixed positions
    for i, exchange in enumerate(chat_history):
        user_pos, assistant_pos = message_positions[i]
        
        print(f"Rendering exchange {i+1} at fixed positions: user={user_pos}, assistant={assistant_pos}")
        
        # User message bubble
        create_chat_bubble(
            ax, exchange['user'], is_user=True, 
            y_position=user_pos, bubble_index=i*2
        )
        
        # Assistant response bubble
        create_chat_bubble(
            ax, exchange['assistant'], is_user=False, 
            y_position=assistant_pos, bubble_index=i*2+1
        )
    
    # Add footer with timestamp and improved styling
    footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"
    ax.text(
        0.5, 0.02, footer_text,
        ha='center', va='bottom',
        fontsize=config['font_size'] - 2,
        fontfamily=config['font_family'],
        color='#888888',  # Darker gray for better contrast
        alpha=0.8,  # Increased opacity
        style='italic'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save outputs if directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"marvis_chat_visualization_{timestamp}"
        
        # Save PNG
        png_path = output_path / f"{base_filename}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor=config['background_color'], edgecolor='none')
        print(f"💾 Saved PNG: {png_path}")
        
        # Save PDF  
        pdf_path = output_path / f"{base_filename}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight',
                   facecolor=config['background_color'], edgecolor='none')
        print(f"💾 Saved PDF: {pdf_path}")
        
        return png_path, pdf_path
    
    return fig

def main():
    """Main function demonstrating chat visualization."""
    print("🎨 MARVIS Chat Visualization Generator")
    print("=" * 50)
    
    # Create sample data for demonstration
    print("📊 Creating sample tabular dataset...")
    X_tabular, y_tabular = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=8,
        n_redundant=1, n_clusters_per_class=1, class_sep=1.2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tabular, y_tabular, test_size=0.3, random_state=42, stratify=y_tabular
    )
    
    class_names = ["Class A", "Class B", "Class C"]
    
    # Initialize MARVIS classifier
    print("🤖 Initializing MARVIS classifier...")
    classifier = MarvisTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        tsne_perplexity=15,
        use_3d=False,
        use_knn_connections=True,
        nn_k=5,
        use_semantic_names=True,
        seed=42
    )
    
    # Train classifier
    print("🏋️ Training classifier...")
    classifier.fit(X_train, y_train, X_test, class_names=class_names, task_type='classification')
    
    # Make some predictions to populate internal state
    print("🔮 Making predictions...")
    results = classifier.evaluate(X_test[:5], y_test[:5], return_detailed=True)
    
    # Conduct sample chat session
    print("💬 Conducting chat session...")
    classifier.chat("How well did the model perform on the test data?")
    classifier.chat("What patterns did you observe in the visualization?")
    classifier.chat("How could we improve the classification results?")
    
    # Get chat history
    chat_history = classifier.get_chat_history()
    
    if chat_history:
        print(f"✅ Retrieved {len(chat_history)} chat exchanges")
        
        # Create visualization
        print("🎨 Creating chat visualization...")
        output_dir = "/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/marvis/figures"
        
        png_path, pdf_path = create_chat_visualization(
            chat_history, 
            title="MARVIS Tabular Classification Chat",
            output_dir=output_dir
        )
        
        print("✅ Chat visualization created successfully!")
        print(f"📁 Files saved to: {output_dir}")
        
        # Show the plot
        plt.show()
        
    else:
        print("⚠️  No chat history available")
    
    print("\n🎉 Chat visualization demo completed!")

if __name__ == "__main__":
    main()