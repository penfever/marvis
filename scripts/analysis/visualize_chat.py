#!/usr/bin/env python3
"""
MARVIS Chat Visualization Script

Generates publication-quality chat bubble visualizations from MARVIS
classifier conversations. Outputs PDF and PNG.

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
import matplotlib.patheffects as path_effects

# Add MARVIS to Python path
script_dir = Path(__file__).parent
marvis_root = script_dir.parent.parent
sys.path.insert(0, str(marvis_root))

from marvis.models.marvis_tsne import MarvisTsneClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Visual config ──────────────────────────────────────────────────
FONT_SIZE = 13          # Large enough for publication
FONT_FAMILY = 'sans-serif'
LINE_SPACING = 1.45
CHARS_PER_LINE = 72     # Wider lines to use space efficiently

USER_COLOR = '#4A90E2'
ASSISTANT_COLOR = '#50B848'
BG_COLOR = '#F5F6F8'

BUBBLE_PAD = 0.015      # Internal padding of FancyBboxPatch
BUBBLE_HMARGIN = 0.04   # Horizontal margin from edge
GAP_LABEL = 0.012       # Gap between label and bubble top
GAP_WITHIN = 0.025      # Gap between user bubble bottom and assistant label
GAP_BETWEEN = 0.045     # Gap between exchanges
TITLE_AREA = 0.06       # Fraction of figure height for title


def wrap_message(text, max_chars=CHARS_PER_LINE, max_lines=None):
    """Wrap text into lines. If max_lines is set, truncate with ellipsis."""
    if not text:
        return ['']

    # Strip markdown bold markers for cleaner rendering
    text = text.replace('**', '')

    all_lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            all_lines.append('')
            continue
        wrapper = textwrap.TextWrapper(
            width=max_chars, break_long_words=True,
            break_on_hyphens=True, drop_whitespace=True,
        )
        all_lines.extend(wrapper.wrap(paragraph))

    if max_lines and len(all_lines) > max_lines:
        all_lines = all_lines[:max_lines]
        last = all_lines[-1]
        if len(last) > max_chars - 3:
            last = last[:max_chars - 3]
        all_lines[-1] = last + '...'

    return all_lines or ['']


def _line_height():
    """Estimated height of one text line in axis-fraction coords."""
    return FONT_SIZE * 0.0022 * LINE_SPACING


def _text_block_height(n_lines):
    """Height of a text block with n_lines."""
    return n_lines * _line_height() + 2 * BUBBLE_PAD + 0.01


def create_chat_visualization(chat_history, title="MARVIS Chat", output_dir=None):
    """Render chat history as a publication-quality figure."""
    if not chat_history:
        print("No chat history to visualize")
        return None

    n = len(chat_history)

    # ── Pre-compute wrapped text and heights ──────────────────────
    exchanges = []
    for ex in chat_history:
        u_lines = wrap_message(ex['user'], max_lines=2)
        a_lines = wrap_message(ex['assistant'], max_lines=12)
        u_h = _text_block_height(len(u_lines))
        a_h = _text_block_height(len(a_lines))
        exchanges.append({
            'u_lines': u_lines, 'a_lines': a_lines,
            'u_h': u_h, 'a_h': a_h,
        })

    # Total content height (in axis-coordinate units)
    label_h = FONT_SIZE * 0.002  # height of "User" / "MARVIS" labels
    content_h = TITLE_AREA
    for i, ex in enumerate(exchanges):
        content_h += label_h + GAP_LABEL + ex['u_h'] + GAP_WITHIN
        content_h += label_h + GAP_LABEL + ex['a_h']
        if i < n - 1:
            content_h += GAP_BETWEEN
    content_h += 0.04  # bottom padding

    # Size figure so content fills nicely: 1 axis-unit ≈ some inches
    fig_width = 11
    inches_per_unit = 11  # how many inches per 1.0 of axis height
    fig_height = max(content_h * inches_per_unit, 6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    # Extend y-axis to fit all content (top = 1, bottom = 1 - content_h)
    y_bottom = 1.0 - content_h
    ax.set_ylim(y_bottom, 1)
    ax.axis('off')
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── Title ─────────────────────────────────────────────────────
    ax.text(0.5, 1 - TITLE_AREA * 0.35, title,
            ha='center', va='center',
            fontsize=FONT_SIZE + 4, fontfamily=FONT_FAMILY,
            weight='bold', color='#2C3E50', zorder=10)

    # ── Draw bubbles ──────────────────────────────────────────────
    y = 1 - TITLE_AREA  # current y cursor (top of next element)

    for i, ex in enumerate(exchanges):
        # ── User bubble (right-aligned) ───────────────────────────
        u_w = 0.52
        u_x = 1.0 - BUBBLE_HMARGIN - u_w

        # Label
        ax.text(u_x + u_w, y, 'User',
                ha='right', va='bottom',
                fontsize=FONT_SIZE - 1, fontfamily=FONT_FAMILY,
                color='#666', weight='bold', zorder=5)
        y -= (label_h + GAP_LABEL)

        # Bubble
        u_h = ex['u_h']
        _draw_bubble(ax, u_x, y - u_h, u_w, u_h,
                     USER_COLOR, 'white', ex['u_lines'], align='right')
        y -= (u_h + GAP_WITHIN)

        # ── Assistant bubble (left-aligned) ───────────────────────
        a_w = 0.70
        a_x = BUBBLE_HMARGIN

        # Label
        ax.text(a_x, y, 'MARVIS',
                ha='left', va='bottom',
                fontsize=FONT_SIZE - 1, fontfamily=FONT_FAMILY,
                color='#666', weight='bold', zorder=5)
        y -= (label_h + GAP_LABEL)

        # Bubble
        a_h = ex['a_h']
        _draw_bubble(ax, a_x, y - a_h, a_w, a_h,
                     ASSISTANT_COLOR, 'black', ex['a_lines'], align='left')
        y -= a_h

        if i < n - 1:
            y -= GAP_BETWEEN

    # ── Save ──────────────────────────────────────────────────────
    plt.tight_layout(pad=0.3)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = "marvis_chat_visualization"
        png = out / f"{base}.png"
        pdf = out / f"{base}.pdf"
        plt.savefig(png, dpi=300, bbox_inches='tight',
                    facecolor=BG_COLOR, edgecolor='none')
        plt.savefig(pdf, bbox_inches='tight',
                    facecolor=BG_COLOR, edgecolor='none')
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")
        return png, pdf

    return fig


def _draw_bubble(ax, x, y, w, h, bg_color, text_color, lines, align='left'):
    """Draw a rounded bubble with text."""
    # Shadow
    ax.add_patch(FancyBboxPatch(
        (x + 0.002, y - 0.002), w, h,
        boxstyle=f"round,pad={BUBBLE_PAD}",
        facecolor='#333', alpha=0.12, zorder=1,
    ))
    # Bubble
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={BUBBLE_PAD}",
        facecolor=bg_color, edgecolor='white',
        linewidth=1.2, alpha=0.95, zorder=2,
    ))
    # Text
    pad_x = 0.025
    if align == 'right':
        tx = x + w - pad_x
        ha = 'right'
    else:
        tx = x + pad_x
        ha = 'left'
    ty = y + h / 2

    txt = ax.text(
        tx, ty, '\n'.join(lines),
        ha=ha, va='center',
        fontsize=FONT_SIZE, fontfamily=FONT_FAMILY,
        color=text_color, weight='medium',
        linespacing=LINE_SPACING, zorder=3,
    )
    # White text gets a subtle stroke for readability
    if text_color == 'white':
        txt.set_path_effects([
            path_effects.withStroke(linewidth=2, foreground=(0, 0, 0, 0.3))
        ])


def main():
    print("MARVIS Chat Visualization Generator")
    print("=" * 50)

    # Create sample data
    print("Creating sample tabular dataset...")
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=8,
        n_redundant=1, n_clusters_per_class=1, class_sep=1.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    class_names = ["Class A", "Class B", "Class C"]

    # Initialize and train MARVIS
    print("Initializing MARVIS classifier...")
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

    print("Training classifier...")
    classifier.fit(X_train, y_train, X_test, class_names=class_names, task_type='classification')

    print("Making predictions...")
    classifier.evaluate(X_test[:5], y_test[:5], return_detailed=True)

    # Chat session — 3 exchanges, last one asks about confidence
    print("Conducting chat session...")
    classifier.chat("How well did the model perform on the test data?")
    classifier.chat("What patterns did you observe in the visualization?")
    classifier.chat("How confident are you in the last prediction, and why?")

    chat_history = classifier.get_chat_history()

    if chat_history:
        print(f"Retrieved {len(chat_history)} chat exchanges")
        output_dir = str(Path(__file__).parent.parent.parent.parent / "marvis_paper" / "figures")
        create_chat_visualization(
            chat_history,
            title="MARVIS: Interactive Classification Chat",
            output_dir=output_dir
        )
        print("Done!")
    else:
        print("No chat history available")


if __name__ == "__main__":
    main()
