"""
Context composition system for multi-visualization reasoning.
"""

from .composer import ContextComposer, CompositionConfig
from .layouts import LayoutManager, LayoutStrategy
# PromptGenerator removed - functionality integrated into unified VLM prompting utilities

__all__ = [
    'ContextComposer',
    'CompositionConfig',
    'LayoutManager', 
    'LayoutStrategy'
]