"""
Context Composer - The core system for combining multiple visualizations.

This module provides the main interface for creating complex, multi-visualization
contexts that can be consumed by VLM backends for enhanced reasoning.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import logging
import time

from ..base import BaseVisualization, VisualizationConfig, VisualizationResult
from .layouts import LayoutManager, LayoutStrategy
# Removed PromptGenerator - using unified VLM prompting utilities

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    get_class_color_name_map
)

logger = logging.getLogger(__name__)


@dataclass
class CompositionConfig:
    """Configuration for context composition."""
    
    # Layout options - use sequential (horizontal) as default to match tsne_functions
    layout_strategy: LayoutStrategy = LayoutStrategy.SEQUENTIAL
    max_visualizations_per_row: int = 2
    subplot_spacing: float = 0.3
    
    # Image output
    composition_figsize: Tuple[int, int] = (16, 12)
    composition_dpi: int = 100
    max_composition_size: int = 4096
    
    # Prompt generation
    include_individual_descriptions: bool = True
    include_cross_references: bool = True
    reasoning_focus: str = "comparison"  # "comparison", "consensus", "divergence"
    
    # Performance options
    parallel_processing: bool = False
    cache_transformations: bool = True
    
    # VLM optimization
    optimize_for_vlm: bool = True
    vlm_model_type: str = "general"  # "general", "vision_specialist", "reasoning_focused"


class ContextComposer:
    """
    Compose multiple visualizations into a unified context for VLM reasoning.
    
    This class coordinates multiple visualization methods to create rich,
    multi-perspective views of data that enable more sophisticated reasoning
    by VLM backends.
    """
    
    def __init__(self, config: Optional[CompositionConfig] = None):
        """
        Initialize the context composer.
        
        Args:
            config: Configuration for composition behavior
        """
        self.config = config or CompositionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-components
        self.layout_manager = LayoutManager(self.config)
        
        # State
        self.visualizations: List[BaseVisualization] = []
        self.visualization_configs: List[Dict[str, Any]] = []
        self.cached_results: Dict[str, Any] = {}
        
        # Data state
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._fitted = False
    
    def add_visualization(
        self,
        viz_type: str,
        config: Optional[Dict[str, Any]] = None,
        viz_config: Optional[VisualizationConfig] = None
    ) -> 'ContextComposer':
        """
        Add a visualization method to the composition.
        
        Args:
            viz_type: Type of visualization (e.g., 'tsne', 'umap', 'pca')
            config: Configuration parameters for the visualization method
            viz_config: VisualizationConfig object for the visualization
            
        Returns:
            Self for method chaining
        """
        # Import visualization classes dynamically
        viz_class = self._get_visualization_class(viz_type)
        
        # Create visualization config
        if viz_config is None:
            viz_config = VisualizationConfig()
            if config:
                viz_config.extra_params.update(config)
        
        # Handle special cases that require additional parameters
        if viz_type == 'decision_regions':
            # Decision regions requires a classifier instance
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            
            # Get classifier type from config
            classifier_type = (config or {}).get('decision_classifier', 'svm')
            
            if classifier_type == 'svm':
                classifier = SVC(kernel='rbf', random_state=viz_config.random_state)
            elif classifier_type == 'rf':
                classifier = RandomForestClassifier(n_estimators=100, random_state=viz_config.random_state)
            else:
                # Default to SVM
                classifier = SVC(kernel='rbf', random_state=viz_config.random_state)
            
            # Create visualization instance with classifier
            visualization = viz_class(classifier=classifier, config=viz_config)
        else:
            # Create standard visualization instance
            visualization = viz_class(viz_config)
        
        self.visualizations.append(visualization)
        self.visualization_configs.append({
            'type': viz_type,
            'config': config or {},
            'viz_config': viz_config
        })
        
        self.logger.info(f"Added {viz_type} visualization to composition")
        return self
    
    def _get_visualization_class(self, viz_type: str):
        """Get the visualization class for a given type string."""
        from ..embeddings.tsne import TSNEVisualization
        from ..embeddings.umap import UMAPVisualization
        from ..embeddings.pca import PCAVisualization
        from ..embeddings.manifold import (
            LocallyLinearEmbeddingVisualization,
            SpectralEmbeddingVisualization,
            IsomapVisualization,
            MDSVisualization
        )
        from ..decision.regions import DecisionRegionsVisualization
        from ..patterns.frequent import FrequentPatternsVisualization
        
        viz_map = {
            'tsne': TSNEVisualization,
            'umap': UMAPVisualization,
            'pca': PCAVisualization,
            'lle': LocallyLinearEmbeddingVisualization,
            'spectral': SpectralEmbeddingVisualization,
            'isomap': IsomapVisualization,
            'mds': MDSVisualization,
            'decision_regions': DecisionRegionsVisualization,
            'frequent_patterns': FrequentPatternsVisualization,
        }
        
        if viz_type not in viz_map:
            raise ValueError(f"Unknown visualization type: {viz_type}. "
                           f"Available types: {list(viz_map.keys())}")
        
        return viz_map[viz_type]
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> 'ContextComposer':
        """
        Fit all visualization methods on the training data.
        
        Args:
            X_train: Training features
            y_train: Training targets (optional)
            X_test: Test features (optional)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting {len(self.visualizations)} visualizations...")
        
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        
        start_time = time.time()
        
        # Fit all visualizations
        for i, viz in enumerate(self.visualizations):
            viz_start = time.time()
            
            try:
                if X_test is not None and not viz.supports_new_data:
                    # For methods that don't support new data (like Spectral, t-SNE),
                    # fit on combined data and split afterwards (like original t-SNE implementation)
                    combined_data = np.vstack([X_train, X_test])
                    if y_train is not None:
                        # Use -1 for test labels to distinguish from training labels
                        combined_labels = np.concatenate([y_train, np.full(len(X_test), -1)])
                    else:
                        combined_labels = None
                    
                    combined_transformed = viz.fit_transform(combined_data, combined_labels)
                    
                    # Split back into train and test
                    n_train = len(X_train)
                    train_transformed = combined_transformed[:n_train]
                    test_transformed = combined_transformed[n_train:]
                    
                    # Cache both results
                    cache_key = f"viz_{i}_train"
                    self.cached_results[cache_key] = train_transformed
                    cache_key_test = f"viz_{i}_test"
                    self.cached_results[cache_key_test] = test_transformed
                else:
                    # Fit and transform training data
                    train_transformed = viz.fit_transform(X_train, y_train)
                    
                    # Cache the result
                    cache_key = f"viz_{i}_train"
                    self.cached_results[cache_key] = train_transformed
                    
                    # Transform test data if available and supported
                    if X_test is not None and viz.supports_new_data:
                        test_transformed = viz.transform(X_test)
                        cache_key_test = f"viz_{i}_test"
                        self.cached_results[cache_key_test] = test_transformed
                
                viz_time = time.time() - viz_start
                self.logger.info(f"Fitted {viz.method_name} in {viz_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to fit {viz.method_name}: {e}")
                # Continue with other visualizations
                continue
        
        total_time = time.time() - start_time
        self.logger.info(f"Fitted all visualizations in {total_time:.2f}s")
        
        self._fitted = True
        return self
    
    def compose_layout(
        self,
        highlight_indices: Optional[List[int]] = None,
        highlight_test_indices: Optional[List[int]] = None,
        layout_strategy: Optional[LayoutStrategy] = None,
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False
    ) -> Image.Image:
        """
        Create a composed layout of all visualizations.
        
        Args:
            highlight_indices: Indices of training points to highlight across all visualizations
            highlight_test_indices: Indices of test points to highlight with red X markers
            layout_strategy: Override the default layout strategy
            class_names: Optional semantic class names for consistent styling
            use_semantic_names: Whether to use semantic names in legends
            
        Returns:
            Composed image
        """
        if not self._fitted:
            raise ValueError("Must call fit() before compose_layout()")
        
        if not self.visualizations:
            raise ValueError("No visualizations added to compose")
        
        strategy = layout_strategy or self.config.layout_strategy
        
        # Generate individual visualization results with explicit figure cleanup
        results = []
        initial_figures = set(plt.get_fignums())
        
        for i, viz in enumerate(self.visualizations):
            train_data = self.cached_results.get(f"viz_{i}_train")
            test_data = self.cached_results.get(f"viz_{i}_test")
            
            if train_data is None:
                self.logger.warning(f"No cached data for visualization {i}, skipping")
                continue
            
            try:
                # Track figures before visualization
                pre_viz_figures = set(plt.get_fignums())
                
                result = viz.generate_plot(
                    transformed_data=train_data,
                    y=self._y_train,
                    highlight_indices=highlight_indices,
                    test_data=test_data,
                    highlight_test_indices=highlight_test_indices,
                    class_names=class_names,
                    use_semantic_names=use_semantic_names
                )
                results.append(result)
                
                # Ensure any new figures created are properly closed
                post_viz_figures = set(plt.get_fignums())
                new_figures = post_viz_figures - pre_viz_figures
                for fignum in new_figures:
                    plt.close(fignum)
                
            except Exception as e:
                self.logger.error(f"Failed to generate plot for {viz.method_name}: {e}")
                # Clean up any figures that might have been created during the failed attempt
                current_figures = set(plt.get_fignums())
                error_figures = current_figures - initial_figures
                for fignum in error_figures:
                    plt.close(fignum)
                continue
        
        if not results:
            raise RuntimeError("No visualizations could be generated")
        
        # Use layout manager to compose the results
        composed_image = self.layout_manager.compose_layout(results, strategy)
        
        # Final cleanup: close any remaining figures that might have been missed
        final_figures = set(plt.get_fignums())
        remaining_figures = final_figures - initial_figures
        if remaining_figures:
            self.logger.debug(f"Closing {len(remaining_figures)} remaining figures to prevent memory leaks")
            for fignum in remaining_figures:
                plt.close(fignum)
        
        return composed_image
    
    def get_aggregated_metadata(self) -> Dict[str, Any]:
        """
        Get aggregated metadata across all visualizations.
        
        Returns:
            Dictionary with aggregated metadata including visible_classes
        """
        if not self._fitted or not self.visualizations:
            return {'visible_classes': [], 'all_classes': []}
        
        all_visible_classes = set()
        plot_types = set()
        
        for viz in self.visualizations:
            if hasattr(viz, 'metadata'):
                visible = viz.metadata.get('visible_classes', [])
                all_visible_classes.update(visible)
                plot_types.add(viz.metadata.get('plot_type', 'classification'))
        
        # Determine overall plot type (if mixed, default to classification)
        overall_plot_type = 'classification'
        if len(plot_types) == 1:
            overall_plot_type = list(plot_types)[0]
        elif 'regression' in plot_types and len(plot_types) > 1:
            self.logger.warning("Mixed plot types detected in multi-viz composition")
        
        return {
            'visible_classes': sorted(list(all_visible_classes)),
            'plot_type': overall_plot_type,
            'n_visualizations': len(self.visualizations),
            'visualization_methods': [viz.method_name for viz in self.visualizations]
        }
    
    def generate_reasoning_prompt(
        self,
        highlight_indices: Optional[List[int]] = None,
        custom_context: Optional[str] = None,
        task_description: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive reasoning prompt for VLM consumption.
        
        Args:
            highlight_indices: Indices of points to highlight
            custom_context: Additional context about the data or task
            task_description: Description of the specific task
            
        Returns:
            Generated prompt string
        """
        if not self._fitted:
            raise ValueError("Must call fit() before generate_reasoning_prompt()")
        
        # Collect information from all visualizations
        viz_info = []
        for i, viz in enumerate(self.visualizations):
            train_data = self.cached_results.get(f"viz_{i}_train")
            if train_data is not None:
                description = viz.get_description(self._X_train, self._y_train)
                viz_info.append({
                    'method': viz.method_name,
                    'description': description,
                    'config': viz.config,
                    'transformed_shape': train_data.shape
                })
        
        # Generate prompt using unified VLM prompting utilities
        from marvis.utils.vlm_prompting import create_comprehensive_multi_viz_prompt
        from marvis.utils.class_name_utils import normalize_class_names_to_class_num
        
        # Prepare multi_viz_info for the unified utilities
        multi_viz_info = []
        for viz in viz_info:
            multi_viz_info.append({
                'method': viz.get('method', 'Unknown'),
                'description': viz.get('description', f"Visualization of data using {viz.get('method', 'unknown')} method")
            })
        
        # Generate class names
        if self._y_train is not None:
            n_classes = len(np.unique(self._y_train))
            class_names = normalize_class_names_to_class_num(n_classes)
        else:
            class_names = ["Class_0", "Class_1", "Class_2"]  # Default
        
        # Build dataset description
        dataset_description = ""
        if self._X_train is not None:
            dataset_description += f"Dataset with {self._X_train.shape[0]} samples and {self._X_train.shape[1]} features. "
        
        if custom_context:
            dataset_description += f"{custom_context} "
            
        if task_description:
            dataset_description += f"Task: {task_description}"
        
        # Generate comprehensive multi-viz prompt
        prompt = create_comprehensive_multi_viz_prompt(
            class_names=class_names,
            multi_viz_info=multi_viz_info,
            modality="tabular",  # Default to tabular
            dataset_description=dataset_description,
            reasoning_focus=self.config.reasoning_focus,
            data_shape=self._X_train.shape if self._X_train is not None else None,
            use_semantic_names=False  # Use Class_X format for consistency
        )
        
        return prompt
    
    def reason_over_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        reasoning_chain: Optional[List[str]] = None,
        highlight_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-visualization reasoning over data.
        
        Args:
            X: Input features
            y: Optional target values
            reasoning_chain: Ordered list of reasoning steps/questions
            highlight_indices: Points to highlight for analysis
            
        Returns:
            Dictionary containing composed image, prompt, and analysis
        """
        # Fit if not already fitted
        if not self._fitted:
            self.fit(X, y)
        
        # Generate composed visualization
        composed_image = self.compose_layout(highlight_indices=highlight_indices)
        
        # Generate reasoning prompt
        reasoning_prompt = self.generate_reasoning_prompt(
            highlight_indices=highlight_indices,
            task_description="Analyze the data patterns across multiple visualization perspectives"
        )
        
        # Add reasoning chain if provided
        if reasoning_chain:
            reasoning_prompt += "\n\nSpecific reasoning steps to consider:\n"
            for i, step in enumerate(reasoning_chain, 1):
                reasoning_prompt += f"{i}. {step}\n"
        
        # Collect metadata from all visualizations
        metadata = {
            'n_visualizations': len(self.visualizations),
            'visualization_methods': [viz.method_name for viz in self.visualizations],
            'data_shape': X.shape,
            'n_classes': len(np.unique(y)) if y is not None else None,
            'highlighted_points': len(highlight_indices) if highlight_indices else 0,
            'composition_config': self.config
        }
        
        return {
            'composed_image': composed_image,
            'reasoning_prompt': reasoning_prompt,
            'metadata': metadata,
            'individual_results': [
                self.cached_results.get(f"viz_{i}_train") 
                for i in range(len(self.visualizations))
            ]
        }
    
    def get_visualization_comparison(self) -> Dict[str, Any]:
        """
        Get a comparison of all visualization methods.
        
        Returns:
            Dictionary with comparison metrics and insights
        """
        if not self._fitted:
            raise ValueError("Must call fit() before getting comparison")
        
        comparison = {
            'methods': [],
            'capabilities': {},
            'performance': {},
            'recommendations': []
        }
        
        for i, viz in enumerate(self.visualizations):
            method_info = {
                'name': viz.method_name,
                'supports_3d': viz.supports_3d,
                'supports_regression': viz.supports_regression,
                'supports_new_data': viz.supports_new_data,
                'config': viz.config.__dict__
            }
            
            # Add performance metrics if available
            if hasattr(viz, '_last_fit_time'):
                method_info['fit_time'] = viz._last_fit_time
            
            comparison['methods'].append(method_info)
        
        # Generate recommendations based on data characteristics
        n_samples, n_features = self._X_train.shape
        
        if n_samples < 100:
            comparison['recommendations'].append("Small dataset: Consider PCA or MDS for stable results")
        if n_features > 1000:
            comparison['recommendations'].append("High-dimensional data: UMAP or t-SNE may be most effective")
        if self._y_train is not None and len(np.unique(self._y_train)) > 10:
            comparison['recommendations'].append("Many classes: Consider spectral embedding or UMAP")
        
        return comparison
    
    def clear_cache(self):
        """Clear cached transformation results."""
        self.cached_results.clear()
        self.logger.info("Cleared visualization cache")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            'composition_config': self.config.__dict__,
            'visualizations': [
                {
                    'type': config['type'],
                    'config': config['config']
                }
                for config in self.visualization_configs
            ]
        }