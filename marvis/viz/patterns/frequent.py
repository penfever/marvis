"""
Frequent patterns visualization using mlxtend.

This module provides visualization for frequent itemsets and association rules
for tabular data analysis.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List
import logging

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

from ..base import BaseVisualization, VisualizationResult

# Import shared styling utilities
from ..utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    create_distinct_color_map,
    get_class_color_name_map
)

logger = logging.getLogger(__name__)


class FrequentPatternsVisualization(BaseVisualization):
    """
    Frequent patterns visualization using mlxtend.
    
    This visualization analyzes frequent itemsets and association rules
    in discretized tabular data, providing insights into feature co-occurrence
    patterns.
    """
    
    @property
    def method_name(self) -> str:
        return "Frequent-Patterns"
    
    @property
    def supports_3d(self) -> bool:
        return False  # Pattern analysis is typically 2D
    
    @property
    def supports_regression(self) -> bool:
        return False  # Patterns are for categorical/discrete data
    
    @property
    def supports_new_data(self) -> bool:
        return True  # Can analyze patterns in new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create pattern analysis transformer."""
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend not available. Install with: pip install mlxtend"
            )
        
        # For frequent patterns, we need to discretize continuous data
        self.n_bins = kwargs.get('n_bins', 5)
        self.min_support = kwargs.get('min_support', 0.1)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
        self.discretization_method = kwargs.get('discretization_method', 'quantile')
        
        # Create a simple transformer that discretizes data
        from sklearn.preprocessing import KBinsDiscretizer
        
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=self.discretization_method,
            random_state=self.config.random_state
        )
        
        self.logger.info(
            f"Creating pattern analyzer with {self.n_bins} bins, "
            f"min_support={self.min_support}, discretization={self.discretization_method}"
        )
        
        return discretizer
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Discretize data and analyze patterns."""
        import time
        start_time = time.time()
        
        # Create transformer if not exists
        if self._transformer is None:
            merged_kwargs = {**self.config.extra_params, **kwargs}
            self._transformer = self._create_transformer(**merged_kwargs)
        
        # Discretize the data
        fit_start = time.time()
        discrete_data = self._transformer.fit_transform(X)
        
        # Convert to DataFrame for easier handling
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        self.discrete_df = pd.DataFrame(discrete_data, columns=feature_names)
        
        # Create binary encoding for frequent pattern mining
        self._create_binary_encoding()
        
        # Find frequent itemsets
        self.frequent_itemsets = apriori(
            self.binary_df, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        # Generate association rules if we have frequent itemsets
        if len(self.frequent_itemsets) > 0:
            self.rules = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
        else:
            self.rules = pd.DataFrame()
        
        # Mark as fitted - IMPORTANT!
        self._fitted = True
        
        # Store timing information
        self._last_fit_time = time.time() - fit_start
        self._last_transform_time = 0.0
        
        self.logger.info(
            f"Found {len(self.frequent_itemsets)} frequent itemsets "
            f"and {len(self.rules)} association rules in {self._last_fit_time:.2f}s"
        )
        
        return discrete_data
    
    def _create_binary_encoding(self):
        """Create binary encoding for frequent pattern mining."""
        # Convert discrete values to binary features
        binary_data = []
        self.item_mapping = {}
        
        for col in self.discrete_df.columns:
            unique_vals = self.discrete_df[col].unique()
            for val in unique_vals:
                item_name = f"{col}={val}"
                self.item_mapping[item_name] = (col, val)
                binary_data.append((self.discrete_df[col] == val).astype(int))
        
        self.binary_df = pd.DataFrame(binary_data).T
        self.binary_df.columns = list(self.item_mapping.keys())
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for frequent patterns."""
        n_itemsets = len(getattr(self, 'frequent_itemsets', []))
        n_rules = len(getattr(self, 'rules', []))
        
        description = (
            f"Frequent patterns analysis of {n_samples} samples with {n_features} features. "
            f"Data discretized into {self.n_bins} bins per feature. "
            f"Found {n_itemsets} frequent itemsets and {n_rules} association rules "
            f"with min_support={self.min_support} and min_confidence={self.min_confidence}."
        )
        
        return description
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[List[int]] = None,
        test_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> VisualizationResult:
        """Generate frequent patterns visualization."""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        # Create subplots for different visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.config.dpi)
        fig.suptitle('Frequent Patterns Analysis', fontsize=16)
        
        # Plot 1: Support vs Length of itemsets using unified styling
        if len(self.frequent_itemsets) > 0:
            itemset_lengths = self.frequent_itemsets['itemsets'].apply(len)
            
            # Use unified color mapping for scatter plot
            color_map = create_distinct_color_map(itemset_lengths.nunique())
            colors = [color_map[length] for length in itemset_lengths]
            
            apply_consistent_point_styling(
                axes[0, 0], itemset_lengths, self.frequent_itemsets['support'], 
                colors=colors, alpha=0.7, s=60
            )
            axes[0, 0].set_xlabel('Itemset Length')
            axes[0, 0].set_ylabel('Support')
            axes[0, 0].set_title('Frequent Itemsets: Support vs Length')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No frequent itemsets found', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Frequent Itemsets')
        
        # Plot 2: Association rules scatter plot using unified styling
        if len(self.rules) > 0:
            # Apply consistent point styling with lift as color coding
            scatter = apply_consistent_point_styling(
                axes[0, 1], self.rules['support'], self.rules['confidence'],
                colors=self.rules['lift'], cmap='viridis', alpha=0.7, s=60
            )
            axes[0, 1].set_xlabel('Support')
            axes[0, 1].set_ylabel('Confidence')
            axes[0, 1].set_title('Association Rules')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Lift')
        else:
            axes[0, 1].text(0.5, 0.5, 'No association rules found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Association Rules')
        
        # Plot 3: Top frequent itemsets
        if len(self.frequent_itemsets) > 0:
            top_itemsets = self.frequent_itemsets.nlargest(10, 'support')
            itemset_labels = [str(set(itemset))[:30] + '...' if len(str(set(itemset))) > 30 
                             else str(set(itemset)) for itemset in top_itemsets['itemsets']]
            
            y_pos = range(len(itemset_labels))
            axes[1, 0].barh(y_pos, top_itemsets['support'])
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(itemset_labels, fontsize=8)
            axes[1, 0].set_xlabel('Support')
            axes[1, 0].set_title('Top Frequent Itemsets')
        else:
            axes[1, 0].text(0.5, 0.5, 'No frequent itemsets to display', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Top Frequent Itemsets')
        
        # Plot 4: Feature discretization heatmap
        if hasattr(self, 'discrete_df'):
            # Show correlation matrix of discretized features
            corr_matrix = self.discrete_df.corr()
            
            # Use seaborn if available, otherwise matplotlib
            try:
                import seaborn as sns
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                           ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
            except ImportError:
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
                axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
                axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
                axes[1, 1].set_yticklabels(corr_matrix.columns)
                plt.colorbar(im, ax=axes[1, 1], label='Correlation')
            
            axes[1, 1].set_title('Discretized Features Correlation')
        else:
            axes[1, 1].text(0.5, 0.5, 'No discretization data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Correlation')
        
        # Apply consistent legend formatting to all subplots
        for ax in axes.flat:
            apply_consistent_legend_formatting(ax, use_3d=False)
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Convert to desired format
        if self.config.image_format == 'RGB' and image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        # Create metadata
        metadata = {
            'n_frequent_itemsets': len(self.frequent_itemsets),
            'n_association_rules': len(self.rules),
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'n_bins': self.n_bins,
            'discretization_method': self.discretization_method
        }
        
        if len(self.frequent_itemsets) > 0:
            metadata['avg_support'] = float(self.frequent_itemsets['support'].mean())
            metadata['max_support'] = float(self.frequent_itemsets['support'].max())
        
        if len(self.rules) > 0:
            metadata['avg_confidence'] = float(self.rules['confidence'].mean())
            metadata['avg_lift'] = float(self.rules['lift'].mean())
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=transformed_data,
            description=self._get_default_description(len(transformed_data), transformed_data.shape[1]),
            method_name=self.method_name,
            config=self.config,
            metadata=metadata,
            legend_text=f"Patterns: {len(self.frequent_itemsets)} itemsets, {len(self.rules)} rules"
        )
        
        return result
    
    def get_top_patterns(self, n_top: int = 10) -> Dict[str, Any]:
        """Get top frequent patterns and rules."""
        result = {
            'top_itemsets': [],
            'top_rules': []
        }
        
        if len(self.frequent_itemsets) > 0:
            top_itemsets = self.frequent_itemsets.nlargest(n_top, 'support')
            result['top_itemsets'] = [
                {
                    'itemset': list(itemset),
                    'support': float(support)
                }
                for itemset, support in zip(top_itemsets['itemsets'], top_itemsets['support'])
            ]
        
        if len(self.rules) > 0:
            top_rules = self.rules.nlargest(n_top, 'confidence')
            result['top_rules'] = [
                {
                    'antecedents': list(row['antecedents']),
                    'consequents': list(row['consequents']),
                    'support': float(row['support']),
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift'])
                }
                for _, row in top_rules.iterrows()
            ]
        
        return result


def analyze_frequent_patterns(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 5,
    min_support: float = 0.1,
    min_confidence: float = 0.6,
    discretization_method: str = 'quantile',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Convenience function to analyze frequent patterns.
    
    Args:
        X: Input data
        feature_names: Names of features
        n_bins: Number of bins for discretization
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for association rules
        discretization_method: Method for discretization
        random_state: Random seed
        
    Returns:
        Dictionary with pattern analysis results
    """
    from ..base import VisualizationConfig
    
    # Create configuration
    config = VisualizationConfig(
        random_state=random_state,
        extra_params={
            'n_bins': n_bins,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'discretization_method': discretization_method
        }
    )
    
    # Create visualization
    viz = FrequentPatternsVisualization(config)
    
    # Analyze patterns
    discrete_data = viz.fit_transform(X)
    
    # Generate visualization
    result = viz.generate_plot(discrete_data)
    
    # Get top patterns
    top_patterns = viz.get_top_patterns()
    
    return {
        'discrete_data': discrete_data,
        'frequent_itemsets': viz.frequent_itemsets,
        'association_rules': viz.rules,
        'top_patterns': top_patterns,
        'visualization_result': result,
        'discretizer': viz._transformer
    }