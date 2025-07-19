"""
Adaptive Nearest Neighbor Mixture Model for Time Series Distribution Visualization.

This module implements a temporally-adaptive approach that samples Student-T 
distribution parameters from training sequences that closely match recent patterns
in the forecast data. This bridges the gap between static mixtures and TOTO's 
dynamic neural approach.

Key idea: For each forecast timestep, find training subsequences with similar
recent patterns and use their fitted parameters to create an adaptive mixture.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.stats import t as student_t
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TrainingPattern:
    """A training pattern with its fitted Student-T parameters."""
    window: np.ndarray  # The training window values
    df: float  # degrees of freedom
    loc: float  # location parameter (mean value)
    scale: float  # scale parameter
    pattern_id: int  # Unique identifier
    source_series_id: int  # Which training series this came from
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample values from this pattern's distribution."""
        if random_state is not None:
            np.random.seed(random_state)
        return student_t.rvs(df=self.df, loc=self.loc, scale=self.scale, size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        return student_t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)


@dataclass
class AdaptiveMixtureComponent:
    """A component in an adaptive mixture with distance-based weighting."""
    pattern: TrainingPattern
    weight: float  # Mixture weight based on similarity
    distance: float  # Distance to current pattern
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from this component."""
        return self.pattern.sample(size, random_state)


@dataclass
class AdaptiveMixtureDistribution:
    """Adaptive mixture that changes based on nearest neighbor patterns."""
    components: List[AdaptiveMixtureComponent]
    current_window: np.ndarray  # The window that generated this mixture
    name: str
    
    def __post_init__(self):
        """Normalize mixture weights."""
        total_weight = sum(comp.weight for comp in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight /= total_weight
    
    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return len(self.components)
    
    @property
    def weights(self) -> np.ndarray:
        """Mixture weights as numpy array."""
        return np.array([comp.weight for comp in self.components])
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the adaptive mixture."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample component indices based on weights
        component_indices = np.random.choice(
            len(self.components), 
            size=size, 
            p=self.weights
        )
        
        # Sample from selected components
        samples = np.zeros(size)
        for i, comp_idx in enumerate(component_indices):
            component = self.components[comp_idx]
            samples[i] = component.sample(1, random_state=random_state + i if random_state else None)[0]
        
        return samples
    
    def forecast_sequence(self, length: int, last_value: float, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast sequence by sampling from adaptive mixture.
        Note: This is a simplified version. Full adaptive approach would 
        update the mixture for each timestep.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample directly from the adaptive mixture
        forecast_values = self.sample(length, random_state)
        
        return forecast_values


class AdaptiveNearestNeighborMixture:
    """
    Adaptive mixture model that finds nearest neighbor patterns in training data
    and uses their parameters to create temporally-adaptive forecasts.
    """
    
    def __init__(
        self, 
        window_size: int = 10,
        n_neighbors: int = 5,
        distance_metric: str = 'euclidean',
        weight_decay: float = 0.5
    ):
        """
        Initialize adaptive nearest neighbor mixture.
        
        Args:
            window_size: Size of the pattern window to match
            n_neighbors: Number of nearest neighbors to use for mixture
            distance_metric: Distance metric ('euclidean', 'dtw', 'trend')
            weight_decay: Exponential decay for distance-based weighting
        """
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.weight_decay = weight_decay
        
        # Storage for training patterns
        self.training_patterns: List[TrainingPattern] = []
        
    def _fit_student_t_to_values(self, values: np.ndarray) -> Dict[str, float]:
        """Fit Student-T distribution to values (not increments)."""
        if len(values) < 2:
            return {'df': 3.0, 'loc': 0.0, 'scale': 1.0}
        
        if np.std(values) == 0:
            return {
                'df': 3.0,
                'loc': np.mean(values) if len(values) > 0 else 0.0,
                'scale': 1.0
            }
        
        # Method of moments initial estimates
        sample_mean = np.mean(values)
        sample_std = np.std(values)
        
        # Initial parameters for value distribution
        df_init = 4.0
        scale_init = sample_std * np.sqrt((df_init - 2) / df_init)
        loc_init = sample_mean
        
        def neg_log_likelihood(params):
            df, loc, scale = params
            if df <= 2.0 or scale <= 0:
                return np.inf
            try:
                log_pdf = student_t.logpdf(values, df=df, loc=loc, scale=scale)
                return -np.sum(log_pdf)
            except (ValueError, RuntimeWarning):
                return np.inf
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    neg_log_likelihood,
                    x0=[df_init, loc_init, scale_init],
                    method='L-BFGS-B',
                    bounds=[(2.1, 50.0), (None, None), (1e-6, None)]
                )
                
                if result.success:
                    df_opt, loc_opt, scale_opt = result.x
                else:
                    df_opt, loc_opt, scale_opt = df_init, loc_init, scale_init
        except Exception:
            df_opt, loc_opt, scale_opt = df_init, loc_init, scale_init
        
        return {
            'df': max(2.1, df_opt),
            'loc': loc_opt,
            'scale': max(1e-6, scale_opt)
        }
    
    def _compute_distance(self, window1: np.ndarray, window2: np.ndarray) -> float:
        """Compute distance between two windows."""
        if self.distance_metric == 'euclidean':
            return euclidean(window1, window2)
        elif self.distance_metric == 'trend':
            # Compare trends (differences)
            trend1 = np.diff(window1) if len(window1) > 1 else np.array([0])
            trend2 = np.diff(window2) if len(window2) > 1 else np.array([0])
            return euclidean(trend1, trend2)
        elif self.distance_metric == 'normalized':
            # Normalize by standard deviation
            std1, std2 = np.std(window1), np.std(window2)
            if std1 > 0 and std2 > 0:
                norm1 = (window1 - np.mean(window1)) / std1
                norm2 = (window2 - np.mean(window2)) / std2
                return euclidean(norm1, norm2)
            else:
                return euclidean(window1, window2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_mixture_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute mixture weights from distances using exponential decay."""
        # Invert distances (smaller distance = higher weight)
        inverted = 1.0 / (1.0 + distances)
        
        # Apply exponential decay
        weights = np.exp(-self.weight_decay * distances)
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def fit_training_patterns(self, training_sequences: List[np.ndarray]):
        """
        Extract and fit patterns from training sequences.
        
        Args:
            training_sequences: List of training time series
        """
        self.training_patterns = []
        pattern_id = 0
        
        for series_id, sequence in enumerate(training_sequences):
            if len(sequence) <= self.window_size:
                # If sequence is too short, use the whole sequence
                window = sequence
                params = self._fit_student_t_to_values(window)
                
                pattern = TrainingPattern(
                    window=window,
                    df=params['df'],
                    loc=params['loc'],
                    scale=params['scale'],
                    pattern_id=pattern_id,
                    source_series_id=series_id
                )
                self.training_patterns.append(pattern)
                pattern_id += 1
            else:
                # Extract sliding windows from the sequence
                for i in range(len(sequence) - self.window_size + 1):
                    window = sequence[i:i + self.window_size]
                    params = self._fit_student_t_to_values(window)
                    
                    pattern = TrainingPattern(
                        window=window,
                        df=params['df'],
                        loc=params['loc'],
                        scale=params['scale'],
                        pattern_id=pattern_id,
                        source_series_id=series_id
                    )
                    self.training_patterns.append(pattern)
                    pattern_id += 1
        
        logger.info(f"Extracted {len(self.training_patterns)} training patterns from {len(training_sequences)} sequences")
    
    def create_adaptive_mixture(
        self, 
        current_sequence: np.ndarray, 
        name: str = "Adaptive Mixture"
    ) -> AdaptiveMixtureDistribution:
        """
        Create adaptive mixture for current sequence based on nearest neighbors.
        
        Args:
            current_sequence: Current time series sequence
            name: Name for the mixture
            
        Returns:
            AdaptiveMixtureDistribution based on nearest neighbors
        """
        if len(self.training_patterns) == 0:
            raise ValueError("No training patterns available. Call fit_training_patterns first.")
        
        # Get recent window from current sequence
        if len(current_sequence) >= self.window_size:
            current_window = current_sequence[-self.window_size:]
        else:
            current_window = current_sequence  # Use what we have
        
        # Compute distances to all training patterns
        distances = []
        for pattern in self.training_patterns:
            # Adjust pattern window size if needed
            pattern_window = pattern.window
            if len(pattern_window) > len(current_window):
                pattern_window = pattern_window[:len(current_window)]
            elif len(pattern_window) < len(current_window):
                # Pad with last value
                padding = np.full(len(current_window) - len(pattern_window), pattern_window[-1])
                pattern_window = np.concatenate([pattern_window, padding])
            
            distance = self._compute_distance(current_window, pattern_window)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_distances = distances[nearest_indices]
        
        # Compute mixture weights
        weights = self._compute_mixture_weights(nearest_distances)
        
        # Create mixture components
        components = []
        for i, (pattern_idx, weight, distance) in enumerate(zip(nearest_indices, weights, nearest_distances)):
            pattern = self.training_patterns[pattern_idx]
            component = AdaptiveMixtureComponent(
                pattern=pattern,
                weight=weight,
                distance=distance
            )
            components.append(component)
        
        mixture = AdaptiveMixtureDistribution(
            components=components,
            current_window=current_window,
            name=name
        )
        
        logger.debug(f"Created adaptive mixture with {len(components)} components, "
                    f"min distance: {np.min(nearest_distances):.2f}, "
                    f"max distance: {np.max(nearest_distances):.2f}")
        
        return mixture


def generate_semantic_adaptive_mixture_name(mixture: AdaptiveMixtureDistribution) -> str:
    """Generate semantic name for adaptive mixture distribution."""
    n_comp = mixture.n_components
    
    # Analyze component diversity
    component_locs = [comp.pattern.loc for comp in mixture.components]
    component_scales = [comp.pattern.scale for comp in mixture.components]
    
    avg_loc = np.mean(component_locs)
    loc_std = np.std(component_locs)
    avg_scale = np.mean(component_scales)
    
    # Analyze distances (similarity to current pattern)
    distances = [comp.distance for comp in mixture.components]
    avg_distance = np.mean(distances)
    
    # Classify similarity
    if avg_distance < 10:  # Adjust thresholds based on data scale
        similarity_desc = "High-Similarity"
    elif avg_distance < 50:
        similarity_desc = "Moderate-Similarity" 
    else:
        similarity_desc = "Low-Similarity"
    
    # Classify diversity
    if loc_std < avg_scale * 0.1:
        diversity_desc = "Focused"
    elif loc_std > avg_scale * 0.5:
        diversity_desc = "Diverse"
    else:
        diversity_desc = "Mixed"
    
    # Classify complexity
    if n_comp == 1:
        complexity_desc = "Single-Pattern"
    elif n_comp <= 3:
        complexity_desc = "Multi-Pattern"
    else:
        complexity_desc = "Complex-Pattern"
    
    return f"{similarity_desc} {diversity_desc} {complexity_desc} Adaptive"