"""
Time Series Distribution Visualization for MARVIS.

This module implements visualization of time series data as classification 
over fitted Student's T distribution functions. Each distribution represents
a potential forecast path that the VLM can select from.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.stats import t as student_t
from scipy.optimize import minimize
import warnings

from .base import BaseVisualization, VisualizationConfig, VisualizationResult
from .value_mixture_distribution import (
    ValueMixtureDistribution, 
    ValueMixtureFitter, 
    generate_semantic_value_mixture_name
)
from .adaptive_mixture_distribution import (
    AdaptiveNearestNeighborMixture,
    AdaptiveMixtureDistribution,
    generate_semantic_adaptive_mixture_name
)

logger = logging.getLogger(__name__)


@dataclass
class StudentTDistribution:
    """Container for Student's T distribution parameters."""
    df: float  # degrees of freedom (>2.0)
    loc: float  # location parameter (mean)
    scale: float  # scale parameter (>0)
    keypoints: np.ndarray  # keypoints used for fitting
    name: str  # semantic name for this distribution
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from this distribution."""
        if random_state is not None:
            np.random.seed(random_state)
        return student_t.rvs(df=self.df, loc=self.loc, scale=self.scale, size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        return student_t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)
    
    def forecast_sequence(self, length: int, last_value: float, random_state: Optional[int] = None) -> np.ndarray:
        """Generate a forecast sequence of given length."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate increments from the distribution
        increments = self.sample(length, random_state)
        
        # Create sequence starting from last observed value
        sequence = np.zeros(length + 1)
        sequence[0] = last_value
        
        for i in range(length):
            # Add increment with some momentum/trend component
            sequence[i + 1] = sequence[i] + increments[i]
        
        return sequence[1:]  # Return without initial value


class TimeSeriesDistributionVisualization(BaseVisualization):
    """
    Visualization of time series data using fitted Student's T distributions.
    
    This visualization:
    1. Fits multiple Student's T distributions to keypoints in training data
    2. Creates forecast paths for each distribution 
    3. Displays them as classification options for VLM selection
    4. Each distribution gets a semantic name based on its characteristics
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the time series distribution visualization."""
        super().__init__(config)
        
        # Time series specific parameters
        self._n_distributions = self.config.extra_params.get('n_distributions', 5)
        self._forecast_horizon = self.config.extra_params.get('forecast_horizon', 12)
        self._n_keypoints = self.config.extra_params.get('n_keypoints', 8)
        self._keypoint_strategy = self.config.extra_params.get('keypoint_strategy', 'uniform')
        self._show_confidence_bands = self.config.extra_params.get('show_confidence_bands', True)
        self._use_mixture_model = self.config.extra_params.get('use_mixture_model', True)
        self._use_adaptive_mixture = self.config.extra_params.get('use_adaptive_mixture', False)
        self._max_mixture_components = self.config.extra_params.get('max_mixture_components', 3)
        self._adaptive_window_size = self.config.extra_params.get('adaptive_window_size', 10)
        self._adaptive_n_neighbors = self.config.extra_params.get('adaptive_n_neighbors', 5)
        
        # Fitted distributions (can be simple, mixture, or adaptive)
        self._distributions: List[Union[StudentTDistribution, ValueMixtureDistribution, AdaptiveMixtureDistribution]] = []
        self._training_data = None
        self._time_points = None
        
    @property
    def method_name(self) -> str:
        """Return the name of the visualization method."""
        return "Time Series Distribution"
    
    @property
    def supports_3d(self) -> bool:
        """Return whether this method supports 3D visualization."""
        return False  # Time series are inherently 2D (time vs value)
    
    @property
    def supports_regression(self) -> bool:
        """Return whether this method supports regression tasks."""
        return True  # Time series forecasting is a regression task
    
    @property
    def supports_new_data(self) -> bool:
        """Return whether this method can transform new data after fitting."""
        return False  # This is a generative visualization, not a transformation
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create the time series distribution fitter."""
        # This is a custom implementation, not using sklearn
        return self
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get a default description for this visualization method."""
        return (f"Time series distribution visualization showing {self._n_distributions} "
                f"fitted Student's T distributions over {n_samples} time points. "
                f"Each distribution represents a potential forecast pattern that can be "
                f"selected for predicting the next {self._forecast_horizon} values.")
    
    def _select_keypoints(self, data: np.ndarray, strategy: str = 'uniform', n_keypoints: Optional[int] = None) -> np.ndarray:
        """
        Select keypoints from time series data for distribution fitting.
        
        Args:
            data: Time series values [n_timesteps]
            strategy: Keypoint selection strategy
            n_keypoints: Number of keypoints to select (if None, uses self._n_keypoints)
            
        Returns:
            Indices of selected keypoints
        """
        n_points = len(data)
        if n_keypoints is None:
            n_keypoints = min(self._n_keypoints, n_points)
        else:
            n_keypoints = min(n_keypoints, n_points)
        
        if strategy == 'uniform':
            # Uniformly spaced keypoints
            indices = np.linspace(0, n_points - 1, n_keypoints, dtype=int)
            
        elif strategy == 'extrema':
            # Select local minima and maxima
            from scipy.signal import find_peaks
            
            # Find peaks (maxima)
            peaks, _ = find_peaks(data)
            # Find troughs (minima)
            troughs, _ = find_peaks(-data)
            
            # Combine and sort
            extrema = np.concatenate([peaks, troughs])
            extrema = np.sort(extrema)
            
            # Add start and end points
            all_points = np.concatenate([[0], extrema, [n_points - 1]])
            all_points = np.unique(all_points)
            
            # Select subset if too many
            if len(all_points) > n_keypoints:
                indices = np.linspace(0, len(all_points) - 1, n_keypoints, dtype=int)
                indices = all_points[indices]
            else:
                indices = all_points
                
        elif strategy == 'random_subset':
            # Select random subset of time points with guaranteed start/end
            # This provides diversity while staying stable
            if n_keypoints <= 2:
                indices = np.array([0, n_points - 1])
            else:
                # Always include start and end
                indices = [0, n_points - 1]
                
                # Randomly select interior points
                interior_points = np.arange(1, n_points - 1)
                n_interior = min(n_keypoints - 2, len(interior_points))
                
                if n_interior > 0:
                    np.random.seed(42)  # For reproducibility
                    selected_interior = np.random.choice(
                        interior_points, size=n_interior, replace=False
                    )
                    indices.extend(selected_interior)
                
                indices = np.sort(np.unique(np.array(indices)))
        
        elif strategy == 'percentile':
            # Select keypoints based on percentiles of the data values
            # This captures the full range of volatility in the data
            if n_keypoints <= 2:
                # Always include start and end
                indices = np.array([0, n_points - 1])
            else:
                # Create percentile-based selection
                # Always include start and end points
                percentiles = np.linspace(0, 100, n_keypoints)
                value_percentiles = np.percentile(data, percentiles)
                
                # Find indices closest to these percentile values
                indices = []
                indices.append(0)  # Always start
                
                for perc_val in value_percentiles[1:-1]:  # Skip first and last
                    # Find closest value in data
                    closest_idx = np.argmin(np.abs(data - perc_val))
                    indices.append(closest_idx)
                
                indices.append(n_points - 1)  # Always end
                indices = np.unique(np.array(indices))
                
        else:
            raise ValueError(f"Unknown keypoint strategy: {strategy}")
            
        return indices
    
    def _fit_student_t_to_keypoints(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Fit Student's T distribution parameters to keypoint increments.
        
        Args:
            keypoints: Selected keypoint values
            
        Returns:
            Dictionary with fitted parameters
        """
        if len(keypoints) < 2:
            # Not enough points, return default parameters
            return {'df': 3.0, 'loc': 0.0, 'scale': 1.0}
        
        # Compute increments
        increments = np.diff(keypoints)
        
        if len(increments) == 0 or np.std(increments) == 0:
            return {'df': 3.0, 'loc': np.mean(increments) if len(increments) > 0 else 0.0, 'scale': 1.0}
        
        # Use method of moments for initial estimates
        sample_mean = np.mean(increments)
        sample_var = np.var(increments)
        sample_std = np.std(increments)
        
        # Initial parameter estimates
        # For Student's T: var = scale^2 * df / (df - 2) for df > 2
        # So scale^2 = var * (df - 2) / df
        df_init = max(3.0, 4.0)  # Start with df=4 for reasonable variance
        scale_init = sample_std * np.sqrt((df_init - 2) / df_init)
        loc_init = sample_mean
        
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            df, loc, scale = params
            
            # Parameter constraints
            if df <= 2.0 or scale <= 0:
                return np.inf
            
            try:
                # Compute negative log-likelihood
                log_pdf = student_t.logpdf(increments, df=df, loc=loc, scale=scale)
                return -np.sum(log_pdf)
            except (ValueError, RuntimeWarning):
                return np.inf
        
        # Optimize parameters
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
                    # Use initial estimates if optimization fails
                    df_opt, loc_opt, scale_opt = df_init, loc_init, scale_init
                    
        except Exception:
            # Fallback to initial estimates
            df_opt, loc_opt, scale_opt = df_init, loc_init, scale_init
        
        return {
            'df': max(2.1, df_opt),
            'loc': loc_opt,
            'scale': max(1e-6, scale_opt)
        }
    
    def _generate_semantic_name(self, distribution: Dict[str, float], keypoints: np.ndarray) -> str:
        """
        Generate semantic name for a distribution based on its characteristics.
        
        Args:
            distribution: Distribution parameters
            keypoints: Keypoints used for fitting
            
        Returns:
            Semantic name string
        """
        df, loc, scale = distribution['df'], distribution['loc'], distribution['scale']
        
        # Analyze trend
        if len(keypoints) >= 2:
            overall_trend = (keypoints[-1] - keypoints[0]) / len(keypoints)
        else:
            overall_trend = 0
        
        # Classify trend
        if abs(overall_trend) < 0.1 * np.std(keypoints) if len(keypoints) > 1 else True:
            trend_desc = "Stable"
        elif overall_trend > 0:
            trend_desc = "Increasing"
        else:
            trend_desc = "Decreasing"
        
        # Classify volatility based on scale and df
        if scale < 0.5:
            volatility_desc = "Low Volatility"
        elif scale > 2.0:
            volatility_desc = "High Volatility"
        else:
            volatility_desc = "Moderate Volatility"
        
        # Classify tail behavior based on degrees of freedom
        if df < 4:
            tail_desc = "Fat Tails"
        elif df > 10:
            tail_desc = "Thin Tails"
        else:
            tail_desc = "Moderate Tails"
        
        return f"{trend_desc} Trend, {volatility_desc}, {tail_desc}"
    
    def _generate_distributions(self, data: np.ndarray) -> List[StudentTDistribution]:
        """
        Generate multiple Student's T distributions for the time series.
        
        Args:
            data: Time series values [n_timesteps]
            
        Returns:
            List of fitted distributions
        """
        distributions = []
        n_points = len(data)
        
        # First trajectory: Use ALL data points as keypoints for maximum fidelity
        try:
            # Use every single data point for maximum resolution
            all_keypoint_indices = np.arange(n_points)
            
            keypoints = data[all_keypoint_indices]
            params = self._fit_student_t_to_keypoints(keypoints)
            name = self._generate_semantic_name(params, keypoints)
            
            dist = StudentTDistribution(
                df=params['df'],
                loc=params['loc'],
                scale=params['scale'],
                keypoints=keypoints,
                name=f"Full Resolution: {name}"
            )
            distributions.append(dist)
            logger.info(f"Generated full resolution distribution with {len(keypoints)} keypoints")
        except Exception as e:
            logger.warning(f"Failed to fit full resolution distribution: {e}")
        
        # Remaining trajectories: Use variable keypoint sampling with random percentiles
        np.random.seed(42)  # For reproducible results
        
        # Define keypoint count ranges for different volatility levels
        keypoint_ranges = [
            (5, 10),   # Low resolution - captures broad trends
            (10, 20),  # Medium resolution - captures moderate detail
            (20, 35),  # High resolution - captures fine detail
        ]
        
        strategies = ['percentile', 'extrema', 'uniform', 'random_subset']
        
        for i in range(min(self._n_distributions - 1, 4)):  # -1 because we already have full resolution
            try:
                # Select strategy and keypoint count
                strategy = strategies[i % len(strategies)]
                keypoint_range = keypoint_ranges[i % len(keypoint_ranges)]
                
                # Randomly sample number of keypoints within range
                n_keypoints = np.random.randint(keypoint_range[0], keypoint_range[1] + 1)
                
                keypoint_indices = self._select_keypoints(data, strategy, n_keypoints)
                keypoints = data[keypoint_indices]
                params = self._fit_student_t_to_keypoints(keypoints)
                name = self._generate_semantic_name(params, keypoints)
                
                dist = StudentTDistribution(
                    df=params['df'],
                    loc=params['loc'],
                    scale=params['scale'],
                    keypoints=keypoints,
                    name=f"{strategy.capitalize()} ({len(keypoints)}pts): {name}"
                )
                distributions.append(dist)
                logger.info(f"Generated {strategy} distribution with {len(keypoints)} keypoints")
            except Exception as e:
                logger.warning(f"Failed to fit distribution {i} with strategy {strategy}: {e}")
        
        # If we still need more distributions, create parameter variations
        if len(distributions) < self._n_distributions and distributions:
            base_dist = distributions[0]  # Use the full resolution as base
            
            for i in range(self._n_distributions - len(distributions)):
                try:
                    # Create variations by adjusting scale parameter
                    variation_factor = 1.5 + 0.5 * i  # 1.5, 2.0, 2.5, ...
                    
                    # Vary scale (volatility) while keeping other parameters
                    new_scale = base_dist.scale * variation_factor
                    new_name = f"Scaled Volatility ({variation_factor:.1f}x): High Variance Pattern"
                    
                    dist = StudentTDistribution(
                        df=base_dist.df,
                        loc=base_dist.loc,
                        scale=new_scale,
                        keypoints=base_dist.keypoints,
                        name=new_name
                    )
                    distributions.append(dist)
                except Exception as e:
                    logger.warning(f"Failed to create variation {i}: {e}")
        
        # Ensure we have at least one distribution (fallback)
        if not distributions:
            logger.warning("No distributions generated successfully, creating default")
            mean_increment = np.mean(np.diff(data)) if len(data) > 1 else 0.0
            std_increment = np.std(np.diff(data)) if len(data) > 1 else 1.0
            
            dist = StudentTDistribution(
                df=4.0,
                loc=mean_increment,
                scale=std_increment,
                keypoints=data[::max(1, len(data)//8)],
                name="Default: Random Walk Pattern"
            )
            distributions.append(dist)
        
        return distributions[:self._n_distributions]
    
    def _generate_adaptive_distributions(self, data: np.ndarray) -> List[AdaptiveMixtureDistribution]:
        """
        Generate adaptive distributions using nearest neighbor pattern matching.
        
        Args:
            data: Time series values [n_timesteps]
            
        Returns:
            List of adaptive mixture distributions
        """
        distributions = []
        
        # Create different adaptive configurations for diversity
        configs = [
            {
                'window_size': self._adaptive_window_size,
                'n_neighbors': self._adaptive_n_neighbors,
                'distance_metric': 'euclidean',
                'name': 'Euclidean Adaptive'
            },
            {
                'window_size': max(5, self._adaptive_window_size // 2),
                'n_neighbors': self._adaptive_n_neighbors,
                'distance_metric': 'trend',
                'name': 'Trend-Based Adaptive'
            },
            {
                'window_size': self._adaptive_window_size,
                'n_neighbors': min(10, self._adaptive_n_neighbors * 2),
                'distance_metric': 'normalized',
                'name': 'Normalized Adaptive'
            },
            {
                'window_size': min(15, self._adaptive_window_size + 5),
                'n_neighbors': max(3, self._adaptive_n_neighbors - 2),
                'distance_metric': 'euclidean',
                'name': 'Long-Window Adaptive'
            },
            {
                'window_size': max(3, self._adaptive_window_size - 5),
                'n_neighbors': min(8, self._adaptive_n_neighbors + 3),
                'distance_metric': 'trend',
                'name': 'Short-Window Adaptive'
            }
        ]
        
        # For adaptive approach, we need to simulate multiple training sequences
        # Since we only have one series, we'll create artificial "training" sequences
        # by using different segments of the data
        training_sequences = self._create_training_sequences_from_data(data)
        
        for i, config in enumerate(configs[:self._n_distributions]):
            try:
                # Create adaptive mixture model
                adaptive_model = AdaptiveNearestNeighborMixture(
                    window_size=config['window_size'],
                    n_neighbors=config['n_neighbors'],
                    distance_metric=config['distance_metric']
                )
                
                # Fit to training patterns
                adaptive_model.fit_training_patterns(training_sequences)
                
                # Create adaptive mixture for the full sequence
                adaptive_mixture = adaptive_model.create_adaptive_mixture(
                    data, 
                    name=config['name']
                )
                
                # Update name with semantic description
                semantic_name = generate_semantic_adaptive_mixture_name(adaptive_mixture)
                adaptive_mixture.name = f"{config['name']}: {semantic_name}"
                
                distributions.append(adaptive_mixture)
                logger.info(f"Generated {config['name']} with {adaptive_mixture.n_components} components")
                
            except Exception as e:
                logger.warning(f"Failed to generate adaptive distribution {i} ({config['name']}): {e}")
        
        # Ensure we have at least one distribution
        if not distributions and len(data) > 10:
            try:
                # Fallback: simple adaptive mixture
                fallback_model = AdaptiveNearestNeighborMixture(
                    window_size=min(10, len(data) // 3),
                    n_neighbors=3,
                    distance_metric='euclidean'
                )
                fallback_model.fit_training_patterns(training_sequences)
                fallback_mixture = fallback_model.create_adaptive_mixture(data, "Fallback Adaptive")
                distributions.append(fallback_mixture)
                logger.info("Generated fallback adaptive distribution")
            except Exception as e:
                logger.error(f"Failed to generate fallback adaptive distribution: {e}")
        
        return distributions[:self._n_distributions]
    
    def _create_training_sequences_from_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create training sequences from a single time series by using overlapping segments.
        This simulates having multiple training series for the adaptive approach.
        """
        training_sequences = []
        
        # Parameters for sequence generation
        min_seq_length = max(20, self._adaptive_window_size * 2)
        max_seq_length = min(len(data) // 2, 100)
        n_sequences = 10  # Number of training sequences to generate
        
        if len(data) < min_seq_length:
            # If data is too short, just return the whole series
            return [data]
        
        # Generate overlapping sequences of different lengths
        np.random.seed(42)  # For reproducibility
        for i in range(n_sequences):
            # Random sequence length
            seq_length = np.random.randint(min_seq_length, min(max_seq_length, len(data) - 10) + 1)
            
            # Random starting position (ensure we don't go past the end)
            max_start = len(data) - seq_length
            if max_start <= 0:
                start_pos = 0
                seq_length = len(data)
            else:
                start_pos = np.random.randint(0, max_start)
            
            # Extract sequence
            sequence = data[start_pos:start_pos + seq_length]
            training_sequences.append(sequence)
        
        logger.debug(f"Created {len(training_sequences)} training sequences from data of length {len(data)}")
        return training_sequences
    
    def _generate_mixture_distributions(self, data: np.ndarray) -> List[ValueMixtureDistribution]:
        """
        Generate multiple Student's T mixture distributions for the time series.
        
        Args:
            data: Time series values [n_timesteps]
            
        Returns:
            List of fitted mixture distributions
        """
        distributions = []
        n_points = len(data)
        
        # Initialize mixture fitter
        mixture_fitter = ValueMixtureFitter(
            max_components=self._max_mixture_components,
            min_components=1
        )
        
        # First trajectory: Use high resolution keypoints for maximum fidelity
        try:
            # Use every single data point for maximum resolution  
            all_keypoint_indices = np.arange(n_points)
            keypoints = data[all_keypoint_indices]
            
            mixture_dist = mixture_fitter.create_value_mixture_distribution(
                keypoints, keypoints, 
                f"FullRes ({len(keypoints)}pts)",
                selection_criterion='bic'
            )
            
            # Update name with semantic description
            semantic_name = generate_semantic_value_mixture_name(mixture_dist, keypoints)
            mixture_dist.name = f"FullRes: {semantic_name}"
            
            distributions.append(mixture_dist)
            logger.info(f"Generated full resolution mixture with {mixture_dist.n_components} components, {len(keypoints)} keypoints")
        except Exception as e:
            logger.warning(f"Failed to fit full resolution mixture: {e}")
        
        # Remaining trajectories: Use variable keypoint sampling with random percentiles
        np.random.seed(42)  # For reproducible results
        
        # Define keypoint count ranges for different resolution levels
        keypoint_ranges = [
            (5, 10),   # Low resolution - captures broad trends
            (10, 20),  # Medium resolution - captures moderate detail
            (20, 35),  # High resolution - captures fine detail
        ]
        
        strategies = ['percentile', 'extrema', 'uniform', 'random_subset']
        
        for i in range(min(self._n_distributions - 1, 4)):  # -1 because we already have full resolution
            try:
                # Select strategy and keypoint count
                strategy = strategies[i % len(strategies)]
                keypoint_range = keypoint_ranges[i % len(keypoint_ranges)]
                
                # Randomly sample number of keypoints within range
                n_keypoints = np.random.randint(keypoint_range[0], keypoint_range[1] + 1)
                
                keypoint_indices = self._select_keypoints(data, strategy, n_keypoints)
                keypoints = data[keypoint_indices]
                
                if len(keypoints) < 2:
                    logger.warning(f"Too few keypoints for mixture {i}, skipping")
                    continue
                
                mixture_dist = mixture_fitter.create_value_mixture_distribution(
                    keypoints, keypoints,
                    f"{strategy.capitalize()}({len(keypoints)}pts)",
                    selection_criterion='bic'
                )
                
                # Update name with semantic description
                semantic_name = generate_semantic_value_mixture_name(mixture_dist, keypoints)
                mixture_dist.name = f"{strategy.capitalize()}({len(keypoints)}pts): {semantic_name}"
                
                distributions.append(mixture_dist)
                logger.info(f"Generated {strategy} mixture with {mixture_dist.n_components} components, {len(keypoints)} keypoints")
            except Exception as e:
                logger.warning(f"Failed to fit mixture {i} with strategy {strategy}: {e}")
        
        # If we still need more distributions, create parameter variations
        if len(distributions) < self._n_distributions and distributions:
            base_dist = distributions[0]  # Use the full resolution as base
            
            for i in range(self._n_distributions - len(distributions)):
                try:
                    # Create variations by adjusting mixture component scales
                    variation_factor = 1.5 + 0.5 * i  # 1.5, 2.0, 2.5, ...
                    
                    # Create new components with scaled volatility
                    new_components = []
                    for j, comp in enumerate(base_dist.components):
                        from .value_mixture_distribution import ValueMixtureComponent
                        new_comp = ValueMixtureComponent(
                            df=comp.df,
                            loc=comp.loc,
                            scale=comp.scale * variation_factor,
                            weight=comp.weight,
                            name=f"ScaledVol_{j+1}"
                        )
                        new_components.append(new_comp)
                    
                    scaled_dist = ValueMixtureDistribution(
                        components=new_components,
                        keypoints=base_dist.keypoints,
                        name=f"Scaled Volatility ({variation_factor:.1f}x): High Variance Mixture",
                        value_range=base_dist.value_range
                    )
                    distributions.append(scaled_dist)
                except Exception as e:
                    logger.warning(f"Failed to create mixture variation {i}: {e}")
        
        # Ensure we have at least one distribution (fallback)
        if not distributions:
            logger.warning("No mixture distributions generated successfully, creating default")
            
            # Create default single-component mixture
            default_values = data[::max(1, len(data)//8)]
            
            try:
                default_dist = mixture_fitter.create_value_mixture_distribution(
                    default_values, default_values,
                    "Default Mixture: Value-based Pattern",
                    selection_criterion='bic'
                )
                distributions.append(default_dist)
            except Exception as e:
                logger.error(f"Failed to create default mixture: {e}")
                # Ultimate fallback - create a very simple mixture manually
                from .value_mixture_distribution import ValueMixtureComponent
                default_comp = ValueMixtureComponent(
                    df=4.0,
                    loc=np.mean(default_values) if len(default_values) > 0 else 0.0,
                    scale=np.std(default_values) if len(default_values) > 0 else 1.0,
                    weight=1.0,
                    name="Default_Component"
                )
                default_dist = ValueMixtureDistribution(
                    components=[default_comp],
                    keypoints=default_values,
                    name="Default: Single Component Mixture",
                    value_range=(np.min(data), np.max(data))
                )
                distributions.append(default_dist)
        
        return distributions[:self._n_distributions]
    
    def _clip_forecast_to_training_bounds(self, forecast: np.ndarray, historical_data: np.ndarray) -> np.ndarray:
        """
        Clip forecast values to reasonable bounds based on training data characteristics.
        
        Instead of filtering out trajectories, this preserves the shape while constraining
        extreme values to be within the observed range of the training data.
        
        Args:
            forecast: Forecast sequence to clip
            historical_data: Historical time series data for reference bounds
            
        Returns:
            Clipped forecast sequence with same shape but bounded values
        """
        if len(forecast) == 0:
            return forecast
        
        # Calculate training data bounds
        hist_min = np.min(historical_data)
        hist_max = np.max(historical_data)
        hist_mean = np.mean(historical_data)
        hist_std = np.std(historical_data)
        
        # Define reasonable bounds based on training data
        # Allow some extrapolation beyond observed range, but not too extreme
        extrapolation_factor = 1.5  # Allow 50% extrapolation beyond observed range
        hist_range = hist_max - hist_min
        
        # Set bounds: training min/max +/- some extrapolation
        lower_bound = hist_min - (extrapolation_factor * hist_range)
        upper_bound = hist_max + (extrapolation_factor * hist_range)
        
        # Also limit based on standard deviations to handle extreme outliers
        std_bound_lower = hist_mean - 6 * hist_std  # 6 sigma lower bound
        std_bound_upper = hist_mean + 6 * hist_std  # 6 sigma upper bound
        
        # Use the more conservative (tighter) bounds
        final_lower = max(lower_bound, std_bound_lower)
        final_upper = min(upper_bound, std_bound_upper)
        
        # Clip the forecast
        clipped_forecast = np.clip(forecast, final_lower, final_upper)
        
        # Handle infinite or NaN values
        clipped_forecast = np.where(~np.isfinite(clipped_forecast), hist_mean, clipped_forecast)
        
        # Log if significant clipping occurred
        n_clipped = np.sum((forecast < final_lower) | (forecast > final_upper))
        if n_clipped > 0:
            logger.info(f"Clipped {n_clipped}/{len(forecast)} forecast points to bounds [{final_lower:.2f}, {final_upper:.2f}]")
        
        return clipped_forecast
    
    def _get_distribution_metadata(self, dist: Union[StudentTDistribution, ValueMixtureDistribution, AdaptiveMixtureDistribution]) -> Dict[str, Any]:
        """Get metadata for a distribution (handles both simple and mixture types)."""
        if hasattr(dist, 'df'):  # Simple StudentTDistribution
            return {
                'type': 'simple',
                'df': dist.df,
                'loc': dist.loc,
                'scale': dist.scale,
                'name': dist.name
            }
        elif hasattr(dist, 'components') and hasattr(dist.components[0], 'df'):  # ValueMixtureDistribution
            return {
                'type': 'mixture',
                'name': dist.name,
                'n_components': dist.n_components,
                'components': [
                    {
                        'df': comp.df,
                        'loc': comp.loc,
                        'scale': comp.scale,
                        'weight': comp.weight,
                        'name': comp.name
                    }
                    for comp in dist.components
                ]
            }
        else:  # AdaptiveMixtureDistribution
            return {
                'type': 'adaptive',
                'name': dist.name,
                'n_components': dist.n_components,
                'components': [
                    {
                        'df': comp.pattern.df,
                        'loc': comp.pattern.loc,
                        'scale': comp.pattern.scale,
                        'weight': comp.weight,
                        'distance': comp.distance,
                        'pattern_id': comp.pattern.pattern_id,
                        'source_series_id': comp.pattern.source_series_id
                    }
                    for comp in dist.components
                ]
            }
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Fit distributions to time series data.
        
        Args:
            X: Time series data [n_timesteps, n_features] or [n_timesteps]
            y: Not used for time series
            **kwargs: Additional parameters
            
        Returns:
            Time points for visualization [n_timesteps, 1]
        """
        import time
        start_time = time.time()
        
        # Handle input format
        if X.ndim == 1:
            data = X
        elif X.ndim == 2 and X.shape[1] == 1:
            data = X.flatten()
        elif X.ndim == 2:
            # For multivariate, use first column or average
            data = X[:, 0] if X.shape[1] > 1 else X.flatten()
        else:
            raise ValueError(f"Unsupported data shape: {X.shape}")
        
        self._training_data = data
        self._time_points = np.arange(len(data))
        
        # Update forecast horizon from kwargs if provided
        self._forecast_horizon = kwargs.get('forecast_horizon', self._forecast_horizon)
        
        # Generate distributions (adaptive, mixture, or simple)
        if self._use_adaptive_mixture:
            self._distributions = self._generate_adaptive_distributions(data)
        elif self._use_mixture_model:
            self._distributions = self._generate_mixture_distributions(data)
        else:
            self._distributions = self._generate_distributions(data)
        
        fit_time = time.time() - start_time
        self._last_fit_time = fit_time
        self._fitted = True
        
        logger.info(f"Fitted {len(self._distributions)} distributions in {fit_time:.2f}s")
        
        # Return time points as "transformed" data (this is a bit of a hack for the interface)
        return self._time_points.reshape(-1, 1)
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[List[int]] = None,
        test_data: Optional[np.ndarray] = None,
        highlight_test_indices: Optional[List[int]] = None,
        **kwargs
    ) -> VisualizationResult:
        """
        Generate time series distribution plot.
        
        Args:
            transformed_data: Time points [n_timesteps, 1]
            y: Not used
            highlight_indices: Time points to highlight
            test_data: Not used for time series
            highlight_test_indices: Not used
            **kwargs: Additional plotting parameters
            
        Returns:
            VisualizationResult with the plot
        """
        import time
        import io
        from PIL import Image
        
        plot_start = time.time()
        
        if self._training_data is None or not self._distributions:
            raise ValueError("Must call fit_transform before generate_plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot historical data
        time_points = self._time_points
        data = self._training_data
        
        ax.plot(time_points, data, 'o-', color='black', linewidth=2, markersize=4, 
                label='Historical Data', alpha=0.8)
        
        # Generate forecast time points
        forecast_start = len(data)
        # Get forecast horizon from config (updated during evaluation)
        forecast_horizon = self.config.extra_params.get('forecast_horizon', self._forecast_horizon)
        forecast_time = np.arange(forecast_start, forecast_start + forecast_horizon)
        last_value = data[-1]
        
        # Colors for distributions
        colors = plt.cm.Set1(np.linspace(0, 1, len(self._distributions)))
        
        # Plot forecast paths for each distribution
        legend_parts = []
        class_names = []
        
        clipped_distributions = []
        clipped_forecasts = []
        all_colors = []
        
        for i, (dist, color) in enumerate(zip(self._distributions, colors)):
            # Generate forecast sequence
            forecast = dist.forecast_sequence(
                forecast_horizon, 
                last_value, 
                random_state=42 + i
            )
            
            # Clip forecast to training data bounds instead of filtering
            clipped_forecast = self._clip_forecast_to_training_bounds(forecast, data)
            
            # Always include the distribution (no filtering, just clipping)
            clipped_distributions.append((i, dist))
            clipped_forecasts.append(clipped_forecast)
            all_colors.append(color)
        
        # Plot all clipped forecast paths
        for (orig_idx, dist), forecast, color in zip(clipped_distributions, clipped_forecasts, all_colors):
            # Plot forecast line
            ax.plot(forecast_time, forecast, 'o-', color=color, linewidth=2, 
                    markersize=3, label=f"Class {orig_idx}: {dist.name}", alpha=0.9)
            
            # Add confidence bands if enabled
            if self._show_confidence_bands:
                # Generate multiple forecast samples for confidence band
                n_samples = 50
                forecast_samples = []
                for j in range(n_samples):
                    sample_forecast = dist.forecast_sequence(forecast_horizon, last_value, random_state=42 + orig_idx + j)
                    clipped_sample = self._clip_forecast_to_training_bounds(sample_forecast, data)
                    forecast_samples.append(clipped_sample)
                
                forecast_samples = np.array(forecast_samples)
                
                # Compute percentiles
                lower = np.percentile(forecast_samples, 25, axis=0)
                upper = np.percentile(forecast_samples, 75, axis=0)
                
                ax.fill_between(forecast_time, lower, upper, color=color, alpha=0.2)
            
            # Store class information
            class_names.append(f"C{orig_idx}: {dist.name}")
            import matplotlib.colors as mcolors
            legend_parts.append(f"C{orig_idx} (Color: {mcolors.to_hex(color)}): {dist.name}")
        
        # Highlight specific time points if requested
        if highlight_indices:
            highlighted_times = time_points[highlight_indices]
            highlighted_values = data[highlight_indices]
            ax.scatter(highlighted_times, highlighted_values, c='red', s=100, 
                      marker='x', linewidths=3, label='Highlighted Points', zorder=5)
        
        # Add vertical line separating history from forecast
        ax.axvline(x=forecast_start - 0.5, color='gray', linestyle='--', alpha=0.7, 
                   label='Forecast Start')
        
        # Styling and axis improvements
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Time Series with {len(clipped_distributions)} Distribution-Based Forecast Paths')
        ax.grid(True, alpha=0.3)
        
        # Apply symlog scale to y-axis for better handling of wide value ranges
        # This is like log scale but handles negative values and values near zero
        all_values = np.concatenate([data, *clipped_forecasts]) if clipped_forecasts else data
        value_range = np.max(all_values) - np.min(all_values)
        if value_range > 0:
            # Use symlog only if we have a significant range
            threshold = value_range / 100  # 1% of range as linear threshold
            ax.set_yscale('symlog', linthresh=threshold)
        
        # Create weighted x-axis to emphasize forecast section
        # Compress historical section, expand forecast section
        total_time_points = len(time_points) + len(forecast_time)
        hist_weight = 0.4  # Historical data gets 40% of x-axis space
        forecast_weight = 0.6  # Forecast gets 60% of x-axis space
        
        # Create new weighted x positions
        hist_positions = np.linspace(0, hist_weight, len(time_points))
        forecast_positions = np.linspace(hist_weight, 1.0, len(forecast_time))
        
        # Clear the current plot and replot with weighted positions
        ax.clear()
        
        # Replot historical data with weighted x positions
        ax.plot(hist_positions, data, 'o-', color='black', linewidth=2, markersize=4, 
                label='Historical Data', alpha=0.8)
        
        # Replot forecast paths with weighted x positions
        for (orig_idx, dist), forecast, color in zip(clipped_distributions, clipped_forecasts, all_colors):
            ax.plot(forecast_positions, forecast, 'o-', color=color, linewidth=2, 
                    markersize=3, label=f"Class {orig_idx}: {dist.name}", alpha=0.9)
        
        # Add vertical line separating history from forecast at weighted position
        ax.axvline(x=hist_weight, color='gray', linestyle='--', alpha=0.7, 
                   label='Forecast Start')
        
        # Reapply styling with weighted axes
        ax.set_xlabel('Time (Weighted: Historical 40% | Forecast 60%)')
        ax.set_ylabel('Value (Symlog Scale)')
        ax.set_title(f'Time Series with {len(clipped_distributions)} Distribution-Based Forecast Paths')
        ax.grid(True, alpha=0.3)
        
        # Apply symlog scale again after clearing
        if value_range > 0:
            threshold = value_range / 100
            ax.set_yscale('symlog', linthresh=threshold)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Create legend text for VLM using clipped distributions
        clipped_legend_parts = []
        clipped_class_names = []
        for (orig_idx, dist), color in zip(clipped_distributions, all_colors):
            import matplotlib.colors as mcolors
            clipped_legend_parts.append(f"C{orig_idx} (Color: {mcolors.to_hex(color)}): {dist.name}")
            clipped_class_names.append(f"C{orig_idx}: {dist.name}")
        
        legend_text = "Available forecast patterns:\n" + "\n".join(clipped_legend_parts)
        class_names = clipped_class_names
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        plot_time = time.time() - plot_start
        
        # Create metadata using clipped distributions
        clipped_indices = [orig_idx for orig_idx, _ in clipped_distributions]
        clipped_dist_objects = [dist for _, dist in clipped_distributions]
        
        metadata = {
            'plot_type': 'time_series_classification',
            'visible_classes': clipped_indices,
            'all_classes': clipped_indices,
            'class_names': class_names,
            'n_distributions': len(clipped_distributions),
            'forecast_horizon': self.config.extra_params.get('forecast_horizon', self._forecast_horizon),
            'distribution_params': [
                self._get_distribution_metadata(d) for d in clipped_dist_objects
            ],
            'clipping_applied': True,  # Indicate that clipping was used instead of filtering
            'filtered_out_count': 0    # No distributions filtered, only clipped
        }
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=transformed_data,
            description=self._get_default_description(len(self._training_data), 1),
            method_name=self.method_name,
            config=self.config,
            fit_time=getattr(self, '_last_fit_time', 0.0),
            transform_time=0.0,
            plot_time=plot_time,
            highlighted_indices=highlight_indices,
            highlighted_coords=time_points[highlight_indices] if highlight_indices else None,
            legend_text=legend_text,
            metadata=metadata
        )
        
        # Store the clipped distributions mapping for predict_from_class
        self._valid_distributions_map = {orig_idx: dist for orig_idx, dist in clipped_distributions}
        
        self._last_result = result
        return result
    
    def predict_from_class(self, class_index: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate prediction from a selected class/distribution.
        
        Args:
            class_index: Index of the selected distribution (from visible classes)
            random_state: Random seed for reproducible predictions
            
        Returns:
            Forecast values [forecast_horizon]
        """
        if not self._fitted or not self._distributions:
            raise ValueError("Must call fit_transform before prediction")
        
        # Use valid distributions map if available (after filtering)
        if hasattr(self, '_valid_distributions_map') and self._valid_distributions_map:
            if class_index not in self._valid_distributions_map:
                available_classes = list(self._valid_distributions_map.keys())
                raise ValueError(f"Invalid class index {class_index}. Available classes after filtering: {available_classes}")
            selected_dist = self._valid_distributions_map[class_index]
        else:
            # Fallback to original behavior
            if class_index < 0 or class_index >= len(self._distributions):
                raise ValueError(f"Invalid class index {class_index}. Available: 0 to {len(self._distributions)-1}")
            selected_dist = self._distributions[class_index]
        
        last_value = self._training_data[-1]
        
        forecast_horizon = self.config.extra_params.get('forecast_horizon', self._forecast_horizon)
        raw_forecast = selected_dist.forecast_sequence(forecast_horizon, last_value, random_state)
        
        # Apply the same clipping used during visualization
        clipped_forecast = self._clip_forecast_to_training_bounds(raw_forecast, self._training_data)
        
        return clipped_forecast
    
    def evaluate_training_fit(self) -> Dict[str, Any]:
        """
        Evaluate how well each distribution fits the training data.
        
        This reveals why T-distributions appear smoother than training data:
        they model increments, not absolute patterns.
        
        Returns:
            Dictionary with fit statistics for each distribution
        """
        if not self._fitted or not self._distributions:
            raise ValueError("Must call fit_transform before evaluation")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        results = {}
        training_data = self._training_data
        training_length = len(training_data) - 1  # -1 because we predict from 2nd point onward
        
        for i, dist in enumerate(self._distributions):
            # Generate prediction for the entire training sequence
            # Start from first training point, predict the rest
            train_prediction = dist.forecast_sequence(
                training_length, 
                training_data[0], 
                random_state=42
            )
            
            # Compare against actual training data (excluding first point)
            actual_training = training_data[1:]
            
            # Calculate metrics
            mse = mean_squared_error(actual_training, train_prediction)
            mae = mean_absolute_error(actual_training, train_prediction)
            
            # Calculate volatility metrics
            actual_volatility = np.std(np.diff(actual_training))
            predicted_volatility = np.std(np.diff(train_prediction))
            
            # Calculate increment statistics
            actual_increments = np.diff(training_data)
            predicted_increments = np.diff(train_prediction)
            
            increment_mse = mean_squared_error(actual_increments[1:], predicted_increments)
            increment_mae = mean_absolute_error(actual_increments[1:], predicted_increments)
            
            # Handle both simple and mixture distributions
            if hasattr(dist, 'df'):  # Simple StudentTDistribution
                distribution_params = {
                    'type': 'simple',
                    'df': dist.df,
                    'loc': dist.loc,
                    'scale': dist.scale
                }
            elif hasattr(dist, 'components') and hasattr(dist.components[0], 'df'):  # ValueMixtureDistribution
                distribution_params = {
                    'type': 'mixture',
                    'n_components': dist.n_components,
                    'components': [
                        {
                            'df': comp.df,
                            'loc': comp.loc,
                            'scale': comp.scale,
                            'weight': comp.weight
                        }
                        for comp in dist.components
                    ]
                }
            else:  # AdaptiveMixtureDistribution
                distribution_params = {
                    'type': 'adaptive',
                    'n_components': dist.n_components,
                    'components': [
                        {
                            'df': comp.pattern.df,
                            'loc': comp.pattern.loc,
                            'scale': comp.pattern.scale,
                            'weight': comp.weight,
                            'distance': comp.distance
                        }
                        for comp in dist.components
                    ]
                }
            
            results[f"distribution_{i}"] = {
                'name': dist.name,
                'n_keypoints': len(getattr(dist, 'keypoints', getattr(dist, 'current_window', []))),
                'distribution_params': distribution_params,
                'value_fit': {
                    'mse': mse,
                    'mae': mae,
                    'mse_vs_test_ratio': None  # Will be filled later if test metrics available
                },
                'volatility_comparison': {
                    'actual_volatility': actual_volatility,
                    'predicted_volatility': predicted_volatility,
                    'volatility_ratio': predicted_volatility / actual_volatility if actual_volatility > 0 else np.inf
                },
                'increment_fit': {
                    'increment_mse': increment_mse,
                    'increment_mae': increment_mae,
                    'actual_increment_std': np.std(actual_increments),
                    'predicted_increment_std': np.std(predicted_increments)
                },
                'training_prediction': train_prediction.tolist(),
                'keypoint_indices': [
                    int(idx) for idx in range(0, len(training_data), max(1, len(training_data) // len(getattr(dist, 'keypoints', getattr(dist, 'current_window', [training_data[-1]])))))
                ][:len(getattr(dist, 'keypoints', getattr(dist, 'current_window', [training_data[-1]])))] if len(getattr(dist, 'keypoints', getattr(dist, 'current_window', [training_data[-1]]))) < len(training_data) else list(range(len(training_data)))
            }
        
        # Add summary statistics
        results['summary'] = {
            'training_data_stats': {
                'mean': np.mean(training_data),
                'std': np.std(training_data),
                'min': np.min(training_data),
                'max': np.max(training_data),
                'length': len(training_data)
            },
            'training_increment_stats': {
                'mean': np.mean(np.diff(training_data)),
                'std': np.std(np.diff(training_data)),
                'min': np.min(np.diff(training_data)),
                'max': np.max(np.diff(training_data))
            },
            'n_distributions': len(self._distributions)
        }
        
        return results
    
    def get_class_names(self) -> List[str]:
        """Get the names of all fitted distribution classes."""
        if not self._distributions:
            return []
        return [f"C{i}: {dist.name}" for i, dist in enumerate(self._distributions)]