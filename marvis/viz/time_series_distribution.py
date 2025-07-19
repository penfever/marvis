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
        
        # Fitted distributions
        self._distributions: List[StudentTDistribution] = []
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
    
    def _select_keypoints(self, data: np.ndarray, strategy: str = 'uniform') -> np.ndarray:
        """
        Select keypoints from time series data for distribution fitting.
        
        Args:
            data: Time series values [n_timesteps]
            strategy: Keypoint selection strategy
            
        Returns:
            Indices of selected keypoints
        """
        n_points = len(data)
        n_keypoints = min(self._n_keypoints, n_points)
        
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
                
        elif strategy == 'changepoints':
            # Use simple changepoint detection
            # Compute differences
            diffs = np.diff(data)
            
            # Find points where trend changes significantly
            change_scores = np.abs(np.diff(diffs))
            
            if len(change_scores) > 0:
                # Select top change points
                top_changes = np.argsort(change_scores)[-min(n_keypoints-2, len(change_scores)):]
                # Add 1 to account for diff operation
                indices = np.concatenate([[0], top_changes + 1, [n_points - 1]])
                indices = np.sort(np.unique(indices))
            else:
                # Fallback to uniform
                indices = np.linspace(0, n_points - 1, n_keypoints, dtype=int)
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
        
        # Strategy 1: Different keypoint strategies
        strategies = ['uniform', 'extrema']
        if len(data) > 10:
            strategies.append('changepoints')
        
        for strategy in strategies[:min(3, self._n_distributions)]:
            try:
                keypoint_indices = self._select_keypoints(data, strategy)
                keypoints = data[keypoint_indices]
                params = self._fit_student_t_to_keypoints(keypoints)
                name = self._generate_semantic_name(params, keypoints)
                
                dist = StudentTDistribution(
                    df=params['df'],
                    loc=params['loc'],
                    scale=params['scale'],
                    keypoints=keypoints,
                    name=f"{strategy.capitalize()}: {name}"
                )
                distributions.append(dist)
            except Exception as e:
                logger.warning(f"Failed to fit distribution with strategy {strategy}: {e}")
        
        # Strategy 2: Different parameter variations for remaining slots
        if len(distributions) < self._n_distributions:
            # Use the best distribution so far and create variations
            base_dist = distributions[0] if distributions else None
            
            if base_dist is not None:
                for i in range(self._n_distributions - len(distributions)):
                    # Create variations by adjusting parameters
                    variation_factor = 1.0 + 0.3 * (i + 1)  # 1.3, 1.6, 1.9, ...
                    
                    # Vary scale (volatility)
                    new_scale = base_dist.scale * variation_factor
                    new_name = f"Variation {i+1}: High Volatility Pattern"
                    
                    dist = StudentTDistribution(
                        df=base_dist.df,
                        loc=base_dist.loc,
                        scale=new_scale,
                        keypoints=base_dist.keypoints,
                        name=new_name
                    )
                    distributions.append(dist)
        
        # Ensure we have at least one distribution
        if not distributions:
            # Create a default distribution
            mean_increment = np.mean(np.diff(data)) if len(data) > 1 else 0.0
            std_increment = np.std(np.diff(data)) if len(data) > 1 else 1.0
            
            dist = StudentTDistribution(
                df=4.0,
                loc=mean_increment,
                scale=std_increment,
                keypoints=data[::max(1, len(data)//4)],
                name="Default: Random Walk Pattern"
            )
            distributions.append(dist)
        
        return distributions[:self._n_distributions]
    
    def _is_forecast_reasonable(self, forecast: np.ndarray, historical_data: np.ndarray) -> bool:
        """
        Check if a forecast trajectory is reasonable and not erratic.
        
        Args:
            forecast: Forecast sequence to validate
            historical_data: Historical time series data for context
            
        Returns:
            True if forecast is reasonable, False if it should be filtered out
        """
        if len(forecast) == 0:
            return False
        
        # Calculate historical statistics for reference
        hist_mean = np.mean(historical_data)
        hist_std = np.std(historical_data)
        hist_range = np.max(historical_data) - np.min(historical_data)
        
        # Check for extreme outliers (more than 5 standard deviations from historical mean)
        outlier_threshold = 5 * hist_std
        if np.any(np.abs(forecast - hist_mean) > outlier_threshold):
            return False
        
        # Check for unrealistic volatility (forecast std > 3x historical std)
        forecast_std = np.std(forecast)
        if forecast_std > 3 * hist_std and hist_std > 0:
            return False
        
        # Check for extreme jumps between consecutive points
        if len(forecast) > 1:
            jumps = np.abs(np.diff(forecast))
            # If any jump is more than 2x the historical range, it's probably unrealistic
            if np.any(jumps > 2 * hist_range) and hist_range > 0:
                return False
        
        # Check for infinite or NaN values
        if not np.all(np.isfinite(forecast)):
            return False
        
        return True
    
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
        
        # Generate distributions
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
        # Ensure forecast_horizon has a valid value
        forecast_horizon = self._forecast_horizon or 48
        forecast_time = np.arange(forecast_start, forecast_start + forecast_horizon)
        last_value = data[-1]
        
        # Colors for distributions
        colors = plt.cm.Set1(np.linspace(0, 1, len(self._distributions)))
        
        # Plot forecast paths for each distribution
        legend_parts = []
        class_names = []
        
        valid_distributions = []
        valid_forecasts = []
        valid_colors = []
        
        for i, (dist, color) in enumerate(zip(self._distributions, colors)):
            # Generate forecast sequence
            forecast = dist.forecast_sequence(
                forecast_horizon, 
                last_value, 
                random_state=42 + i
            )
            
            # Validate forecast trajectory
            if self._is_forecast_reasonable(forecast, data):
                valid_distributions.append((i, dist))
                valid_forecasts.append(forecast)
                valid_colors.append(color)
            else:
                logger.warning(f"Filtered out erratic trajectory for distribution {i}: {dist.name}")
        
        # Plot only valid forecast paths
        for (orig_idx, dist), forecast, color in zip(valid_distributions, valid_forecasts, valid_colors):
            # Plot forecast line
            ax.plot(forecast_time, forecast, 'o-', color=color, linewidth=2, 
                    markersize=3, label=f"Class {orig_idx}: {dist.name}", alpha=0.9)
            
            # Add confidence bands if enabled
            if self._show_confidence_bands:
                # Generate multiple forecast samples for confidence band
                n_samples = 50
                forecast_samples = np.array([
                    dist.forecast_sequence(forecast_horizon, last_value, random_state=42 + orig_idx + j)
                    for j in range(n_samples)
                ])
                
                # Compute percentiles
                lower = np.percentile(forecast_samples, 25, axis=0)
                upper = np.percentile(forecast_samples, 75, axis=0)
                
                ax.fill_between(forecast_time, lower, upper, color=color, alpha=0.2)
            
            # Store class information
            class_names.append(f"Class {orig_idx}: {dist.name}")
            import matplotlib.colors as mcolors
            legend_parts.append(f"Class {orig_idx} (Color: {mcolors.to_hex(color)}): {dist.name}")
        
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
        ax.set_title(f'Time Series with {len(valid_distributions)} Distribution-Based Forecast Paths')
        ax.grid(True, alpha=0.3)
        
        # Apply symlog scale to y-axis for better handling of wide value ranges
        # This is like log scale but handles negative values and values near zero
        all_values = np.concatenate([data, *valid_forecasts]) if valid_forecasts else data
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
        for (orig_idx, dist), forecast, color in zip(valid_distributions, valid_forecasts, valid_colors):
            ax.plot(forecast_positions, forecast, 'o-', color=color, linewidth=2, 
                    markersize=3, label=f"Class {orig_idx}: {dist.name}", alpha=0.9)
        
        # Add vertical line separating history from forecast at weighted position
        ax.axvline(x=hist_weight, color='gray', linestyle='--', alpha=0.7, 
                   label='Forecast Start')
        
        # Reapply styling with weighted axes
        ax.set_xlabel('Time (Weighted: Historical 40% | Forecast 60%)')
        ax.set_ylabel('Value (Symlog Scale)')
        ax.set_title(f'Time Series with {len(valid_distributions)} Distribution-Based Forecast Paths')
        ax.grid(True, alpha=0.3)
        
        # Apply symlog scale again after clearing
        if value_range > 0:
            threshold = value_range / 100
            ax.set_yscale('symlog', linthresh=threshold)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Create legend text for VLM using valid distributions
        valid_legend_parts = []
        valid_class_names = []
        for (orig_idx, dist), color in zip(valid_distributions, valid_colors):
            import matplotlib.colors as mcolors
            valid_legend_parts.append(f"Class {orig_idx} (Color: {mcolors.to_hex(color)}): {dist.name}")
            valid_class_names.append(f"Class {orig_idx}: {dist.name}")
        
        legend_text = "Available forecast patterns:\n" + "\n".join(valid_legend_parts)
        class_names = valid_class_names
        
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
        
        # Create metadata using valid distributions
        valid_indices = [orig_idx for orig_idx, _ in valid_distributions]
        valid_dist_objects = [dist for _, dist in valid_distributions]
        
        metadata = {
            'plot_type': 'time_series_classification',
            'visible_classes': valid_indices,
            'all_classes': valid_indices,
            'class_names': class_names,
            'n_distributions': len(valid_distributions),
            'forecast_horizon': self._forecast_horizon or 48,
            'distribution_params': [
                {'df': d.df, 'loc': d.loc, 'scale': d.scale, 'name': d.name}
                for d in valid_dist_objects
            ],
            'filtered_out_count': len(self._distributions) - len(valid_distributions)
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
        
        # Store the valid distributions mapping for predict_from_class
        self._valid_distributions_map = {orig_idx: dist for orig_idx, dist in valid_distributions}
        
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
        
        forecast_horizon = self._forecast_horizon or 48
        return selected_dist.forecast_sequence(forecast_horizon, last_value, random_state)
    
    def get_class_names(self) -> List[str]:
        """Get the names of all fitted distribution classes."""
        if not self._distributions:
            return []
        return [f"Class {i}: {dist.name}" for i, dist in enumerate(self._distributions)]