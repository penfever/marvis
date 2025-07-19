"""
Value-Based Student-T Mixture Model for Time Series Distribution Visualization.

This module implements mixtures of Student-T distributions fitted directly to 
actual training values (not increments), inspired by TOTO's approach.

The key insight: Instead of modeling increments and doing random walks,
we fit distributions to the actual values we want to predict and sample
directly from them to get realistic trajectories.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.stats import t as student_t
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ValueMixtureComponent:
    """Single component of a value-based Student-T mixture distribution."""
    df: float  # degrees of freedom (>2.0)
    loc: float  # location parameter (mean value)
    scale: float  # scale parameter (>0)
    weight: float  # mixture weight (0-1)
    name: str  # component name
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample values directly from this component."""
        if random_state is not None:
            np.random.seed(random_state)
        return student_t.rvs(df=self.df, loc=self.loc, scale=self.scale, size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        return student_t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density function."""
        return student_t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)


@dataclass 
class ValueMixtureDistribution:
    """Mixture of Student's T distributions for direct value modeling."""
    components: List[ValueMixtureComponent]
    keypoints: np.ndarray  # Original keypoints used for context
    name: str
    value_range: Tuple[float, float]  # (min, max) of training data
    
    def __post_init__(self):
        """Validate mixture components."""
        # Normalize weights to sum to 1
        total_weight = sum(comp.weight for comp in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight /= total_weight
        
        if not self.components:
            raise ValueError("Mixture must have at least one component")
    
    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return len(self.components)
    
    @property
    def weights(self) -> np.ndarray:
        """Mixture weights as numpy array."""
        return np.array([comp.weight for comp in self.components])
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample values directly from the mixture distribution."""
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
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute mixture probability density function."""
        pdf_values = np.zeros_like(x)
        for component in self.components:
            pdf_values += component.weight * component.pdf(x)
        return pdf_values
    
    def forecast_sequence(self, length: int, last_value: float, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate a forecast sequence by sampling directly from the value distribution.
        
        This is fundamentally different from random walks - we sample actual values
        that should be in the realistic range of the training data.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample directly from the value mixture distribution
        forecast_values = self.sample(length, random_state)
        
        return forecast_values


class ValueMixtureFitter:
    """Fit mixture of Student-T distributions directly to time series values."""
    
    def __init__(self, max_components: int = 5, min_components: int = 1):
        """Initialize the value mixture fitter.
        
        Args:
            max_components: Maximum number of mixture components to consider
            min_components: Minimum number of mixture components  
        """
        self.max_components = max_components
        self.min_components = min_components
    
    def _fit_single_component(self, values: np.ndarray) -> Dict[str, float]:
        """Fit a single Student-T component to values."""
        if len(values) < 2:
            return {'df': 3.0, 'loc': 0.0, 'scale': 1.0, 'weight': 1.0}
        
        if np.std(values) == 0:
            return {
                'df': 3.0, 
                'loc': np.mean(values) if len(values) > 0 else 0.0, 
                'scale': 1.0,
                'weight': 1.0
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
            'scale': max(1e-6, scale_opt),
            'weight': 1.0
        }
    
    def _fit_mixture_em(self, values: np.ndarray, n_components: int, max_iter: int = 100, tol: float = 1e-4) -> List[Dict[str, float]]:
        """Fit mixture using Expectation-Maximization algorithm."""
        n_data = len(values)
        
        # Initialize components using quantile-based centers
        components = []
        
        if n_components == 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0.1, 0.9, n_components)
        
        for i in range(n_components):
            # Initialize around different quantiles of the VALUE distribution
            center = np.quantile(values, quantiles[i])
            
            # Create subset around this quantile
            distances = np.abs(values - center)
            subset_mask = distances <= np.percentile(distances, 70)  # Take 70% closest points
            subset = values[subset_mask] if np.any(subset_mask) else values
            
            # Fit single component to subset
            comp_params = self._fit_single_component(subset)
            comp_params['weight'] = 1.0 / n_components
            components.append(comp_params)
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            responsibilities = np.zeros((n_data, n_components))
            
            for j, comp in enumerate(components):
                try:
                    log_pdf = student_t.logpdf(values, df=comp['df'], loc=comp['loc'], scale=comp['scale'])
                    responsibilities[:, j] = comp['weight'] * np.exp(log_pdf)
                except:
                    responsibilities[:, j] = 1e-10
            
            # Normalize responsibilities
            row_sums = np.sum(responsibilities, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-10
            responsibilities /= row_sums
            
            # M-step: update parameters
            new_components = []
            for j in range(n_components):
                resp_j = responsibilities[:, j]
                weight_j = np.mean(resp_j)
                
                if weight_j < 1e-10:
                    # Reinitialize this component
                    comp_params = self._fit_single_component(values)
                    comp_params['weight'] = 1e-3
                    new_components.append(comp_params)
                    continue
                
                # Weighted statistics for this component
                weighted_mean = np.sum(resp_j * values) / np.sum(resp_j)
                weighted_var = np.sum(resp_j * (values - weighted_mean)**2) / np.sum(resp_j)
                weighted_std = np.sqrt(max(weighted_var, 1e-6))
                
                # Update component parameters
                new_comp = {
                    'df': max(2.1, components[j]['df']),  # Keep df stable
                    'loc': weighted_mean,
                    'scale': max(1e-6, weighted_std),
                    'weight': weight_j
                }
                new_components.append(new_comp)
            
            # Check convergence
            converged = True
            for j in range(n_components):
                old_comp = components[j]
                new_comp = new_components[j]
                
                param_changes = [
                    abs(new_comp['loc'] - old_comp['loc']) / (abs(old_comp['loc']) + 1e-6),
                    abs(new_comp['scale'] - old_comp['scale']) / (abs(old_comp['scale']) + 1e-6),
                    abs(new_comp['weight'] - old_comp['weight'])
                ]
                
                if any(change > tol for change in param_changes):
                    converged = False
                    break
            
            components = new_components
            
            if converged:
                logger.debug(f"EM converged after {iteration + 1} iterations")
                break
        
        # Normalize weights
        total_weight = sum(comp['weight'] for comp in components)
        if total_weight > 0:
            for comp in components:
                comp['weight'] /= total_weight
        
        return components
    
    def _compute_aic_bic(self, values: np.ndarray, components: List[Dict[str, float]]) -> Tuple[float, float]:
        """Compute AIC and BIC for model selection."""
        n_data = len(values)
        n_params = len(components) * 4 - 1  # 4 params per component minus 1 weight constraint
        
        # Compute log-likelihood
        log_likelihood = 0.0
        for x in values:
            mixture_prob = 0.0
            for comp in components:
                try:
                    comp_prob = comp['weight'] * student_t.pdf(x, df=comp['df'], loc=comp['loc'], scale=comp['scale'])
                    mixture_prob += comp_prob
                except:
                    comp_prob = 1e-10
                    mixture_prob += comp_prob
            
            log_likelihood += np.log(max(mixture_prob, 1e-10))
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_data) * n_params - 2 * log_likelihood
        
        return aic, bic
    
    def fit_best_mixture(self, values: np.ndarray, selection_criterion: str = 'bic') -> List[Dict[str, float]]:
        """Fit mixture with optimal number of components using model selection.
        
        Args:
            values: Time series values to fit to
            selection_criterion: 'aic' or 'bic' for model selection
            
        Returns:
            List of fitted component parameters
        """
        if len(values) < 3:
            logger.warning("Too few values for mixture fitting, using single component")
            return [self._fit_single_component(values)]
        
        best_components = None
        best_score = np.inf
        best_n_components = 1
        
        # Try different numbers of components
        max_reasonable_components = min(self.max_components, len(values) // 5, 5)  # Reasonable upper bound
        
        for n_comp in range(self.min_components, max_reasonable_components + 1):
            try:
                components = self._fit_mixture_em(values, n_comp)
                aic, bic = self._compute_aic_bic(values, components)
                
                score = bic if selection_criterion == 'bic' else aic
                
                logger.debug(f"n_components={n_comp}, AIC={aic:.2f}, BIC={bic:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_components = components
                    best_n_components = n_comp
                    
            except Exception as e:
                logger.warning(f"Failed to fit {n_comp} components: {e}")
                continue
        
        if best_components is None:
            logger.warning("All mixture fitting failed, using single component fallback")
            best_components = [self._fit_single_component(values)]
            best_n_components = 1
        
        logger.info(f"Selected {best_n_components} components with {selection_criterion.upper()}={best_score:.2f}")
        return best_components
    
    def create_value_mixture_distribution(
        self, 
        values: np.ndarray, 
        keypoints: np.ndarray, 
        name: str,
        selection_criterion: str = 'bic'
    ) -> ValueMixtureDistribution:
        """Create a fitted value mixture distribution.
        
        Args:
            values: Time series values to fit to
            keypoints: Keypoints used for context (not fitting)
            name: Name for the distribution
            selection_criterion: Model selection criterion ('aic' or 'bic')
            
        Returns:
            Fitted value mixture distribution
        """
        # Fit mixture components to actual values
        component_params = self.fit_best_mixture(values, selection_criterion)
        
        # Create component objects
        components = []
        for i, params in enumerate(component_params):
            component = ValueMixtureComponent(
                df=params['df'],
                loc=params['loc'],
                scale=params['scale'],
                weight=params['weight'],
                name=f"ValueComp_{i+1}"
            )
            components.append(component)
        
        return ValueMixtureDistribution(
            components=components,
            keypoints=keypoints,
            name=name,
            value_range=(np.min(values), np.max(values))
        )


def generate_semantic_value_mixture_name(distribution: ValueMixtureDistribution, keypoints: np.ndarray) -> str:
    """Generate semantic name for value mixture distribution."""
    n_comp = distribution.n_components
    
    # Analyze value range and central tendency
    component_locs = [comp.loc for comp in distribution.components]
    avg_loc = np.mean(component_locs)
    value_min, value_max = distribution.value_range
    value_center = (value_min + value_max) / 2
    
    # Classify central tendency relative to training range
    if avg_loc < value_center - 0.2 * (value_max - value_min):
        level_desc = "Low-Value"
    elif avg_loc > value_center + 0.2 * (value_max - value_min):
        level_desc = "High-Value"
    else:
        level_desc = "Mid-Value"
    
    # Analyze mixture complexity
    if n_comp == 1:
        complexity_desc = "Focused"
    elif n_comp <= 3:
        complexity_desc = "Multi-Modal" 
    else:
        complexity_desc = "Complex"
    
    # Analyze spread from component scales
    avg_scale = np.mean([comp.scale for comp in distribution.components])
    value_range = value_max - value_min
    relative_scale = avg_scale / value_range if value_range > 0 else 0
    
    if relative_scale < 0.1:
        spread_desc = "Narrow"
    elif relative_scale > 0.3:
        spread_desc = "Wide"
    else:
        spread_desc = "Moderate"
    
    return f"{level_desc} {complexity_desc} {spread_desc} Distribution"