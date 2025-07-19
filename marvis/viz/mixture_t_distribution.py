"""
Enhanced Student-T Mixture Model for Time Series Distribution Visualization.

This module implements a mixture of Student-T distributions inspired by the TOTO paper
and codebase, providing much more flexible modeling of time series increments
than single Student-T distributions.

Based on:
- TOTO paper: 2407.07874v2.pdf 
- TOTO codebase: /toto/toto/model/distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.stats import t as student_t
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StudentTMixtureComponent:
    """Single component of a Student-T mixture distribution."""
    df: float  # degrees of freedom (>2.0)
    loc: float  # location parameter (mean)
    scale: float  # scale parameter (>0)
    weight: float  # mixture weight (0-1)
    name: str  # component name
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from this component."""
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
class StudentTMixtureDistribution:
    """Mixture of Student's T distributions for more flexible time series modeling."""
    components: List[StudentTMixtureComponent]
    keypoints: np.ndarray
    name: str
    
    def __post_init__(self):
        """Validate mixture components."""
        # Normalize weights to sum to 1
        total_weight = sum(comp.weight for comp in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight /= total_weight
        
        # Ensure we have at least one component
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
        """Sample from the mixture distribution."""
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
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute mixture log probability density function."""
        # Use log-sum-exp trick for numerical stability
        log_pdfs = np.zeros((len(self.components), len(x)))
        for i, component in self.components:
            log_pdfs[i] = np.log(component.weight) + component.log_pdf(x)
        
        # Log-sum-exp
        max_log_pdf = np.max(log_pdfs, axis=0)
        return max_log_pdf + np.log(np.sum(np.exp(log_pdfs - max_log_pdf), axis=0))
    
    def forecast_sequence(self, length: int, last_value: float, random_state: Optional[int] = None) -> np.ndarray:
        """Generate a forecast sequence using the mixture distribution."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate increments from the mixture
        increments = self.sample(length, random_state)
        
        # Create sequence starting from last observed value
        sequence = np.zeros(length + 1)
        sequence[0] = last_value
        
        for i in range(length):
            sequence[i + 1] = sequence[i] + increments[i]
        
        return sequence[1:]  # Return without initial value


class StudentTMixtureFitter:
    """Fit mixture of Student-T distributions to time series increment data."""
    
    def __init__(self, max_components: int = 5, min_components: int = 1):
        """Initialize the mixture fitter.
        
        Args:
            max_components: Maximum number of mixture components to consider
            min_components: Minimum number of mixture components  
        """
        self.max_components = max_components
        self.min_components = min_components
    
    def _fit_single_component(self, increments: np.ndarray) -> Dict[str, float]:
        """Fit a single Student-T component to increments."""
        if len(increments) < 2:
            return {'df': 3.0, 'loc': 0.0, 'scale': 1.0, 'weight': 1.0}
        
        if np.std(increments) == 0:
            return {
                'df': 3.0, 
                'loc': np.mean(increments) if len(increments) > 0 else 0.0, 
                'scale': 1.0,
                'weight': 1.0
            }
        
        # Method of moments initial estimates
        sample_mean = np.mean(increments)
        sample_std = np.std(increments)
        
        # Initial parameters
        df_init = 4.0
        scale_init = sample_std * np.sqrt((df_init - 2) / df_init)
        loc_init = sample_mean
        
        def neg_log_likelihood(params):
            df, loc, scale = params
            if df <= 2.0 or scale <= 0:
                return np.inf
            try:
                log_pdf = student_t.logpdf(increments, df=df, loc=loc, scale=scale)
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
    
    def _fit_mixture_em(self, increments: np.ndarray, n_components: int, max_iter: int = 100, tol: float = 1e-4) -> List[Dict[str, float]]:
        """Fit mixture using Expectation-Maximization algorithm."""
        n_data = len(increments)
        
        # Initialize components using k-means-like approach
        components = []
        
        # Initialize with quantile-based means
        if n_components == 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0.1, 0.9, n_components)
        
        for i in range(n_components):
            # Initialize around different quantiles
            center = np.quantile(increments, quantiles[i])
            subset_mask = np.abs(increments - center) <= np.std(increments)
            subset = increments[subset_mask] if np.any(subset_mask) else increments
            
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
                    log_pdf = student_t.logpdf(increments, df=comp['df'], loc=comp['loc'], scale=comp['scale'])
                    responsibilities[:, j] = comp['weight'] * np.exp(log_pdf)
                except:
                    responsibilities[:, j] = 1e-10  # Avoid numerical issues
            
            # Normalize responsibilities
            row_sums = np.sum(responsibilities, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-10  # Avoid division by zero
            responsibilities /= row_sums
            
            # M-step: update parameters
            new_components = []
            for j in range(n_components):
                resp_j = responsibilities[:, j]
                weight_j = np.mean(resp_j)
                
                if weight_j < 1e-10:  # Component has no support
                    # Reinitialize this component
                    comp_params = self._fit_single_component(increments)
                    comp_params['weight'] = 1e-3
                    new_components.append(comp_params)
                    continue
                
                # Weighted data for this component
                weighted_mean = np.sum(resp_j * increments) / np.sum(resp_j)
                weighted_var = np.sum(resp_j * (increments - weighted_mean)**2) / np.sum(resp_j)
                weighted_std = np.sqrt(max(weighted_var, 1e-6))
                
                # Update component parameters
                new_comp = {
                    'df': max(2.1, components[j]['df']),  # Keep df from previous iteration
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
                    abs(new_comp['loc'] - old_comp['loc']),
                    abs(new_comp['scale'] - old_comp['scale']),
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
    
    def _compute_aic_bic(self, increments: np.ndarray, components: List[Dict[str, float]]) -> Tuple[float, float]:
        """Compute AIC and BIC for model selection."""
        n_data = len(increments)
        n_params = len(components) * 4 - 1  # 4 params per component minus 1 weight constraint
        
        # Compute log-likelihood
        log_likelihood = 0.0
        for x in increments:
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
    
    def fit_best_mixture(self, increments: np.ndarray, selection_criterion: str = 'bic') -> List[Dict[str, float]]:
        """Fit mixture with optimal number of components using model selection.
        
        Args:
            increments: Time series increment data
            selection_criterion: 'aic' or 'bic' for model selection
            
        Returns:
            List of fitted component parameters
        """
        if len(increments) < 3:
            logger.warning("Too few increments for mixture fitting, using single component")
            return [self._fit_single_component(increments)]
        
        best_components = None
        best_score = np.inf
        best_n_components = 1
        
        # Try different numbers of components
        for n_comp in range(self.min_components, min(self.max_components + 1, len(increments) // 2)):
            try:
                components = self._fit_mixture_em(increments, n_comp)
                aic, bic = self._compute_aic_bic(increments, components)
                
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
            best_components = [self._fit_single_component(increments)]
            best_n_components = 1
        
        logger.info(f"Selected {best_n_components} components with {selection_criterion.upper()}={best_score:.2f}")
        return best_components
    
    def create_mixture_distribution(
        self, 
        increments: np.ndarray, 
        keypoints: np.ndarray, 
        name: str,
        selection_criterion: str = 'bic'
    ) -> StudentTMixtureDistribution:
        """Create a fitted mixture distribution.
        
        Args:
            increments: Time series increment data to fit
            keypoints: Keypoints used for fitting
            name: Name for the distribution
            selection_criterion: Model selection criterion ('aic' or 'bic')
            
        Returns:
            Fitted mixture distribution
        """
        # Fit mixture components
        component_params = self.fit_best_mixture(increments, selection_criterion)
        
        # Create component objects
        components = []
        for i, params in enumerate(component_params):
            component = StudentTMixtureComponent(
                df=params['df'],
                loc=params['loc'],
                scale=params['scale'],
                weight=params['weight'],
                name=f"Component_{i+1}"
            )
            components.append(component)
        
        return StudentTMixtureDistribution(
            components=components,
            keypoints=keypoints,
            name=name
        )


def generate_semantic_mixture_name(distribution: StudentTMixtureDistribution, keypoints: np.ndarray) -> str:
    """Generate semantic name for mixture distribution based on characteristics."""
    n_comp = distribution.n_components
    
    # Analyze trend from keypoints
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
    
    # Analyze mixture complexity
    if n_comp == 1:
        complexity_desc = "Simple"
    elif n_comp <= 3:
        complexity_desc = "Moderate Mixture"
    else:
        complexity_desc = "Complex Mixture"
    
    # Analyze volatility from mixture scale parameters
    avg_scale = np.mean([comp.scale for comp in distribution.components])
    
    if avg_scale < 0.5:
        volatility_desc = "Low Volatility"
    elif avg_scale > 2.0:
        volatility_desc = "High Volatility"
    else:
        volatility_desc = "Moderate Volatility"
    
    return f"{trend_desc} Trend, {complexity_desc}, {volatility_desc}"