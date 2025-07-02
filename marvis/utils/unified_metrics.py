#!/usr/bin/env python
"""
Unified metrics logging and naming scheme for MARVIS.

This module provides a consistent interface for logging metrics across all MARVIS scripts,
ensuring that metric names are standardized and compatible with the failure_as_random 
functionality in analyze_cc18_results_wandb.py.

The hierarchical naming scheme follows the pattern used in evaluate_llm_baselines.py:
    model/{model_name}/dataset/{dataset_name}/{metric_name}

Standard metrics supported:
- Core performance: accuracy, balanced_accuracy, f1_macro, f1_micro, f1_weighted
- Class-level: precision, recall, precision_macro, recall_macro
- Probabilistic: roc_auc, log_loss
- Dataset info: num_features, num_samples, num_classes, num_test_samples
- Execution: training_time, prediction_time, total_time, completion_rate, completed_samples
- Status: timeout, error
- Explanation: explanation_type, mean_probability, explanation_samples

Timing metric guidelines:
- training_time: Time spent in actual model training (traditional ML only)
- prediction_time: Time spent on inference/predictions (all models)
- total_time: Total execution time (training + prediction for traditional ML, prediction only for LLMs)

Usage:
    from marvis.utils.unified_metrics import MetricsLogger
    
    # Initialize for a specific model and dataset
    metrics = MetricsLogger(model_name="marvis", dataset_name="har", use_wandb=True)
    
    # Log core metrics
    metrics.log_core_metrics(accuracy=0.85, balanced_accuracy=0.84)
    
    # Log optional metrics
    metrics.log_optional_metrics(roc_auc=0.92, f1_macro=0.83)
    
    # Log dataset info
    metrics.log_dataset_info(num_features=561, num_samples=10299, num_classes=6)
    
    # Log timing (use appropriate metrics for your model type)
    metrics.log_timing(training_time=120.5, prediction_time=2.3)  # Traditional ML
    metrics.log_timing(prediction_time=45.2, total_time=45.2)     # LLMs
    
    # Log explanation metrics (for models with explanations)
    metrics.log_explanation_metrics(explanation_type="counterfactual", mean_probability=0.87)
    
    # Log all at once
    metrics.log_all_metrics(result_dict)
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Unified metrics logger that standardizes metric names and logging across MARVIS scripts.
    """
    
    # Standard metric names (these should match what analyze_cc18_results_wandb.py expects)
    CORE_METRICS = {
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'
    }
    
    OPTIONAL_METRICS = {
        'precision', 'recall', 'precision_macro', 'recall_macro', 
        'roc_auc', 'log_loss', 'confusion_matrix'
    }
    
    REGRESSION_METRICS = {
        'r2_score', 'mae', 'mse', 'rmse', 'mean_absolute_percentage_error'
    }
    
    DATASET_INFO_METRICS = {
        'num_features', 'num_samples', 'num_classes', 'num_test_samples',
        'dataset_name', 'dataset_id', 'task_id', 'task_type'
    }
    
    EXECUTION_METRICS = {
        'training_time', 'prediction_time', 'total_time', 
        'completion_rate', 'completed_samples'
    }
    
    STATUS_METRICS = {
        'timeout', 'error'
    }
    
    EXPLANATION_METRICS = {
        'explanation_type', 'mean_probability', 'explanation_samples'
    }
    
    # Metric name mappings for backward compatibility
    METRIC_ALIASES = {
        'f1_score': 'f1_macro',
        'f1': 'f1_macro',
        'auc': 'roc_auc',
        'precision_score': 'precision',
        'recall_score': 'recall'
    }
    
    def __init__(
        self, 
        model_name: str, 
        dataset_name: str, 
        use_wandb: bool = False,
        use_hierarchical: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize metrics logger for a specific model and dataset.
        
        Args:
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset being used
            use_wandb: Whether to log to Weights & Biases
            use_hierarchical: Whether to use hierarchical metric names (recommended)
            logger: Logger instance to use for logging
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_hierarchical = use_hierarchical
        self.logger = logger or logging.getLogger(__name__)
        
        # Storage for metrics
        self.logged_metrics = {}
        
        if self.use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases requested but not available. Metrics will not be logged to W&B.")
    
    def _get_metric_key(self, metric_name: str) -> str:
        """
        Get the standardized metric key for logging.
        
        Args:
            metric_name: Raw metric name
            
        Returns:
            Standardized metric key
        """
        # Apply aliases
        standardized_name = self.METRIC_ALIASES.get(metric_name, metric_name)
        
        if self.use_hierarchical:
            return f"model/{self.model_name}/dataset/{self.dataset_name}/{standardized_name}"
        else:
            return standardized_name
    
    def _log_metric(self, metric_name: str, value: Any) -> None:
        """
        Internal method to log a single metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to log
        """
        key = self._get_metric_key(metric_name)
        
        # Convert numpy types and other problematic types for JSON serialization
        from marvis.utils.json_utils import convert_for_json_serialization
        value = convert_for_json_serialization(value)
        
        self.logged_metrics[key] = value
        
        if self.use_wandb:
            wandb.log({key: value})
        
        self.logger.debug(f"Logged metric: {key} = {value}")
    
    def log_core_metrics(
        self,
        accuracy: Optional[float] = None,
        balanced_accuracy: Optional[float] = None,
        f1_macro: Optional[float] = None,
        f1_micro: Optional[float] = None,
        f1_weighted: Optional[float] = None
    ) -> None:
        """
        Log core performance metrics.
        
        Args:
            accuracy: Overall accuracy
            balanced_accuracy: Balanced accuracy (adjusts for class imbalance)
            f1_macro: Macro-averaged F1 score
            f1_micro: Micro-averaged F1 score
            f1_weighted: Weighted F1 score
        """
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_optional_metrics(
        self,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        precision_macro: Optional[float] = None,
        recall_macro: Optional[float] = None,
        roc_auc: Optional[float] = None,
        log_loss: Optional[float] = None
    ) -> None:
        """
        Log optional performance metrics.
        
        Args:
            precision: Overall precision
            recall: Overall recall
            precision_macro: Macro-averaged precision
            recall_macro: Macro-averaged recall
            roc_auc: Area under the ROC curve
            log_loss: Logarithmic loss
        """
        metrics = {
            'precision': precision,
            'recall': recall,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'roc_auc': roc_auc,
            'log_loss': log_loss
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_regression_metrics(
        self,
        r2_score: Optional[float] = None,
        mae: Optional[float] = None,
        mse: Optional[float] = None,
        rmse: Optional[float] = None,
        mean_absolute_percentage_error: Optional[float] = None
    ) -> None:
        """
        Log regression-specific metrics.
        
        Args:
            r2_score: R-squared (coefficient of determination)
            mae: Mean Absolute Error
            mse: Mean Squared Error
            rmse: Root Mean Squared Error
            mean_absolute_percentage_error: MAPE
        """
        metrics = {
            'r2_score': r2_score,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mean_absolute_percentage_error': mean_absolute_percentage_error
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_dataset_info(
        self,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
        num_classes: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        task_type: Optional[str] = None
    ) -> None:
        """
        Log dataset information metrics.
        
        Args:
            num_features: Number of features in the dataset
            num_samples: Total number of samples
            num_classes: Number of target classes (None for regression)
            num_test_samples: Number of test samples
            task_type: Type of task ('classification' or 'regression')
        """
        metrics = {
            'num_features': num_features,
            'num_samples': num_samples,
            'num_classes': num_classes,
            'num_test_samples': num_test_samples,
            'task_type': task_type
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_timing(
        self,
        training_time: Optional[float] = None,
        prediction_time: Optional[float] = None,
        total_time: Optional[float] = None
    ) -> None:
        """
        Log timing metrics.
        
        Args:
            training_time: Time spent training (seconds)
            prediction_time: Time spent on predictions (seconds)
            total_time: Total execution time (seconds)
        """
        metrics = {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_time': total_time
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_execution_metrics(
        self,
        completion_rate: Optional[float] = None,
        completed_samples: Optional[int] = None,
        timeout: Optional[bool] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log execution status metrics.
        
        Args:
            completion_rate: Rate of successful completions (0-1)
            completed_samples: Number of samples successfully processed
            timeout: Whether the execution timed out
            error: Error message if any
        """
        metrics = {
            'completion_rate': completion_rate,
            'completed_samples': completed_samples,
            'timeout': timeout,
            'error': error
        }
        
        for name, value in metrics.items():
            if value is not None:
                self._log_metric(name, value)
    
    def log_confusion_matrix(self, confusion_matrix: Optional[np.ndarray] = None) -> None:
        """
        Log confusion matrix with special handling for W&B.
        
        Args:
            confusion_matrix: Confusion matrix as numpy array
        """
        if confusion_matrix is not None:
            # For W&B, we might want to create a visualization
            if self.use_wandb:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix - {self.model_name} - {self.dataset_name}')
                    
                    key = self._get_metric_key('confusion_matrix')
                    wandb.log({key: wandb.Image(plt)})
                    plt.close()
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create confusion matrix visualization: {e}")
            
            # Store the raw matrix
            self.logged_metrics[self._get_metric_key('confusion_matrix')] = confusion_matrix
    
    def log_explanation_metrics(
        self, 
        explanation_type: Optional[str] = None,
        mean_probability: Optional[float] = None,
        explanation_samples: Optional[list] = None
    ) -> None:
        """
        Log explanation-specific metrics for models that generate explanations.
        
        Args:
            explanation_type: Type of explanation used
            mean_probability: Mean prediction probability
            explanation_samples: List of explanation samples to log as a table
        """
        if explanation_type is not None:
            self._log_metric('explanation_type', explanation_type)
        
        if mean_probability is not None:
            self._log_metric('mean_probability', mean_probability)
        
        if explanation_samples is not None and self.use_wandb:
            try:
                import pandas as pd
                key = self._get_metric_key('explanation_samples')
                wandb.log({key: wandb.Table(dataframe=pd.DataFrame(explanation_samples))})
            except Exception as e:
                self.logger.warning(f"Failed to log explanation samples: {e}")
    
    def log_custom_metric(self, metric_name: str, value: Any) -> None:
        """
        Log a custom metric that doesn't fit into standard categories.
        
        Args:
            metric_name: Name of the custom metric
            value: Value to log
        """
        self._log_metric(metric_name, value)
    
    def log_all_metrics(self, result_dict: Dict[str, Any]) -> None:
        """
        Log all metrics from a result dictionary with automatic mapping.
        
        Args:
            result_dict: Dictionary containing metric values
        """
        # Core metrics
        self.log_core_metrics(
            accuracy=result_dict.get('accuracy'),
            balanced_accuracy=result_dict.get('balanced_accuracy'),
            f1_macro=result_dict.get('f1_macro', result_dict.get('f1_score', result_dict.get('f1'))),
            f1_micro=result_dict.get('f1_micro'),
            f1_weighted=result_dict.get('f1_weighted')
        )
        
        # Optional metrics
        self.log_optional_metrics(
            precision=result_dict.get('precision'),
            recall=result_dict.get('recall'),
            precision_macro=result_dict.get('precision_macro'),
            recall_macro=result_dict.get('recall_macro'),
            roc_auc=result_dict.get('roc_auc', result_dict.get('auc')),
            log_loss=result_dict.get('log_loss')
        )
        
        # Regression metrics
        self.log_regression_metrics(
            r2_score=result_dict.get('r2_score'),
            mae=result_dict.get('mae'),
            mse=result_dict.get('mse'),
            rmse=result_dict.get('rmse'),
            mean_absolute_percentage_error=result_dict.get('mean_absolute_percentage_error')
        )
        
        # Dataset info
        self.log_dataset_info(
            num_features=result_dict.get('num_features'),
            num_samples=result_dict.get('num_samples'),
            num_classes=result_dict.get('num_classes'),
            num_test_samples=result_dict.get('num_test_samples'),
            task_type=result_dict.get('task_type')
        )
        
        # Timing
        self.log_timing(
            training_time=result_dict.get('training_time'),
            prediction_time=result_dict.get('prediction_time'),
            total_time=result_dict.get('total_time')
        )
        
        # Execution metrics
        self.log_execution_metrics(
            completion_rate=result_dict.get('completion_rate'),
            completed_samples=result_dict.get('completed_samples'),
            timeout=result_dict.get('timeout'),
            error=result_dict.get('error')
        )
        
        # Confusion matrix
        self.log_confusion_matrix(result_dict.get('confusion_matrix'))
        
        # Explanation metrics
        self.log_explanation_metrics(
            explanation_type=result_dict.get('explanation_type'),
            mean_probability=result_dict.get('mean_probability')
            # Note: explanation_samples are handled separately as they require special formatting
        )
        
        # Log any additional metrics not covered above
        all_standard_metrics = (
            self.CORE_METRICS | self.OPTIONAL_METRICS | self.REGRESSION_METRICS |
            self.DATASET_INFO_METRICS | self.EXECUTION_METRICS | 
            self.STATUS_METRICS | self.EXPLANATION_METRICS | {'confusion_matrix'}
        )
        
        for key, value in result_dict.items():
            # Convert key to JSON-safe string 
            from marvis.utils.json_utils import convert_for_json_serialization
            if not isinstance(key, str):
                key = str(convert_for_json_serialization(key))
                
            if key not in all_standard_metrics and key not in self.METRIC_ALIASES:
                # Log unknown metrics with a warning
                self.logger.debug(f"Logging non-standard metric: {key}")
                self._log_metric(key, value)
    
    def get_logged_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics that have been logged.
        
        Returns:
            Dictionary of all logged metrics
        """
        return self.logged_metrics.copy()
    
    def log_aggregated_metrics(
        self, 
        model_results: list, 
        prefix: str = "average"
    ) -> None:
        """
        Log aggregated metrics across multiple results.
        
        Args:
            model_results: List of result dictionaries
            prefix: Prefix for aggregated metric names
        """
        if not model_results:
            return
        
        # Calculate aggregated metrics
        metric_values = {}
        for result in model_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)
        
        # Log means
        for metric, values in metric_values.items():
            if values:
                mean_value = np.mean(values)
                self._log_metric(f"{prefix}_{metric}", mean_value)


# Convenience functions for quick usage
def log_metrics_for_result(
    result_dict: Dict[str, Any],
    model_name: str,
    dataset_name: str,
    use_wandb: bool = False,
    logger: Optional[logging.Logger] = None
) -> MetricsLogger:
    """
    Convenience function to log all metrics from a result dictionary.
    
    Args:
        result_dict: Dictionary containing metric values
        model_name: Name of the model
        dataset_name: Name of the dataset
        use_wandb: Whether to log to W&B
        logger: Logger instance
        
    Returns:
        MetricsLogger instance used for logging
    """
    metrics_logger = MetricsLogger(
        model_name=model_name,
        dataset_name=dataset_name,
        use_wandb=use_wandb,
        logger=logger
    )
    
    metrics_logger.log_all_metrics(result_dict)
    return metrics_logger


def get_standard_metric_names() -> Dict[str, set]:
    """
    Get all standard metric names organized by category.
    
    Returns:
        Dictionary mapping category names to sets of metric names
    """
    return {
        'core_metrics': MetricsLogger.CORE_METRICS,
        'optional_metrics': MetricsLogger.OPTIONAL_METRICS,
        'regression_metrics': MetricsLogger.REGRESSION_METRICS,
        'dataset_info_metrics': MetricsLogger.DATASET_INFO_METRICS,
        'execution_metrics': MetricsLogger.EXECUTION_METRICS,
        'status_metrics': MetricsLogger.STATUS_METRICS,
        'metric_aliases': set(MetricsLogger.METRIC_ALIASES.keys())
    }