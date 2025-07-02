#!/usr/bin/env python3
"""
Unified Results Management System for MARVIS

This module provides centralized, consistent results storage across all modalities (tabular, audio, vision).
Integrates with the existing resource manager to provide standardized directory structures, 
file naming conventions, and metadata handling.

Key features:
- Consistent directory structure: <base_output_dir>/<modality>/<dataset_id>/<model_name>/
- Standardized file formats with comprehensive metadata
- Modality-specific artifact support (visualizations, audio files, etc.)
- Integration with existing resource manager
- Backward compatibility with existing scripts
- Thread-safe operations for concurrent evaluations

Usage examples:
    # Basic usage
    results_manager = get_results_manager()
    results_manager.save_evaluation_results(
        model_name="tabllm",
        dataset_id="adult", 
        modality="tabular",
        results=evaluation_results,
        experiment_metadata=experiment_config
    )
    
    # With artifacts
    results_manager.save_evaluation_results(
        model_name="marvis_tsne",
        dataset_id="cifar10",
        modality="vision", 
        results=evaluation_results,
        artifacts={
            "visualizations": ["tsne_plot.png", "knn_plot.png"],
            "raw_responses": "vlm_responses.json"
        }
    )
    
    # Load previous results
    previous_results = results_manager.load_evaluation_results(
        model_name="jolt",
        dataset_id="diabetes",
        modality="tabular"
    )
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
import datetime
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from .resource_manager import get_resource_manager, MarvisResourceManager
from .json_utils import safe_json_dump, safe_json_dumps, convert_for_json_serialization

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for a machine learning experiment."""
    model_name: str
    dataset_id: str
    modality: str  # "tabular", "vision", "audio"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Dataset information
    num_samples_train: Optional[int] = None
    num_samples_test: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    task_type: Optional[str] = None  # "classification", "regression"
    
    # Model configuration
    model_config: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Experiment setup
    random_seed: Optional[int] = None
    k_shot: Optional[int] = None  # For few-shot learning
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None
    
    # Computational resources
    device: Optional[str] = None
    training_time_seconds: Optional[float] = None
    evaluation_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Version information
    marvis_version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None
    
    # Additional context
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass 
class EvaluationResults:
    """Standardized evaluation results structure."""
    # Core metrics (at least one should be present)
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    r2_score: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    
    # Classification-specific metrics
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None
    classification_report: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Per-class metrics
    per_class_accuracy: Optional[Dict[str, float]] = None
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    per_class_f1: Optional[Dict[str, float]] = None
    
    # Additional metrics
    completion_rate: Optional[float] = None  # For LLM evaluations that may fail
    prediction_time_per_sample: Optional[float] = None
    total_prediction_time: Optional[float] = None
    
    # Raw outputs (for detailed analysis)
    predictions: Optional[List[Any]] = None
    prediction_probabilities: Optional[List[List[float]]] = None
    true_labels: Optional[List[Any]] = None
    
    # Error analysis
    failed_predictions: Optional[List[int]] = None  # Indices of failed predictions
    error_messages: Optional[List[str]] = None
    
    # Model-specific outputs
    raw_responses: Optional[List[str]] = None  # For LLM models
    embeddings: Optional[List[List[float]]] = None  # For embedding-based models
    visualization_paths: Optional[List[str]] = None  # Paths to saved visualizations
    
    # Status and diagnostics
    status: str = "completed"  # "completed", "failed", "partial"
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None


@dataclass
class ResultsArtifacts:
    """Container for additional artifacts beyond core results."""
    # Visualization files
    plots: Optional[List[str]] = None  # Paths to plot files
    visualizations: Optional[Dict[str, str]] = None  # Name -> path mapping
    
    # Model outputs
    raw_outputs: Optional[str] = None  # Path to raw model outputs file
    processed_outputs: Optional[str] = None  # Path to processed outputs
    
    # Data files
    predictions_csv: Optional[str] = None
    metrics_csv: Optional[str] = None
    
    # Model artifacts
    model_checkpoint: Optional[str] = None
    embeddings_file: Optional[str] = None
    
    # Modality-specific artifacts
    audio_files: Optional[List[str]] = None  # For audio experiments
    image_files: Optional[List[str]] = None  # For vision experiments
    data_files: Optional[List[str]] = None   # For tabular experiments


class ResultsManager:
    """Unified results management system for MARVIS experiments."""
    
    def __init__(self, resource_manager: Optional[MarvisResourceManager] = None):
        self.resource_manager = resource_manager or get_resource_manager()
        self._lock = threading.Lock()
        self._supported_modalities = {"tabular", "vision", "audio"}
        
        logger.debug(f"Initialized ResultsManager with base dir: {self.resource_manager.path_resolver.get_base_dir()}")
    
    def get_results_base_dir(self) -> Path:
        """Get base directory for all results storage."""
        base_dir = self.resource_manager.path_resolver.get_base_dir() / "results"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    
    def get_experiment_dir(self, model_name: str, dataset_id: str, modality: str) -> Path:
        """
        Get standardized directory structure for an experiment.
        
        Structure: <results_base>/<modality>/<dataset_id>/<model_name>/
        """
        if modality not in self._supported_modalities:
            logger.warning(f"Unsupported modality '{modality}'. Supported: {self._supported_modalities}")
        
        experiment_dir = (
            self.get_results_base_dir() / 
            modality / 
            str(dataset_id) / 
            model_name
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    def save_evaluation_results(
        self, 
        model_name: str,
        dataset_id: str, 
        modality: str,
        results: Union[Dict[str, Any], EvaluationResults],
        experiment_metadata: Optional[Union[Dict[str, Any], ExperimentMetadata]] = None,
        artifacts: Optional[Union[Dict[str, Any], ResultsArtifacts]] = None,
        overwrite: bool = True
    ) -> Path:
        """
        Save evaluation results with standardized structure and metadata.
        
        Args:
            model_name: Name of the model (e.g., "tabllm", "marvis_tsne")
            dataset_id: Dataset identifier (e.g., "adult", "cifar10", "1590")
            modality: Experiment modality ("tabular", "vision", "audio")
            results: Evaluation results (dict or EvaluationResults object)
            experiment_metadata: Experiment metadata (dict or ExperimentMetadata object)
            artifacts: Additional artifacts (dict or ResultsArtifacts object)
            overwrite: Whether to overwrite existing results
            
        Returns:
            Path to the saved results directory
        """
        with self._lock:
            experiment_dir = self.get_experiment_dir(model_name, dataset_id, modality)
            
            # Check if results already exist
            results_file = experiment_dir / "results.json"
            if results_file.exists() and not overwrite:
                logger.warning(f"Results already exist at {results_file} and overwrite=False")
                return experiment_dir
            
            # Convert inputs to standardized formats
            if isinstance(results, dict):
                # Convert dict to EvaluationResults, preserving all fields
                results_obj = self._dict_to_evaluation_results(results)
            else:
                results_obj = results
            
            if experiment_metadata is None:
                metadata_obj = ExperimentMetadata(
                    model_name=model_name,
                    dataset_id=dataset_id,
                    modality=modality
                )
            elif isinstance(experiment_metadata, dict):
                # Merge provided metadata with required fields
                metadata_dict = {
                    'model_name': model_name,
                    'dataset_id': dataset_id, 
                    'modality': modality,
                    **experiment_metadata
                }
                metadata_obj = ExperimentMetadata(**metadata_dict)
            else:
                metadata_obj = experiment_metadata
            
            if artifacts is None:
                artifacts_obj = ResultsArtifacts()
            elif isinstance(artifacts, dict):
                artifacts_obj = self._dict_to_artifacts(artifacts)
            else:
                artifacts_obj = artifacts
            
            # Create combined results structure
            combined_results = {
                'metadata': asdict(metadata_obj),
                'results': asdict(results_obj),
                'artifacts': asdict(artifacts_obj),
                'format_version': '1.0'
            }
            
            # Save main results file
            success = safe_json_dump(
                combined_results,
                str(results_file),
                logger=logger,
                minimal_fallback=True,
                indent=2
            )
            
            if not success:
                logger.error(f"Failed to save results to {results_file}")
                return experiment_dir
            
            # Save separate metadata file for easy access
            metadata_file = experiment_dir / "metadata.json"
            safe_json_dump(
                asdict(metadata_obj),
                str(metadata_file),
                logger=logger,
                minimal_fallback=True,
                indent=2
            )
            
            # Save metrics summary for quick reference
            self._save_metrics_summary(results_obj, experiment_dir)
            
            # Handle artifacts
            if artifacts_obj and self._has_artifacts(artifacts_obj):
                self._save_artifacts_manifest(artifacts_obj, experiment_dir)
            
            logger.info(f"Successfully saved results for {model_name} on {dataset_id} to {experiment_dir}")
            return experiment_dir
    
    def load_evaluation_results(
        self, 
        model_name: str, 
        dataset_id: str, 
        modality: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results for a specific experiment.
        
        Returns:
            Combined dictionary with 'metadata', 'results', and 'artifacts' keys,
            or None if not found
        """
        experiment_dir = self.get_experiment_dir(model_name, dataset_id, modality)
        results_file = experiment_dir / "results.json"
        
        if not results_file.exists():
            logger.debug(f"No results found for {model_name} on {dataset_id} in {experiment_dir}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {results_file}: {e}")
            return None
    
    def list_experiments(
        self, 
        modality: Optional[str] = None,
        dataset_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        List all experiments matching the given criteria.
        
        Returns:
            List of experiment info dictionaries with keys: 
            'modality', 'dataset_id', 'model_name', 'path'
        """
        results_base = self.get_results_base_dir()
        experiments = []
        
        if not results_base.exists():
            return experiments
        
        # Traverse the directory structure
        for modality_dir in results_base.iterdir():
            if not modality_dir.is_dir():
                continue
            if modality and modality_dir.name != modality:
                continue
                
            for dataset_dir in modality_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                if dataset_id and dataset_dir.name != dataset_id:
                    continue
                    
                for model_dir in dataset_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    if model_name and model_dir.name != model_name:
                        continue
                    
                    # Check if this is a valid experiment (has results.json)
                    if (model_dir / "results.json").exists():
                        experiments.append({
                            'modality': modality_dir.name,
                            'dataset_id': dataset_dir.name,
                            'model_name': model_dir.name,
                            'path': str(model_dir)
                        })
        
        return sorted(experiments, key=lambda x: (x['modality'], x['dataset_id'], x['model_name']))
    
    def save_artifacts(
        self,
        model_name: str,
        dataset_id: str,
        modality: str,
        artifacts: Dict[str, Union[str, List[str]]],
        copy_files: bool = True
    ) -> Dict[str, str]:
        """
        Save additional artifacts (files, visualizations, etc.) for an experiment.
        
        Args:
            model_name: Name of the model
            dataset_id: Dataset identifier
            modality: Experiment modality
            artifacts: Dictionary mapping artifact names to file paths
            copy_files: Whether to copy files to the experiment directory
            
        Returns:
            Dictionary mapping artifact names to their final paths
        """
        experiment_dir = self.get_experiment_dir(model_name, dataset_id, modality)
        artifacts_dir = experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        final_paths = {}
        
        for artifact_name, artifact_path in artifacts.items():
            if isinstance(artifact_path, list):
                # Handle multiple files for one artifact
                final_paths[artifact_name] = []
                for i, path in enumerate(artifact_path):
                    if copy_files and os.path.exists(path):
                        dest_path = artifacts_dir / f"{artifact_name}_{i}_{Path(path).name}"
                        shutil.copy2(path, dest_path)
                        final_paths[artifact_name].append(str(dest_path))
                    else:
                        final_paths[artifact_name].append(path)
            else:
                # Single file
                if copy_files and os.path.exists(artifact_path):
                    dest_path = artifacts_dir / f"{artifact_name}_{Path(artifact_path).name}"
                    shutil.copy2(artifact_path, dest_path)
                    final_paths[artifact_name] = str(dest_path)
                else:
                    final_paths[artifact_name] = artifact_path
        
        # Update artifacts manifest
        manifest_file = artifacts_dir / "manifest.json"
        try:
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
            else:
                manifest = {}
            
            manifest.update(final_paths)
            
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update artifacts manifest: {e}")
        
        return final_paths
    
    def create_summary_report(
        self, 
        modality: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a summary report of all experiments.
        
        Args:
            modality: Filter by modality, or None for all
            output_file: Optional file to save the report
            
        Returns:
            Summary report dictionary
        """
        experiments = self.list_experiments(modality=modality)
        
        report = {
            'summary': {
                'total_experiments': len(experiments),
                'modalities': set(),
                'datasets': set(),
                'models': set(),
                'generated_at': datetime.datetime.now().isoformat()
            },
            'experiments': []
        }
        
        for exp in experiments:
            report['summary']['modalities'].add(exp['modality'])
            report['summary']['datasets'].add(exp['dataset_id'])
            report['summary']['models'].add(exp['model_name'])
            
            # Load experiment details
            exp_results = self.load_evaluation_results(
                exp['model_name'], exp['dataset_id'], exp['modality']
            )
            
            if exp_results:
                exp_summary = {
                    'modality': exp['modality'],
                    'dataset_id': exp['dataset_id'],
                    'model_name': exp['model_name'],
                    'timestamp': exp_results.get('metadata', {}).get('timestamp'),
                    'status': exp_results.get('results', {}).get('status', 'unknown')
                }
                
                # Add key metrics
                results = exp_results.get('results', {})
                if results.get('accuracy') is not None:
                    exp_summary['accuracy'] = results['accuracy']
                if results.get('r2_score') is not None:
                    exp_summary['r2_score'] = results['r2_score']
                if results.get('training_time_seconds') is not None:
                    exp_summary['training_time'] = results['training_time_seconds']
                
                report['experiments'].append(exp_summary)
        
        # Convert sets to lists for JSON serialization
        report['summary']['modalities'] = sorted(list(report['summary']['modalities']))
        report['summary']['datasets'] = sorted(list(report['summary']['datasets']))
        report['summary']['models'] = sorted(list(report['summary']['models']))
        
        if output_file:
            safe_json_dump(report, output_file, logger=logger, indent=2)
            logger.info(f"Summary report saved to {output_file}")
        
        return report
    
    def cleanup_old_results(self, days_old: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up old experiment results.
        
        Args:
            days_old: Remove results older than this many days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        results_base = self.get_results_base_dir()
        
        stats = {
            'scanned': 0,
            'marked_for_deletion': 0,
            'deleted': 0,
            'errors': 0,
            'total_size_mb': 0.0,
            'freed_size_mb': 0.0
        }
        
        experiments = self.list_experiments()
        
        for exp in experiments:
            stats['scanned'] += 1
            exp_path = Path(exp['path'])
            
            try:
                # Check modification time of results.json
                results_file = exp_path / "results.json"
                if results_file.exists():
                    mod_time = datetime.datetime.fromtimestamp(results_file.stat().st_mtime)
                    
                    if mod_time < cutoff_date:
                        stats['marked_for_deletion'] += 1
                        
                        # Calculate size
                        size_mb = sum(f.stat().st_size for f in exp_path.rglob('*') if f.is_file()) / (1024 * 1024)
                        stats['total_size_mb'] += size_mb
                        
                        if not dry_run:
                            shutil.rmtree(exp_path)
                            stats['deleted'] += 1
                            stats['freed_size_mb'] += size_mb
                            logger.info(f"Deleted old experiment: {exp['modality']}/{exp['dataset_id']}/{exp['model_name']}")
                        else:
                            logger.info(f"Would delete: {exp['modality']}/{exp['dataset_id']}/{exp['model_name']} ({size_mb:.1f} MB)")
                            
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error processing {exp_path}: {e}")
        
        logger.info(f"Cleanup completed: {stats['deleted']}/{stats['marked_for_deletion']} deleted, {stats['freed_size_mb']:.1f} MB freed")
        return stats
    
    def _dict_to_evaluation_results(self, data: Dict[str, Any]) -> EvaluationResults:
        """Convert dictionary to EvaluationResults, preserving all fields."""
        # Get all fields from EvaluationResults
        result_fields = {f.name for f in EvaluationResults.__dataclass_fields__.values()}
        
        # Extract matching fields from input data
        kwargs = {k: v for k, v in data.items() if k in result_fields}
        
        return EvaluationResults(**kwargs)
    
    def _dict_to_artifacts(self, data: Dict[str, Any]) -> ResultsArtifacts:
        """Convert dictionary to ResultsArtifacts."""
        artifact_fields = {f.name for f in ResultsArtifacts.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in artifact_fields}
        return ResultsArtifacts(**kwargs)
    
    def _has_artifacts(self, artifacts: ResultsArtifacts) -> bool:
        """Check if artifacts object contains any actual artifacts."""
        for field_name, field_value in asdict(artifacts).items():
            if field_value is not None and field_value != []:
                return True
        return False
    
    def _save_metrics_summary(self, results: EvaluationResults, experiment_dir: Path):
        """Save a quick metrics summary for easy access."""
        metrics = {}
        
        # Core metrics
        for metric in ['accuracy', 'balanced_accuracy', 'r2_score', 'mae', 'rmse']:
            value = getattr(results, metric, None)
            if value is not None:
                metrics[metric] = value
        
        # Classification metrics
        for metric in ['precision_macro', 'recall_macro', 'f1_macro', 
                      'precision_weighted', 'recall_weighted', 'f1_weighted']:
            value = getattr(results, metric, None)
            if value is not None:
                metrics[metric] = value
        
        # Timing and completion
        for metric in ['completion_rate', 'total_prediction_time', 'prediction_time_per_sample']:
            value = getattr(results, metric, None)
            if value is not None:
                metrics[metric] = value
        
        if metrics:
            summary_file = experiment_dir / "metrics_summary.json"
            safe_json_dump(metrics, str(summary_file), logger=logger, indent=2)
    
    def _save_artifacts_manifest(self, artifacts: ResultsArtifacts, experiment_dir: Path):
        """Save artifacts manifest for easy discovery."""
        artifacts_dict = asdict(artifacts)
        # Remove None values
        artifacts_dict = {k: v for k, v in artifacts_dict.items() if v is not None}
        
        if artifacts_dict:
            manifest_file = experiment_dir / "artifacts_manifest.json"
            safe_json_dump(artifacts_dict, str(manifest_file), logger=logger, indent=2)


# Global results manager instance
_results_manager: Optional[ResultsManager] = None
_results_manager_lock = threading.Lock()


def get_results_manager(resource_manager: Optional[MarvisResourceManager] = None) -> ResultsManager:
    """Get global results manager instance."""
    global _results_manager
    
    with _results_manager_lock:
        if _results_manager is None:
            _results_manager = ResultsManager(resource_manager)
        return _results_manager


def reset_results_manager():
    """Reset global results manager instance (mainly for testing)."""
    global _results_manager
    with _results_manager_lock:
        _results_manager = None


# Convenience functions for backward compatibility

def save_results_unified(
    results: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    model_name: Optional[str] = None,
    modality: str = "tabular",
    **kwargs
) -> Path:
    """
    Backward-compatible results saving function.
    
    This function provides compatibility with the existing save_results() pattern
    while using the new unified system.
    """
    if model_name is None:
        # Extract model name from results if available
        model_name = results.get('model_name', 'unknown_model')
    
    # Use the results manager but save to the specified output directory
    results_manager = get_results_manager()
    
    # Temporarily override the base directory
    original_base_dir = results_manager.resource_manager.path_resolver.get_base_dir
    
    def custom_base_dir():
        return Path(output_dir)
    
    results_manager.resource_manager.path_resolver.get_base_dir = custom_base_dir
    
    try:
        return results_manager.save_evaluation_results(
            model_name=model_name,
            dataset_id=dataset_name,
            modality=modality,
            results=results,
            **kwargs
        )
    finally:
        # Restore original base directory function
        results_manager.resource_manager.path_resolver.get_base_dir = original_base_dir