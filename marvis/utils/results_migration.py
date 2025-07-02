#!/usr/bin/env python3
"""
Results Migration Utilities

This module provides utilities to help migrate existing evaluation scripts 
to use the new unified results management system while maintaining 
backward compatibility.

Key functions:
- Auto-detect result format and convert to standardized structure
- Migrate existing result directories to new standardized structure
- Validate and repair result files
- Provide adapters for different evaluation scripts
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import datetime
import re

from .results_manager import (
    get_results_manager, 
    ExperimentMetadata, 
    EvaluationResults, 
    ResultsArtifacts
)
from .json_utils import safe_json_dump, convert_for_json_serialization

logger = logging.getLogger(__name__)


class ResultsFormatDetector:
    """Detects and standardizes different result formats used across MARVIS."""
    
    @staticmethod
    def detect_format(result_dict: Dict[str, Any]) -> str:
        """
        Detect the format of a result dictionary.
        
        Returns:
            Format type: "tabular_llm", "vision_marvis", "audio_marvis", "legacy", "unknown"
        """
        # Check for specific format indicators
        if "model_name" in result_dict and "dataset_name" in result_dict:
            if "accuracy" in result_dict and "completion_rate" in result_dict:
                return "tabular_llm"
            elif "visualizations_saved" in result_dict or "tsne_plot" in result_dict:
                return "vision_marvis"
            elif "embedding_model" in result_dict or "audio_duration" in result_dict:
                return "audio_marvis"
        
        # Check for legacy format
        if "accuracy" in result_dict or "r2_score" in result_dict:
            return "legacy"
            
        return "unknown"
    
    @staticmethod
    def extract_metadata(result_dict: Dict[str, Any], format_type: str) -> ExperimentMetadata:
        """Extract metadata from result dictionary based on format."""
        
        # Common fields
        model_name = result_dict.get('model_name', 'unknown_model')
        dataset_id = result_dict.get('dataset_id') or result_dict.get('dataset_name', 'unknown_dataset')
        
        # Detect modality from format or content
        if format_type == "tabular_llm":
            modality = "tabular"
        elif format_type == "vision_marvis":
            modality = "vision"
        elif format_type == "audio_marvis":
            modality = "audio"
        else:
            # Try to infer from model name or other indicators
            model_lower = model_name.lower()
            if any(x in model_lower for x in ['tabllm', 'jolt', 'tabula']):
                modality = "tabular"
            elif any(x in model_lower for x in ['marvis_tsne', 'dinov2', 'vlm', 'vision']):
                modality = "vision"
            elif any(x in model_lower for x in ['whisper', 'clap', 'audio']):
                modality = "audio"
            else:
                modality = "tabular"  # Default fallback
        
        metadata = ExperimentMetadata(
            model_name=model_name,
            dataset_id=str(dataset_id),
            modality=modality
        )
        
        # Extract additional fields based on format
        if format_type == "tabular_llm":
            metadata.num_samples_test = result_dict.get('num_test_samples')
            metadata.num_classes = result_dict.get('num_classes')
            metadata.class_names = result_dict.get('class_names')
            metadata.k_shot = result_dict.get('k_shot')
            metadata.training_time_seconds = result_dict.get('training_time')
            metadata.evaluation_time_seconds = result_dict.get('prediction_time')
            
        elif format_type in ["vision_marvis", "audio_marvis"]:
            config = result_dict.get('config', {})
            metadata.model_config = config
            metadata.training_time_seconds = result_dict.get('training_time')
            metadata.evaluation_time_seconds = result_dict.get('prediction_time')
            
            if 'dataset_info' in result_dict:
                dataset_info = result_dict['dataset_info']
                metadata.num_samples_train = dataset_info.get('train_samples')
                metadata.num_samples_test = dataset_info.get('test_samples')
                metadata.num_classes = dataset_info.get('num_classes')
                metadata.class_names = dataset_info.get('class_names')
                metadata.k_shot = dataset_info.get('k_shot')
        
        # Common fields
        metadata.random_seed = result_dict.get('seed') or result_dict.get('random_seed')
        metadata.use_wandb = result_dict.get('use_wandb', False)
        metadata.wandb_run_id = result_dict.get('wandb_run_id')
        
        return metadata
    
    @staticmethod
    def extract_results(result_dict: Dict[str, Any], format_type: str) -> EvaluationResults:
        """Extract standardized results from result dictionary."""
        
        results = EvaluationResults()
        
        # Core metrics
        results.accuracy = result_dict.get('accuracy')
        results.balanced_accuracy = result_dict.get('balanced_accuracy')
        results.r2_score = result_dict.get('r2_score')
        results.mae = result_dict.get('mae')
        results.rmse = result_dict.get('rmse')
        
        # Classification metrics
        results.precision_macro = result_dict.get('precision_macro')
        results.recall_macro = result_dict.get('recall_macro')
        results.f1_macro = result_dict.get('f1_macro')
        results.precision_weighted = result_dict.get('precision_weighted')
        results.recall_weighted = result_dict.get('recall_weighted')
        results.f1_weighted = result_dict.get('f1_weighted')
        
        # Classification report and confusion matrix
        results.classification_report = result_dict.get('classification_report')
        results.confusion_matrix = result_dict.get('confusion_matrix')
        
        # Additional metrics
        results.completion_rate = result_dict.get('completion_rate')
        results.total_prediction_time = result_dict.get('prediction_time')
        results.prediction_time_per_sample = result_dict.get('prediction_time_per_sample')
        
        # Model-specific outputs
        results.predictions = result_dict.get('predictions')
        results.prediction_probabilities = result_dict.get('prediction_probabilities')
        results.true_labels = result_dict.get('true_labels')
        results.raw_responses = result_dict.get('raw_responses')
        results.embeddings = result_dict.get('embeddings')
        
        # Handle errors and status
        if 'error' in result_dict:
            results.status = "failed"
            results.error_message = result_dict['error']
        elif result_dict.get('timeout', False):
            results.status = "failed"
            results.error_message = "Evaluation timed out"
        else:
            results.status = "completed"
        
        # Handle visualization paths for vision/audio models
        if format_type in ["vision_marvis", "audio_marvis"]:
            vis_paths = []
            if result_dict.get('visualizations_saved'):
                output_dir = result_dict.get('output_directory')
                if output_dir:
                    vis_paths.append(f"{output_dir}/visualizations")
            results.visualization_paths = vis_paths if vis_paths else None
        
        return results
    
    @staticmethod
    def extract_artifacts(result_dict: Dict[str, Any], format_type: str) -> ResultsArtifacts:
        """Extract artifacts from result dictionary."""
        
        artifacts = ResultsArtifacts()
        
        if format_type in ["vision_marvis", "audio_marvis"]:
            # Handle visualization artifacts
            if result_dict.get('visualizations_saved'):
                output_dir = result_dict.get('output_directory')
                if output_dir:
                    artifacts.visualizations = {"main_plot": f"{output_dir}/main_visualization.png"}
            
            # Handle raw outputs
            if 'raw_responses' in result_dict:
                artifacts.raw_outputs = "raw_vlm_responses.json"
        
        # Handle modality-specific artifacts
        if format_type == "vision_marvis":
            artifacts.image_files = result_dict.get('visualization_paths')
        elif format_type == "audio_marvis":
            artifacts.audio_files = result_dict.get('spectrogram_paths')
        
        return artifacts


class ResultsMigrator:
    """Migrates existing results to the new standardized format."""
    
    def __init__(self, results_manager=None):
        self.results_manager = results_manager or get_results_manager()
        self.detector = ResultsFormatDetector()
    
    def migrate_file(
        self, 
        file_path: str, 
        target_model_name: Optional[str] = None,
        target_dataset_id: Optional[str] = None,
        target_modality: Optional[str] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Migrate a single result file to the new format.
        
        Args:
            file_path: Path to the result file to migrate
            target_model_name: Override model name detection
            target_dataset_id: Override dataset ID detection
            target_modality: Override modality detection
            dry_run: If True, only validate and report what would be done
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                result_dict = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load result file {file_path}: {e}")
            return False
        
        # Detect format
        format_type = self.detector.detect_format(result_dict)
        logger.info(f"Detected format: {format_type} for {file_path}")
        
        # Extract components
        try:
            metadata = self.detector.extract_metadata(result_dict, format_type)
            results = self.detector.extract_results(result_dict, format_type)
            artifacts = self.detector.extract_artifacts(result_dict, format_type)
            
            # Apply overrides
            if target_model_name:
                metadata.model_name = target_model_name
            if target_dataset_id:
                metadata.dataset_id = target_dataset_id
            if target_modality:
                metadata.modality = target_modality
            
            if dry_run:
                logger.info(f"Would migrate: {metadata.model_name} on {metadata.dataset_id} ({metadata.modality})")
                return True
            
            # Save using new format
            experiment_dir = self.results_manager.save_evaluation_results(
                model_name=metadata.model_name,
                dataset_id=metadata.dataset_id,
                modality=metadata.modality,
                results=results,
                experiment_metadata=metadata,
                artifacts=artifacts
            )
            
            logger.info(f"Successfully migrated {file_path} to {experiment_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            return False
    
    def migrate_directory(
        self, 
        source_dir: str, 
        pattern: str = "*_results.json",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Migrate all result files in a directory.
        
        Args:
            source_dir: Source directory containing result files
            pattern: File pattern to match (default: "*_results.json")
            dry_run: If True, only validate and report what would be done
            
        Returns:
            Migration statistics
        """
        source_path = Path(source_dir)
        stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return stats
        
        # Find all matching files
        result_files = list(source_path.rglob(pattern))
        stats['total_files'] = len(result_files)
        
        logger.info(f"Found {len(result_files)} result files to migrate")
        
        for file_path in result_files:
            try:
                # Try to extract metadata from file path
                model_name, dataset_id = self._extract_info_from_path(file_path)
                
                success = self.migrate_file(
                    str(file_path),
                    target_model_name=model_name,
                    target_dataset_id=dataset_id,
                    dry_run=dry_run
                )
                
                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(f"{file_path}: {e}")
                logger.error(f"Error migrating {file_path}: {e}")
        
        logger.info(f"Migration complete: {stats['successful']}/{stats['total_files']} successful")
        return stats
    
    def _extract_info_from_path(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Extract model name and dataset ID from file path."""
        
        # Try to extract from filename
        filename = file_path.stem
        
        # Pattern: <model_name>_results
        if filename.endswith('_results'):
            model_name = filename[:-8]  # Remove '_results'
        else:
            model_name = None
        
        # Try to extract dataset from parent directory
        dataset_id = None
        parent_name = file_path.parent.name
        
        # Pattern: dataset_<name>
        if parent_name.startswith('dataset_'):
            dataset_id = parent_name[8:]  # Remove 'dataset_'
        else:
            # Try to find dataset name in path components
            for part in file_path.parts:
                if any(part.startswith(prefix) for prefix in ['dataset_', 'data_']):
                    dataset_id = part.split('_', 1)[1] if '_' in part else part
                    break
        
        return model_name, dataset_id


def create_migration_adapters():
    """Create adapters for common evaluation script patterns."""
    
    def tabular_llm_adapter(
        model_name: str,
        dataset_name: str, 
        results: Dict[str, Any],
        args: Any = None
    ) -> Dict[str, Any]:
        """Adapter for tabular LLM evaluation scripts."""
        
        # Add modality and standardize field names
        results['modality'] = 'tabular'
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name
        
        # Extract metadata from args if available
        if args:
            results['k_shot'] = getattr(args, 'k_shot', None)
            results['num_few_shot_examples'] = getattr(args, 'num_few_shot_examples', None)
            results['max_context_length'] = getattr(args, 'max_context_length', None)
            results['seed'] = getattr(args, 'seed', None)
            results['use_wandb'] = getattr(args, 'use_wandb', False)
        
        return results
    
    def vision_marvis_adapter(
        model_name: str,
        dataset_name: str,
        results: Dict[str, Any], 
        args: Any = None
    ) -> Dict[str, Any]:
        """Adapter for vision MARVIS evaluation scripts."""
        
        results['modality'] = 'vision'
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name
        
        if args:
            results['use_3d_tsne'] = getattr(args, 'use_3d_tsne', False)
            results['use_knn_connections'] = getattr(args, 'use_knn_connections', False)
            results['knn_k'] = getattr(args, 'nn_k', 5)
            results['use_pca_backend'] = getattr(args, 'use_pca_backend', False)
            results['dinov2_model'] = getattr(args, 'dinov2_model', None)
            results['seed'] = getattr(args, 'seed', None)
        
        return results
    
    def audio_marvis_adapter(
        model_name: str,
        dataset_name: str,
        results: Dict[str, Any],
        args: Any = None
    ) -> Dict[str, Any]:
        """Adapter for audio MARVIS evaluation scripts."""
        
        results['modality'] = 'audio'
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name
        
        if args:
            results['embedding_model'] = getattr(args, 'embedding_model', 'whisper')
            results['whisper_model'] = getattr(args, 'whisper_model', None)
            results['clap_version'] = getattr(args, 'clap_version', None)
            results['k_shot'] = getattr(args, 'k_shot', None)
            results['audio_duration'] = getattr(args, 'audio_duration', None)
            results['seed'] = getattr(args, 'seed', None)
        
        return results
    
    return {
        'tabular_llm': tabular_llm_adapter,
        'vision_marvis': vision_marvis_adapter,
        'audio_marvis': audio_marvis_adapter
    }


# Convenience functions for common migration tasks

def migrate_legacy_results(
    source_dir: str,
    pattern: str = "*_results.json",
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to migrate legacy results.
    
    Args:
        source_dir: Directory containing legacy result files
        pattern: File pattern to match
        dry_run: If True, only report what would be done
        
    Returns:
        Migration statistics
    """
    migrator = ResultsMigrator()
    return migrator.migrate_directory(source_dir, pattern, dry_run)


def validate_result_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a result file and report its structure.
    
    Args:
        file_path: Path to result file
        
    Returns:
        Validation report
    """
    detector = ResultsFormatDetector()
    
    try:
        with open(file_path, 'r') as f:
            result_dict = json.load(f)
        
        format_type = detector.detect_format(result_dict)
        metadata = detector.extract_metadata(result_dict, format_type)
        results = detector.extract_results(result_dict, format_type)
        artifacts = detector.extract_artifacts(result_dict, format_type)
        
        return {
            'status': 'valid',
            'format_type': format_type,
            'model_name': metadata.model_name,
            'dataset_id': metadata.dataset_id,
            'modality': metadata.modality,
            'has_accuracy': results.accuracy is not None,
            'has_artifacts': artifacts.plots is not None or artifacts.visualizations is not None,
            'warnings': []
        }
        
    except Exception as e:
        return {
            'status': 'invalid',
            'error': str(e),
            'warnings': []
        }