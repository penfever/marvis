#!/usr/bin/env python
"""
Comprehensive VLM Prompting Test and Demo

This script generates 20 prompts and responses using a small QwenVL model
across various single-viz and multi-viz permutations. It saves all inputs,
outputs, and visualizations for manual inspection.

Usage:
    python tests/test_comprehensive_vlm_prompting.py --output_dir ./test_outputs --dataset_id 31
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marvis.models.marvis_tsne import MarvisTsneClassifier
from marvis.utils.vlm_prompting import create_classification_prompt, create_regression_prompt
from marvis.utils.unified_metrics import MetricsLogger
from marvis.utils.json_utils import safe_json_dump
from marvis.utils.class_name_utils import extract_class_names_from_labels
from marvis.utils.task_detection import detect_task_type, VISION_CLASSIFICATION_TASK_ID, AUDIO_CLASSIFICATION_TASK_ID
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error

# Audio/Vision dataset imports
try:
    from examples.audio.audio_datasets import ESC50Dataset, UrbanSound8KDataset, RAVDESSDataset
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    
try:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMPromptingTestSuite:
    """Comprehensive test suite for VLM prompting with various configurations."""
    
    def __init__(self, output_dir: str, task_id_or_dataset_name: str = "23", vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct", 
                 num_tests: int = None, num_samples_per_test: int = 10, backend: str = "auto", zoom_factor: float = 6.5,
                 task_type: Optional[str] = None):
        """
        Initialize the test suite.
        
        Args:
            output_dir: Directory to save outputs
            task_id_or_dataset_name: Dataset identifier - can be OpenML task ID (e.g. "23"), dataset name (e.g. "cifar10", "esc50") 
            vlm_model: VLM model to use (default: small Qwen model)
            num_tests: Number of test configurations to run (default: None, runs all available)
            num_samples_per_test: Number of test samples per configuration (default: 10)
            backend: Backend to use for VLM inference (default: auto)
            zoom_factor: Zoom factor for t-SNE visualizations (default: 6.5)
            task_type: Override task type ("classification" or "regression")
        """
        self.output_dir = Path(output_dir).resolve()
        self.task_id_or_dataset_name = task_id_or_dataset_name
        self.vlm_model = vlm_model
        self.num_tests = num_tests
        self.num_samples_per_test = num_samples_per_test
        self.backend = backend
        self.zoom_factor = zoom_factor
        self.task_type_override = task_type
        
        # Parse dataset identifier to determine modality and task_id
        self.modality, self.task_id = self._parse_dataset_identifier(task_id_or_dataset_name)
        
        # Create output directories with nested structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "responses").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "ground_truth").mkdir(exist_ok=True)
        
        # Fixed test indices for consistency across all tests
        self.test_indices = None
        
        # Store test results
        self.test_results = []
        
        # Number of responses per test (use the provided parameter)
        self.responses_per_test = num_samples_per_test
        
        # Load dataset based on modality
        self.X, self.y, self.dataset_info = self._load_dataset()
        
        # Determine task type
        self.task_type, self.task_type_method = self._determine_task_type()
        
        logger.info(f"Initialized test suite with dataset: {task_id_or_dataset_name}, modality: {self.modality}")
        logger.info(f"Dataset shape: {self.X.shape if hasattr(self.X, 'shape') else len(self.X) if isinstance(self.X, list) else 'N/A'}")
        logger.info(f"Task type: {self.task_type} (detected via: {self.task_type_method})")
        if self.task_type == 'classification':
            logger.info(f"Classes: {len(np.unique(self.y))}")
        else:
            logger.info(f"Target range: [{np.min(self.y):.3f}, {np.max(self.y):.3f}]")
    
    def _parse_dataset_identifier(self, identifier: str) -> Tuple[str, Optional[int]]:
        """
        Parse dataset identifier to determine modality and task_id.
        
        Args:
            identifier: Dataset identifier (e.g. "23", "cifar10", "esc50")
            
        Returns:
            Tuple of (modality, task_id)
        """
        # Known audio datasets
        audio_datasets = {'esc50', 'ravdess', 'urbansound8k'}
        
        # Known vision datasets
        vision_datasets = {'cifar10', 'cifar100', 'mnist', 'imagenet'}
        
        # Check if it's a known audio dataset
        if identifier.lower() in audio_datasets:
            return "audio", AUDIO_CLASSIFICATION_TASK_ID
        
        # Check if it's a known vision dataset
        if identifier.lower() in vision_datasets:
            return "vision", VISION_CLASSIFICATION_TASK_ID
        
        # Try to parse as numeric task_id (tabular)
        try:
            task_id = int(identifier)
            return "tabular", task_id
        except ValueError:
            pass
        
        # Default to tabular with no task_id (will try to resolve later)
        logger.warning(f"Unknown dataset identifier '{identifier}', defaulting to tabular modality")
        return "tabular", None
    
    def _determine_task_type(self) -> Tuple[str, str]:
        """
        Determine task type (classification vs regression).
        
        Returns:
            Tuple of (task_type, detection_method)
        """
        # Manual override takes precedence
        if self.task_type_override:
            return self.task_type_override, "manual_override"
        
        # Use task detection utilities
        try:
            task_type, method = detect_task_type(
                task_id=self.task_id,
                y=self.y,
                dataset_info=self.dataset_info
            )
            return task_type, method
        except Exception as e:
            logger.warning(f"Task type detection failed: {e}, defaulting to classification")
            return "classification", "fallback_default"
    
    def _validate_detailed_vlm_outputs(self, detailed_output_path: Path) -> Tuple[bool, str]:
        """
        Validate that detailed_vlm_outputs.json was properly saved and formatted.
        
        Args:
            detailed_output_path: Path to the detailed_vlm_outputs.json file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not detailed_output_path.exists():
                return False, f"detailed_vlm_outputs.json file not found at {detailed_output_path}"
            
            # Check if file is readable and properly formatted JSON
            with open(detailed_output_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return False, "detailed_vlm_outputs.json file is empty"
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    return False, f"detailed_vlm_outputs.json contains invalid JSON: {e}"
            
            # Validate required fields structure (based on MARVIS's actual output format)
            required_fields = ['test_config', 'prediction_details']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return False, f"detailed_vlm_outputs.json missing required fields: {missing_fields}"
            
            # Optional fields that may be present from test logic but not required by MARVIS
            # dataset_info and model_config are added by test logic but not by MARVIS itself
            
            # Validate prediction_details structure
            prediction_details = data.get('prediction_details', [])
            if not isinstance(prediction_details, list):
                return False, "prediction_details should be a list"
            
            # Check if this is a failed test case
            pipeline_failed = data.get('pipeline_failed', False)
            if len(prediction_details) == 0:
                if pipeline_failed:
                    # For failed tests, empty prediction_details is expected
                    return True, "detailed_vlm_outputs.json validation passed (failed test case with expected empty predictions)"
                else:
                    return False, "prediction_details is empty - no predictions were recorded"
            
            # Validate each prediction entry has required fields
            for i, pred in enumerate(prediction_details):
                if not isinstance(pred, dict):
                    return False, f"prediction_details[{i}] should be a dictionary"
                
                # Check for actual MARVIS output fields (based on real output structure)
                pred_required_fields = ['test_point_idx', 'vlm_response']
                pred_missing_fields = [field for field in pred_required_fields if field not in pred]
                if pred_missing_fields:
                    return False, f"prediction_details[{i}] missing required fields: {pred_missing_fields}"
            
            # Validate metrics if present
            if 'metrics' in data and data['metrics'] is not None:
                metrics = data['metrics']
                if not isinstance(metrics, dict):
                    return False, "metrics should be a dictionary"
                
                # Check for basic accuracy metric
                if 'accuracy' not in metrics:
                    return False, "metrics missing required 'accuracy' field"
            
            return True, "detailed_vlm_outputs.json validation passed"
            
        except Exception as e:
            return False, f"Error validating detailed_vlm_outputs.json: {str(e)}"
    
    def _load_semantic_class_names(self, task_id: int, num_classes: int) -> Optional[List[str]]:
        """
        Load semantic class names from CC18 semantic data directory.
        
        Args:
            task_id: OpenML task ID
            num_classes: Number of classes in the dataset
            
        Returns:
            List of semantic class names if found, None otherwise
        """
        # Try to find semantic file using the general search
        try:
            from marvis.utils.metadata_loader import get_metadata_loader
            loader = get_metadata_loader()
            semantic_file = loader.detect_metadata_file(task_id)
        except Exception as e:
            logger.debug(f"Could not use metadata loader: {e}")
            # Fallback to hardcoded path
            semantic_file = Path(__file__).parent.parent / "data" / "cc18_semantic" / f"{task_id}.json"
        
        if not semantic_file or not semantic_file.exists():
            logger.info(f"No semantic file found for task {task_id}")
            return None
        
        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Try different structures to extract class names
            class_names = None
            
            # Method 1: target_values (maps numeric labels to semantic names)
            if 'target_values' in data:
                target_values = data['target_values']
                if isinstance(target_values, dict):
                    # Sort by key (handle both numeric and string keys)
                    try:
                        sorted_items = sorted(target_values.items(), key=lambda x: int(x[0]))
                    except ValueError:
                        # If keys are strings, sort lexicographically
                        sorted_items = sorted(target_values.items(), key=lambda x: x[0])
                    class_names = [item[1] for item in sorted_items]
            
            # Method 2: target_classes (list with name/meaning)
            if class_names is None and 'target_classes' in data:
                target_classes = data['target_classes']
                if isinstance(target_classes, list):
                    # Use 'meaning' if available, otherwise 'name'
                    class_names = []
                    for tc in target_classes:
                        if isinstance(tc, dict):
                            name = tc.get('meaning', tc.get('name', ''))
                            class_names.append(name)
                        else:
                            class_names.append(str(tc))
            
            # Method 3: instances_per_class keys
            if class_names is None and 'instances_per_class' in data:
                instances_per_class = data['instances_per_class']
                if isinstance(instances_per_class, dict):
                    class_names = list(instances_per_class.keys())
            
            # Validate and truncate to match number of classes
            if class_names:
                # Clean up class names (remove extra whitespace, etc.)
                class_names = [name.strip() for name in class_names if name.strip()]
                
                # Truncate to match actual number of classes
                if len(class_names) >= num_classes:
                    class_names = class_names[:num_classes]
                    logger.info(f"Loaded {len(class_names)} semantic class names for task {task_id}: {class_names}")
                    return class_names
                else:
                    logger.warning(f"Found {len(class_names)} semantic names but need {num_classes} for task {task_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load semantic file for task {task_id}: {e}")
        
        return None
    
    def _calculate_metrics(self, prediction_details, ground_truth_labels, config_name):
        """Calculate comprehensive metrics from prediction details."""
        if not prediction_details:
            return None
        
        # Extract predictions and true labels
        predictions = []
        true_labels = []
        
        for detail in prediction_details:
            if 'parsed_prediction' in detail and 'true_label' in detail:
                try:
                    if self.task_type == 'classification':
                        pred = int(detail['parsed_prediction'])
                        true = int(detail['true_label'])
                    else:  # regression
                        pred = float(detail['parsed_prediction'])
                        true = float(detail['true_label'])
                    predictions.append(pred)
                    true_labels.append(true)
                except (ValueError, TypeError):
                    # Skip invalid predictions
                    continue
        
        if len(predictions) == 0:
            return None
        
        # Completion rate
        completion_rate = len(predictions) / len(prediction_details)
        
        if self.task_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(true_labels, predictions)
            balanced_acc = balanced_accuracy_score(true_labels, predictions)
            
            # Calculate precision, recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='macro', zero_division=0
            )
            
            # Calculate per-class metrics
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                true_labels, predictions, average='micro', zero_division=0
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            return {
                'config_name': config_name,
                'task_type': 'classification',
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'precision_macro': float(precision),
                'recall_macro': float(recall), 
                'f1_macro': float(f1),
                'precision_micro': float(precision_micro),
                'recall_micro': float(recall_micro),
                'f1_micro': float(f1_micro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'confusion_matrix': cm.tolist() if cm is not None else [],
                'completion_rate': float(completion_rate),
                'num_test_samples': len(predictions),
                'num_classes': len(np.unique(true_labels)),
                'support': support.tolist() if support is not None else []
            }
        else:
            # Regression metrics
            mse = mean_squared_error(true_labels, predictions)
            mae = mean_absolute_error(true_labels, predictions)
            rmse = np.sqrt(mse)
            
            # R-squared
            ss_res = np.sum((true_labels - predictions) ** 2)
            ss_tot = np.sum((true_labels - np.mean(true_labels)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'config_name': config_name,
                'task_type': 'regression',
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'completion_rate': float(completion_rate),
                'num_test_samples': len(predictions),
                'target_range': [float(np.min(true_labels)), float(np.max(true_labels))],
                'prediction_range': [float(np.min(predictions)), float(np.max(predictions))]
            }
    
    def _load_dataset(self):
        """Load OpenML dataset with robust error handling."""
        try:
            # Create data directory if it doesn't exist
            data_dir = Path.home() / ".marvis" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset from OpenML using task_id
            # First resolve task_id to dataset_id using the resource manager
            from marvis.utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            identifiers = rm.resolve_openml_identifiers(task_id=self.task_id)
            
            dataset_id = identifiers.get('dataset_id')
            if not dataset_id:
                # Fallback: try to get task and dataset_id directly from OpenML
                import openml
                task = openml.tasks.get_task(self.task_id)
                dataset_id = task.dataset_id
            
            # Load dataset using dataset_id (not task_id)
            dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
            X = dataset.data
            y = dataset.target
            
            # Handle categorical targets
            if hasattr(y, 'cat'):
                y = y.cat.codes
            elif y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Convert to numpy with proper data type handling
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            
            # Ensure X is numeric and handle mixed data types
            from sklearn.preprocessing import LabelEncoder
            X_processed = []
            for col in range(X.shape[1]):
                col_data = X[:, col]
                if col_data.dtype == 'object' or col_data.dtype.kind in 'SU':  # String/Unicode
                    # Encode categorical columns
                    le = LabelEncoder()
                    col_data = le.fit_transform(col_data.astype(str))
                else:
                    # Convert to float and handle any remaining issues
                    col_data = pd.to_numeric(col_data, errors='coerce')
                X_processed.append(col_data)
            
            X = np.column_stack(X_processed).astype(float)
            y = y.astype(int)
                
            # Handle missing values (now safe since everything is numeric)
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Limit size for testing
            if len(X) > 500:
                indices = np.random.choice(len(X), 500, replace=False)
                X = X[indices]
                y = y[indices]
            
            # Extract feature names from OpenML dataset
            feature_names = None
            if hasattr(dataset, 'feature_names') and dataset.feature_names is not None:
                feature_names = list(dataset.feature_names)
            elif hasattr(dataset, 'data') and hasattr(dataset.data, 'columns'):
                feature_names = list(dataset.data.columns)
            
            dataset_info = {
                'name': dataset.DESCR if hasattr(dataset, 'DESCR') else f'OpenML_Task_{self.task_id}',
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'task_id': self.task_id,  # OpenML task ID for metadata loading
                'data_source': 'openml',
                'feature_names': feature_names
            }
            
            logger.info(f"Successfully loaded OpenML dataset {self.task_id}")
            return X, y, dataset_info
            
        except Exception as e:
            logger.warning(f"Failed to load OpenML dataset {self.task_id}: {e}")
            logger.info("Using synthetic dataset as fallback")
            
            # Fallback to synthetic data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=300, n_features=10, n_classes=3, 
                                     n_informative=6, random_state=42)
            dataset_info = {
                'name': 'synthetic_fallback',
                'n_samples': len(X),
                'n_features': X.shape[1], 
                'n_classes': len(np.unique(y)),
                'task_id': None,
                'data_source': 'synthetic',
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
            }
            return X, y, dataset_info
    
    def _load_audio_dataset(self):
        """Load audio dataset with robust error handling."""
        if not AUDIO_AVAILABLE:
            raise ImportError("Audio datasets not available. Install required dependencies for audio support.")
        
        try:
            dataset_name = self.task_id_or_dataset_name.lower()
            
            # Default data directory
            data_dir = Path.home() / ".marvis" / "audio_datasets" / dataset_name
            
            if dataset_name == 'esc50':
                dataset = ESC50Dataset(str(data_dir), download=True)
            elif dataset_name == 'ravdess':
                dataset = RAVDESSDataset(str(data_dir), download=True)
            elif dataset_name == 'urbansound8k':
                dataset = UrbanSound8KDataset(str(data_dir), download=True)
            else:
                raise ValueError(f"Unsupported audio dataset: {dataset_name}")
            
            # Create few-shot splits
            splits = dataset.create_few_shot_split(
                k_shot=5,  # Small for testing
                val_size=0.2,
                test_size=0.3,
                random_state=42
            )
            
            # Combine train and test for our purposes
            audio_paths = splits['train'][0] + splits['test'][0]
            labels = splits['train'][1] + splits['test'][1]
            class_names = splits['class_names']
            
            # Convert to numpy arrays
            X = np.array(audio_paths)  # Store as paths
            y = np.array(labels)
            
            # Limit size for testing
            if len(X) > 100:
                indices = np.random.choice(len(X), 100, replace=False)
                X = X[indices]
                y = y[indices]
            
            dataset_info = {
                'name': f'{dataset_name}_audio',
                'n_samples': len(X),
                'n_features': 1,  # Audio path
                'n_classes': len(class_names),
                'task_id': self.task_id,
                'data_source': 'audio',
                'class_names': class_names,
                'modality': 'audio'
            }
            
            logger.info(f"Successfully loaded {dataset_name} audio dataset")
            return X, y, dataset_info
            
        except Exception as e:
            logger.warning(f"Failed to load audio dataset {self.task_id_or_dataset_name}: {e}")
            logger.info("Using synthetic dataset as fallback")
            
            # Fallback to synthetic data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=50, n_features=10, n_classes=3, 
                                     n_informative=6, random_state=42)
            dataset_info = {
                'name': 'synthetic_audio_fallback',
                'n_samples': len(X),
                'n_features': X.shape[1], 
                'n_classes': len(np.unique(y)),
                'task_id': None,
                'data_source': 'synthetic',
                'class_names': [f'audio_class_{i}' for i in range(len(np.unique(y)))],
                'modality': 'audio'
            }
            return X, y, dataset_info
    
    def _load_vision_dataset(self):
        """Load vision dataset with robust error handling."""
        if not VISION_AVAILABLE:
            raise ImportError("Vision datasets not available. Install required dependencies for vision support.")
        
        try:
            dataset_name = self.task_id_or_dataset_name.lower()
            
            # Use resource manager to prepare dataset
            from marvis.utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            
            if dataset_name in ['cifar10', 'cifar100']:
                train_paths, train_labels, test_paths, test_labels, class_names = rm.prepare_cifar_dataset(dataset_name)
                
                # Combine train and test for our purposes
                image_paths = train_paths + test_paths
                labels = train_labels + test_labels
                
                # Convert to numpy arrays
                X = np.array(image_paths)  # Store as paths
                y = np.array(labels)
                
                # Limit size for testing
                if len(X) > 200:
                    indices = np.random.choice(len(X), 200, replace=False)
                    X = X[indices]
                    y = y[indices]
                
                dataset_info = {
                    'name': f'{dataset_name}_vision',
                    'n_samples': len(X),
                    'n_features': 1,  # Image path
                    'n_classes': len(class_names),
                    'task_id': self.task_id,
                    'data_source': 'vision',
                    'class_names': class_names,
                    'modality': 'vision'
                }
                
                logger.info(f"Successfully loaded {dataset_name} vision dataset")
                return X, y, dataset_info
            else:
                raise ValueError(f"Unsupported vision dataset: {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load vision dataset {self.task_id_or_dataset_name}: {e}")
            logger.info("Using synthetic dataset as fallback")
            
            # Fallback to synthetic data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=50, n_features=10, n_classes=3, 
                                     n_informative=6, random_state=42)
            dataset_info = {
                'name': 'synthetic_vision_fallback',
                'n_samples': len(X),
                'n_features': X.shape[1], 
                'n_classes': len(np.unique(y)),
                'task_id': None,
                'data_source': 'synthetic',
                'class_names': [f'vision_class_{i}' for i in range(len(np.unique(y)))],
                'modality': 'vision'
            }
            return X, y, dataset_info
    
    def _get_test_configurations(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of test configurations."""
        configs = []
        
        # Single visualization configurations
        single_viz_configs = [
            # Basic t-SNE
            {
                'name': 'basic_tsne',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # t-SNE with KNN
            {
                'name': 'tsne_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': True,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D t-SNE
            {
                'name': 'tsne_3d',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D t-SNE with KNN
            {
                'name': 'tsne_3d_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': True,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Different perplexity
            {
                'name': 'tsne_high_perplexity',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 25
            },
            # Basic UMAP
            {
                'name': 'basic_umap',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'visualization_method': 'umap'
            },
            # UMAP with KNN
            {
                'name': 'umap_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': True,
                'nn_k': 30,
                'visualization_method': 'umap'
            }
        ]
        
        # Multi-visualization configurations
        multi_viz_configs = [
            # Basic multi-viz (PCA + t-SNE)
            {
                'name': 'multi_pca_tsne',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # Three methods
            {
                'name': 'multi_pca_tsne_spectral',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'consensus'
            },
            # Linear vs nonlinear focus
            {
                'name': 'multi_linear_nonlinear',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'isomap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'divergence'
            },
            # Local vs global methods
            {
                'name': 'multi_local_global',
                'enable_multi_viz': True,
                'visualization_methods': ['tsne', 'isomap', 'mds'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # Comprehensive multi-viz
            {
                'name': 'multi_comprehensive',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral', 'isomap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'consensus'
            },
            # Different layout strategies
            {
                'name': 'multi_grid_layout',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'grid',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # With UMAP if available
            {
                'name': 'multi_with_umap',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'umap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            }
        ]
        
        # Semantic naming variations
        semantic_configs = [
            # Single viz with semantic names - loaded from CC18 semantic data
            {
                'name': 'tsne_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,  # Load from CC18 semantic directory
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with semantic names
            {
                'name': 'multi_semantic',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'nn_k': 30,
                'load_semantic_from_cc18': True  # Load from CC18 semantic directory
            },
            # Semantic class names test - single visualization with semantic names only
            {
                'name': 'tsne_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # UMAP with semantic names
            {
                'name': 'umap_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'visualization_method': 'umap'
            },
            # Semantic axes test - single visualization with axes interpretation
            {
                'name': 'tsne_semantic_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,  # NEW: Enable semantic axes computation
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Use metadata test - incorporate metadata into prompts
            {
                'name': 'tsne_use_metadata',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'use_metadata': True,  # NEW: Enable metadata incorporation
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Combined new features test
            {
                'name': 'tsne_semantic_metadata_combined',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,  # NEW: Enable semantic axes
                'use_metadata': True,   # NEW: Enable metadata
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with semantic names only
            {
                'name': 'multi_tsne_semantic',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'nn_k': 30,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True
            },
            # Multi-viz with new features
            {
                'name': 'multi_semantic_axes_metadata',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'nn_k': 30,
                'semantic_axes': True,  # NEW: Enable semantic axes
                'use_metadata': True,   # NEW: Enable metadata
                'auto_load_metadata': True
            },
            # Perturbation semantic axes test - single visualization with perturbation method
            {
                'name': 'tsne_perturbation_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_smart_feature_selection': True,  # NEW: Enable smart feature selection
                'max_features_to_test': 8,  # NEW: Test 8 features instead of all
                'feature_selection_method': 'pca_variance',  # NEW: Use PCA-based selection
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Feature importance semantic axes test - alternative method
            {
                'name': 'tsne_importance_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'feature_importance',  # NEW: Use feature importance method
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Perturbation + metadata test - combine perturbation method with metadata
            {
                'name': 'tsne_perturbation_metadata',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_smart_feature_selection': True,  # NEW: Enable smart feature selection
                'max_features_to_test': 12,  # NEW: Test more features for metadata combo
                'feature_selection_method': 'mutual_info',  # NEW: Use mutual info selection
                'use_metadata': True,  # NEW: Enable metadata
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D perturbation test - test perturbation method with 3D visualization
            {
                'name': 'tsne_3d_perturbation',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_smart_feature_selection': True,  # NEW: Enable smart feature selection
                'max_features_to_test': 10,  # NEW: Standard limit for 3D
                'feature_selection_method': 'f_score',  # NEW: Use F-score selection
                'use_3d_tsne': True,  # NEW: 3D visualization
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with perturbation method
            {
                'name': 'multi_perturbation_axes',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'nn_k': 30,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_smart_feature_selection': True,  # NEW: Enable smart feature selection
                'max_features_to_test': 15,  # NEW: More features for multi-viz
                'feature_selection_method': 'pca_variance'  # NEW: Use PCA-based selection
            }
        ]
        
        # Different modality parameters and mlxtend methods
        parameter_configs = [
            # High DPI visualization
            {
                'name': 'tsne_high_dpi',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15,
                'image_dpi': 150
            },
            # Different zoom factor
            {
                'name': 'tsne_zoomed',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15,
                'tsne_zoom_factor': 3.0
            },
            # MLxtend frequent patterns
            {
                'name': 'frequent_patterns',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'frequent_patterns'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # MLxtend decision regions with SVM
            {
                'name': 'decision_regions_svm',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'decision_regions'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'nn_k': 30,
                'decision_classifier': 'svm'
            },
            # Metadata testing with comprehensive info
            {
                'name': 'metadata_comprehensive',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'isomap'],
                'layout_strategy': 'hierarchical',
                'reasoning_focus': 'comparison',
                'include_metadata': True,
                'nn_k': 30,
                'metadata_types': ['quality_metrics', 'timing_info', 'method_params']
            }
        ]
        
        # Add modality-specific configurations
        modality_configs = []
        
        if self.modality == "audio":
            # Audio-specific configurations
            modality_configs.extend([
                {
                    'name': 'audio_basic_tsne',
                    'enable_multi_viz': False,
                    'use_3d_tsne': False,
                    'use_knn_connections': False,
                    'nn_k': 10,
                    'tsne_perplexity': 10,
                    'include_spectrogram': True
                },
                {
                    'name': 'audio_tsne_knn',
                    'enable_multi_viz': False,
                    'use_3d_tsne': False,
                    'use_knn_connections': True,
                    'nn_k': 10,
                    'tsne_perplexity': 10,
                    'include_spectrogram': True
                }
            ])
        elif self.modality == "vision":
            # Vision-specific configurations
            modality_configs.extend([
                {
                    'name': 'vision_basic_tsne',
                    'enable_multi_viz': False,
                    'use_3d_tsne': False,
                    'use_knn_connections': False,
                    'nn_k': 10,
                    'tsne_perplexity': 10
                },
                {
                    'name': 'vision_tsne_knn',
                    'enable_multi_viz': False,
                    'use_3d_tsne': False,
                    'use_knn_connections': True,
                    'nn_k': 10,
                    'tsne_perplexity': 10
                }
            ])
        
        # For non-tabular modalities, use simpler configs by default
        if self.modality != "tabular":
            all_configs = modality_configs + single_viz_configs[:3]  # Just a few basic configs
        else:
            # Combine all configurations - semantic tests first for tabular
            all_configs = (semantic_configs + single_viz_configs + multi_viz_configs + 
                          parameter_configs)
        
        return all_configs
    
    def run_single_test(self, config: Dict[str, Any], test_idx: int) -> Dict[str, Any]:
        """Run a single test configuration."""
        logger.info(f"Running test {test_idx + 1}: {config['name']}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Use fixed test indices for all tests
        if self.test_indices is None:
            # First test - establish the indices
            if len(X_test) > self.responses_per_test:
                self.test_indices = np.random.choice(len(X_test), self.responses_per_test, replace=False)
                self.test_indices.sort()  # Keep sorted for consistency
            else:
                self.test_indices = np.arange(len(X_test))
        
        # Use the same indices for all tests
        X_test = X_test[self.test_indices]
        y_test = y_test[self.test_indices]
        
        # Create test-specific directories
        test_dir = self.output_dir / f"test_{test_idx:02d}_{config['name']}"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "responses").mkdir(exist_ok=True)
        (test_dir / "prompts").mkdir(exist_ok=True)
        
        try:
            # Load semantic class names if requested
            semantic_class_names = None
            if config.get('load_semantic_from_cc18', False):
                semantic_class_names = self._load_semantic_class_names(self.task_id, len(np.unique(y_train)))
                if semantic_class_names:
                    config['semantic_class_names'] = semantic_class_names
                    logger.info(f"Loaded semantic class names for test {test_idx + 1}: {semantic_class_names}")
                else:
                    logger.info(f"No semantic class names found for task {self.task_id}, using Class_<NUM> fallback")
                    config['semantic_class_names'] = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
                    config['use_semantic_names'] = False  # Disable semantic names since we couldn't load them
            
            # Create classifier with configuration
            classifier_config = {
                'modality': self.modality,
                'vlm_model_id': self.vlm_model,
                'tsne_perplexity': config.get('tsne_perplexity', 15),
                'tsne_max_iter': 500,  # Reduced for speed
                'seed': 42,
                'max_vlm_image_size': 1024,  # Reduced for speed
                'image_dpi': config.get('image_dpi', 100),
                'zoom_factor': config.get('zoom_factor', self.zoom_factor),  # Use instance variable as default
                'use_semantic_names': config.get('use_semantic_names', False),
                # NEW: Add semantic axes and metadata parameters
                'semantic_axes': config.get('semantic_axes', False),
                'semantic_axes_method': config.get('semantic_axes_method', 'pca_loadings'),  # NEW: Support different semantic axes methods
                'use_smart_feature_selection': config.get('use_smart_feature_selection', True),  # NEW: Enable smart feature selection by default
                'max_features_to_test': config.get('max_features_to_test', 10),  # NEW: Limit features for perturbation testing
                'feature_selection_method': config.get('feature_selection_method', 'pca_variance'),  # NEW: Feature selection method
                'use_metadata': config.get('use_metadata', False),
                # Pass feature names for semantic axes computation
                'feature_names': self.dataset_info.get('feature_names', None),
                'auto_load_metadata': config.get('auto_load_metadata', True),
                # VLM model parameters to avoid KV cache issues
                'max_model_len': 16384,
                'gpu_memory_utilization': 0.7,  # Reduced to be safer
                'backend': self.backend  # Use backend from command line
            }
            
            # Add single or multi-viz specific parameters
            if config.get('enable_multi_viz', False):
                # Build multi_viz_config with method-specific parameters
                multi_viz_config = {}
                
                # Check if decision_regions is in the visualization methods
                if 'decision_regions' in config.get('visualization_methods', []):
                    multi_viz_config['decision_regions'] = {
                        'decision_classifier': config.get('decision_classifier', 'svm'),
                        'embedding_method': 'pca'
                    }
                
                classifier_config.update({
                    'enable_multi_viz': True,
                    'visualization_methods': config.get('visualization_methods', ['tsne']),
                    'layout_strategy': config.get('layout_strategy', 'sequential'),
                    'reasoning_focus': config.get('reasoning_focus', 'classification'),
                    'multi_viz_config': multi_viz_config
                })
            else:
                classifier_config.update({
                    'enable_multi_viz': False,
                    'use_3d': config.get('use_3d_tsne', False),  # Updated parameter name
                    'use_knn_connections': config.get('use_knn_connections', False),
                    'nn_k': config.get('knn_k', 5)  # Updated parameter name
                })
            
            classifier = MarvisTsneClassifier(**classifier_config)
            
            # Fit the classifier first
            # Pass semantic class names if provided
            fit_kwargs = {
                'dataset_info': self.dataset_info  # Pass dataset info for metadata loading
            }
            if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                fit_kwargs['class_names'] = config['semantic_class_names'][:len(np.unique(y_train))]
            
            classifier.fit(X_train, y_train, X_test, **fit_kwargs)
            
            # Use MARVIS's evaluate method to get detailed prediction information
            try:
                results = classifier.evaluate(
                    X_test, 
                    y_test, 
                    return_detailed=True,
                    save_outputs=True,
                    output_dir=str(test_dir.resolve()),
                    visualization_save_cadence=3
                )
                
                # Close any remaining matplotlib figures to prevent memory leaks
                import matplotlib.pyplot as plt
                plt.close('all')
                
                # Extract ground truth labels
                ground_truth_labels = [int(label) for label in y_test]
                
                # Extract actual VLM responses and prompts from prediction_details
                all_responses = []
                all_prompts = []
                
                if 'prediction_details' in results and results['prediction_details']:
                    prediction_details = results['prediction_details']
                    
                    # Extract VLM responses
                    for detail in prediction_details:
                        if 'vlm_response' in detail:
                            all_responses.append(detail['vlm_response'])
                        else:
                            # Fallback if vlm_response not available
                            parsed_pred = detail.get('parsed_prediction', 'UNKNOWN')
                            all_responses.append(f"Class: Class_{parsed_pred} | Reasoning: Parsed prediction from MARVIS")
                    
                    logger.info(f"Extracted {len(all_responses)} actual VLM responses from prediction_details")
                    
                    # For prompts, we'll generate a representative one since MARVIS doesn't store the exact prompts
                    # in prediction_details, but we know what prompt structure was used
                    if config.get('enable_multi_viz', False):
                        # Multi-viz prompt
                        multi_viz_info = []
                        if hasattr(classifier, 'context_composer') and classifier.context_composer:
                            for viz in classifier.context_composer.visualizations:
                                multi_viz_info.append({
                                    'method': viz.method_name,
                                    'description': f"{viz.method_name} visualization"
                                })
                        
                        from marvis.utils.vlm_prompting import create_classification_prompt
                        # Use semantic class names if available, otherwise generic ones
                        if self.task_type == 'classification':
                            prompt_class_names = config.get('semantic_class_names', [f"Class_{i}" for i in range(len(np.unique(y_train)))])
                            sample_prompt = create_classification_prompt(
                                class_names=prompt_class_names,
                                modality=self.modality,
                                dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                                use_semantic_names=config.get('use_semantic_names', False),
                                multi_viz_info=multi_viz_info
                            )
                        else:  # regression
                            sample_prompt = create_regression_prompt(
                                modality=self.modality,
                                dataset_description=f"Test dataset for regression",
                                target_min=float(np.min(y_train)),
                                target_max=float(np.max(y_train)),
                                target_mean=float(np.mean(y_train)),
                                target_std=float(np.std(y_train)),
                                multi_viz_info=multi_viz_info
                            )
                    else:
                        # Single viz prompt
                        from marvis.utils.vlm_prompting import create_classification_prompt
                        # Use semantic class names if available, otherwise generic ones
                        if self.task_type == 'classification':
                            prompt_class_names = config.get('semantic_class_names', [f"Class_{i}" for i in range(len(np.unique(y_train)))])
                            sample_prompt = create_classification_prompt(
                                class_names=prompt_class_names,
                                modality=self.modality,
                                use_knn=config.get('use_knn_connections', False),
                                use_3d=config.get('use_3d_tsne', False),
                                nn_k=config.get('knn_k', 5),
                                dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                                use_semantic_names=config.get('use_semantic_names', False)
                            )
                        else:  # regression
                            sample_prompt = create_regression_prompt(
                                modality=self.modality,
                                use_knn=config.get('use_knn_connections', False),
                                use_3d=config.get('use_3d_tsne', False),
                                nn_k=config.get('knn_k', 5),
                                dataset_description=f"Test dataset for regression",
                                target_min=float(np.min(y_train)),
                                target_max=float(np.max(y_train)),
                                target_mean=float(np.mean(y_train)),
                                target_std=float(np.std(y_train))
                            )
                    
                    # Use the same prompt for all samples (this is typically how MARVIS works)
                    all_prompts = [sample_prompt] * len(all_responses)
                
                else:
                    # Fallback if no prediction_details available
                    logger.warning("No prediction_details found in results")
                    predictions = results.get('predictions', [])
                    if isinstance(predictions, (list, np.ndarray)):
                        all_responses = [f"Class: Class_{pred} | Reasoning: MARVIS prediction (no detailed response)" for pred in predictions]
                    else:
                        all_responses = [f"Class: UNKNOWN | Reasoning: No prediction details available" for _ in range(len(X_test))]
                    
                    # Generate fallback prompts
                    # Generate appropriate class names for fallback
                    if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                        fallback_class_names = config['semantic_class_names'][:len(np.unique(y_train))]
                    else:
                        fallback_class_names = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
                    
                    if self.task_type == 'classification':
                        from marvis.utils.vlm_prompting import create_classification_prompt
                        sample_prompt = create_classification_prompt(
                            class_names=fallback_class_names,
                            modality=self.modality,
                            dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                            use_semantic_names=config.get('use_semantic_names', False)
                        )
                    else:  # regression
                        from marvis.utils.vlm_prompting import create_regression_prompt
                        sample_prompt = create_regression_prompt(
                            modality=self.modality,
                            dataset_description=f"Test dataset for regression",
                            target_min=float(np.min(y_train)),
                            target_max=float(np.max(y_train)),
                            target_mean=float(np.mean(y_train)),
                            target_std=float(np.std(y_train))
                        )
                    all_prompts = [sample_prompt] * len(all_responses)
                
                # Save the actual outputs manually since MARVIS's save_outputs isn't implemented
                # Save detailed VLM information to the test directory
                if 'prediction_details' in results and results['prediction_details']:
                    detailed_output = {
                        'test_config': config,
                        'ground_truth': ground_truth_labels,
                        'prediction_details': results['prediction_details'],
                        'completion_rate': results.get('completion_rate', 1.0),
                        'test_indices': self.test_indices.tolist() if self.test_indices is not None else []
                    }
                    
                    detailed_output_path = test_dir / "detailed_vlm_outputs.json"
                    with open(detailed_output_path, 'w') as f:
                        json.dump(detailed_output, f, indent=2, default=str)
                    logger.info(f"Saved detailed VLM outputs to {detailed_output_path}")
                    
                    # Calculate and log metrics using unified metrics system
                    metrics = self._calculate_metrics(
                        results['prediction_details'], 
                        ground_truth_labels, 
                        config['name']
                    )
                    
                    if metrics:
                        # Log metrics using unified metrics logger
                        metrics_logger = MetricsLogger(
                            model_name=f"MARVIS-{config['name']}",
                            dataset_name=self.dataset_info.get('name', f'dataset_{self.task_id}'),
                            use_wandb=False,  # Disable W&B for test script
                            logger=logger
                        )
                        
                        # Log all calculated metrics
                        metrics_logger.log_all_metrics(metrics)
                        
                        # Add metrics to detailed output
                        detailed_output['metrics'] = metrics
                        
                        # Save updated detailed output with metrics
                        with open(detailed_output_path, 'w') as f:
                            json.dump(detailed_output, f, indent=2, default=str)
                        
                        logger.info(f"✓ Test {test_idx + 1} metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")
                    else:
                        logger.warning(f"Could not calculate metrics for test {test_idx + 1}")
                        metrics = {'error': 'Could not calculate metrics'}
                
                # Save individual response and prompt files in test directory
                for i, (response, prompt) in enumerate(zip(all_responses, all_prompts)):
                    response_file = test_dir / "responses" / f"response_{i:02d}.txt"
                    with open(response_file, 'w') as f:
                        f.write(response)
                    
                    prompt_file = test_dir / "prompts" / f"prompt_{i:02d}.txt"
                    with open(prompt_file, 'w') as f:
                        f.write(prompt)
                
                # Find and copy MARVIS's generated visualizations
                viz_path = self.output_dir / "visualizations" / f"test_{test_idx:02d}_{config['name']}.png"
                
                # MARVIS saves visualizations in its temp directory during prediction
                viz_found = False
                if hasattr(classifier, 'temp_dir') and classifier.temp_dir and os.path.exists(classifier.temp_dir):
                    temp_viz_files = list(Path(classifier.temp_dir).glob("*.png"))
                    if temp_viz_files:
                        # Copy the most recent visualization file
                        latest_viz = max(temp_viz_files, key=lambda p: p.stat().st_mtime)
                        shutil.copy2(latest_viz, viz_path)
                        logger.info(f"Copied MARVIS visualization from {latest_viz} to {viz_path}")
                        viz_found = True
                        
                        # Also copy any additional visualizations for multi-viz tests
                        if len(temp_viz_files) > 1:
                            for i, viz_file in enumerate(temp_viz_files):
                                additional_viz_path = test_dir / f"visualization_{i}.png"
                                shutil.copy2(viz_file, additional_viz_path)
                
                if not viz_found:
                    logger.warning(f"No visualization found for test {test_idx + 1}. Check MARVIS's temp_dir: {getattr(classifier, 'temp_dir', 'Not set')}")
                
                logger.info(f"✓ Test {test_idx + 1} completed successfully using real MARVIS pipeline")
                
            except Exception as e:
                logger.error(f"✗ MARVIS pipeline failed for test {test_idx + 1}: {e}")
                # Close any figures that might have been created before the error
                import matplotlib.pyplot as plt
                plt.close('all')
                # Create fallback data for summary
                ground_truth_labels = [int(label) for label in y_test]
                all_responses = [f"Class: FAILED | Reasoning: Pipeline error: {str(e)}"] * len(X_test)
                all_prompts = ["FAILED"] * len(X_test)
                viz_path = self.output_dir / "visualizations" / f"test_{test_idx:02d}_{config['name']}_failed.png"
                
                # Create a minimal detailed_vlm_outputs.json for failed tests to ensure consistency
                test_dir = self.output_dir / f"test_{test_idx:02d}_{config['name']}"
                test_dir.mkdir(exist_ok=True)
                detailed_output_path = test_dir / "detailed_vlm_outputs.json"
                failed_detailed_output = {
                    'test_config': config,
                    'dataset_info': self.dataset_info,
                    'model_config': {'vlm_model': self.vlm_model, 'backend': self.backend},
                    'prediction_details': [],  # Empty because pipeline failed
                    'error': f"MARVIS pipeline failed: {str(e)}",
                    'pipeline_failed': True
                }
                
                try:
                    with open(detailed_output_path, 'w') as f:
                        json.dump(failed_detailed_output, f, indent=2, default=str)
                    logger.info(f"Created minimal detailed_vlm_outputs.json for failed test: {detailed_output_path}")
                except Exception as json_error:
                    logger.error(f"Failed to create detailed_vlm_outputs.json for failed test: {json_error}")
            
            # Save aggregated prompt and responses
            prompt_path = self.output_dir / "prompts" / f"test_{test_idx:02d}_{config['name']}_all.txt"
            with open(prompt_path, 'w') as f:
                f.write("\n\n=== PROMPT ===\n\n".join(all_prompts))
            
            response_path = self.output_dir / "responses" / f"test_{test_idx:02d}_{config['name']}_all.txt"
            with open(response_path, 'w') as f:
                f.write("\n\n=== RESPONSE ===\n\n".join(all_responses))
            
            # Save ground truth
            # Use semantic class names if provided
            if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                gt_class_names = config['semantic_class_names'][:len(np.unique(y_train))]
            else:
                gt_class_names = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
            
            ground_truth_path = self.output_dir / "ground_truth" / f"test_{test_idx:02d}_{config['name']}.json"
            ground_truth_data = {
                'test_indices': self.test_indices.tolist() if self.test_indices is not None else [],
                'ground_truth_labels': ground_truth_labels,
                'class_names': gt_class_names
            }
            
            # Add metadata information if this is a metadata test
            if config.get('include_metadata', False):
                ground_truth_data['metadata_config'] = {
                    'metadata_types': config.get('metadata_types', []),
                    'visualization_methods': config.get('visualization_methods', []),
                    'note': 'This test includes comprehensive metadata from visualization methods'
                }
            
            with open(ground_truth_path, 'w') as f:
                json.dump(ground_truth_data, f, indent=2)
            
            # Save configuration
            config_path = self.output_dir / "configs" / f"test_{test_idx:02d}_{config['name']}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Validate detailed_vlm_outputs.json before marking test as successful
            detailed_output_path = (test_dir / "detailed_vlm_outputs.json").resolve()
            vlm_outputs_valid, validation_error = self._validate_detailed_vlm_outputs(detailed_output_path)
            
            # Check if this was a pipeline failure by examining the detailed outputs
            pipeline_failed = False
            if vlm_outputs_valid and detailed_output_path.exists():
                try:
                    with open(detailed_output_path, 'r') as f:
                        data = json.load(f)
                        pipeline_failed = data.get('pipeline_failed', False)
                except:
                    pass  # If we can't read it, vlm_outputs_valid will already be False
            
            # Determine final test success status
            # Test is successful only if VLM outputs are valid AND pipeline didn't fail
            test_success = vlm_outputs_valid and not pipeline_failed
            
            if not vlm_outputs_valid:
                test_error = f"detailed_vlm_outputs.json validation failed: {validation_error}"
            elif pipeline_failed:
                test_error = "MARVIS pipeline failed during test execution"
            else:
                test_error = None
            
            # Collect test result
            result = {
                'test_idx': test_idx,
                'config_name': config['name'],
                'success': test_success,
                'num_test_samples': len(X_test),
                'test_indices': self.test_indices.tolist() if self.test_indices is not None else [],
                'ground_truth_labels': ground_truth_labels,
                'avg_prompt_length': np.mean([len(p) for p in all_prompts]),
                'visualization_path': str(viz_path.relative_to(self.output_dir)),
                'prompt_path': str(prompt_path.relative_to(self.output_dir)),
                'response_path': str(response_path.relative_to(self.output_dir)),
                'ground_truth_path': str(ground_truth_path.relative_to(self.output_dir)),
                'config_path': str(config_path.relative_to(self.output_dir)),
                'test_directory': str(test_dir.relative_to(self.output_dir)),
                'is_multi_viz': config.get('enable_multi_viz', False),
                'visualization_methods': config.get('visualization_methods', ['tsne']),
                'metrics': metrics if 'metrics' in locals() else None,
                'error': test_error,
                'detailed_vlm_outputs_valid': vlm_outputs_valid,
                'detailed_vlm_outputs_path': str(detailed_output_path.relative_to(self.output_dir))
            }
            
            if test_success:
                logger.info(f"✓ Test {test_idx + 1} completed successfully")
                logger.info(f"  detailed_vlm_outputs.json validation: PASSED")
                logger.info(f"  MARVIS pipeline execution: SUCCESS")
            else:
                logger.error(f"✗ Test {test_idx + 1} FAILED")
                if not vlm_outputs_valid:
                    logger.error(f"  detailed_vlm_outputs.json validation: FAILED - {validation_error}")
                else:
                    logger.error(f"  detailed_vlm_outputs.json validation: PASSED")
                
                if pipeline_failed:
                    logger.error(f"  MARVIS pipeline execution: FAILED")
                
                logger.error(f"  Overall error: {test_error}")
            
            # Final cleanup: ensure all figures are closed after this test
            import matplotlib.pyplot as plt
            plt.close('all')
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Test {test_idx + 1} failed: {e}")
            result = {
                'test_idx': test_idx,
                'config_name': config['name'],
                'success': False,
                'error': str(e)
            }
            return result
    
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations."""
        logger.info("Starting comprehensive VLM prompting test suite")
        logger.info(f"Dataset: {self.dataset_info['name']} (OpenML ID: {self.task_id})")
        logger.info(f"VLM Model: {self.vlm_model}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        configs = self._get_test_configurations()
        
        # Handle specific target config selection
        if hasattr(self, '_target_configs') and self._target_configs:
            matching_configs = [config for config in configs if config['name'] in self._target_configs]
            if matching_configs:
                configs = matching_configs
                if len(self._target_configs) == 1:
                    logger.info(f"Running specific test configuration: {self._target_configs[0]}")
                else:
                    logger.info(f"Running {len(self._target_configs)} specific test configurations: {', '.join(self._target_configs)}")
            else:
                logger.error(f"Target configurations not found: {', '.join(self._target_configs)}")
                return {}
        # Handle legacy single target config (for backward compatibility)
        elif hasattr(self, '_target_config') and self._target_config:
            matching_configs = [config for config in configs if config['name'] == self._target_config]
            if matching_configs:
                configs = [matching_configs[0]]
                logger.info(f"Running specific test configuration: {self._target_config}")
            else:
                logger.error(f"Target configuration '{self._target_config}' not found")
                return {}
        else:
            # Limit the number of test configurations if specified
            if self.num_tests is not None and self.num_tests < len(configs):
                configs = configs[:self.num_tests]
                logger.info(f"Limited to {self.num_tests} test configurations (out of {len(self._get_test_configurations())} available)")
        
        logger.info(f"Running {len(configs)} test configurations with {self.num_samples_per_test} samples each...")
        
        for i, config in enumerate(configs):
            result = self.run_single_test(config, i)
            self.test_results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save summary using unified JSON utilities
        summary_path = self.output_dir / "test_summary.json"
        safe_json_dump(summary, str(summary_path), logger=logger, indent=2)
        
        # Save detailed results using unified JSON utilities
        results_path = self.output_dir / "detailed_results.json"
        safe_json_dump(self.test_results, str(results_path), logger=logger, indent=2)
        
        logger.info(f"Test suite completed. Results saved to {self.output_dir}")
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary with comparative analysis."""
        successful_tests = [r for r in self.test_results if r.get('success', False)]
        failed_tests = [r for r in self.test_results if not r.get('success', False)]
        
        # Analyze detailed VLM outputs validation results
        vlm_outputs_valid_tests = [r for r in self.test_results if r.get('detailed_vlm_outputs_valid', False)]
        vlm_outputs_invalid_tests = [r for r in self.test_results if not r.get('detailed_vlm_outputs_valid', False)]
        pipeline_failed_tests = [r for r in failed_tests if 'pipeline failed' in r.get('error', '').lower()]
        validation_failed_tests = [r for r in failed_tests if 'validation failed' in r.get('error', '').lower()]
        
        single_viz_tests = [r for r in successful_tests if not r.get('is_multi_viz', False)]
        multi_viz_tests = [r for r in successful_tests if r.get('is_multi_viz', False)]
        
        # Count visualization methods used
        method_counts = {}
        for result in multi_viz_tests:
            methods = result.get('visualization_methods', [])
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        # Aggregate metrics for comparative analysis
        config_metrics = {}
        overall_metrics = {'accuracy': [], 'f1_macro': [], 'balanced_accuracy': []}
        
        for result in successful_tests:
            if result.get('metrics'):
                config_name = result['config_name']
                metrics = result['metrics']
                
                config_metrics[config_name] = metrics
                
                # Collect for overall statistics
                if metrics.get('accuracy') is not None:
                    overall_metrics['accuracy'].append(metrics['accuracy'])
                if metrics.get('f1_macro') is not None:
                    overall_metrics['f1_macro'].append(metrics['f1_macro'])
                if metrics.get('balanced_accuracy') is not None:
                    overall_metrics['balanced_accuracy'].append(metrics['balanced_accuracy'])
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        for metric_name, values in overall_metrics.items():
            if values:
                aggregate_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Find best performing configurations
        best_configs = {}
        for metric_name in ['accuracy', 'f1_macro', 'balanced_accuracy']:
            best_score = -1
            best_config = None
            for config_name, metrics in config_metrics.items():
                if metrics.get(metric_name, -1) > best_score:
                    best_score = metrics[metric_name]
                    best_config = config_name
            if best_config:
                best_configs[metric_name] = {
                    'config': best_config,
                    'score': best_score
                }
        
        # Compare single-viz vs multi-viz performance
        single_viz_metrics = [r['metrics'] for r in single_viz_tests if r.get('metrics')]
        multi_viz_metrics = [r['metrics'] for r in multi_viz_tests if r.get('metrics')]
        
        comparison = {}
        if single_viz_metrics and multi_viz_metrics:
            for metric_name in ['accuracy', 'f1_macro', 'balanced_accuracy']:
                single_scores = [m[metric_name] for m in single_viz_metrics if m.get(metric_name) is not None]
                multi_scores = [m[metric_name] for m in multi_viz_metrics if m.get(metric_name) is not None]
                
                if single_scores and multi_scores:
                    comparison[metric_name] = {
                        'single_viz_mean': np.mean(single_scores),
                        'multi_viz_mean': np.mean(multi_scores),
                        'difference': np.mean(multi_scores) - np.mean(single_scores),
                        'single_viz_count': len(single_scores),
                        'multi_viz_count': len(multi_scores)
                    }
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.dataset_info,
            'vlm_model': self.vlm_model,
            'total_tests': len(self.test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(self.test_results) * 100,
            'single_viz_tests': len(single_viz_tests),
            'multi_viz_tests': len(multi_viz_tests),
            'visualization_method_counts': method_counts,
            'average_prompt_length': np.mean([r.get('avg_prompt_length', 0) for r in successful_tests]),
            'total_test_samples': sum([r.get('num_test_samples', 0) for r in successful_tests]),
            'output_directory': str(self.output_dir),
            
            # Detailed VLM outputs validation statistics
            'validation_statistics': {
                'vlm_outputs_valid_tests': len(vlm_outputs_valid_tests),
                'vlm_outputs_invalid_tests': len(vlm_outputs_invalid_tests),
                'pipeline_failed_tests': len(pipeline_failed_tests),
                'validation_failed_tests': len(validation_failed_tests),
                'validation_success_rate': len(vlm_outputs_valid_tests) / len(self.test_results) * 100 if self.test_results else 0
            },
            
            # New metrics analysis
            'overall_metrics': aggregate_stats,
            'best_performing_configs': best_configs,
            'single_vs_multi_viz_comparison': comparison,
            'config_performance': config_metrics,
            
            # Analysis insights
            'insights': {
                'best_overall_config': best_configs.get('accuracy', {}).get('config'),
                'multi_viz_advantage': {
                    metric: comp.get('difference', 0) > 0 
                    for metric, comp in comparison.items()
                } if comparison else {},
                'most_tested_viz_method': max(method_counts, key=method_counts.get) if method_counts else None,
                'configs_with_metrics': len(config_metrics)
            },
            
            'files_generated': {
                'visualizations': len(list((self.output_dir / "visualizations").glob("*.png"))),
                'prompts': len(list((self.output_dir / "prompts").glob("*.txt"))),
                'responses': len(list((self.output_dir / "responses").glob("*.txt"))),
                'configs': len(list((self.output_dir / "configs").glob("*.json"))),
                'ground_truth': len(list((self.output_dir / "ground_truth").glob("*.json"))),
                'test_directories': len([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
            }
        }
        
        return summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Comprehensive VLM Prompting Test Suite")
    parser.add_argument("--output_dir", type=str, default="./test_vlm_outputs",
                       help="Directory to save test outputs")
    # Deprecated argument - will raise error if used
    parser.add_argument("--task_id", type=int, default=None,
                       help="(DEPRECATED) This argument is no longer supported. Use --task_id_or_dataset_name instead")
    parser.add_argument("--task_id_or_dataset_name", type=str, default="23",
                       help="Dataset identifier - can be OpenML task ID (e.g. '23'), dataset name (e.g. 'cifar10', 'esc50')")
    parser.add_argument("--task_type", type=str, default=None, choices=["classification", "regression"],
                       help="Override task type (classification or regression)")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="VLM model to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_tests", type=int, default=None,
                       help="Number of test configurations to run (default: None, runs all available)")
    parser.add_argument("--num_samples_per_test", type=int, default=10,
                       help="Number of test samples to process per configuration (default: 10)")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "vllm", "transformers"],
                       help="Backend to use for VLM inference (default: auto)")
    parser.add_argument("--zoom_factor", type=float, default=6.5,
                       help="Zoom factor for t-SNE visualizations (default: 6.5)")
    parser.add_argument("--test_config", type=str, nargs='*', default=None,
                       help="Run only specific test configurations by name (e.g., 'tsne_3d_knn' 'basic_tsne'). Can specify multiple configs separated by spaces.")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available test configuration names and exit")
    
    args = parser.parse_args()
    
    # Check for deprecated argument usage
    if args.task_id is not None:
        print("\nERROR: The --task_id argument is deprecated and no longer supported.")
        print("Please use --task_id_or_dataset_name instead.")
        print("\nExamples:")
        print("  # For OpenML tabular datasets:")
        print("  --task_id_or_dataset_name 23")
        print("  # For vision datasets:")
        print("  --task_id_or_dataset_name cifar10")
        print("  # For audio datasets:")
        print("  --task_id_or_dataset_name esc50")
        return 1
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create test suite
    test_suite = VLMPromptingTestSuite(
        output_dir=args.output_dir,
        task_id_or_dataset_name=args.task_id_or_dataset_name,
        vlm_model=args.vlm_model,
        num_tests=args.num_tests,
        num_samples_per_test=args.num_samples_per_test,
        backend=args.backend,
        zoom_factor=args.zoom_factor,
        task_type=args.task_type
    )
    
    # Handle list_configs option
    if args.list_configs:
        test_configs = test_suite._get_test_configurations()
        print("\nAvailable test configurations:")
        print("=" * 50)
        for i, config in enumerate(test_configs):
            semantic_method = config.get('semantic_axes_method', 'pca_loadings')
            semantic_axes = config.get('semantic_axes', False)
            multi_viz = config.get('enable_multi_viz', False)
            metadata = config.get('use_metadata', False)
            
            status = []
            if semantic_axes:
                status.append(f"semantic_axes({semantic_method})")
            if metadata:
                status.append("metadata")
            if multi_viz:
                status.append("multi_viz")
            
            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"{i+1:2d}. {config['name']}{status_str}")
        
        print(f"\nTotal: {len(test_configs)} configurations")
        print("\nUse --test_config <name1> [<name2> ...] to run specific configuration(s)")
        print("Examples:")
        print("  --test_config tsne_3d_knn")
        print("  --test_config basic_tsne tsne_knn tsne_3d")
        return
    
    # Handle specific test config option
    if args.test_config:
        test_configs = test_suite._get_test_configurations()
        
        # Find matching configurations for all specified config names
        matching_configs = []
        missing_configs = []
        
        for config_name in args.test_config:
            matches = [config for config in test_configs if config['name'] == config_name]
            if matches:
                matching_configs.extend(matches)
            else:
                missing_configs.append(config_name)
        
        if missing_configs:
            print(f"Error: Test configuration(s) not found: {', '.join(missing_configs)}")
            print("Use --list_configs to see available configurations.")
            return 1
        
        if len(args.test_config) == 1:
            print(f"Running specific test configuration: {args.test_config[0]}")
        else:
            print(f"Running {len(args.test_config)} specific test configurations: {', '.join(args.test_config)}")
        
        # Override num_tests to run only the specific configs
        test_suite.num_tests = len(matching_configs)
        test_suite._target_configs = args.test_config  # Store list of target configs
    
    summary = test_suite.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VLM PROMPTING TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Dataset: {summary['dataset_info']['name']} ({test_suite.modality})")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Single-viz Tests: {summary['single_viz_tests']}")
    print(f"Multi-viz Tests: {summary['multi_viz_tests']}")
    print(f"Total Test Samples: {summary['total_test_samples']}")
    print(f"Average Prompt Length: {summary['average_prompt_length']:.0f} characters")
    
    # Print validation statistics
    validation_stats = summary['validation_statistics']
    print(f"\nDetailed VLM Outputs Validation:")
    print(f"Valid VLM Outputs: {validation_stats['vlm_outputs_valid_tests']}")
    print(f"Invalid VLM Outputs: {validation_stats['vlm_outputs_invalid_tests']}")
    print(f"Pipeline Failures: {validation_stats['pipeline_failed_tests']}")
    print(f"Validation Failures: {validation_stats['validation_failed_tests']}")
    print(f"Validation Success Rate: {validation_stats['validation_success_rate']:.1f}%")
    print(f"\nFiles Generated:")
    for file_type, count in summary['files_generated'].items():
        print(f"  {file_type}: {count}")
    print(f"\nOutput Directory: {summary['output_directory']}")
    print(f"Summary saved to: {args.output_dir}/test_summary.json")
    print(f"Detailed results saved to: {args.output_dir}/detailed_results.json")
    
    # Print metrics analysis
    if summary.get('overall_metrics'):
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS ANALYSIS")
        print("=" * 60)
        
        for metric_name, stats in summary['overall_metrics'].items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Configs tested: {stats['count']}")
        
        # Best performing configs
        if summary.get('best_performing_configs'):
            print("\nBEST PERFORMING CONFIGURATIONS:")
            for metric, best in summary['best_performing_configs'].items():
                print(f"  {metric}: {best['config']} (score: {best['score']:.3f})")
        
        # Single vs Multi-viz comparison
        if summary.get('single_vs_multi_viz_comparison'):
            print("\nSINGLE-VIZ vs MULTI-VIZ COMPARISON:")
            for metric, comp in summary['single_vs_multi_viz_comparison'].items():
                advantage = "Multi-viz" if comp['difference'] > 0 else "Single-viz"
                print(f"  {metric}: Single={comp['single_viz_mean']:.3f}, Multi={comp['multi_viz_mean']:.3f}")
                print(f"    Advantage: {advantage} (+{abs(comp['difference']):.3f})")
        
        # Key insights
        if summary.get('insights'):
            insights = summary['insights']
            print("\nKEY INSIGHTS:")
            if insights.get('best_overall_config'):
                print(f"  • Best overall configuration: {insights['best_overall_config']}")
            if insights.get('most_tested_viz_method'):
                print(f"  • Most tested visualization method: {insights['most_tested_viz_method']}")
            if insights.get('multi_viz_advantage'):
                multi_advantages = [k for k, v in insights['multi_viz_advantage'].items() if v]
                if multi_advantages:
                    print(f"  • Multi-viz shows advantage in: {', '.join(multi_advantages)}")
    
    if summary['visualization_method_counts']:
        print(f"\nVisualization Methods Used:")
        for method, count in summary['visualization_method_counts'].items():
            print(f"  {method}: {count} tests")
    
    print("\n🎉 Test suite completed successfully!")
    print("\nTo inspect results:")
    print(f"  ls {args.output_dir}/")
    print(f"  cat {args.output_dir}/test_summary.json")
    print(f"  cat {args.output_dir}/detailed_results.json")


if __name__ == "__main__":
    main()