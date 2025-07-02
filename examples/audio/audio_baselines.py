"""
Audio classification baseline methods.

Includes:
1. WhisperKNNClassifier: Whisper embeddings + KNN classification
2. CLAPZeroShotClassifier: CLAP zero-shot classification
"""

import os
import numpy as np
import torch
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.data.audio_embeddings import get_whisper_embeddings
from marvis.utils.device_utils import detect_optimal_device
from marvis.utils.class_name_utils import get_semantic_class_names_or_fallback

logger = logging.getLogger(__name__)


class WhisperKNNClassifier:
    """
    Whisper + KNN baseline for audio classification.
    
    Uses Whisper Large-v2 embeddings with K-Nearest Neighbors classification.
    """
    
    def __init__(
        self,
        whisper_model: str = "large-v2",
        n_neighbors: int = 5,
        metric: str = "cosine",
        weights: str = "distance",
        standardize: bool = True,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize Whisper KNN classifier.
        
        Args:
            whisper_model: Whisper model variant to use
            n_neighbors: Number of neighbors for KNN
            metric: Distance metric ('cosine', 'euclidean', etc.)
            weights: Weight function ('uniform', 'distance')
            standardize: Whether to standardize embeddings
            cache_dir: Directory for caching embeddings
            device: Device for Whisper model
            seed: Random seed
        """
        self.whisper_model = whisper_model
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.standardize = standardize
        self.cache_dir = cache_dir
        self.device = device or detect_optimal_device()
        self.seed = seed
        
        # Initialize KNN classifier
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Optional scaler
        self.scaler = StandardScaler() if standardize else None
        
        # To be set during fit
        self.train_embeddings = None
        self.train_labels = None
        self.class_names = None
        
    def fit(self, train_paths: List[str], train_labels: List[int], 
            class_names: Optional[List[str]] = None, use_semantic_names: bool = False):
        """
        Fit the classifier on training data.
        
        Args:
            train_paths: List of paths to training audio files
            train_labels: List of training labels
            class_names: Optional list of class names
            use_semantic_names: Whether to use semantic class names
        """
        logger.info(f"Fitting Whisper KNN classifier with {len(train_paths)} training samples")
        
        # Store training data
        self.train_labels = np.array(train_labels)
        
        if class_names is None:
            # Use new utility to extract class names with semantic support
            unique_labels = np.unique(train_labels).tolist()
            from marvis.utils.class_name_utils import extract_class_names_from_labels
            self.class_names, _ = extract_class_names_from_labels(
                labels=unique_labels,
                dataset_name=getattr(self, 'dataset_name', None),
                use_semantic=use_semantic_names
            )
        else:
            self.class_names = class_names
            
        # Extract Whisper embeddings
        logger.info(f"Extracting Whisper {self.whisper_model} embeddings...")
        self.train_embeddings = get_whisper_embeddings(
            train_paths,
            model_name=self.whisper_model,
            cache_dir=self.cache_dir,
            device=self.device
        )
        
        # Standardize if requested
        if self.scaler is not None:
            self.train_embeddings = self.scaler.fit_transform(self.train_embeddings)
            
        # Fit KNN
        self.knn.fit(self.train_embeddings, self.train_labels)
        logger.info("Whisper KNN classifier fitted successfully")
        
    def predict(self, test_paths: List[str]) -> np.ndarray:
        """
        Predict labels for test audio files.
        
        Args:
            test_paths: List of paths to test audio files
            
        Returns:
            predictions: Array of predicted labels
        """
        if self.train_embeddings is None:
            raise ValueError("Classifier must be fitted before prediction")
            
        # Extract test embeddings
        logger.info(f"Extracting embeddings for {len(test_paths)} test samples...")
        test_embeddings = get_whisper_embeddings(
            test_paths,
            model_name=self.whisper_model,
            cache_dir=self.cache_dir,
            device=self.device
        )
        
        # Standardize if needed
        if self.scaler is not None:
            test_embeddings = self.scaler.transform(test_embeddings)
            
        # Predict
        predictions = self.knn.predict(test_embeddings)
        
        return predictions
        
    def predict_proba(self, test_paths: List[str]) -> np.ndarray:
        """
        Predict class probabilities for test audio files.
        
        Args:
            test_paths: List of paths to test audio files
            
        Returns:
            probabilities: Array of shape (n_samples, n_classes)
        """
        if self.train_embeddings is None:
            raise ValueError("Classifier must be fitted before prediction")
            
        # Extract test embeddings
        test_embeddings = get_whisper_embeddings(
            test_paths,
            model_name=self.whisper_model,
            cache_dir=self.cache_dir,
            device=self.device
        )
        
        # Standardize if needed
        if self.scaler is not None:
            test_embeddings = self.scaler.transform(test_embeddings)
            
        # Predict probabilities
        probabilities = self.knn.predict_proba(test_embeddings)
        
        return probabilities
        
    def evaluate(self, test_paths: List[str], test_labels: List[int],
                return_detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            test_paths: List of test audio paths
            test_labels: List of test labels
            return_detailed: Return detailed results
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_paths)} test samples")
        
        start_time = time.time()
        predictions = self.predict(test_paths)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': prediction_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions.tolist(),
            'true_labels': test_labels
        }
        
        if return_detailed:
            # Only use class names for classes that appear in the test set
            unique_test_labels = np.unique(test_labels)
            test_class_names = [self.class_names[i] for i in unique_test_labels]
            
            report = classification_report(
                test_labels, predictions,
                labels=unique_test_labels,
                target_names=test_class_names,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = report
            
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
        
        return results
        
    def get_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        return {
            'whisper_model': self.whisper_model,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'weights': self.weights,
            'standardize': self.standardize,
            'device': str(self.device),
            'seed': self.seed
        }


class CLAPZeroShotClassifier:
    """
    CLAP (Contrastive Language-Audio Pre-training) zero-shot classifier.
    
    Uses Microsoft's CLAP model for zero-shot audio classification.
    """
    
    def __init__(
        self,
        version: str = "2023",
        use_cuda: Optional[bool] = None,
        batch_size: int = 8
    ):
        """
        Initialize CLAP zero-shot classifier.
        
        Args:
            version: CLAP model version ('2022', '2023', or 'clapcap')
            use_cuda: Whether to use CUDA (auto-detected if None)
            batch_size: Batch size for processing
        """
        self.version = version
        self.device = detect_optimal_device()
        self.use_cuda = use_cuda if use_cuda is not None else (self.device == "cuda")
        self.batch_size = batch_size
        
        self.model = None
        self.class_names = None
        self.text_embeddings = None
        
    def _load_model(self):
        """Load CLAP model using msclap library."""
        if self.model is not None:
            return
            
        try:
            from msclap import CLAP
            
            logger.info(f"Loading CLAP model version: {self.version}")
            
            self.model = CLAP(version=self.version, use_cuda=self.use_cuda)
            
            logger.info("CLAP model loaded successfully")
            
        except ImportError as e:
            logger.error("msclap library not found. Please install with: pip install msclap")
            raise ImportError("msclap library required. Install with: pip install msclap") from e
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {e}")
            raise
            
    def _create_text_prompts(self, class_names: List[str]) -> List[str]:
        """
        Create text prompts for each class.
        
        Args:
            class_names: List of class names
            
        Returns:
            prompts: List of text prompts (one per class)
        """
        # Use simple, direct prompts for audio classification
        # CLAP works best with clear, descriptive prompts
        return [f"sound of {class_name}" for class_name in class_names]
        
    def fit(self, train_paths: List[str], train_labels: List[int], 
            class_names: Optional[List[str]] = None, use_semantic_names: bool = False):
        """
        Prepare the classifier (compute text embeddings).
        
        Note: CLAP is zero-shot, so we don't use training data directly,
        but we need class names to create text prompts.
        
        Args:
            train_paths: Ignored (kept for API consistency)
            train_labels: Used to infer class names if not provided
            class_names: List of class names
            use_semantic_names: Whether to use semantic class names
        """
        # Load model if needed
        self._load_model()
        
        # Set class names
        if class_names is None:
            # Use new utility to extract class names with semantic support
            unique_labels = np.unique(train_labels).tolist()
            from marvis.utils.class_name_utils import extract_class_names_from_labels
            self.class_names, _ = extract_class_names_from_labels(
                labels=unique_labels,
                dataset_name=getattr(self, 'dataset_name', None),
                use_semantic=use_semantic_names
            )
        else:
            self.class_names = class_names
            
        # Create text prompts
        text_prompts = self._create_text_prompts(self.class_names)
        
        # Compute text embeddings using msclap
        logger.info(f"Computing text embeddings for {len(self.class_names)} classes...")
        text_embeddings = self.model.get_text_embeddings(text_prompts)
        
        # Ensure text embeddings are detached from computation graph
        import torch
        if torch.is_tensor(text_embeddings):
            self.text_embeddings = text_embeddings.detach()
        else:
            self.text_embeddings = text_embeddings
        
        logger.info("CLAP zero-shot classifier ready")
        
    def predict(self, test_paths: List[str]) -> np.ndarray:
        """
        Predict labels for test audio files using zero-shot classification.
        
        Args:
            test_paths: List of paths to test audio files
            
        Returns:
            predictions: Array of predicted labels
        """
        if self.text_embeddings is None:
            raise ValueError("Classifier must be fitted before prediction")
            
        logger.info(f"Predicting labels for {len(test_paths)} audio files...")
        
        import torch
        
        # Process in batches to avoid OOM
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(test_paths), self.batch_size):
                batch_paths = test_paths[i:i + self.batch_size]
                
                if i % (self.batch_size * 10) == 0:  # Log every 10 batches
                    logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(test_paths)-1)//self.batch_size + 1}")
                
                # Get audio embeddings for this batch
                audio_embeddings = self.model.get_audio_embeddings(batch_paths)
                
                # Ensure audio embeddings are detached from computation graph
                if torch.is_tensor(audio_embeddings):
                    audio_embeddings = audio_embeddings.detach()
                
                # Compute similarities using msclap
                similarities = self.model.compute_similarity(audio_embeddings, self.text_embeddings)
                
                # Ensure similarities are on CPU for numpy operations and detach from computation graph
                if torch.is_tensor(similarities):
                    similarities = similarities.detach().cpu().numpy()
                
                # Get predictions (argmax along class dimension)
                batch_predictions = np.argmax(similarities, axis=1)
                all_predictions.extend(batch_predictions)
                
                # Clear cache to free memory
                torch.cuda.empty_cache() if self.use_cuda else None
        
        logger.info("CLAP prediction completed")
        return np.array(all_predictions)
        
    def predict_proba(self, test_paths: List[str]) -> np.ndarray:
        """
        Get prediction probabilities using softmax on similarities.
        
        Args:
            test_paths: List of paths to test audio files
            
        Returns:
            probabilities: Array of shape (n_samples, n_classes)
        """
        if self.text_embeddings is None:
            raise ValueError("Classifier must be fitted before prediction")
            
        logger.info(f"Computing probabilities for {len(test_paths)} audio files...")
        
        import torch
        temperature = 0.07  # CLAP default temperature
        
        # Process in batches to avoid OOM
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(test_paths), self.batch_size):
                batch_paths = test_paths[i:i + self.batch_size]
                
                if i % (self.batch_size * 10) == 0:  # Log every 10 batches
                    logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(test_paths)-1)//self.batch_size + 1}")
                
                # Get audio embeddings for this batch
                audio_embeddings = self.model.get_audio_embeddings(batch_paths)
                
                # Ensure audio embeddings are detached from computation graph
                if torch.is_tensor(audio_embeddings):
                    audio_embeddings = audio_embeddings.detach()
                
                # Compute similarities using msclap
                similarities = self.model.compute_similarity(audio_embeddings, self.text_embeddings)
                
                # Ensure similarities are on CPU for numpy conversion and detach from computation graph
                if torch.is_tensor(similarities):
                    similarities_tensor = similarities.detach().cpu()
                else:
                    similarities_tensor = torch.tensor(similarities)
                
                # Apply temperature scaling and softmax to get probabilities
                batch_probs = torch.softmax(similarities_tensor / temperature, dim=-1)
                all_probs.append(batch_probs.cpu().numpy())
                
                # Clear cache to free memory
                torch.cuda.empty_cache() if self.use_cuda else None
        
        logger.info("CLAP probability computation completed")
        return np.vstack(all_probs)
        
    def evaluate(self, test_paths: List[str], test_labels: List[int],
                return_detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            test_paths: List of test audio paths
            test_labels: List of test labels
            return_detailed: Return detailed results
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating CLAP zero-shot on {len(test_paths)} test samples")
        
        start_time = time.time()
        predictions = self.predict(test_paths)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': prediction_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions.tolist(),
            'true_labels': test_labels
        }
        
        if return_detailed:
            # Only use class names for classes that appear in the test set
            unique_test_labels = np.unique(test_labels)
            test_class_names = [self.class_names[i] for i in unique_test_labels]
            
            report = classification_report(
                test_labels, predictions,
                labels=unique_test_labels,
                target_names=test_class_names,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = report
            
        logger.info(f"CLAP evaluation completed - Accuracy: {accuracy:.4f}")
        
        return results
        
    def get_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        return {
            'clap_version': self.version,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'use_cuda': self.use_cuda
        }