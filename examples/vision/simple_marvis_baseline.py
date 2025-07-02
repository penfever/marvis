"""
Simple MARVIS baseline for image classification using DINOV2 embeddings directly.

This version skips t-SNE due to numerical instability and uses embeddings directly.
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import time
import os

# Import MARVIS utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.data.embeddings import get_dinov2_embeddings

logger = logging.getLogger(__name__)


class SimpleMarvisImageClassifier:
    """
    Simple MARVIS classifier for image classification using DINOV2 embeddings directly.
    
    This approach:
    1. Extracts DINOV2 embeddings from images
    2. Optionally applies PCA for dimensionality reduction
    3. Uses k-NN for classification in the embedding space
    """
    
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        embedding_size: int = 512,
        use_pca: bool = True,
        pca_components: int = 50,
        knn_neighbors: int = 5,
        knn_weights: str = 'distance',
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize simple MARVIS image classifier.
        
        Args:
            dinov2_model: DINOV2 model variant to use
            embedding_size: Target embedding size
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components
            knn_neighbors: Number of neighbors for k-NN
            knn_weights: Weight function for k-NN
            cache_dir: Directory for caching embeddings
            device: Device for model inference
        """
        self.dinov2_model = dinov2_model
        self.embedding_size = embedding_size
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.knn_neighbors = knn_neighbors
        self.knn_weights = knn_weights
        self.cache_dir = cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        if self.use_pca:
            self.pca = PCA(n_components=pca_components, random_state=42)
        else:
            self.pca = None
        self.knn = KNeighborsClassifier(
            n_neighbors=knn_neighbors,
            weights=knn_weights,
            n_jobs=-1
        )
        
        # State variables
        self.is_fitted = False
        self.class_names = None
        
    def fit(
        self,
        train_image_paths: List[str],
        train_labels: List[int],
        class_names: Optional[List[str]] = None
    ) -> 'SimpleMarvisImageClassifier':
        """
        Fit the simple MARVIS image classifier.
        
        Args:
            train_image_paths: List of training image paths
            train_labels: List of training labels
            class_names: Optional list of class names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting simple MARVIS image classifier on {len(train_image_paths)} training images")
        start_time = time.time()
        
        # Store class names
        self.class_names = class_names
        
        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        
        # Extract DINOV2 embeddings
        logger.info("Extracting DINOV2 embeddings...")
        train_embeddings = get_dinov2_embeddings(
            train_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="train",
            device=self.device
        )
        
        # Standardize embeddings
        logger.info("Standardizing embeddings...")
        train_embeddings_scaled = self.scaler.fit_transform(train_embeddings)
        
        # Apply PCA if requested
        if self.use_pca:
            logger.info(f"Applying PCA (components={self.pca_components})...")
            train_embeddings_final = self.pca.fit_transform(train_embeddings_scaled)
        else:
            train_embeddings_final = train_embeddings_scaled
        
        # Fit k-NN classifier
        logger.info(f"Fitting k-NN classifier (k={self.knn_neighbors})...")
        self.knn.fit(train_embeddings_final, train_labels_encoded)
        
        self.is_fitted = True
        elapsed_time = time.time() - start_time
        logger.info(f"Simple MARVIS image classifier fitted in {elapsed_time:.2f} seconds")
        
        return self
    
    def predict(self, test_image_paths: List[str]) -> np.ndarray:
        """
        Predict labels for test images.
        
        Args:
            test_image_paths: List of test image paths
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        logger.info(f"Predicting labels for {len(test_image_paths)} test images")
        
        # Extract DINOV2 embeddings
        test_embeddings = get_dinov2_embeddings(
            test_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="test",
            device=self.device
        )
        
        # Standardize embeddings
        test_embeddings_scaled = self.scaler.transform(test_embeddings)
        
        # Apply PCA if used during training
        if self.use_pca:
            test_embeddings_final = self.pca.transform(test_embeddings_scaled)
        else:
            test_embeddings_final = test_embeddings_scaled
        
        # Predict using k-NN
        predictions_encoded = self.knn.predict(test_embeddings_final)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, test_image_paths: List[str]) -> np.ndarray:
        """
        Predict class probabilities for test images.
        
        Args:
            test_image_paths: List of test image paths
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Extract and transform embeddings
        test_embeddings = get_dinov2_embeddings(
            test_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="test",
            device=self.device
        )
        
        test_embeddings_scaled = self.scaler.transform(test_embeddings)
        
        if self.use_pca:
            test_embeddings_final = self.pca.transform(test_embeddings_scaled)
        else:
            test_embeddings_final = test_embeddings_scaled
        
        # Get probabilities
        probabilities = self.knn.predict_proba(test_embeddings_final)
        
        return probabilities
    
    def evaluate(
        self,
        test_image_paths: List[str],
        test_labels: List[int],
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_image_paths: List of test image paths
            test_labels: List of true test labels
            return_detailed: Whether to return detailed results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        
        # Make predictions
        predictions = self.predict(test_image_paths)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels)
        }
        
        if return_detailed:
            results.update({
                'classification_report': classification_report(
                    test_labels, predictions, 
                    target_names=self.class_names,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(test_labels, predictions),
                'predictions': predictions,
                'true_labels': test_labels
            })
        
        logger.info(f"Simple MARVIS image classifier accuracy: {accuracy:.4f}")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            'dinov2_model': self.dinov2_model,
            'embedding_size': self.embedding_size,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'knn_neighbors': self.knn_neighbors,
            'knn_weights': self.knn_weights,
            'device': self.device
        }