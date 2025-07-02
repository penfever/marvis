"""
MARVIS t-SNE baseline for image classification using DINOV2 embeddings.
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import os

# Import MARVIS utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.data.embeddings import get_dinov2_embeddings, load_dinov2_model, prepare_image_for_dinov2
from examples.vision.image_utils import extract_features_from_loader

logger = logging.getLogger(__name__)


class MarvisImageClassifier:
    """
    MARVIS t-SNE classifier for image classification using pre-trained embeddings.
    
    This approach:
    1. Extracts DINOV2 embeddings from images
    2. Applies t-SNE for dimensionality reduction
    3. Uses k-NN for classification in the t-SNE space
    """
    
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        embedding_size: int = 1000,
        tsne_components: int = 2,
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: str = 'auto',
        tsne_random_state: int = 42,
        knn_neighbors: int = 5,
        knn_weights: str = 'distance',
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize MARVIS image classifier.
        
        Args:
            dinov2_model: DINOV2 model variant to use
            embedding_size: Target embedding size
            tsne_components: Number of t-SNE components
            tsne_perplexity: t-SNE perplexity parameter
            tsne_learning_rate: t-SNE learning rate
            tsne_random_state: Random seed for t-SNE
            knn_neighbors: Number of neighbors for k-NN
            knn_weights: Weight function for k-NN
            cache_dir: Directory for caching embeddings
            device: Device for model inference
        """
        self.dinov2_model = dinov2_model
        self.embedding_size = embedding_size
        self.tsne_components = tsne_components
        self.tsne_perplexity = tsne_perplexity
        self.tsne_learning_rate = tsne_learning_rate
        self.tsne_random_state = tsne_random_state
        self.knn_neighbors = knn_neighbors
        self.knn_weights = knn_weights
        self.cache_dir = cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tsne = TSNE(
            n_components=tsne_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            random_state=tsne_random_state,
            n_jobs=1,  # Use single thread for stability
            init='pca',  # Use PCA initialization for better stability
            method='exact' if tsne_perplexity < 50 else 'barnes_hut'  # Use exact method for small perplexity
        )
        self.knn = KNeighborsClassifier(
            n_neighbors=knn_neighbors,
            weights=knn_weights,
            n_jobs=-1
        )
        
        # State variables
        self.is_fitted = False
        self.train_tsne_embeddings = None
        self.class_names = None
        self.pca = None
        
    def fit(
        self,
        train_image_paths: List[str],
        train_labels: List[int],
        class_names: Optional[List[str]] = None
    ) -> 'MarvisImageClassifier':
        """
        Fit the MARVIS image classifier.
        
        Args:
            train_image_paths: List of training image paths
            train_labels: List of training labels
            class_names: Optional list of class names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting MARVIS image classifier on {len(train_image_paths)} training images")
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
        
        # Store original embeddings for better test projection
        self.train_embeddings_original = train_embeddings.copy()
        
        # Standardize embeddings with clipping to avoid numerical issues
        logger.info("Standardizing embeddings...")
        train_embeddings_scaled = self.scaler.fit_transform(train_embeddings)
        
        # Clip extreme values to avoid numerical instability in t-SNE
        train_embeddings_scaled = np.clip(train_embeddings_scaled, -5.0, 5.0)
        
        # First apply PCA to reduce dimensionality before t-SNE for stability
        logger.info("Applying PCA preprocessing...")
        n_pca_components = min(50, train_embeddings_scaled.shape[1])  # Reduce to 50 dims first
        self.pca = PCA(n_components=n_pca_components, random_state=self.tsne_random_state)
        train_embeddings_pca = self.pca.fit_transform(train_embeddings_scaled)
        
        # Apply t-SNE with numerical stability improvements
        logger.info(f"Applying t-SNE (components={self.tsne_components}, perplexity={self.tsne_perplexity})...")
        # Add small noise to break ties and improve numerical stability
        train_embeddings_pca += np.random.normal(0, 1e-6, train_embeddings_pca.shape)
        self.train_tsne_embeddings = self.tsne.fit_transform(train_embeddings_pca)
        
        # Fit k-NN classifier
        logger.info(f"Fitting k-NN classifier (k={self.knn_neighbors})...")
        self.knn.fit(self.train_tsne_embeddings, train_labels_encoded)
        
        self.is_fitted = True
        elapsed_time = time.time() - start_time
        logger.info(f"MARVIS image classifier fitted in {elapsed_time:.2f} seconds")
        
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
        
        # Standardize embeddings with same clipping as training
        test_embeddings_scaled = self.scaler.transform(test_embeddings)
        test_embeddings_scaled = np.clip(test_embeddings_scaled, -5.0, 5.0)
        
        # Apply PCA preprocessing
        test_embeddings_pca = self.pca.transform(test_embeddings_scaled)
        
        # Apply t-SNE transform using training data
        # Note: For proper t-SNE, we need to retrain on combined data
        # For efficiency, we'll use a simpler approach here
        logger.info("Projecting test embeddings to t-SNE space...")
        
        # Combine train and test embeddings for joint t-SNE
        combined_embeddings = np.vstack([
            self.scaler.transform(self.scaler.inverse_transform(self.scaler.transform(
                np.random.random((len(self.train_tsne_embeddings), test_embeddings_scaled.shape[1]))
            )))[:len(self.train_tsne_embeddings)],  # Placeholder for train embeddings
            test_embeddings_pca
        ])
        
        # For efficiency, we'll use a different approach:
        # Use the nearest neighbors in the original embedding space and map to t-SNE space
        test_tsne_embeddings = self._project_to_tsne_space(test_embeddings_pca)
        
        # Predict using k-NN
        predictions_encoded = self.knn.predict(test_tsne_embeddings)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def _project_to_tsne_space(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Project test embeddings to t-SNE space using training embeddings as reference.
        
        For better accuracy, we'll retrain t-SNE on combined data, then extract
        only the test portion.
        
        Args:
            test_embeddings: Test embeddings in PCA space
            
        Returns:
            Test embeddings projected to t-SNE space
        """
        logger.info("Re-fitting t-SNE on combined train+test data for better projection...")
        
        # Get original training embeddings and apply same preprocessing
        train_embeddings_scaled = None
        if hasattr(self, 'train_embeddings_original'):
            train_embeddings_scaled = self.scaler.transform(self.train_embeddings_original)
            train_embeddings_scaled = np.clip(train_embeddings_scaled, -5.0, 5.0)
            train_embeddings_scaled = self.pca.transform(train_embeddings_scaled)
        
        if train_embeddings_scaled is None:
            logger.warning("Original training embeddings not stored, using approximation")
            # Fallback: use nearest neighbor approach
            from sklearn.neighbors import NearestNeighbors
            
            # Use k-NN to map test points to t-SNE space
            nn = NearestNeighbors(n_neighbors=min(3, len(self.train_tsne_embeddings)))
            # We need to use some proxy for original embeddings - use random projection back
            proxy_train = np.random.normal(0, 1, (len(self.train_tsne_embeddings), test_embeddings.shape[1]))
            nn.fit(proxy_train)
            
            distances, indices = nn.kneighbors(test_embeddings)
            
            # Weighted average of nearest neighbors in t-SNE space
            test_tsne_embeddings = np.zeros((len(test_embeddings), self.tsne_components))
            
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                # Use inverse distance weighting
                weights = 1.0 / (dists + 1e-3)
                weights = weights / weights.sum()
                
                test_tsne_embeddings[i] = np.average(
                    self.train_tsne_embeddings[idxs], 
                    weights=weights, 
                    axis=0
                )
            
            return test_tsne_embeddings
        
        # Combine train and test embeddings (both are now PCA-transformed)
        combined_embeddings = np.vstack([train_embeddings_scaled, test_embeddings])
        
        # Clip combined embeddings for numerical stability
        combined_embeddings = np.clip(combined_embeddings, -5.0, 5.0)
        
        # Add small noise to break ties
        combined_embeddings += np.random.normal(0, 1e-6, combined_embeddings.shape)
        
        # Create new t-SNE with same parameters but more conservative settings
        adjusted_perplexity = min(self.tsne_perplexity, (len(combined_embeddings) - 1) / 3)
        adjusted_perplexity = max(5.0, adjusted_perplexity)  # Minimum perplexity of 5
        
        combined_tsne = TSNE(
            n_components=self.tsne_components,
            perplexity=adjusted_perplexity,
            learning_rate=self.tsne_learning_rate,
            random_state=self.tsne_random_state,
            n_jobs=1,  # Use single thread for stability
            init='pca',  # Use PCA initialization for stability
            method='exact' if len(combined_embeddings) < 1000 else 'barnes_hut'  # Use exact method for small datasets
        )
        
        # Fit t-SNE on combined data
        combined_tsne_embeddings = combined_tsne.fit_transform(combined_embeddings)
        
        # Extract test portion
        n_train = len(train_embeddings_scaled)
        test_tsne_embeddings = combined_tsne_embeddings[n_train:]
        
        return test_tsne_embeddings
    
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
        
        # Extract and project embeddings
        test_embeddings = get_dinov2_embeddings(
            test_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="test",
            device=self.device
        )
        
        test_embeddings_scaled = self.scaler.transform(test_embeddings)
        test_tsne_embeddings = self._project_to_tsne_space(test_embeddings_pca)
        
        # Get probabilities
        probabilities = self.knn.predict_proba(test_tsne_embeddings)
        
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
        
        logger.info(f"MARVIS image classifier accuracy: {accuracy:.4f}")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            'dinov2_model': self.dinov2_model,
            'embedding_size': self.embedding_size,
            'tsne_components': self.tsne_components,
            'tsne_perplexity': self.tsne_perplexity,
            'tsne_learning_rate': self.tsne_learning_rate,
            'tsne_random_state': self.tsne_random_state,
            'knn_neighbors': self.knn_neighbors,
            'knn_weights': self.knn_weights,
            'device': self.device
        }