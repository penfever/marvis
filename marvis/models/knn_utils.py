"""
KNN utilities for TabPFN embeddings analysis.

This module provides utilities for performing K-nearest neighbors analysis
on TabPFN embeddings, including finding nearest neighbors and analyzing
influential training points.
"""

import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler

__all__ = [
    'find_knn_in_embedding_space',
    'find_influential_training_points',
    'get_neighbor_distances_and_indices'
]

logger = logging.getLogger(__name__)


def find_knn_in_embedding_space(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 5,
    metric: str = 'euclidean',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find K-nearest neighbors for query points in the embedding space.
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        query_embeddings: Query embeddings [n_query, embedding_dim]
        k: Number of nearest neighbors to find
        metric: Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
        normalize: Whether to normalize embeddings before KNN
        
    Returns:
        distances: Distance to k-nearest neighbors [n_query, k]
        indices: Indices of k-nearest neighbors in train_embeddings [n_query, k]
    """
    # Ensure query_embeddings is always 2D (fix for single sample case)
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)
        logger.debug(f"Reshaped 1D query embedding to 2D: {query_embeddings.shape}")
    
    # Reduce logging verbosity - only log for the first query or larger batches
    if len(query_embeddings) > 1:
        logger.info(f"Finding {k} nearest neighbors for {len(query_embeddings)} query points")
    
    # Normalize embeddings if requested
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        query_embeddings_scaled = scaler.transform(query_embeddings)
    else:
        train_embeddings_scaled = train_embeddings
        query_embeddings_scaled = query_embeddings
    
    # Adjust k if necessary
    max_k = min(k, len(train_embeddings))
    if max_k != k:
        logger.warning(f"Adjusting k from {k} to {max_k} due to limited training data")
        k = max_k
    
    # Create and fit KNN model
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(train_embeddings_scaled)
    
    # Find neighbors
    distances, indices = knn.kneighbors(query_embeddings_scaled)
    
    # Only log completion for larger batches
    if len(query_embeddings) > 1:
        logger.info(f"KNN completed. Average distance to nearest neighbor: {np.mean(distances[:, 0]):.4f}")
    
    return distances, indices


def find_influential_training_points(
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    k: int = 5,
    influence_threshold_factor: float = 1.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Find influential training points based on KNN analysis.
    
    This function identifies training points that are frequently selected as
    nearest neighbors for validation points, indicating their importance.
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        val_embeddings: Validation embeddings [n_val, embedding_dim]
        k: Number of nearest neighbors to consider
        influence_threshold_factor: Multiplier for influence threshold
        
    Returns:
        influential_indices: Indices of influential training points
        metadata: Dictionary with analysis metadata
    """
    logger.info(f"Analyzing influential training points with k={k}")
    
    # Find KNN for validation points
    distances, indices = find_knn_in_embedding_space(
        train_embeddings, val_embeddings, k=k
    )
    
    # Track how many times each training point is a nearest neighbor
    influence_count = np.zeros(len(train_embeddings))
    
    # Count neighbor occurrences
    for neighbors in indices:
        for neighbor_idx in neighbors:
            if neighbor_idx < len(influence_count):
                influence_count[neighbor_idx] += 1
    
    # Calculate influence threshold
    influence_count_nonzero = influence_count[influence_count > 0]
    if len(influence_count_nonzero) > 0:
        avg_count = np.mean(influence_count_nonzero)
        influence_threshold = avg_count * influence_threshold_factor
    else:
        influence_threshold = 1
    
    # Find highly influential points
    influential_indices = np.where(influence_count >= influence_threshold)[0]
    
    # Also include points that are nearest neighbors to uncertain validation points
    distance_threshold = np.percentile(distances[:, 0], 75)  # 75th percentile
    uncertain_val_indices = np.where(distances[:, 0] > distance_threshold)[0]
    uncertain_neighbors = indices[uncertain_val_indices, 0]  # First neighbor only
    
    # Combine and remove duplicates
    all_influential = np.unique(np.concatenate([influential_indices, uncertain_neighbors]))
    
    # Create metadata
    metadata = {
        'total_training_points': len(train_embeddings),
        'total_validation_points': len(val_embeddings),
        'k': k,
        'influence_threshold': influence_threshold,
        'avg_influence_count': avg_count if len(influence_count_nonzero) > 0 else 0,
        'n_influential_points': len(all_influential),
        'influence_percentage': len(all_influential) / len(train_embeddings) * 100,
        'distance_threshold': distance_threshold,
        'n_uncertain_points': len(uncertain_val_indices)
    }
    
    logger.info(f"Found {len(all_influential)} influential points ({metadata['influence_percentage']:.2f}%)")
    
    return all_influential, metadata


def get_neighbor_distances_and_indices(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k: int = 5,
    return_class_distribution: bool = True
) -> Dict[str, Any]:
    """
    Get detailed neighbor information for each test point.
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        train_labels: Training labels [n_train]
        k: Number of nearest neighbors
        return_class_distribution: Whether to include class distribution info
        
    Returns:
        Dictionary containing neighbor information for each test point
    """
    logger.info(f"Getting detailed neighbor information for {len(test_embeddings)} test points")
    
    # Find KNN
    distances, indices = find_knn_in_embedding_space(
        train_embeddings, test_embeddings, k=k
    )
    
    neighbor_info = {
        'distances': distances,
        'indices': indices,
        'k': k
    }
    
    if return_class_distribution:
        # Get class distribution for neighbors of each test point
        neighbor_classes = []
        class_distributions = []
        
        for i in range(len(test_embeddings)):
            # Get classes of neighbors for this test point
            neighbor_idx = indices[i]
            neighbor_class = train_labels[neighbor_idx]
            neighbor_classes.append(neighbor_class)
            
            # Calculate class distribution
            unique_classes, counts = np.unique(neighbor_class, return_counts=True)
            class_dist = dict(zip(unique_classes, counts / len(neighbor_class)))
            class_distributions.append(class_dist)
        
        neighbor_info['neighbor_classes'] = neighbor_classes
        neighbor_info['class_distributions'] = class_distributions
        
        # Calculate overall statistics
        all_neighbor_classes = np.concatenate(neighbor_classes)
        overall_unique, overall_counts = np.unique(all_neighbor_classes, return_counts=True)
        neighbor_info['overall_class_distribution'] = dict(zip(overall_unique, overall_counts))
    
    return neighbor_info


def analyze_knn_performance(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Analyze KNN classification performance on embeddings.
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        train_labels: Training labels [n_train]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        test_labels: Test labels [n_test]
        k_values: List of k values to test (default: [1, 3, 5, 10, 15])
        
    Returns:
        Dictionary with performance results for different k values
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    if k_values is None:
        k_values = [1, 3, 5, 10, 15]
    
    logger.info(f"Analyzing KNN performance for k values: {k_values}")
    
    results = {}
    
    for k in k_values:
        if k > len(train_embeddings):
            logger.warning(f"Skipping k={k} (larger than training set size)")
            continue
            
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings, train_labels)
        
        # Make predictions
        predictions = knn.predict(test_embeddings)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        results[f'k_{k}'] = {
            'k': k,
            'accuracy': accuracy,
            'predictions': predictions
        }
        
        logger.info(f"KNN (k={k}) accuracy: {accuracy:.4f}")
    
    # Find best k
    best_k = max(results.keys(), key=lambda x: results[x]['accuracy'])
    results['best_k'] = results[best_k]['k']
    results['best_accuracy'] = results[best_k]['accuracy']
    
    logger.info(f"Best KNN performance: k={results['best_k']}, accuracy={results['best_accuracy']:.4f}")
    
    return results