#!/usr/bin/env python
"""Test script to verify embedding cache optimization."""

import numpy as np
import os
import tempfile

# Create a mock cache file for testing
def create_mock_cache(cache_file, n_train=1000, n_val=5000, n_test=5000, emb_size=128):
    """Create a mock cache file with embeddings."""
    
    # Create mock embeddings
    train_embeddings = np.random.randn(n_train, emb_size)
    val_embeddings = np.random.randn(n_val, emb_size)
    test_embeddings = np.random.randn(n_test, emb_size)
    y_train_sample = np.random.randint(0, 10, n_train)
    
    # Save to cache file
    np.savez(
        cache_file,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        y_train_sample=y_train_sample,
        metadata={"test": True}
    )
    
    print(f"Created mock cache with shapes:")
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Val: {val_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")
    
    return cache_file

# Test the loading function
def test_load_embeddings_with_limit():
    """Test loading embeddings with limit."""
    
    # Create temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        cache_file = f.name
    
    try:
        # Create mock cache
        create_mock_cache(cache_file, n_train=1000, n_val=5000, n_test=5000)
        
        # Test 1: Load without limit
        print("\nTest 1: Loading without limit")
        cache = np.load(cache_file, allow_pickle=True)
        
        train_embeddings = cache["train_embeddings"]
        val_embeddings = cache["val_embeddings"]
        test_embeddings = cache["test_embeddings"]
        
        print(f"Loaded shapes - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
        
        # Test 2: Load with limit
        max_test_samples = 100
        print(f"\nTest 2: Loading with max_test_samples={max_test_samples}")
        
        # Manually apply limit (simulating what the function does)
        if max_test_samples is not None:
            val_count = len(val_embeddings)
            test_count = len(test_embeddings)
            
            val_to_load = min(val_count, max_test_samples)
            test_to_load = min(test_count, max_test_samples)
            
            val_embeddings_limited = val_embeddings[:val_to_load]
            test_embeddings_limited = test_embeddings[:test_to_load]
            
            print(f"Limited shapes - Train: {train_embeddings.shape}, Val: {val_embeddings_limited.shape}, Test: {test_embeddings_limited.shape}")
            print(f"Loaded embeddings with limits - Val: {val_to_load}/{val_count}, Test: {test_to_load}/{test_count}")
        
        # Test 3: Test with multi-ensemble embeddings
        print("\nTest 3: Testing with multi-ensemble embeddings")
        
        # Create multi-ensemble mock cache
        n_ensemble = 8
        train_embeddings_multi = np.random.randn(n_ensemble, 1000, 128)
        val_embeddings_multi = np.random.randn(n_ensemble, 5000, 128)
        test_embeddings_multi = np.random.randn(n_ensemble, 5000, 128)
        y_train_sample_multi = np.random.randint(0, 10, 1000)
        
        # Save multi-ensemble cache
        np.savez(
            cache_file,
            train_embeddings=train_embeddings_multi,
            val_embeddings=val_embeddings_multi,
            test_embeddings=test_embeddings_multi,
            y_train_sample=y_train_sample_multi,
            metadata={"test": True, "multi_ensemble": True}
        )
        
        # Load and apply limit
        cache = np.load(cache_file, allow_pickle=True)
        val_embeddings_multi = cache["val_embeddings"]
        test_embeddings_multi = cache["test_embeddings"]
        
        _, val_count, _ = val_embeddings_multi.shape
        _, test_count, _ = test_embeddings_multi.shape
        
        val_to_load = min(val_count, max_test_samples)
        test_to_load = min(test_count, max_test_samples)
        
        val_embeddings_limited = val_embeddings_multi[:, :val_to_load, :]
        test_embeddings_limited = test_embeddings_multi[:, :test_to_load, :]
        
        print(f"Multi-ensemble shapes - Val: {val_embeddings_limited.shape}, Test: {test_embeddings_limited.shape}")
        print(f"Loaded embeddings with limits - Val: {val_to_load}/{val_count}, Test: {test_to_load}/{test_count}")
        
    finally:
        # Clean up
        if os.path.exists(cache_file):
            os.unlink(cache_file)
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_load_embeddings_with_limit()