#!/usr/bin/env python
"""
Simple test script to verify JOLT integration works.
"""

import sys
import os
import numpy as np
import pandas as pd
from argparse import Namespace

# Add paths (dynamically resolve current directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
jolt_dir = os.path.join(project_root, 'examples', 'tabular', 'llm_baselines', 'jolt')
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, jolt_dir)

def test_jolt_integration():
    """Test that JOLT integration works with a simple dataset."""
    print("Testing JOLT integration...")
    
    try:
        from official_jolt_wrapper import evaluate_jolt_official
        
        # Create a simple test dataset
        np.random.seed(42)
        n_samples = 50
        X = np.random.randn(n_samples, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification
        
        # Create test dataset in expected format
        dataset = {
            'name': 'test_dataset',
            'id': 'test',
            'X': pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3']),
            'y': pd.Series(y, name='target'),
            'attribute_names': ['feature1', 'feature2', 'feature3']
        }
        
        # Create test args
        args = Namespace(
            seed=42,
            max_test_samples=10,  # Keep it small for testing
            jolt_model='meta-llama/Llama-2-7b-hf',
            device='cpu'  # Use CPU for testing
        )
        
        print("Running JOLT evaluation...")
        result = evaluate_jolt_official(dataset, args)
        
        print("JOLT evaluation completed!")
        print(f"Result keys: {list(result.keys())}")
        
        if 'error' in result:
            print(f"Error occurred: {result['error']}")
            return False
        else:
            print(f"Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"Completed samples: {result.get('completed_samples', 'N/A')}")
            print(f"Used official JOLT: {result.get('used_official_jolt', False)}")
            return True
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jolt_integration()
    if success:
        print("\n✅ JOLT integration test passed!")
    else:
        print("\n❌ JOLT integration test failed!")