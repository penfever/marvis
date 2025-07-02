#!/usr/bin/env python
"""
Test MPS detection and usage on Mac M4.

This script verifies that MPS is properly detected and used by the transformers backend.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marvis.utils.device_utils import detect_optimal_device, log_platform_info
from marvis.utils.model_loader import model_loader
from marvis.models.marvis_tsne import MarvisTsneClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_platform_detection():
    """Test platform and device detection."""
    print("=== Testing Platform Detection ===")
    
    # Log platform info
    platform_info = log_platform_info(logger)
    
    print(f"Platform: {platform_info['platform']}")
    print(f"PyTorch version: {platform_info['torch_version']}")
    print(f"CUDA available: {platform_info['cuda_available']}")
    print(f"MPS available: {platform_info['mps_available']}")
    print(f"Optimal device: {platform_info['optimal_device']}")
    
    # Test device detection
    device = detect_optimal_device(prefer_mps=True)
    print(f"\nDetected optimal device (prefer_mps=True): {device}")
    
    # Check if MPS is actually available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS backend is available")
        
        # Test tensor creation on MPS
        try:
            test_tensor = torch.randn(10, 10, device='mps')
            print(f"✅ Successfully created tensor on MPS: shape={test_tensor.shape}, device={test_tensor.device}")
        except Exception as e:
            print(f"❌ Failed to create tensor on MPS: {e}")
    else:
        print("❌ MPS backend is not available")
    
    return platform_info['mps_available']


def test_model_loading_with_mps():
    """Test loading a small model with MPS."""
    print("\n=== Testing Model Loading with MPS ===")
    
    # Force VLLM to be unavailable
    os.environ['VLLM_AVAILABLE'] = 'false'
    
    try:
        # Try to load a small model
        print("Loading a small VLM model with transformers backend...")
        
        # Create a simple classifier to test model loading
        classifier = MarvisTsneClassifier(
            vlm_model_id='microsoft/Phi-3.5-vision-instruct',  # Small VLM model
            backend='transformers',
            device='auto',  # Should detect MPS
            max_vlm_image_size=512,
            seed=42
        )
        
        # Check if model was loaded
        if hasattr(classifier, 'vlm_wrapper') and classifier.vlm_wrapper:
            print("✅ Model loaded successfully")
            
            # Check the device of the loaded model
            if hasattr(classifier.vlm_wrapper, '_model') and classifier.vlm_wrapper._model:
                try:
                    # Get device from model parameters
                    device = next(classifier.vlm_wrapper._model.parameters()).device
                    print(f"✅ Model is on device: {device}")
                    
                    if device.type == 'mps':
                        print("✅ Model successfully loaded on MPS!")
                        return True
                    else:
                        print(f"⚠️  Model loaded on {device.type} instead of MPS")
                except Exception as e:
                    print(f"❌ Could not determine model device: {e}")
        else:
            print("❌ Model wrapper not initialized")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    return False


def test_simple_mps_operations():
    """Test simple operations on MPS."""
    print("\n=== Testing Simple MPS Operations ===")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS not available, skipping MPS operations test")
        return False
    
    try:
        # Create tensors on MPS
        a = torch.randn(100, 100, device='mps')
        b = torch.randn(100, 100, device='mps')
        
        # Perform operations
        c = torch.matmul(a, b)
        d = torch.nn.functional.softmax(c, dim=1)
        
        print(f"✅ Matrix multiplication successful: output shape = {c.shape}")
        print(f"✅ Softmax successful: output shape = {d.shape}")
        
        # Test moving tensors between devices
        cpu_tensor = a.cpu()
        mps_tensor_back = cpu_tensor.to('mps')
        
        print("✅ Successfully moved tensors between CPU and MPS")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS operations failed: {e}")
        return False


def main():
    """Run all MPS tests."""
    print("🧪 MPS Detection and Usage Test")
    print("=" * 50)
    
    tests = [
        ("Platform Detection", test_platform_detection),
        ("Simple MPS Operations", test_simple_mps_operations),
        ("Model Loading with MPS", test_model_loading_with_mps)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "⚠️  PARTIAL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR in {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    # Overall assessment
    mps_available = any(result for name, result in results if name == "Platform Detection")
    if mps_available:
        print("\n✅ MPS is available on this system")
        print("📝 Note: Model loading may still use CPU if the model doesn't support MPS")
    else:
        print("\n❌ MPS is not available on this system")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)