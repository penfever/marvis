#!/usr/bin/env python3
"""
Comprehensive audio testing suite.
Tests Whisper embeddings, t-SNE visualization, baseline classifiers, and MARVIS audio pipeline.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_whisper_embeddings():
    """Test Whisper embedding extraction."""
    logger.info("Testing Whisper embedding extraction...")
    
    try:
        from marvis.data.audio_embeddings import load_whisper_model, get_whisper_embeddings
        from examples.audio.audio_datasets import ESC50Dataset
        
        # Load small dataset
        dataset = ESC50Dataset("./esc50_test_data", download=True)
        paths, labels, class_names = dataset.get_samples()
        
        # Use just 10 samples for testing
        test_paths = paths[:10]
        test_labels = labels[:10]
        
        logger.info(f"Testing with {len(test_paths)} audio samples")
        
        # Extract embeddings with tiny model
        embeddings = get_whisper_embeddings(
            test_paths,
            model_name="tiny",
            cache_dir="./cache",
            device="cpu"
        )
        
        logger.info(f"✓ Whisper embeddings extracted: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Whisper embedding test failed: {e}")
        return False


def test_tsne_visualization():
    """Test t-SNE visualization creation."""
    logger.info("Testing t-SNE visualization...")
    
    try:
        from marvis.viz.tsne_functions import create_tsne_visualization
        
        # Create dummy embeddings
        n_samples = 20
        embedding_dim = 384  # Whisper tiny embedding size
        
        train_embeddings = np.random.randn(n_samples, embedding_dim)
        train_labels = np.random.randint(0, 5, n_samples)  # 5 classes
        test_embeddings = np.random.randn(2, embedding_dim)
        
        # Create visualization
        train_tsne, test_tsne, fig = create_tsne_visualization(
            train_embeddings, train_labels, test_embeddings,
            perplexity=min(10, n_samples // 3),
            max_iter=250,  # Minimum allowed by sklearn
            figsize=(8, 6)
        )
        
        logger.info(f"✓ t-SNE visualization created: train={train_tsne.shape}, test={test_tsne.shape}")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ t-SNE visualization test failed: {e}")
        return False


def test_audio_utils():
    """Test audio utility functions."""
    logger.info("Testing audio utilities...")
    
    try:
        from marvis.utils.audio_utils import create_spectrogram, load_audio
        
        # Create dummy audio signal
        sr = 16000
        duration = 2  # seconds
        audio = np.random.randn(sr * duration) * 0.1  # Low amplitude
        
        # Test spectrogram creation
        spec = create_spectrogram(audio, sr, n_mels=128, db_scale=True)
        
        logger.info(f"✓ Spectrogram created: {spec.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Audio utilities test failed: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading."""
    logger.info("Testing dataset loading...")
    
    try:
        from examples.audio.audio_datasets import ESC50Dataset
        
        dataset = ESC50Dataset("./esc50_test_data", download=True)
        
        # Test few-shot split
        splits = dataset.create_few_shot_split(k_shot=2, test_size=0.1, random_state=42)
        
        train_paths, train_labels = splits['train']
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"✓ Dataset loaded: {len(train_paths)} train, {len(test_paths)} test samples")
        logger.info(f"✓ Classes: {len(class_names)} - {', '.join(class_names[:5])}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🎵 Starting MARVIS Audio Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Audio Utilities", test_audio_utils),
        ("t-SNE Visualization", test_tsne_visualization),
        ("Whisper Embeddings", test_whisper_embeddings),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 TEST RESULTS:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🏆 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Audio pipeline is working correctly.")
        logger.info("Ready to run full tests with:")
        logger.info("  python examples/audio/test_esc50.py --quick_test")
    else:
        logger.info("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(tests)


def create_synthetic_test_data(num_samples_per_class=2, num_classes=3, duration=1.0, sample_rate=16000):
    """Create synthetic audio test data."""
    import tempfile
    import shutil
    from pathlib import Path
    from marvis.utils.audio_utils import create_synthetic_audio
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_paths = []
        labels = []
        class_names = [f"class_{i}" for i in range(num_classes)]
        
        for class_idx in range(num_classes):
            for sample_idx in range(num_samples_per_class):
                # Create synthetic audio with different frequencies for each class
                base_freq = 200 + class_idx * 100  # 200, 300, 400 Hz
                audio = create_synthetic_audio(
                    frequency=base_freq + sample_idx * 50,
                    duration=duration,
                    sample_rate=sample_rate
                )
                
                # Save to temporary file
                audio_path = Path(temp_dir) / f"class_{class_idx}_sample_{sample_idx}.wav"
                
                import soundfile as sf
                sf.write(str(audio_path), audio, sample_rate)
                
                audio_paths.append(str(audio_path))
                labels.append(class_idx)
        
        # Convert to absolute paths and copy to persistent location
        persistent_dir = Path("./temp_test_audio")
        persistent_dir.mkdir(exist_ok=True)
        
        persistent_paths = []
        for path in audio_paths:
            dest_path = persistent_dir / Path(path).name
            shutil.copy2(path, dest_path)
            persistent_paths.append(str(dest_path))
        
        return persistent_paths, labels, class_names


def test_whisper_knn_baseline():
    """Test Whisper KNN baseline classifier."""
    logger.info("Testing Whisper KNN baseline classifier...")
    
    try:
        from examples.audio.audio_baselines import WhisperKNNClassifier
        
        # Create test data
        audio_paths, labels, class_names = create_synthetic_test_data(num_samples_per_class=3, num_classes=2)
        
        # Split into train/test
        train_paths = audio_paths[:4]  # 2 per class
        train_labels = labels[:4]
        test_paths = audio_paths[4:]
        test_labels = labels[4:]
        
        # Initialize classifier
        classifier = WhisperKNNClassifier(
            whisper_model="tiny",  # Use tiny for speed
            n_neighbors=3,
            metric="cosine",
            weights="distance",
            standardize=True,
            device="cpu",  # Force CPU for compatibility
            seed=42
        )
        
        # Fit classifier
        logger.info(f"Fitting on {len(train_paths)} training samples...")
        classifier.fit(train_paths, train_labels, class_names)
        
        # Evaluate
        logger.info(f"Evaluating on {len(test_paths)} test samples...")
        results = classifier.evaluate(test_paths, test_labels, return_detailed=True)
        
        logger.info(f"✓ Whisper KNN results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Training time: {results.get('training_time', 0):.2f}s")
        logger.info(f"  Prediction time: {results['prediction_time']:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Whisper KNN baseline test failed: {e}")
        return False


def test_clap_zero_shot_baseline():
    """Test CLAP zero-shot baseline classifier."""
    logger.info("Testing CLAP zero-shot baseline classifier...")
    
    try:
        from examples.audio.audio_baselines import CLAPZeroShotClassifier
        
        # Create test data
        audio_paths, labels, class_names = create_synthetic_test_data(num_samples_per_class=2, num_classes=2)
        
        # Use all data for testing (zero-shot doesn't need training)
        test_paths = audio_paths
        test_labels = labels
        
        # Initialize classifier
        classifier = CLAPZeroShotClassifier(
            version="2023",  # Use 2023 version
            use_cuda=False,  # Force CPU for compatibility
            batch_size=2  # Small batch size
        )
        
        # "Fit" classifier (just sets up class names)
        logger.info("Setting up CLAP classifier...")
        classifier.fit([], [], class_names)  # Empty training data for zero-shot
        
        # Evaluate
        logger.info(f"Evaluating on {len(test_paths)} test samples...")
        results = classifier.evaluate(test_paths, test_labels, return_detailed=True)
        
        logger.info(f"✓ CLAP zero-shot results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Training time: {results.get('training_time', 0):.2f}s")
        logger.info(f"  Prediction time: {results['prediction_time']:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"CLAP zero-shot baseline test failed: {e}")
        return False


def test_marvis_audio_minimal():
    """Test MARVIS audio classifier with minimal synthetic data."""
    logger.info("Testing MARVIS audio classifier with minimal synthetic data...")
    
    try:
        from examples.audio.marvis_tsne_audio_baseline import MarvisAudioTsneClassifier
        from marvis.utils.audio_utils import create_synthetic_audio
        from pathlib import Path
        import shutil
        
        # Create minimal test data
        temp_dir = Path("./temp_minimal_test")
        temp_dir.mkdir(exist_ok=True)
        
        audio_paths = []
        labels = []
        
        # Create 4 files: 2 classes, 2 samples each
        for class_idx in range(2):
            for sample_idx in range(2):
                # Create synthetic audio
                frequency = 200 + class_idx * 200  # 200Hz, 400Hz
                audio = create_synthetic_audio(
                    frequency=frequency,
                    duration=1.0,
                    sample_rate=16000
                )
                
                # Save to file
                import soundfile as sf
                audio_path = temp_dir / f"class_{class_idx}_sample_{sample_idx}.wav"
                sf.write(str(audio_path), audio, 16000)
                
                audio_paths.append(str(audio_path))
                labels.append(class_idx)
        
        class_names = ["class_0", "class_1"]
        
        # Split into train/test
        train_paths = audio_paths[:2]  # 1 per class
        train_labels = labels[:2]
        test_paths = audio_paths[2:]   # 1 per class for testing
        test_labels = labels[2:]
        
        logger.info(f"Train: {len(train_paths)} samples, Test: {len(test_paths)} samples")
        
        # Initialize classifier with minimal settings
        classifier = MarvisAudioTsneClassifier(
            whisper_model="tiny",  # Fastest model
            embedding_layer="encoder_last",
            tsne_perplexity=1.0,  # Very small for 2 points
            tsne_max_iter=250,      # Minimum
            vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            use_3d_tsne=False,
            use_knn_connections=True,  # Test the KNN fix
            nn_k=1,              # Only 1 neighbor available
            max_vlm_image_size=512,
            zoom_factor=2.0,
            use_pca_backend=False,
            include_spectrogram=False,
            audio_duration=1.0,
            cache_dir="./temp_cache",
            device="cpu",  # Force CPU for compatibility
            seed=42
        )
        
        # Fit classifier
        logger.info("Fitting classifier...")
        classifier.fit(train_paths, train_labels, test_paths, class_names)
        
        # Test prediction on just one sample
        logger.info("Testing prediction...")
        prediction = classifier.predict([test_paths[0]])
        
        logger.info(f"✓ Prediction successful: {prediction}")
        
        # Cleanup
        shutil.rmtree("./temp_minimal_test", ignore_errors=True)
        shutil.rmtree("./temp_cache", ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"MARVIS audio minimal test failed: {e}")
        # Cleanup
        import shutil
        shutil.rmtree("./temp_minimal_test", ignore_errors=True)
        shutil.rmtree("./temp_cache", ignore_errors=True)
        return False


def cleanup_test_data():
    """Clean up temporary test data."""
    import shutil
    from pathlib import Path
    
    test_dirs = ["./temp_test_audio", "./temp_minimal_test", "./temp_cache"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    logger.info("Cleaned up test data")


def run_comprehensive_tests():
    """Run comprehensive audio tests including baselines."""
    logger.info("Running comprehensive audio tests...")
    
    # Run original tests
    original_success = main()
    
    # Run baseline tests
    baseline_tests = [
        ("Whisper KNN Baseline", test_whisper_knn_baseline),
        ("CLAP Zero-Shot Baseline", test_clap_zero_shot_baseline),
        ("MARVIS Audio Minimal", test_marvis_audio_minimal),
    ]
    
    baseline_results = []
    for test_name, test_func in baseline_tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}...")
        try:
            success = test_func()
            baseline_results.append((test_name, success))
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            baseline_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE AUDIO TEST RESULTS")
    logger.info("="*60)
    
    logger.info(f"Original Pipeline Tests: {'✅ PASS' if original_success else '❌ FAIL'}")
    
    baseline_passed = 0
    for test_name, success in baseline_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            baseline_passed += 1
    
    total_passed = (1 if original_success else 0) + baseline_passed
    total_tests = 1 + len(baseline_tests)
    
    logger.info(f"\n🏆 Overall: {total_passed}/{total_tests} test suites passed")
    
    # Cleanup
    cleanup_test_data()
    
    if total_passed == total_tests:
        logger.info("🎉 All comprehensive audio tests passed!")
        return True
    else:
        logger.info("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    import sys
    
    # Check for comprehensive test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        success = run_comprehensive_tests()
    else:
        success = main()
    
    sys.exit(0 if success else 1)