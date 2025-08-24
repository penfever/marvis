#!/usr/bin/env python
"""
Pytest configuration and fixtures for MARVIS tests.

This file contains shared test configuration, fixtures, and hardware detection
logic for the MARVIS test suite.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add MARVIS to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest with hardware-specific markers and settings."""

    # Import torch for hardware detection (only if available)
    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False

    # Hardware detection
    cuda_available = torch_available and torch.cuda.is_available()
    mps_available = (
        torch_available
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    # Add dynamic marker information
    config.addinivalue_line(
        "markers", f"cuda_available: CUDA available={cuda_available}"
    )
    config.addinivalue_line("markers", f"mps_available: MPS available={mps_available}")

    # Register custom markers
    markers = [
        "unit: Fast unit tests with no external dependencies",
        "integration: Integration tests with limited dependencies",
        "e2e: Full end-to-end workflow tests",
        "gpu: Tests requiring GPU/CUDA",
        "mps: Tests requiring Apple Metal Performance Shaders",
        "cpu: CPU-only tests",
        "api: Tests requiring API keys",
        "slow: Tests that may take >30 seconds",
        "manual: Manual tests not run in CI",
        "llamacpp: Tests for LlamaCPP/GGUF functionality",
        "audio: Audio processing tests",
        "vision: Vision processing tests",
        "tabular: Tabular data processing tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)

    # Environment-based test skipping
    if not cuda_available:
        config.addinivalue_line("markers", "skip_no_cuda: Skip if CUDA not available")

    if not mps_available:
        config.addinivalue_line("markers", "skip_no_mps: Skip if MPS not available")

    # Add environment info to test report
    print(f"\n=== MARVIS Test Environment ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch available: {torch_available}")
    if torch_available:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"MPS available: {mps_available}")
    print(f"================================\n")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        cuda_available = False
        mps_available = False

    # Skip conditions
    skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_no_mps = pytest.mark.skip(reason="MPS not available")
    skip_no_api_key = pytest.mark.skip(reason="API key not available")

    for item in items:
        # Auto-apply markers based on file location
        test_path = Path(item.fspath)

        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)

        # Hardware-specific skipping
        if "gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_no_cuda)

        if "mps" in item.keywords and not mps_available:
            item.add_marker(skip_no_mps)

        # API key skipping
        if "api" in item.keywords:
            openai_key = os.getenv("OPENAI_API_KEY")
            google_key = os.getenv("GOOGLE_API_KEY")
            if not openai_key and not google_key:
                item.add_marker(skip_no_api_key)


def pytest_runtest_setup(item):
    """Setup hook called for each test item."""

    # Skip manual tests unless explicitly requested
    if "manual" in item.keywords:
        if not item.config.getoption("--run-manual", default=False):
            pytest.skip("Manual test - use --run-manual to run")

    # Skip slow tests in fast mode
    if "slow" in item.keywords:
        if item.config.getoption("--fast", default=False):
            pytest.skip("Slow test - remove --fast to run")


def pytest_addoption(parser):
    """Add custom command-line options."""

    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run manual tests that require user interaction",
    )

    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests for faster execution",
    )

    parser.addoption(
        "--gpu-only", action="store_true", default=False, help="Only run GPU tests"
    )

    parser.addoption(
        "--cpu-only", action="store_true", default=False, help="Only run CPU tests"
    )


@pytest.fixture(scope="session")
def test_environment():
    """Provide test environment information."""
    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        torch_available = False
        cuda_available = False
        mps_available = False

    return {
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "openai_key_available": bool(os.getenv("OPENAI_API_KEY")),
        "google_key_available": bool(os.getenv("GOOGLE_API_KEY")),
    }


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="marvis_test_"))
    yield temp_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def sample_tabular_data():
    """Generate sample tabular data for testing."""
    try:
        from sklearn.datasets import make_classification
        import numpy as np

        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_classes=3,
            n_informative=4,
            n_redundant=1,
            random_state=42,
            class_sep=1.2,
        )

        return {
            "X": X,
            "y": y,
            "n_samples": 50,
            "n_features": 5,
            "n_classes": 3,
            "class_names": ["Class A", "Class B", "Class C"],
        }
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture
def mock_vlm_response():
    """Provide a mock VLM response for testing."""
    return {
        "response": "Class A",
        "reasoning": "Based on the clustering pattern, this sample belongs to Class A.",
        "confidence": 0.85,
    }


@pytest.fixture(scope="session")
def marvis_test_config():
    """Provide test configuration for MARVIS."""
    return {
        "use_cache": False,  # Disable caching in tests
        "device": "cpu",  # Force CPU for reproducibility
        "seed": 42,  # Fixed seed for reproducible tests
        "timeout": 30,  # Shorter timeout for tests
        "test_mode": True,  # Enable test-specific behavior
    }


# Hardware-specific fixtures
@pytest.fixture
def cuda_device():
    """Provide CUDA device if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def mps_device():
    """Provide MPS device if available."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            pytest.skip("MPS not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def api_keys():
    """Provide API keys for testing."""
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
    }

    if not any(keys.values()):
        pytest.skip("No API keys available")

    return keys


# Test data fixtures
@pytest.fixture(scope="session")
def sample_audio_data():
    """Generate sample audio data for testing."""
    try:
        import numpy as np

        # Generate synthetic audio (simple sine waves)
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Generate different frequency tones
        audio_samples = []
        frequencies = [440, 880, 1320]  # A4, A5, E6

        for freq in frequencies:
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio_samples.append(audio)

        return {
            "audio_samples": audio_samples,
            "sample_rate": sample_rate,
            "frequencies": frequencies,
            "class_names": ["Low Tone", "Mid Tone", "High Tone"],
        }
    except ImportError:
        pytest.skip("NumPy not available")


@pytest.fixture(scope="session")
def sample_image_data():
    """Generate sample image data for testing."""
    try:
        from sklearn.datasets import load_digits

        digits = load_digits(n_class=3)  # Only first 3 classes
        return {
            "images": digits.data[:30],  # First 30 samples
            "labels": digits.target[:30],
            "image_shape": digits.images[0].shape,
            "class_names": ["Digit 0", "Digit 1", "Digit 2"],
        }
    except ImportError:
        pytest.skip("scikit-learn not available")


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield

    # Clean up any temporary files or GPU memory
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Clean up any environment variables that might have been set
    test_env_vars = [key for key in os.environ.keys() if key.startswith("MARVIS_TEST_")]
    for key in test_env_vars:
        os.environ.pop(key, None)
