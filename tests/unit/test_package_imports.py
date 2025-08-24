#!/usr/bin/env python
"""
Unit tests to verify that the marvis package imports are working correctly.
"""

import sys

import pytest


@pytest.mark.unit
def test_marvis_package_import():
    """Test that the main marvis package can be imported."""
    import marvis

    assert hasattr(marvis, "__version__")
    assert hasattr(marvis, "__file__")


@pytest.mark.unit
def test_marvis_submodules_import():
    """Test that all marvis submodules can be imported."""
    modules = ["data", "models", "train", "utils", "viz"]

    for module_name in modules:
        module = __import__(f"marvis.{module_name}", fromlist=[""])
        assert hasattr(module, "__file__")


@pytest.mark.unit
def test_core_classes_import():
    """Test that core MARVIS classes can be imported."""

    # Test MarvisTsneClassifier
    from marvis.models.marvis_tsne import MarvisTsneClassifier

    assert MarvisTsneClassifier is not None

    # Test ModelLoader
    from marvis.utils.model_loader import ModelLoader

    assert ModelLoader is not None

    # Test utilities
    from marvis.utils.vlm_prompting import create_classification_prompt

    assert create_classification_prompt is not None


@pytest.mark.unit
def test_data_functions_import():
    """Test that data functions can be imported."""

    try:
        from marvis.data import load_dataset

        assert callable(load_dataset)
    except ImportError:
        pytest.skip("load_dataset not available")

    try:
        from marvis.data import get_tabpfn_embeddings

        assert callable(get_tabpfn_embeddings)
    except ImportError:
        pytest.skip("get_tabpfn_embeddings not available")

    try:
        from marvis.data import create_llm_dataset

        assert callable(create_llm_dataset)
    except ImportError:
        pytest.skip("create_llm_dataset not available")


@pytest.mark.unit
def test_optional_imports():
    """Test optional imports that may not be available."""

    # Test LlamaCPP imports (may not be available)
    try:
        from marvis.utils.model_loader import LLAMACPP_AVAILABLE

        # Should not raise an error even if False
        assert isinstance(LLAMACPP_AVAILABLE, bool)
    except ImportError:
        pytest.fail("LLAMACPP_AVAILABLE should be importable even if False")

    # Test GGUF utils
    from marvis.utils.gguf_utils import is_gguf_url, parse_gguf_url

    assert callable(is_gguf_url)
    assert callable(parse_gguf_url)


@pytest.mark.unit
def test_version_format():
    """Test that the version string is properly formatted."""
    import marvis

    version = marvis.__version__
    assert isinstance(version, str)
    assert len(version) > 0

    # Basic semantic versioning check (major.minor.patch)
    parts = version.split(".")
    assert len(parts) >= 2  # At least major.minor


def main():
    """Legacy main function for backward compatibility."""
    print("Running MARVIS package import tests...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Run the tests using pytest programmatically
    exit_code = pytest.main([__file__, "-v"])

    if exit_code == 0:
        print("\n✅ All package import tests passed!")
    else:
        print("\n❌ Some package import tests failed!")

    return exit_code == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
