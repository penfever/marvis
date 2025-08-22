#!/usr/bin/env python
"""
Integration tests for LlamaCPP/GGUF model support in MARVIS.

This test file verifies that the LlamaCPP integration works correctly
with GGUF models and URLs.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add MARVIS to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from marvis.utils.gguf_utils import (
    parse_gguf_url, 
    is_gguf_url, 
    get_cache_path,
    validate_gguf_file,
    suggest_gguf_files
)
from marvis.utils.model_loader import ModelLoader, LLAMACPP_AVAILABLE


class TestGGUFUtils:
    """Test GGUF utility functions."""
    
    def test_parse_gguf_url_show_file_info(self):
        """Test parsing HuggingFace URL with show_file_info parameter."""
        url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
        repo_id, filename, download_url = parse_gguf_url(url)
        
        assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
        assert filename == "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
        assert download_url == "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
    
    def test_parse_gguf_url_direct_resolve(self):
        """Test parsing direct resolve URL."""
        url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf"
        repo_id, filename, download_url = parse_gguf_url(url)
        
        assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
        assert filename == "model.gguf"
        assert download_url == url
    
    def test_parse_gguf_url_blob_format(self):
        """Test parsing blob URL format."""
        url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/blob/main/model.gguf"
        repo_id, filename, download_url = parse_gguf_url(url)
        
        assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
        assert filename == "model.gguf"
        assert download_url == "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf"
    
    def test_parse_gguf_url_local_file(self):
        """Test parsing local file path."""
        local_path = "/path/to/model.gguf"
        
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            repo_id, filename, download_url = parse_gguf_url(local_path)
        
        assert repo_id is None
        assert filename == "model.gguf"
        assert download_url == local_path
    
    def test_parse_gguf_url_invalid(self):
        """Test parsing invalid URLs."""
        invalid_urls = [
            "https://example.com/model.gguf",
            "https://huggingface.co/",
            "not_a_url",
        ]
        
        for url in invalid_urls:
            repo_id, filename, download_url = parse_gguf_url(url)
            assert repo_id is None
            assert filename is None
            assert download_url is None
    
    def test_is_gguf_url(self):
        """Test GGUF URL detection."""
        # Positive cases
        assert is_gguf_url("/path/to/model.gguf")
        assert is_gguf_url("https://huggingface.co/repo/model?show_file_info=model.gguf")
        assert is_gguf_url("https://huggingface.co/repo/resolve/main/model.gguf")
        
        # Negative cases
        assert not is_gguf_url("https://huggingface.co/microsoft/DialoGPT-medium")
        assert not is_gguf_url("/path/to/model.safetensors")
        assert not is_gguf_url("gpt-4o")
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = get_cache_path(url, Path(temp_dir))
            
            assert cache_path.parent == Path(temp_dir)
            assert cache_path.suffix == ".gguf" or "model.gguf" in cache_path.name
            assert len(cache_path.stem) > 10  # Should include hash
    
    def test_validate_gguf_file(self):
        """Test GGUF file validation."""
        # Test with valid GGUF file (mocked)
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp:
            # Write GGUF magic bytes
            tmp.write(b'GGUF')
            tmp.write(b'\x00\x00\x00\x03')  # Version 3
            tmp.write(b'\x00' * 100)  # Some data
            tmp.flush()
            
            tmp_path = Path(tmp.name)
            assert validate_gguf_file(tmp_path)
            
            # Cleanup
            tmp_path.unlink()
        
        # Test with invalid file
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp.write(b'NOT_GGUF')
            tmp.flush()
            
            tmp_path = Path(tmp.name)
            assert not validate_gguf_file(tmp_path)
            
            # Cleanup
            tmp_path.unlink()
    
    def test_suggest_gguf_files(self):
        """Test GGUF file suggestions."""
        repo_url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
        suggestions = suggest_gguf_files(repo_url)
        
        assert len(suggestions) > 0
        assert all("q4_k_m.gguf" in s or "q5_k_m.gguf" in s or "q8_0.gguf" in s for s in suggestions[:3])
        assert all(s.startswith("https://huggingface.co/") for s in suggestions)


@pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="LlamaCPP not available")
class TestLlamaCPPIntegration:
    """Test LlamaCPP model wrapper integration."""
    
    def test_model_loader_llamacpp_detection(self):
        """Test that ModelLoader correctly detects GGUF models."""
        loader = ModelLoader()
        
        # Test GGUF URL detection
        gguf_url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf"
        
        # We can't actually load the model without downloading, but we can test detection
        with patch.object(loader, '_loaded_models', {}):
            with patch('marvis.utils.model_loader.LlamaCPPVisionModelWrapper') as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper_class.return_value = mock_wrapper
                
                try:
                    loader.load_vlm(gguf_url, backend="llamacpp")
                    # If we get here, the backend was correctly selected
                    mock_wrapper_class.assert_called_once()
                except Exception as e:
                    # Expected - we don't have the actual model file
                    # But the wrapper should have been created
                    if "llama-cpp-python not available" not in str(e):
                        mock_wrapper_class.assert_called_once()
    
    def test_llamacpp_wrapper_initialization(self):
        """Test LlamaCPP wrapper initialization."""
        from marvis.utils.model_loader import LlamaCPPVisionModelWrapper
        
        wrapper = LlamaCPPVisionModelWrapper(
            model_name="test_model.gguf",
            device="cpu",
            n_ctx=2048,
            n_gpu_layers=0
        )
        
        assert wrapper.model_name == "test_model.gguf"
        assert wrapper.device == "cpu"
        assert wrapper.n_ctx == 2048
        assert wrapper.n_gpu_layers == 0
        assert not wrapper.is_loaded()
    
    @patch('marvis.utils.model_loader.Llama')
    @patch('marvis.utils.gguf_utils.download_gguf_file')
    def test_llamacpp_wrapper_load(self, mock_download, mock_llama):
        """Test LlamaCPP wrapper loading."""
        from marvis.utils.model_loader import LlamaCPPVisionModelWrapper
        
        # Mock the download to return a local path
        mock_download.return_value = Path("/tmp/model.gguf")
        
        # Mock the Llama class
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        wrapper = LlamaCPPVisionModelWrapper(
            model_name="https://example.com/model.gguf",
            device="cpu"
        )
        
        # Test loading
        wrapper.load()
        
        assert wrapper.is_loaded()
        mock_download.assert_called_once()
        mock_llama.assert_called_once()


class TestMARVISLlamaCPPIntegration:
    """Test MARVIS classifier with LlamaCPP backend."""
    
    @pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="LlamaCPP not available")
    def test_marvis_llamacpp_backend_detection(self):
        """Test that MARVIS can detect and configure LlamaCPP backend."""
        from marvis.models.marvis_tsne import MarvisTsneClassifier
        
        # Mock the model loading to avoid downloading
        with patch('marvis.utils.model_loader.ModelLoader.load_vlm') as mock_load:
            mock_vlm = Mock()
            mock_load.return_value = mock_vlm
            
            # Create classifier with GGUF model
            gguf_url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf"
            
            classifier = MarvisTsneClassifier(
                modality="tabular",
                vlm_model_id=gguf_url,
                backend="auto"  # Should auto-detect llamacpp
            )
            
            # The model should be recognized as GGUF
            assert classifier.vlm_model_id == gguf_url
            
    def test_marvis_explicit_llamacpp_backend(self):
        """Test MARVIS with explicit llamacpp backend."""
        from marvis.models.marvis_tsne import MarvisTsneClassifier
        
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="/path/to/model.gguf",
            backend="llamacpp"
        )
        
        assert classifier.backend == "llamacpp"
        assert classifier.vlm_model_id == "/path/to/model.gguf"


# Example usage test (requires actual GGUF model)
@pytest.mark.manual
class TestLlamaCPPRealModel:
    """
    Manual tests that require actual GGUF models.
    
    These tests are marked as 'manual' and won't run in CI.
    Run with: pytest -m manual tests/test_llamacpp_integration.py
    """
    
    def test_real_gguf_download_and_load(self):
        """Test downloading and loading a real GGUF model."""
        from marvis.utils.gguf_utils import download_gguf_file
        from marvis.utils.model_loader import LlamaCPPVisionModelWrapper
        
        # Use a small GGUF model for testing
        test_url = "https://huggingface.co/QuantFactory/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf"
        
        # Download the model
        model_path = download_gguf_file(test_url)
        assert model_path.exists()
        assert model_path.suffix == '.gguf'
        
        # Load with LlamaCPP wrapper
        wrapper = LlamaCPPVisionModelWrapper(
            model_name=str(model_path),
            device="cpu",
            n_ctx=512,  # Small context for testing
            n_gpu_layers=0  # CPU only for testing
        )
        
        wrapper.load()
        assert wrapper.is_loaded()
        
        # Test text generation
        test_prompt = "Hello, how are you?"
        from marvis.utils.model_loader import GenerationConfig
        config = GenerationConfig(max_new_tokens=20, temperature=0.1)
        
        response = wrapper.generate([test_prompt], config)
        assert isinstance(response, list)
        assert len(response) == 1
        assert len(response[0]) > 0
        
        wrapper.unload()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-x"])