#!/usr/bin/env python
"""
Unit tests for GGUF utilities and URL parsing.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_parse_gguf_url_show_file_info():
    """Test parsing HuggingFace URL with show_file_info parameter."""
    from marvis.utils.gguf_utils import parse_gguf_url

    url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
    repo_id, filename, download_url = parse_gguf_url(url)

    assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
    assert filename == "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
    assert (
        download_url
        == "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
    )


@pytest.mark.unit
def test_parse_gguf_url_direct_resolve():
    """Test parsing direct resolve URL."""
    from marvis.utils.gguf_utils import parse_gguf_url

    url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf"
    repo_id, filename, download_url = parse_gguf_url(url)

    assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
    assert filename == "model.gguf"
    assert download_url == url


@pytest.mark.unit
def test_parse_gguf_url_blob_format():
    """Test parsing blob URL format."""
    from marvis.utils.gguf_utils import parse_gguf_url

    url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/blob/main/model.gguf"
    repo_id, filename, download_url = parse_gguf_url(url)

    assert repo_id == "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
    assert filename == "model.gguf"
    assert (
        download_url
        == "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf"
    )


@pytest.mark.unit
def test_parse_gguf_url_local_file():
    """Test parsing local file path."""
    from marvis.utils.gguf_utils import parse_gguf_url

    local_path = "/path/to/model.gguf"

    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True):
        repo_id, filename, download_url = parse_gguf_url(local_path)

    assert repo_id is None
    assert filename == "model.gguf"
    assert download_url == local_path


@pytest.mark.unit
def test_parse_gguf_url_invalid():
    """Test parsing invalid URLs."""
    from marvis.utils.gguf_utils import parse_gguf_url

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


@pytest.mark.unit
def test_is_gguf_url():
    """Test GGUF URL detection."""
    from marvis.utils.gguf_utils import is_gguf_url

    # Positive cases
    assert is_gguf_url("/path/to/model.gguf")
    assert is_gguf_url("https://huggingface.co/repo/model?show_file_info=model.gguf")
    assert is_gguf_url("https://huggingface.co/repo/resolve/main/model.gguf")

    # Negative cases
    assert not is_gguf_url("https://huggingface.co/microsoft/DialoGPT-medium")
    assert not is_gguf_url("/path/to/model.safetensors")
    assert not is_gguf_url("gpt-4o")


@pytest.mark.unit
def test_get_cache_path():
    """Test cache path generation."""
    from marvis.utils.gguf_utils import get_cache_path

    url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = get_cache_path(url, Path(temp_dir))

        assert cache_path.parent == Path(temp_dir)
        assert cache_path.suffix == ".gguf" or "model.gguf" in cache_path.name
        assert len(cache_path.stem) > 10  # Should include hash


@pytest.mark.unit
def test_validate_gguf_file():
    """Test GGUF file validation."""
    from marvis.utils.gguf_utils import validate_gguf_file

    # Test with valid GGUF file (mocked)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        # Write GGUF magic bytes
        tmp.write(b"GGUF")
        tmp.write(b"\x00\x00\x00\x03")  # Version 3
        tmp.write(b"\x00" * 100)  # Some data
        tmp.flush()

        tmp_path = Path(tmp.name)
        assert validate_gguf_file(tmp_path)

        # Cleanup
        tmp_path.unlink()

    # Test with invalid file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(b"NOT_GGUF")
        tmp.flush()

        tmp_path = Path(tmp.name)
        assert not validate_gguf_file(tmp_path)

        # Cleanup
        tmp_path.unlink()


@pytest.mark.unit
def test_suggest_gguf_files():
    """Test GGUF file suggestions."""
    from marvis.utils.gguf_utils import suggest_gguf_files

    repo_url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
    suggestions = suggest_gguf_files(repo_url)

    # Debug output
    print(f"Number of suggestions: {len(suggestions)}")
    if suggestions:
        print(f"First 3 suggestions: {suggestions[:3]}")

    assert len(suggestions) > 0, f"Expected suggestions but got {len(suggestions)}"

    # Check the pattern matching more explicitly
    first_three = suggestions[:3]
    # Include more GGUF patterns that are actually generated
    pattern_matches = [
        any(
            pattern in s
            for pattern in [
                "q4_k_m.gguf",
                "q4_k_s.gguf",
                "q5_k_m.gguf",
                "q5_k_s.gguf",
                "q6_k.gguf",
                "q8_0.gguf",
            ]
        )
        for s in first_three
    ]
    print(f"Pattern matches for first 3: {pattern_matches}")

    assert all(pattern_matches), f"Pattern check failed for: {first_three}"
    assert all(
        s.startswith("https://huggingface.co/") for s in suggestions
    ), f"URL format check failed"


@pytest.mark.unit
def test_get_gguf_metadata():
    """Test GGUF metadata extraction."""
    from marvis.utils.gguf_utils import get_gguf_metadata

    # Test with valid GGUF file (mocked)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        # Write GGUF magic bytes and version
        tmp.write(b"GGUF")
        tmp.write(b"\x03\x00\x00\x00")  # Version 3 in little endian
        tmp.write(b"\x00" * 100)  # Some data
        tmp.flush()

        tmp_path = Path(tmp.name)
        metadata = get_gguf_metadata(tmp_path)

        assert metadata["valid"] is True
        assert metadata["magic"] == "GGUF"
        assert metadata["version"] == 3
        assert metadata["file_size"] > 0
        assert "file_size_mb" in metadata

        # Cleanup
        tmp_path.unlink()

    # Test with non-existent file
    non_existent = Path("/non/existent/file.gguf")
    metadata = get_gguf_metadata(non_existent)
    assert metadata["valid"] is False
