#!/usr/bin/env python
"""
GGUF utilities for handling GGUF model files and URLs.

This module provides functionality to:
- Parse HuggingFace GGUF URLs
- Download and cache GGUF files
- Validate GGUF format and integrity
"""

import hashlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "marvis" / "gguf"


def parse_gguf_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a HuggingFace GGUF URL to extract repository and file information.

    Supports formats:
    - https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf
    - https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf
    - https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/blob/main/model.gguf

    Args:
        url: HuggingFace URL to parse

    Returns:
        Tuple of (repo_id, filename, direct_download_url)
    """
    if not url.startswith("https://huggingface.co/"):
        # Local file path - return as-is
        if url.endswith(".gguf") and os.path.exists(url):
            return None, os.path.basename(url), url
        return None, None, None

    try:
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            logger.warning(f"Invalid HuggingFace URL format: {url}")
            return None, None, None

        # Extract repository ID (first two path components)
        repo_id = "/".join(path_parts[:2])

        # Case 1: URL with show_file_info parameter
        if "show_file_info" in parsed.query:
            query_params = parse_qs(parsed.query)
            filename = query_params["show_file_info"][0]
            # Convert to direct download URL
            direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            return repo_id, filename, direct_url

        # Case 2: Direct resolve/main URL
        elif "resolve" in path_parts and "main" in path_parts:
            resolve_idx = path_parts.index("resolve")
            main_idx = path_parts.index("main")
            if main_idx == resolve_idx + 1 and len(path_parts) > main_idx + 1:
                filename = "/".join(path_parts[main_idx + 1 :])
                return repo_id, filename, url

        # Case 3: Blob URL - convert to resolve
        elif "blob" in path_parts and "main" in path_parts:
            blob_idx = path_parts.index("blob")
            main_idx = path_parts.index("main")
            if main_idx == blob_idx + 1 and len(path_parts) > main_idx + 1:
                filename = "/".join(path_parts[main_idx + 1 :])
                direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                return repo_id, filename, direct_url

        # Case 4: Repository root - look for common GGUF files
        else:
            logger.info(f"Repository URL without specific file: {url}")
            return repo_id, None, None

    except Exception as e:
        logger.error(f"Error parsing GGUF URL {url}: {e}")
        return None, None, None

    logger.warning(f"Could not parse GGUF URL format: {url}")
    return None, None, None


def get_cache_path(url: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Generate a cache path for a GGUF model URL.

    Args:
        url: Model URL or path
        cache_dir: Cache directory (defaults to ~/.cache/marvis/gguf)

    Returns:
        Path to cached file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a hash of the URL for the cache key
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]

    # Extract filename from URL or use hash
    _, filename, _ = parse_gguf_url(url)
    if filename:
        # Clean filename and add hash prefix for uniqueness
        clean_filename = re.sub(r"[^\w\-_\.]", "_", filename)
        cache_filename = f"{url_hash}_{clean_filename}"
    else:
        cache_filename = f"{url_hash}.gguf"

    return cache_dir / cache_filename


def download_gguf_file(
    url: str, cache_dir: Optional[Path] = None, force_download: bool = False
) -> Path:
    """
    Download a GGUF file from URL and cache it locally.

    Args:
        url: URL to download from
        cache_dir: Directory to cache files (defaults to ~/.cache/marvis/gguf)
        force_download: Whether to re-download even if cached

    Returns:
        Path to the downloaded/cached file

    Raises:
        RuntimeError: If download fails
        FileNotFoundError: If local file doesn't exist
    """
    # Handle local files
    if not url.startswith("http"):
        if os.path.exists(url):
            return Path(url)
        else:
            raise FileNotFoundError(f"Local GGUF file not found: {url}")

    # Get cache path
    cache_path = get_cache_path(url, cache_dir)

    # Return cached file if it exists and not forcing download
    if cache_path.exists() and not force_download:
        logger.info(f"Using cached GGUF file: {cache_path}")
        return cache_path

    # Parse URL to get download URL
    repo_id, filename, download_url = parse_gguf_url(url)

    if not download_url:
        raise ValueError(f"Could not determine download URL from: {url}")

    logger.info(f"Downloading GGUF file from: {download_url}")
    logger.info(f"Cache location: {cache_path}")

    try:
        # Import here to avoid dependency issues if not installed
        import requests
        from tqdm import tqdm

        # Download with progress bar
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tmp_file:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        pbar.update(len(chunk))

            tmp_path = Path(tmp_file.name)

        # Validate the downloaded file
        if validate_gguf_file(tmp_path):
            # Move to final cache location
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_path), str(cache_path))
            logger.info(f"Successfully downloaded and cached GGUF file: {cache_path}")
            return cache_path
        else:
            # Clean up invalid file
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file failed GGUF validation: {download_url}"
            )

    except ImportError:
        logger.error(
            "Missing dependencies for GGUF download. Install with: pip install requests tqdm"
        )
        raise RuntimeError(
            "Cannot download GGUF files without requests and tqdm packages"
        )
    except Exception as e:
        logger.error(f"Failed to download GGUF file from {download_url}: {e}")
        raise RuntimeError(f"GGUF download failed: {e}")


def validate_gguf_file(file_path: Path) -> bool:
    """
    Validate that a file is a proper GGUF format.

    Args:
        file_path: Path to file to validate

    Returns:
        True if valid GGUF file, False otherwise
    """
    try:
        # Check file exists and has minimum size
        if not file_path.exists() or file_path.stat().st_size < 16:
            return False

        # Check GGUF magic bytes (first 4 bytes should be "GGUF")
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                logger.warning(f"File {file_path} does not have GGUF magic bytes")
                return False

        logger.debug(f"GGUF file validation passed: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating GGUF file {file_path}: {e}")
        return False


def get_gguf_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract basic metadata from a GGUF file.

    Args:
        file_path: Path to GGUF file

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "file_path": str(file_path),
        "file_size": file_path.stat().st_size,
        "valid": False,
    }

    try:
        if validate_gguf_file(file_path):
            with open(file_path, "rb") as f:
                # Read GGUF header
                magic = f.read(4)  # "GGUF"
                version = int.from_bytes(f.read(4), byteorder="little")

                metadata.update(
                    {
                        "valid": True,
                        "magic": magic.decode("ascii"),
                        "version": version,
                        "file_size_mb": round(
                            file_path.stat().st_size / 1024 / 1024, 1
                        ),
                    }
                )

    except Exception as e:
        logger.error(f"Error reading GGUF metadata from {file_path}: {e}")

    return metadata


def is_gguf_url(model_name: str) -> bool:
    """
    Check if a model name/path refers to a GGUF file.

    Args:
        model_name: Model identifier to check

    Returns:
        True if this appears to be a GGUF model
    """
    # Local file check
    if model_name.endswith(".gguf"):
        return True

    # HuggingFace URL check
    if "huggingface.co" in model_name:
        repo_id, filename, _ = parse_gguf_url(model_name)
        if filename and filename.endswith(".gguf"):
            return True

    return False


def suggest_gguf_files(repo_url: str) -> list[str]:
    """
    Suggest common GGUF filenames for a HuggingFace repository.

    Args:
        repo_url: Repository URL

    Returns:
        List of suggested GGUF file URLs
    """
    repo_id, _, _ = parse_gguf_url(repo_url)
    if not repo_id:
        return []

    # Common GGUF quantization patterns
    common_patterns = [
        "q4_k_m.gguf",
        "q4_k_s.gguf",
        "q5_k_m.gguf",
        "q5_k_s.gguf",
        "q6_k.gguf",
        "q8_0.gguf",
        "f16.gguf",
        "f32.gguf",
    ]

    # Extract model name from repo
    model_name = repo_id.split("/")[-1].replace("-GGUF", "").replace("-gguf", "")

    suggestions = []
    for pattern in common_patterns:
        filename = f"{model_name}-{pattern}"
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        suggestions.append(url)

    return suggestions


# Example usage and testing
if __name__ == "__main__":
    # Test URL parsing
    test_urls = [
        "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=Qwen2.5-VL-3B-Instruct-q4_k_m.gguf",
        "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf",
        "/path/to/local/model.gguf",
    ]

    for url in test_urls:
        repo_id, filename, download_url = parse_gguf_url(url)
        print(f"URL: {url}")
        print(f"  Repo: {repo_id}")
        print(f"  File: {filename}")
        print(f"  Download: {download_url}")
        print()
