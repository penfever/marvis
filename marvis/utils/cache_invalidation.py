#!/usr/bin/env python
"""
Centralized cache invalidation system for MARVIS.

This module provides automatic cache invalidation when the data directory changes,
ensuring that all caches remain up-to-date when new datasets or benchmarks are added.

The system works by:
1. Computing a hash of all files in the data directory
2. Storing this hash in a central location
3. Checking the hash whenever the resource manager is initialized
4. Invalidating all relevant caches when the directory changes

This approach is data-type agnostic and works for any benchmark or dataset type.
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache invalidation settings
DEFAULT_CHECK_INTERVAL = 300  # 5 minutes between hash checks
DEFAULT_CACHE_FILES = [
    "openml_mappings.pkl",
    "metadata_cache.pkl", 
    "dataset_cache.pkl"
]


class DataDirectoryWatcher:
    """
    Monitors data directory for changes and invalidates caches as needed.
    
    Uses file hashing to detect when the data directory contents change,
    then invalidates all relevant cache files automatically.
    """
    
    def __init__(self, data_dir: Path, cache_dir: Path, check_interval: int = DEFAULT_CHECK_INTERVAL):
        """
        Initialize the data directory watcher.
        
        Args:
            data_dir: Path to the data directory to monitor
            cache_dir: Path to the cache directory where hash file is stored
            check_interval: Minimum seconds between hash computations
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.check_interval = check_interval
        self.hash_file = self.cache_dir / "data_directory.hash"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_directory_hash(self) -> str:
        """
        Compute SHA-256 hash of data directory contents.
        
        Includes all files (recursively) with their names, sizes, and modification times
        to create a comprehensive fingerprint of the directory state.
        
        Returns:
            Hex string of SHA-256 hash
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return "empty_directory"
        
        hasher = hashlib.sha256()
        
        try:
            # Get all files recursively, sorted for consistent ordering
            all_files = []
            for root, dirs, files in os.walk(self.data_dir):
                # Sort for consistent ordering
                dirs.sort()
                files.sort()
                
                for file in files:
                    file_path = Path(root) / file
                    try:
                        # Include relative path, size, and modification time
                        rel_path = file_path.relative_to(self.data_dir)
                        stat = file_path.stat()
                        file_info = f"{rel_path}:{stat.st_size}:{stat.st_mtime}"
                        all_files.append(file_info)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Could not stat file {file_path}: {e}")
                        continue
            
            # Hash all file information
            for file_info in sorted(all_files):
                hasher.update(file_info.encode('utf-8'))
            
            hash_value = hasher.hexdigest()
            logger.debug(f"Computed directory hash for {len(all_files)} files: {hash_value[:16]}...")
            return hash_value
            
        except Exception as e:
            logger.error(f"Error computing directory hash: {e}")
            return "error_computing_hash"
    
    def load_stored_hash(self) -> Optional[Dict]:
        """
        Load previously stored hash information.
        
        Returns:
            Dictionary with hash info or None if not found/invalid
        """
        if not self.hash_file.exists():
            return None
        
        try:
            with open(self.hash_file, 'r') as f:
                hash_info = json.load(f)
            
            # Validate required fields
            if not isinstance(hash_info, dict) or 'directory_hash' not in hash_info:
                logger.warning("Invalid hash file format, will regenerate")
                return None
            
            return hash_info
            
        except Exception as e:
            logger.warning(f"Could not load stored hash: {e}")
            return None
    
    def save_hash_info(self, directory_hash: str, invalidated_caches: List[str] = None) -> None:
        """
        Save current hash information to disk.
        
        Args:
            directory_hash: Current directory hash
            invalidated_caches: List of cache files that were invalidated
        """
        hash_info = {
            'directory_hash': directory_hash,
            'last_checked': time.time(),
            'check_interval': self.check_interval,
            'invalidated_caches': invalidated_caches or [],
            'version': 1
        }
        
        try:
            # Atomic write using temporary file
            temp_file = self.hash_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(hash_info, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.hash_file)
            logger.debug(f"Saved directory hash: {directory_hash[:16]}...")
            
        except Exception as e:
            logger.error(f"Could not save hash info: {e}")
    
    def should_check_hash(self) -> bool:
        """
        Determine if enough time has passed to warrant checking the hash.
        
        Returns:
            True if hash should be checked
        """
        stored_info = self.load_stored_hash()
        if not stored_info:
            return True
        
        last_checked = stored_info.get('last_checked', 0)
        time_since_check = time.time() - last_checked
        
        return time_since_check >= self.check_interval
    
    def find_cache_files(self, cache_patterns: List[str] = None) -> List[Path]:
        """
        Find cache files that should be invalidated.
        
        Args:
            cache_patterns: List of cache file patterns to look for
            
        Returns:
            List of existing cache file paths
        """
        if cache_patterns is None:
            cache_patterns = DEFAULT_CACHE_FILES
        
        cache_files = []
        
        for pattern in cache_patterns:
            # Check exact filename
            cache_file = self.cache_dir / pattern
            if cache_file.exists():
                cache_files.append(cache_file)
            
            # Check for pattern matches (e.g., *.pkl)
            if '*' in pattern:
                for match in self.cache_dir.glob(pattern):
                    if match.is_file():
                        cache_files.append(match)
        
        return cache_files
    
    def invalidate_caches(self, cache_patterns: List[str] = None) -> List[str]:
        """
        Remove cache files to force regeneration.
        
        Args:
            cache_patterns: List of cache file patterns to invalidate
            
        Returns:
            List of cache files that were actually removed
        """
        cache_files = self.find_cache_files(cache_patterns)
        invalidated = []
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                invalidated.append(cache_file.name)
                logger.info(f"Invalidated cache file: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Could not remove cache file {cache_file}: {e}")
        
        return invalidated
    
    def check_and_invalidate(self, cache_patterns: List[str] = None, force: bool = False) -> bool:
        """
        Check if data directory has changed and invalidate caches if needed.
        
        Args:
            cache_patterns: List of cache file patterns to check
            force: Force hash check even if interval hasn't passed
            
        Returns:
            True if caches were invalidated, False otherwise
        """
        # Skip check if not enough time has passed (unless forced)
        if not force and not self.should_check_hash():
            logger.debug("Skipping directory hash check (interval not reached)")
            return False
        
        # Compute current hash
        current_hash = self.compute_directory_hash()
        
        # Load stored hash
        stored_info = self.load_stored_hash()
        stored_hash = stored_info.get('directory_hash') if stored_info else None
        
        # Check if hash has changed
        if stored_hash == current_hash:
            logger.debug("Data directory unchanged, no cache invalidation needed")
            # Update timestamp but keep same hash
            self.save_hash_info(current_hash, [])
            return False
        
        # Hash has changed - invalidate caches
        logger.info(f"Data directory changed (hash: {stored_hash[:16] if stored_hash else 'None'} -> {current_hash[:16]})")
        invalidated_caches = self.invalidate_caches(cache_patterns)
        
        # Save new hash
        self.save_hash_info(current_hash, invalidated_caches)
        
        if invalidated_caches:
            logger.info(f"Invalidated {len(invalidated_caches)} cache files: {', '.join(invalidated_caches)}")
        else:
            logger.info("No cache files found to invalidate")
        
        return True


def create_data_directory_watcher(data_dir: Optional[Path] = None, 
                                cache_dir: Optional[Path] = None) -> DataDirectoryWatcher:
    """
    Create a DataDirectoryWatcher with default paths.
    
    Args:
        data_dir: Data directory to monitor (defaults to project_root/data)
        cache_dir: Cache directory (defaults to ~/.marvis/cache)
        
    Returns:
        DataDirectoryWatcher instance
    """
    if data_dir is None:
        # Auto-detect project root and data directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up to project root
        data_dir = project_root / "data"
    
    if cache_dir is None:
        cache_dir = Path.home() / ".marvis" / "cache"
    
    return DataDirectoryWatcher(data_dir, cache_dir)


def check_and_invalidate_caches(data_dir: Optional[Path] = None,
                               cache_dir: Optional[Path] = None,
                               cache_patterns: List[str] = None,
                               force: bool = False) -> bool:
    """
    Convenience function to check and invalidate caches.
    
    Args:
        data_dir: Data directory to monitor
        cache_dir: Cache directory
        cache_patterns: Cache file patterns to check
        force: Force check even if interval hasn't passed
        
    Returns:
        True if caches were invalidated
    """
    watcher = create_data_directory_watcher(data_dir, cache_dir)
    return watcher.check_and_invalidate(cache_patterns, force)