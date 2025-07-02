"""
Robust resource management system for MARVIS with concurrent access support.

This module provides centralized, package-aware resource management that eliminates
fragile path-guessing logic and provides consistent resource organization across
different deployment environments.

Key features:
- Package-aware path resolution using importlib.resources
- Consistent dataset workspace isolation
- Environment variable configuration support
- Backward compatibility with existing code
- Centralized config and metadata management
- Unified dataset preparation with intelligent caching and checking
- Thread-safe dataset registry with concurrent access support

The DatasetRegistry provides safe concurrent access through:
- File locking with timeout and retry mechanisms
- Atomic write operations using temporary files + rename
- Distributed registry architecture with local registry files
- Automatic merging of registry updates from multiple processes
- Graceful degradation when registry is unavailable

Environment Variables for Registry Configuration:
- MARVIS_REGISTRY_LOCK_TIMEOUT: Lock acquisition timeout in seconds (default: 10.0)
- MARVIS_REGISTRY_RETRY_ATTEMPTS: Number of retry attempts (default: 3)
- MARVIS_REGISTRY_SYNC_INTERVAL: Registry refresh interval in seconds (default: 30.0)
- MARVIS_DISABLE_REGISTRY: Set to 'true' to disable registry entirely (default: false)

The DatasetPreparer class provides a unified interface for dataset preparation that:
- Checks if datasets are already prepared before re-downloading
- Separates download and organization steps with intelligent caching
- Provides consistent logging and error handling
- Integrates with the dataset registry for tracking

Usage examples:
    # Simple CIFAR preparation using convenience function
    train_paths, train_labels, test_paths, test_labels, class_names = prepare_cifar_dataset("cifar10")
    
    # Custom dataset preparation using the DatasetPreparer
    resource_manager = get_resource_manager()
    success = resource_manager.dataset_preparer.prepare_dataset(
        dataset_id="my_dataset",
        dataset_type="vision",
        check_function=my_check_function,
        download_function=my_download_function,
        organize_function=my_organize_function
    )
    
    # Registry management
    resource_manager.cleanup_registry(max_age_days=7)  # Clean up old entries
    status = resource_manager.get_registry_status()    # Get registry info
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import importlib.resources as resources

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Configuration for MARVIS resource management."""
    base_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    configs_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'ResourceConfig':
        """Create config from environment variables."""
        return cls(
            base_dir=os.environ.get('MARVIS_BASE_DIR'),
            cache_dir=os.environ.get('MARVIS_CACHE_DIR'),
            datasets_dir=os.environ.get('MARVIS_DATASETS_DIR'),
            configs_dir=os.environ.get('MARVIS_CONFIGS_DIR'),
            temp_dir=os.environ.get('MARVIS_TEMP_DIR'),
        )


@dataclass
class DatasetMetadata:
    """Metadata for dataset management and caching."""
    dataset_id: str
    name: Optional[str] = None
    task_type: Optional[str] = None
    feature_count: Optional[int] = None
    sample_count: Optional[int] = None
    class_count: Optional[int] = None
    openml_task_id: Optional[int] = None
    source: Optional[str] = None
    cached_at: Optional[str] = None
    file_paths: Optional[Dict[str, str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'task_type': self.task_type,
            'feature_count': self.feature_count,
            'sample_count': self.sample_count,
            'class_count': self.class_count,
            'openml_task_id': self.openml_task_id,
            'source': self.source,
            'cached_at': self.cached_at,
            'file_paths': self.file_paths or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create from dictionary."""
        return cls(**data)


class PathResolver:
    """Robust path resolution using package-aware methods."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self._base_dir = None
    
    def get_base_dir(self) -> Path:
        """Get base MARVIS directory."""
        if self._base_dir is None:
            if self.config.base_dir:
                self._base_dir = Path(self.config.base_dir).expanduser().resolve()
            else:
                self._base_dir = Path.home() / '.marvis'
            self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir
    
    def get_cache_dir(self) -> Path:
        """Get cache directory."""
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir).expanduser().resolve()
        else:
            cache_dir = self.get_base_dir() / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_datasets_dir(self) -> Path:
        """Get datasets directory."""
        if self.config.datasets_dir:
            datasets_dir = Path(self.config.datasets_dir).expanduser().resolve()
        else:
            datasets_dir = self.get_base_dir() / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        return datasets_dir
    
    def get_configs_dir(self) -> Path:
        """Get configs directory."""
        if self.config.configs_dir:
            configs_dir = Path(self.config.configs_dir).expanduser().resolve()
        else:
            configs_dir = self.get_base_dir() / 'configs'
        configs_dir.mkdir(parents=True, exist_ok=True)
        return configs_dir
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        if self.config.temp_dir:
            temp_dir = Path(self.config.temp_dir).expanduser().resolve()
        else:
            temp_dir = self.get_base_dir() / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def get_dataset_dir(self, dataset_id: str) -> Path:
        """Get directory for a specific dataset."""
        dataset_dir = self.get_datasets_dir() / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    
    def get_embedding_dir(self, dataset_id: str) -> Path:
        """Get embeddings directory for a specific dataset."""
        embed_dir = self.get_dataset_dir(dataset_id) / 'embeddings'
        embed_dir.mkdir(parents=True, exist_ok=True)
        return embed_dir
    
    def get_dataset_cache_dir(self, dataset_id: str) -> Path:
        """Get cache directory for a specific dataset."""
        cache_dir = self.get_dataset_dir(dataset_id) / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_config_path(self, model_type: str, filename: str) -> Optional[Path]:
        """Get config file path with fallback to package resources."""
        # First try managed configs directory
        config_path = self.get_configs_dir() / model_type / filename
        if config_path.exists():
            return config_path
        
        # Try package resources for backward compatibility
        try:
            # Handle different config patterns
            if model_type == 'jolt':
                package_path = f'marvis.examples.tabular.llm_baselines.jolt'
            elif model_type == 'tabllm':
                package_path = f'marvis.examples.tabular.llm_baselines.tabllm_like'
            elif model_type == 'cc18_semantic':
                package_path = f'marvis.data.cc18_semantic'
            else:
                return None
            
            # Try to get path from package resources
            try:
                with resources.path(package_path, filename) as path:
                    if path.exists():
                        return path
            except (ModuleNotFoundError, FileNotFoundError, AttributeError):
                # Fallback: try to find package directory manually
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                
                if model_type == 'jolt':
                    fallback_path = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'jolt' / filename
                elif model_type == 'tabllm':
                    fallback_path = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'tabllm_like' / filename
                elif model_type == 'cc18_semantic':
                    fallback_path = project_root / 'data' / 'cc18_semantic' / filename
                else:
                    return None
                
                if fallback_path.exists():
                    return fallback_path
                    
        except Exception as e:
            logger.debug(f"Error accessing package resources for {model_type}/{filename}: {e}")
        
        return None


class DatasetRegistry:
    """Thread-safe registry for tracking dataset metadata and locations with concurrent access support."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self._registry = {}
        self._lock_timeout = float(os.environ.get('MARVIS_REGISTRY_LOCK_TIMEOUT', '10.0'))
        self._retry_attempts = int(os.environ.get('MARVIS_REGISTRY_RETRY_ATTEMPTS', '3'))
        self._sync_interval = float(os.environ.get('MARVIS_REGISTRY_SYNC_INTERVAL', '30.0'))
        self._last_sync_time = 0
        self._disable_registry = os.environ.get('MARVIS_DISABLE_REGISTRY', '').lower() in ('true', '1', 'yes')
        
        if not self._disable_registry:
            self._load_registry()
    
    def _get_registry_path(self) -> Path:
        """Get path to main registry file."""
        return self.path_resolver.get_base_dir() / 'dataset_registry.json'
    
    def _get_local_registry_path(self) -> Path:
        """Get path to process-specific local registry file."""
        import time
        pid = os.getpid()
        timestamp = int(time.time())
        return self.path_resolver.get_base_dir() / f'dataset_registry_{pid}_{timestamp}.json'
    
    def _get_lock_path(self) -> Path:
        """Get path to registry lock file."""
        return self.path_resolver.get_base_dir() / 'dataset_registry.lock'
    
    def _acquire_file_lock(self, lock_path: Path, timeout: float = None) -> Optional[int]:
        """
        Acquire file lock with timeout support.
        Returns file descriptor if successful, None if failed.
        """
        if timeout is None:
            timeout = self._lock_timeout
            
        try:
            import fcntl
            import time
            
            # Create lock file if it doesn't exist
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_fd
                except (IOError, OSError):
                    time.sleep(0.1)
                    
            # Timeout reached
            os.close(lock_fd)
            return None
            
        except ImportError:
            # Windows - use msvcrt if available
            try:
                import msvcrt
                import time
                
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
                        msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                        return lock_fd
                    except (IOError, OSError):
                        if 'lock_fd' in locals():
                            try:
                                os.close(lock_fd)
                            except:
                                pass
                        time.sleep(0.1)
                        
                return None
                
            except ImportError:
                # No file locking available - log warning and continue
                logger.warning("File locking not available on this platform. Registry may have race conditions.")
                return -1  # Dummy fd to indicate "success" without locking
        except Exception as e:
            logger.debug(f"Failed to acquire file lock: {e}")
            return None
    
    def _release_file_lock(self, lock_fd: int, lock_path: Path):
        """Release file lock."""
        if lock_fd == -1:  # Dummy fd from platforms without locking
            return
            
        try:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except ImportError:
            try:
                import msvcrt
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                os.close(lock_fd)
            except ImportError:
                pass
        except Exception as e:
            logger.debug(f"Error releasing file lock: {e}")
        
        # Clean up lock file
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
    
    def _atomic_write_json(self, data: dict, file_path: Path) -> bool:
        """Write JSON data atomically using temporary file + rename."""
        import tempfile
        import shutil
        
        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=file_path.parent,
                delete=False,
                suffix='.tmp'
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)
            
            # Atomic rename
            shutil.move(str(tmp_path), str(file_path))
            return True
            
        except Exception as e:
            logger.warning(f"Error in atomic write to {file_path}: {e}")
            # Clean up temporary file if it exists
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False
    
    def _validate_registry_data(self, data: dict) -> bool:
        """Validate registry data structure."""
        if not isinstance(data, dict):
            return False
            
        for dataset_id, entry in data.items():
            if not isinstance(dataset_id, str) or not isinstance(entry, dict):
                return False
            if 'registered_at' not in entry or 'workspace_dir' not in entry:
                return False
                
        return True
    
    def _load_registry_with_retry(self) -> dict:
        """Load registry with retry logic and corruption handling."""
        registry_path = self._get_registry_path()
        
        for attempt in range(self._retry_attempts):
            try:
                if not registry_path.exists():
                    return {}
                    
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    
                if self._validate_registry_data(data):
                    return data
                else:
                    logger.warning(f"Registry data validation failed on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Error loading registry on attempt {attempt + 1}: {e}")
                
            if attempt < self._retry_attempts - 1:
                import time
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        logger.warning("Failed to load registry after all retry attempts, starting with empty registry")
        return {}
    
    def _load_registry(self):
        """Load registry from disk with concurrency safety."""
        if self._disable_registry:
            self._registry = {}
            return
            
        try:
            # Try to merge any pending local registries first
            self._merge_local_registries()
            
            # Load main registry
            self._registry = self._load_registry_with_retry()
            self._last_sync_time = time.time()
            
        except Exception as e:
            logger.warning(f"Error loading dataset registry: {e}")
            self._registry = {}
    
    def _merge_local_registries(self):
        """Merge local registry files into main registry."""
        try:
            base_dir = self.path_resolver.get_base_dir()
            local_registry_pattern = 'dataset_registry_*.json'
            
            # Find all local registry files
            local_registries = list(base_dir.glob(local_registry_pattern))
            if not local_registries:
                return
                
            logger.debug(f"Found {len(local_registries)} local registry files to merge")
            
            lock_path = self._get_lock_path()
            lock_fd = self._acquire_file_lock(lock_path, timeout=5.0)
            
            if lock_fd is None:
                logger.debug("Could not acquire lock for registry merge, skipping")
                return
                
            try:
                # Load current main registry
                main_registry = self._load_registry_with_retry()
                
                # Merge local registries
                merged_count = 0
                for local_path in local_registries:
                    try:
                        with open(local_path, 'r') as f:
                            local_data = json.load(f)
                            
                        if self._validate_registry_data(local_data):
                            # Merge with timestamp-based conflict resolution
                            for dataset_id, entry in local_data.items():
                                existing = main_registry.get(dataset_id)
                                if (not existing or 
                                    entry.get('registered_at', '') > existing.get('registered_at', '')):
                                    main_registry[dataset_id] = entry
                                    merged_count += 1
                        
                        # Remove successfully merged local registry
                        local_path.unlink()
                        
                    except Exception as e:
                        logger.debug(f"Error merging local registry {local_path}: {e}")
                
                # Save merged registry
                if merged_count > 0:
                    registry_path = self._get_registry_path()
                    if self._atomic_write_json(main_registry, registry_path):
                        logger.debug(f"Successfully merged {merged_count} registry entries")
                        
            finally:
                self._release_file_lock(lock_fd, lock_path)
                
        except Exception as e:
            logger.debug(f"Error in registry merge: {e}")
    
    def _save_registry_safe(self) -> bool:
        """Save registry with file locking and atomic operations."""
        if self._disable_registry:
            return True
            
        registry_path = self._get_registry_path()
        lock_path = self._get_lock_path()
        
        # Try to acquire lock
        lock_fd = self._acquire_file_lock(lock_path)
        
        if lock_fd is None:
            # Can't get lock - save to local registry instead
            logger.debug("Registry locked, saving to local registry file")
            return self._save_to_local_registry()
        
        try:
            # We have the lock - perform atomic write
            success = self._atomic_write_json(self._registry, registry_path)
            if success:
                logger.debug("Successfully saved main registry")
            return success
            
        finally:
            self._release_file_lock(lock_fd, lock_path)
    
    def _save_to_local_registry(self) -> bool:
        """Save registry entry to local process-specific file."""
        try:
            local_path = self._get_local_registry_path()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = self._atomic_write_json(self._registry, local_path)
            if success:
                logger.debug(f"Saved registry to local file: {local_path}")
            return success
            
        except Exception as e:
            logger.warning(f"Error saving to local registry: {e}")
            return False
    
    def _refresh_if_needed(self):
        """Refresh registry from disk if enough time has passed."""
        import time
        
        if (self._disable_registry or 
            time.time() - self._last_sync_time < self._sync_interval):
            return
            
        try:
            # Try to merge and reload
            old_registry = self._registry.copy()
            self._load_registry()
            
            # Check if anything changed
            if self._registry != old_registry:
                logger.debug("Registry refreshed with external changes")
                
        except Exception as e:
            logger.debug(f"Error refreshing registry: {e}")
    
    def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]):
        """Register a dataset with metadata using safe concurrent operations."""
        if self._disable_registry:
            logger.debug("Registry disabled, skipping dataset registration")
            return
            
        import datetime
        
        # Refresh registry to get latest state
        self._refresh_if_needed()
        
        # Add dataset entry
        entry = {
            'metadata': metadata,
            'registered_at': datetime.datetime.now().isoformat(),
            'workspace_dir': str(self.path_resolver.get_dataset_dir(dataset_id))
        }
        
        self._registry[dataset_id] = entry
        
        # Try to save safely
        success = self._save_registry_safe()
        if not success:
            logger.warning(f"Could not save registry for dataset {dataset_id}, but operation will continue")
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information with fresh data."""
        if self._disable_registry:
            return None
            
        self._refresh_if_needed()
        return self._registry.get(dataset_id)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets with fresh data."""
        if self._disable_registry:
            return []
            
        self._refresh_if_needed()
        return list(self._registry.keys())
    
    def unregister_dataset(self, dataset_id: str):
        """Unregister a dataset using safe concurrent operations."""
        if self._disable_registry:
            return
            
        self._refresh_if_needed()
        
        if dataset_id in self._registry:
            del self._registry[dataset_id]
            success = self._save_registry_safe()
            if not success:
                logger.warning(f"Could not save registry after unregistering dataset {dataset_id}")
    
    def cleanup_stale_entries(self, max_age_days: int = 30):
        """Clean up stale registry entries and local registry files."""
        if self._disable_registry:
            return
            
        try:
            import datetime
            import time
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            cutoff_iso = cutoff_date.isoformat()
            
            # Clean up main registry
            original_count = len(self._registry)
            self._registry = {
                k: v for k, v in self._registry.items()
                if v.get('registered_at', '') > cutoff_iso
            }
            
            removed_count = original_count - len(self._registry)
            if removed_count > 0:
                self._save_registry_safe()
                logger.info(f"Cleaned up {removed_count} stale registry entries")
            
            # Clean up old local registry files
            base_dir = self.path_resolver.get_base_dir()
            local_files = list(base_dir.glob('dataset_registry_*.json'))
            current_time = time.time()
            
            for local_file in local_files:
                try:
                    # Extract timestamp from filename
                    parts = local_file.stem.split('_')
                    if len(parts) >= 4:
                        timestamp = int(parts[-1])
                        age_days = (current_time - timestamp) / (24 * 3600)
                        
                        if age_days > max_age_days:
                            local_file.unlink()
                            logger.debug(f"Removed stale local registry file: {local_file}")
                            
                except Exception as e:
                    logger.debug(f"Error cleaning up local registry file {local_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error during registry cleanup: {e}")


class CacheManager:
    """Manager for caching data with smart organization."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
    
    def get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        import hashlib
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_type: str, cache_key: str, extension: str = '') -> Path:
        """Get path for cached data."""
        cache_dir = self.path_resolver.get_cache_dir() / cache_type
        cache_dir.mkdir(exist_ok=True)
        filename = f"{cache_key}{extension}"
        return cache_dir / filename
    
    def cache_exists(self, cache_type: str, cache_key: str, extension: str = '') -> bool:
        """Check if cache file exists."""
        cache_path = self.get_cache_path(cache_type, cache_key, extension)
        return cache_path.exists()
    
    def save_to_cache(self, cache_type: str, cache_key: str, data: Any, extension: str = '') -> bool:
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key, extension)
            
            if extension == '.json':
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif extension == '.npz':
                import numpy as np
                if isinstance(data, dict):
                    np.savez(cache_path, **data)
                else:
                    np.savez(cache_path, data=data)
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
            return False
    
    def load_from_cache(self, cache_type: str, cache_key: str, extension: str = '') -> Optional[Any]:
        """Load data from cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key, extension)
            if not cache_path.exists():
                return None
            
            if extension == '.json':
                with open(cache_path, 'r') as f:
                    return json.load(f)
            elif extension == '.npz':
                import numpy as np
                return np.load(cache_path, allow_pickle=True)
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None


class ConfigManager:
    """Manager for config file discovery and validation."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
    
    def find_jolt_config(self, dataset_name: str) -> Optional[Path]:
        """Find JOLT config for dataset."""
        filename = f'jolt_config_{dataset_name}.json'
        return self.path_resolver.get_config_path('jolt', filename)
    
    def find_tabllm_template(self, dataset_name: str) -> Optional[Path]:
        """Find TabLLM template for dataset."""
        filename = f'templates_{dataset_name}.yaml'
        return self.path_resolver.get_config_path('tabllm', filename)
    
    def find_tabllm_notes(self, dataset_name: str) -> Optional[Path]:
        """Find TabLLM notes for dataset."""
        filename = f'notes_{dataset_name}.jsonl'
        # Notes are in a subdirectory
        notes_path = self.path_resolver.get_configs_dir() / 'tabllm' / 'notes' / filename
        if notes_path.exists():
            return notes_path
        
        # Fallback to package resources
        return self.path_resolver.get_config_path('tabllm', f'notes/{filename}')
    
    def find_cc18_semantic(self, dataset_name: str) -> Optional[Path]:
        """Find CC18 semantic metadata for dataset."""
        filename = f'{dataset_name}.json'
        return self.path_resolver.get_config_path('cc18_semantic', filename)
    
    def find_semantic_metadata(self, dataset_id: Union[str, int]) -> Optional[Path]:
        """Find semantic metadata for dataset ID using recursive search in data directory."""
        from .metadata_loader import get_metadata_loader
        
        # Use the updated MetadataLoader which searches recursively
        loader = get_metadata_loader()
        return loader.detect_metadata_file(dataset_id)
    
    def get_openml_task_mapping(self, model_type: str) -> Optional[Dict[str, int]]:
        """Get OpenML task mapping for a model type."""
        filename = 'openml_task_mapping.json'
        mapping_path = self.path_resolver.get_config_path(model_type, filename)
        
        if mapping_path and mapping_path.exists():
            try:
                with open(mapping_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading OpenML task mapping for {model_type}: {e}")
        
        return None


class MarvisResourceManager:
    """Main resource manager for MARVIS."""
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        if config is None:
            config = ResourceConfig.from_environment()
        
        self.config = config
        self.path_resolver = PathResolver(config)
        
        # Check for data directory changes and invalidate caches if needed
        self._check_and_invalidate_caches()
        
        self.dataset_registry = DatasetRegistry(self.path_resolver)
        self.config_manager = ConfigManager(self.path_resolver)
        self.cache_manager = CacheManager(self.path_resolver)
        self.dataset_preparer = DatasetPreparer(self.path_resolver, self.dataset_registry, self.cache_manager)
        
        logger.debug(f"Initialized MARVIS resource manager with base dir: {self.path_resolver.get_base_dir()}")
    
    def _check_and_invalidate_caches(self) -> None:
        """
        Check if data directory has changed and invalidate caches if needed.
        
        This method is called during resource manager initialization to ensure
        that all caches are up-to-date with the current data directory state.
        """
        try:
            from .cache_invalidation import check_and_invalidate_caches
            
            # Get data directory from project structure
            base_dir = self.path_resolver.get_base_dir()
            data_dir = base_dir / "data"
            cache_dir = Path.home() / ".marvis" / "cache"
            
            # Define cache patterns to check
            cache_patterns = [
                "openml_mappings.pkl",
                "metadata_cache.pkl",
                "dataset_cache.pkl",
                "*.cache",  # Any other cache files
                "*.pkl"     # Any pickle cache files
            ]
            
            # Check and invalidate if needed
            caches_invalidated = check_and_invalidate_caches(
                data_dir=data_dir,
                cache_dir=cache_dir,
                cache_patterns=cache_patterns,
                force=False  # Respect check interval
            )
            
            if caches_invalidated:
                logger.info("Data directory changes detected, caches have been invalidated")
            
        except Exception as e:
            # Don't fail initialization if cache checking fails
            logger.warning(f"Could not check data directory for cache invalidation: {e}")
    
    def get_dataset_workspace(self, dataset_id: str) -> Path:
        """Get workspace directory for a dataset."""
        return self.path_resolver.get_dataset_dir(dataset_id)
    
    def find_csv_file(self, dataset_id: str, additional_search_dirs: Optional[List[str]] = None) -> Optional[Path]:
        """Find CSV file using robust search strategy."""
        search_dirs = []
        
        # Priority 1: Dataset's own directory
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_id)
        search_dirs.append(dataset_dir)
        
        # Priority 2: Additional search directories provided by user
        if additional_search_dirs:
            search_dirs.extend([Path(d) for d in additional_search_dirs])
        
        # Priority 3: Common data directories
        search_dirs.extend([
            self.path_resolver.get_datasets_dir(),
            Path.cwd() / 'data',
            Path.cwd(),
        ])
        
        # Search for CSV files
        csv_patterns = [
            f'{dataset_id}.csv',
            f'dataset_{dataset_id}.csv',
            f'{dataset_id}_processed.csv',
            f'{dataset_id}_clean.csv',
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for pattern in csv_patterns:
                csv_path = search_dir / pattern
                if csv_path.exists():
                    logger.debug(f"Found CSV file: {csv_path}")
                    return csv_path
                
                # Also check subdirectories
                for subdir in search_dir.iterdir():
                    if subdir.is_dir():
                        csv_path = subdir / pattern
                        if csv_path.exists():
                            logger.debug(f"Found CSV file in subdirectory: {csv_path}")
                            return csv_path
        
        logger.debug(f"CSV file not found for dataset {dataset_id} in search directories")
        return None
    
    def resolve_openml_identifiers(self, task_id: Optional[int] = None, 
                                  dataset_id: Optional[int] = None, 
                                  dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve OpenML identifiers to get complete mapping.
        
        Given any one of task_id, dataset_id, or dataset_name, returns all three.
        
        Args:
            task_id: OpenML task ID
            dataset_id: OpenML dataset ID  
            dataset_name: Dataset name string
            
        Returns:
            Dictionary with keys: task_id, dataset_id, dataset_name
            Returns None values for any that couldn't be resolved
        """
        from marvis.utils.openml_mapping import (
            get_openml_cc18_mapping, 
            impute_task_id_from_dataset_id,
            impute_task_id_from_dataset_name
        )
        
        result = {
            'task_id': task_id,
            'dataset_id': dataset_id,
            'dataset_name': dataset_name
        }
        
        # Get the complete mapping
        mapping = get_openml_cc18_mapping()
        
        # If we have task_id, prioritize OpenML API resolution
        if task_id:
            # Try OpenML API first for accurate resolution
            from marvis.utils.openml_mapping import resolve_task_id_from_openml_api
            api_result = resolve_task_id_from_openml_api(task_id)
            if api_result:
                result['dataset_id'] = api_result.get('dataset_id')
                result['dataset_name'] = api_result.get('dataset_name')
                logger.info(f"Task ID {task_id} resolves to dataset ID {result['dataset_id']}")
                return result
            
            # Fallback to mapping only if API fails
            if task_id in mapping:
                info = mapping[task_id]
                result['dataset_id'] = info.get('dataset_id')
                result['dataset_name'] = info.get('dataset_name')
                logger.warning(f"Using fallback mapping for task {task_id} -> dataset {result['dataset_id']}")
                return result
            
            logger.error(f"Could not resolve task_id {task_id} from either OpenML API or fallback mapping")
        
        # DISABLED: If we have dataset_id, find task_id first - requires working fallback mappings
        # if dataset_id and not task_id:
        #     imputed_task_id = impute_task_id_from_dataset_id(dataset_id)
        #     if imputed_task_id:
        #         result['task_id'] = imputed_task_id
        #         if imputed_task_id in mapping:
        #             info = mapping[imputed_task_id]
        #             result['dataset_name'] = info.get('dataset_name')
        #     return result
        
        # DISABLED: If we have dataset_name, find task_id first - requires working fallback mappings  
        # if dataset_name and not task_id:
        #     imputed_task_id = impute_task_id_from_dataset_name(dataset_name)
        #     if imputed_task_id:
        #         result['task_id'] = imputed_task_id
        #         if imputed_task_id in mapping:
        #             info = mapping[imputed_task_id]
        #             result['dataset_id'] = info.get('dataset_id')
        #     return result
        
        logger.warning("Resolution by dataset_id or dataset_name is temporarily disabled to force task_id-based OpenML API usage")
        
        return result
    
    def validate_model_metadata(self, openml_task_id: int, model_type: str) -> Dict[str, Any]:
        """Validate metadata for a specific model type."""
        result = {
            'valid': True,
            'missing_files': [],
            'errors': [],
            'warnings': [],
            'dataset_name': None
        }
        
        # Handle JOLT with new task ID-based approach
        if model_type == 'jolt':
            # For JOLT, check directly for task ID-based config file
            jolt_dir = self.path_resolver.get_config_path('jolt', '')
            if jolt_dir and jolt_dir.exists():
                config_path = jolt_dir / f"jolt_config_task_{openml_task_id}.json"
                if config_path.exists():
                    # Try to load and validate the config
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        result['dataset_name'] = config_data.get('dataset_name', f'task_{openml_task_id}')
                        logger.debug(f"Found JOLT config for task {openml_task_id}: {config_path}")
                    except Exception as e:
                        result['valid'] = False
                        result['errors'].append(f"JOLT config file is invalid for task {openml_task_id}: {e}")
                else:
                    # Try fallback to old dataset name approach
                    task_mapping = self.config_manager.get_openml_task_mapping('jolt')
                    if task_mapping:
                        dataset_name = None
                        for name, task_id in task_mapping.items():
                            if task_id == openml_task_id:
                                dataset_name = name
                                break
                        
                        if dataset_name:
                            old_config_path = jolt_dir / f"jolt_config_{dataset_name}.json"
                            if old_config_path.exists():
                                result['dataset_name'] = dataset_name
                                result['warnings'].append(f"Using old JOLT config format for {dataset_name}")
                            else:
                                result['valid'] = False
                                result['errors'].append(f"No JOLT config found for OpenML task ID {openml_task_id}")
                        else:
                            result['valid'] = False
                            result['errors'].append(f"No JOLT config found for OpenML task ID {openml_task_id}")
                    else:
                        result['valid'] = False
                        result['errors'].append(f"No JOLT config found for OpenML task ID {openml_task_id}")
            else:
                result['valid'] = False
                result['errors'].append(f"JOLT directory not found")
        
        # Handle TabLLM with new task ID-based approach  
        elif model_type == 'tabllm':
            # For TabLLM, check directly for task ID-based template file
            tabllm_dir = self.path_resolver.get_config_path('tabllm_like', '')
            if tabllm_dir and tabllm_dir.exists():
                template_path = tabllm_dir / f"templates_task_{openml_task_id}.yaml"
                if template_path.exists():
                    result['dataset_name'] = f'task_{openml_task_id}'
                    logger.debug(f"Found TabLLM template for task {openml_task_id}: {template_path}")
                else:
                    # Try fallback to old dataset name approach
                    task_mapping = self.config_manager.get_openml_task_mapping('tabllm')
                    if task_mapping:
                        dataset_name = None
                        for name, task_id in task_mapping.items():
                            if task_id == openml_task_id:
                                dataset_name = name
                                break
                        
                        if dataset_name:
                            old_template_path = tabllm_dir / f"templates_{dataset_name}.yaml"
                            if old_template_path.exists():
                                result['dataset_name'] = dataset_name
                                result['warnings'].append(f"Using old TabLLM template format for {dataset_name}")
                            else:
                                result['valid'] = False
                                result['errors'].append(f"No TabLLM template found for OpenML task ID {openml_task_id}")
                        else:
                            result['valid'] = False
                            result['errors'].append(f"No TabLLM template found for OpenML task ID {openml_task_id}")
                    else:
                        result['valid'] = False
                        result['errors'].append(f"No TabLLM template found for OpenML task ID {openml_task_id}")
            else:
                result['valid'] = False
                result['errors'].append(f"TabLLM directory not found")
        
        else:
            result['valid'] = False
            result['errors'].append(f"Unknown model type: {model_type}")
        
        return result
    
    def cleanup_registry(self, max_age_days: int = 30):
        """Clean up stale registry entries and local registry files."""
        self.dataset_registry.cleanup_stale_entries(max_age_days)
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the dataset registry."""
        try:
            registry_path = self.dataset_registry._get_registry_path()
            base_dir = self.path_resolver.get_base_dir()
            
            # Count local registry files
            local_files = list(base_dir.glob('dataset_registry_*.json'))
            
            status = {
                'registry_enabled': not self.dataset_registry._disable_registry,
                'main_registry_exists': registry_path.exists(),
                'main_registry_path': str(registry_path),
                'local_registry_files': len(local_files),
                'total_datasets': len(self.dataset_registry.list_datasets()),
                'sync_interval': self.dataset_registry._sync_interval,
                'lock_timeout': self.dataset_registry._lock_timeout,
                'retry_attempts': self.dataset_registry._retry_attempts
            }
            
            if registry_path.exists():
                import os
                stat = os.stat(registry_path)
                status['main_registry_size'] = stat.st_size
                status['main_registry_modified'] = stat.st_mtime
            
            return status
            
        except Exception as e:
            logger.warning(f"Error getting registry status: {e}")
            return {
                'error': str(e),
                'registry_enabled': not self.dataset_registry._disable_registry
            }


class DatasetPreparer:
    """Unified dataset preparation with intelligent caching and checking."""
    
    def __init__(self, path_resolver: PathResolver, dataset_registry: Optional['DatasetRegistry'] = None, cache_manager: Optional['CacheManager'] = None):
        self.path_resolver = path_resolver
        self.dataset_registry = dataset_registry
        self.cache_manager = cache_manager
    
    def prepare_dataset(
        self,
        dataset_id: str,
        dataset_type: str,
        check_function: callable,
        download_function: callable,
        organize_function: callable = None,
        force_redownload: bool = False,
        force_reorganize: bool = False,
        min_samples: int = 100
    ) -> bool:
        """
        Unified dataset preparation with intelligent caching.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_type: Type of dataset (e.g., 'vision', 'audio', 'tabular')
            check_function: Function that returns True if dataset is ready
            download_function: Function that downloads/extracts the dataset
            organize_function: Optional function that organizes downloaded data
            force_redownload: Force re-download even if dataset exists
            force_reorganize: Force re-organization even if already organized
            min_samples: Minimum number of samples expected in prepared dataset
            
        Returns:
            True if dataset is ready, False otherwise
        """
        logger.info(f"Preparing {dataset_type} dataset: {dataset_id}")
        
        # Get dataset workspace
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_id)
        
        # Check if dataset is already prepared
        if not force_redownload and not force_reorganize:
            try:
                if check_function(dataset_dir):
                    logger.info(f"Dataset {dataset_id} already prepared, skipping preparation")
                    return True
            except Exception as e:
                logger.debug(f"Dataset check failed for {dataset_id}: {e}")
        
        # Check if we need to download
        download_needed = force_redownload
        if not download_needed:
            try:
                # Check if raw data exists (but might need organization)
                download_needed = not self._check_raw_data_exists(dataset_dir, dataset_id)
            except Exception as e:
                logger.debug(f"Raw data check failed for {dataset_id}: {e}")
                download_needed = True
        
        # Download if needed
        if download_needed:
            try:
                logger.info(f"Downloading dataset {dataset_id}...")
                download_function(dataset_dir)
                logger.info(f"Download completed for {dataset_id}")
            except Exception as e:
                logger.error(f"Download failed for {dataset_id}: {e}")
                return False
        else:
            logger.info(f"Raw data for {dataset_id} already exists, skipping download")
        
        # Organize if needed and function provided
        if organize_function:
            organize_needed = force_reorganize
            if not organize_needed:
                try:
                    organize_needed = not check_function(dataset_dir)
                except Exception as e:
                    logger.debug(f"Organization check failed for {dataset_id}: {e}")
                    organize_needed = True
            
            if organize_needed:
                try:
                    logger.info(f"Organizing dataset {dataset_id}...")
                    organize_function(dataset_dir)
                    logger.info(f"Organization completed for {dataset_id}")
                except Exception as e:
                    logger.error(f"Organization failed for {dataset_id}: {e}")
                    return False
            else:
                logger.info(f"Dataset {dataset_id} already organized, skipping organization")
        
        # Final check
        try:
            if check_function(dataset_dir):
                logger.info(f"Dataset {dataset_id} successfully prepared")
                
                # Register dataset in registry
                metadata = {
                    'dataset_id': dataset_id,
                    'dataset_type': dataset_type,
                    'status': 'prepared',
                    'min_samples': min_samples,
                    'workspace_dir': str(dataset_dir)
                }
                
                # Register dataset in registry if available
                if self.dataset_registry:
                    self.dataset_registry.register_dataset(dataset_id, metadata)
                else:
                    logger.debug("No dataset registry available, skipping registration")
                
                return True
            else:
                logger.error(f"Dataset {dataset_id} preparation verification failed")
                return False
        except Exception as e:
            logger.error(f"Final check failed for {dataset_id}: {e}")
            return False
    
    def _check_raw_data_exists(self, dataset_dir: Path, dataset_id: str) -> bool:
        """Check if raw downloaded data exists (before organization)."""
        # Look for common indicators of downloaded data
        indicators = [
            # Zip files
            list(dataset_dir.glob("*.zip")),
            # Tar files
            list(dataset_dir.glob("*.tar*")),
            # Extracted directories (excluding organized structure)
            [d for d in dataset_dir.iterdir() 
             if d.is_dir() and d.name not in ['images', 'audio', 'train', 'test', 'val', 'validation', '__pycache__']],
            # Raw data files
            list(dataset_dir.glob("*.csv")),
            list(dataset_dir.glob("*.json")),
            list(dataset_dir.glob("*.txt")),
        ]
        
        # If any indicator exists, consider raw data present
        return any(indicator for indicator in indicators)
    
    def prepare_cifar_dataset(self, dataset_type: str = "cifar10", force_redownload: bool = False) -> tuple:
        """
        Prepare CIFAR-10 or CIFAR-100 dataset using unified management.
        
        Args:
            dataset_type: "cifar10" or "cifar100"
            force_redownload: Force re-download even if dataset exists
            
        Returns:
            Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
        """
        from pathlib import Path
        import torchvision
        import torchvision.transforms as transforms
        
        if dataset_type == "cifar10":
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            min_images = 1000
            dataset_class = torchvision.datasets.CIFAR10
        elif dataset_type == "cifar100":
            class_names = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm'
            ]
            min_images = 5000
            dataset_class = torchvision.datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        def check_cifar_prepared(dataset_dir: Path) -> bool:
            """Check if CIFAR dataset is properly prepared."""
            images_dir = dataset_dir / "images"
            if not images_dir.exists():
                return False
            
            # Check that we have train and test directories with images
            train_count = len(list(images_dir.glob("train/*/*.png")))
            test_count = len(list(images_dir.glob("test/*/*.png")))
            return train_count > min_images and test_count > 0
        
        def download_and_organize_cifar(dataset_dir: Path):
            """Download and organize CIFAR dataset in one step."""
            # Download dataset using torchvision
            transform = transforms.Compose([transforms.ToTensor()])
            
            train_dataset = dataset_class(
                root=str(dataset_dir), train=True, download=True, transform=transform
            )
            test_dataset = dataset_class(
                root=str(dataset_dir), train=False, download=True, transform=transform
            )
            
            # Organize into ImageNet-style structure
            images_dir = dataset_dir / "images"
            
            # Create directory structure
            for split in ['train', 'test']:
                for class_name in class_names:
                    (images_dir / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Convert and save images
            def save_cifar_images(dataset, base_dir: Path, class_names: list, split: str) -> tuple:
                """Save CIFAR images to disk in ImageNet-style structure."""
                paths = []
                labels = []
                
                logger.info(f"Saving {split} images...")
                
                for idx, (image_tensor, label) in enumerate(dataset):
                    if idx % 10000 == 0:
                        logger.info(f"Processed {idx}/{len(dataset)} {split} images")
                    
                    # Convert tensor to PIL Image
                    image = transforms.ToPILImage()(image_tensor)
                    
                    # Save image
                    class_name = class_names[label]
                    image_path = base_dir / class_name / f"{idx:05d}.png"
                    image.save(image_path)
                    
                    paths.append(str(image_path))
                    labels.append(label)
                
                return paths, labels
            
            train_paths, train_labels = save_cifar_images(
                train_dataset, images_dir / 'train', class_names, 'train'
            )
            test_paths, test_labels = save_cifar_images(
                test_dataset, images_dir / 'test', class_names, 'test'
            )
            
            logger.info(f"{dataset_type.upper()} prepared: {len(train_paths)} train, {len(test_paths)} test images")
        
        # Use unified preparation (combining download and organize into one step)
        success = self.prepare_dataset(
            dataset_id=dataset_type,
            dataset_type="vision",
            check_function=check_cifar_prepared,
            download_function=download_and_organize_cifar,
            organize_function=None,  # Already handled in download step
            force_redownload=force_redownload,
            min_samples=min_images
        )
        
        if not success:
            raise RuntimeError(f"Failed to prepare {dataset_type} dataset")
        
        # Load and return the prepared data
        def load_existing_cifar(images_dir: Path, class_names: list) -> tuple:
            """Load existing CIFAR directory structure."""
            train_paths, train_labels = [], []
            test_paths, test_labels = [], []
            
            for split, (paths_list, labels_list) in [('train', (train_paths, train_labels)), 
                                                    ('test', (test_paths, test_labels))]:
                for label, class_name in enumerate(class_names):
                    class_dir = images_dir / split / class_name
                    if class_dir.exists():
                        for img_path in sorted(class_dir.glob("*.png")):
                            paths_list.append(str(img_path))
                            labels_list.append(label)
            
            return train_paths, train_labels, test_paths, test_labels, class_names
        
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_type)
        return load_existing_cifar(dataset_dir / "images", class_names)
    
    def get_cached_color_mapping(self, dataset_id: str, unique_classes: list) -> dict:
        """
        Get cached color mapping for a dataset, generating and caching if needed.
        
        Args:
            dataset_id: Dataset identifier (e.g., task_id or dataset name)
            unique_classes: List of unique class labels
            
        Returns:
            Dictionary with:
                - class_to_color: Maps class labels to color names
                - color_to_class: Maps color names to class labels
        """
        # Create stable cache key based on dataset and sorted classes
        cache_key_data = {
            'dataset_id': str(dataset_id),
            'unique_classes': sorted([str(c) for c in unique_classes])
        }
        
        # Check if cache_manager is available
        if self.cache_manager is None:
            logger.warning("No cache manager available, generating color mapping without caching")
            # Generate mappings directly without caching
            try:
                from marvis.viz.utils.styling import get_class_color_name_map, get_color_to_class_map
                import numpy as np
                unique_classes_array = np.array(unique_classes)
                class_to_color = get_class_color_name_map(unique_classes_array)
                color_to_class = get_color_to_class_map(unique_classes_array)
                return {
                    'class_to_color': class_to_color,
                    'color_to_class': color_to_class
                }
            except Exception as e:
                logger.error(f"Error generating color mapping: {e}")
                return {'class_to_color': {}, 'color_to_class': {}}
        
        cache_key = self.cache_manager.get_cache_key(**cache_key_data)
        
        # Try to load from cache
        cached_mapping = self.cache_manager.load_from_cache(
            cache_type='color_mappings',
            cache_key=cache_key,
            extension='.json'
        )
        
        if cached_mapping is not None:
            logger.debug(f"Loaded color mapping from cache for dataset {dataset_id}")
            # Convert keys back to original types if needed
            return self._restore_color_mapping_types(cached_mapping, unique_classes)
        
        # Generate new mapping if not cached
        logger.info(f"Generating new color mapping for dataset {dataset_id} with {len(unique_classes)} classes")
        
        try:
            from marvis.viz.utils.styling import get_class_color_name_map, get_color_to_class_map
            
            # Convert to numpy array for consistent handling
            import numpy as np
            unique_classes_array = np.array(unique_classes)
            
            # Generate mappings
            class_to_color = get_class_color_name_map(unique_classes_array)
            color_to_class = get_color_to_class_map(unique_classes_array)
            
            # Convert to serializable format
            mapping_data = {
                'class_to_color': {str(k): v for k, v in class_to_color.items()},
                'color_to_class': {k: str(v) for k, v in color_to_class.items()},
                'dataset_id': str(dataset_id),
                'num_classes': len(unique_classes)
            }
            
            # Save to cache
            success = self.cache_manager.save_to_cache(
                cache_type='color_mappings',
                cache_key=cache_key,
                data=mapping_data,
                extension='.json'
            )
            
            if success:
                logger.info(f"Cached color mapping for dataset {dataset_id}")
            else:
                logger.warning(f"Failed to cache color mapping for dataset {dataset_id}")
            
            # Return mapping in original format
            return {
                'class_to_color': class_to_color,
                'color_to_class': color_to_class
            }
            
        except Exception as e:
            logger.error(f"Error generating color mapping for dataset {dataset_id}: {e}")
            # Return empty mappings as fallback
            return {
                'class_to_color': {},
                'color_to_class': {}
            }
    
    def _restore_color_mapping_types(self, cached_mapping: dict, unique_classes: list) -> dict:
        """
        Restore proper types for cached color mapping based on original class types.
        
        Args:
            cached_mapping: Cached mapping with string keys
            unique_classes: Original unique classes with proper types
            
        Returns:
            Mapping with restored types
        """
        # Create mapping from string to original type
        str_to_original = {str(c): c for c in unique_classes}
        
        # Restore class_to_color mapping
        class_to_color = {}
        for str_class, color in cached_mapping.get('class_to_color', {}).items():
            if str_class in str_to_original:
                class_to_color[str_to_original[str_class]] = color
        
        # Restore color_to_class mapping
        color_to_class = {}
        for color, str_class in cached_mapping.get('color_to_class', {}).items():
            if str_class in str_to_original:
                color_to_class[color] = str_to_original[str_class]
        
        return {
            'class_to_color': class_to_color,
            'color_to_class': color_to_class
        }


# Global resource manager instance
_resource_manager: Optional[MarvisResourceManager] = None


def get_resource_manager(config: Optional[ResourceConfig] = None) -> MarvisResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = MarvisResourceManager(config)
    return _resource_manager


def reset_resource_manager():
    """Reset global resource manager instance (mainly for testing)."""
    global _resource_manager
    _resource_manager = None


# Convenience functions for common operations
def prepare_cifar_dataset(dataset_type: str = "cifar10", force_redownload: bool = False) -> tuple:
    """
    Convenience function to prepare CIFAR datasets using unified management.
    
    Args:
        dataset_type: "cifar10" or "cifar100"
        force_redownload: Force re-download even if dataset exists
        
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    resource_manager = get_resource_manager()
    return resource_manager.dataset_preparer.prepare_cifar_dataset(dataset_type, force_redownload)