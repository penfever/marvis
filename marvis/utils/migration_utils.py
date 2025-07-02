"""
Migration utilities for transitioning to the new resource management system.

This module helps migrate existing caches, configs, and datasets to the
new organized structure without losing existing work.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .resource_manager import (
    ResourceManager, 
    DatasetMetadata, 
    get_resource_manager
)


logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages migration of existing resources to new structure."""
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.rm = resource_manager or get_resource_manager()
    
    def migrate_all(self, dry_run: bool = False) -> Dict[str, int]:
        """Migrate all discoverable resources."""
        results = {
            'failed_datasets': 0,
            'embedding_caches': 0,
            'jolt_configs': 0,
            'tabllm_configs': 0,
            'cc18_semantic': 0,
            'errors': 0
        }
        
        try:
            results['failed_datasets'] = self.migrate_failed_dataset_cache(dry_run)
            results['embedding_caches'] = self.migrate_embedding_caches(dry_run)
            results['jolt_configs'] = self.migrate_jolt_configs(dry_run)
            results['tabllm_configs'] = self.migrate_tabllm_configs(dry_run)
            results['cc18_semantic'] = self.migrate_cc18_semantic(dry_run)
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results['errors'] += 1
        
        return results
    
    def migrate_failed_dataset_cache(self, dry_run: bool = False) -> int:
        """Migrate the global failed dataset cache."""
        old_cache_path = Path.home() / '.marvis_failed_datasets.json'
        
        if not old_cache_path.exists():
            return 0
        
        try:
            with open(old_cache_path, 'r') as f:
                failed_ids = json.load(f)
            
            if not isinstance(failed_ids, list):
                return 0
            
            # Store in new cache system
            cache_key = self.rm.cache_manager.get_cache_key(type='failed_datasets')
            
            if not dry_run:
                success = self.rm.cache_manager.save_to_cache(
                    'system', cache_key, failed_ids, '.json'
                )
                
                if success:
                    # Backup old file before removing
                    backup_path = old_cache_path.with_suffix('.json.bak')
                    shutil.move(old_cache_path, backup_path)
                    logger.info(f"Migrated failed dataset cache, backed up to {backup_path}")
                else:
                    logger.warning("Failed to migrate failed dataset cache")
                    return 0
            else:
                logger.info(f"[DRY RUN] Would migrate {len(failed_ids)} failed dataset IDs")
            
            return len(failed_ids)
        
        except Exception as e:
            logger.error(f"Error migrating failed dataset cache: {e}")
            return 0
    
    def migrate_embedding_caches(self, dry_run: bool = False) -> int:
        """Migrate existing embedding cache files."""
        # Look for embedding caches in common locations
        search_dirs = [
            Path('./data'),
            Path('./cache'),
            Path.cwd(),
            self.rm.path_resolver.base_dir.parent / 'data'  # Check if there's a sibling data dir
        ]
        
        migrated_count = 0
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Look for TabPFN embedding files
            for pattern in ['*tabpfn_embeddings*.npz', '*_embeddings.npz', 'prefix_data.npz']:
                for cache_file in search_dir.glob(pattern):
                    try:
                        if self._migrate_embedding_file(cache_file, dry_run):
                            migrated_count += 1
                    except Exception as e:
                        logger.error(f"Error migrating {cache_file}: {e}")
        
        return migrated_count
    
    def _migrate_embedding_file(self, cache_file: Path, dry_run: bool) -> bool:
        """Migrate a single embedding cache file."""
        try:
            # Try to extract dataset info from filename
            filename = cache_file.stem
            
            # Common patterns in embedding filenames
            dataset_id = None
            if 'tabpfn_embeddings' in filename:
                # Extract hash or dataset name
                parts = filename.split('_')
                if len(parts) > 2:
                    dataset_id = parts[-1]  # Usually the hash
            elif filename == 'prefix_data':
                # This is typically in a dataset-specific directory
                dataset_id = cache_file.parent.name
            
            if not dataset_id:
                logger.debug(f"Could not determine dataset ID for {cache_file}")
                return False
            
            # Create a cache key based on file content hash
            import hashlib
            with open(cache_file, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()[:16]
            
            cache_key = f"{dataset_id}_{content_hash}"
            
            if not dry_run:
                # Copy to new cache location
                new_cache_path = self.rm.cache_manager.get_cache_path(
                    'embeddings', cache_key, '.npz'
                )
                
                if not new_cache_path.exists():
                    shutil.copy2(cache_file, new_cache_path)
                    logger.info(f"Migrated embedding cache: {cache_file} -> {new_cache_path}")
                    return True
            else:
                logger.info(f"[DRY RUN] Would migrate embedding cache: {cache_file}")
                return True
        
        except Exception as e:
            logger.error(f"Error migrating embedding file {cache_file}: {e}")
            return False
        
        return False
    
    def migrate_jolt_configs(self, dry_run: bool = False) -> int:
        """Migrate JOLT configuration files."""
        # Find JOLT configs in the codebase
        project_root = self._find_project_root()
        if not project_root:
            return 0
        
        jolt_dir = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'jolt'
        if not jolt_dir.exists():
            return 0
        
        migrated_count = 0
        
        for config_file in jolt_dir.glob('jolt_config_*.json'):
            try:
                # Extract dataset name from filename
                filename = config_file.stem
                dataset_name = filename.replace('jolt_config_', '')
                
                if not dry_run:
                    # Copy to new managed location
                    new_config_path = self.rm.path_resolver.get_config_path('jolt', dataset_name)
                    if new_config_path and not new_config_path.exists():
                        new_config_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, new_config_path)
                        logger.info(f"Migrated JOLT config: {config_file} -> {new_config_path}")
                        migrated_count += 1
                else:
                    logger.info(f"[DRY RUN] Would migrate JOLT config: {config_file}")
                    migrated_count += 1
            
            except Exception as e:
                logger.error(f"Error migrating JOLT config {config_file}: {e}")
        
        return migrated_count
    
    def migrate_tabllm_configs(self, dry_run: bool = False) -> int:
        """Migrate TabLLM configuration files."""
        project_root = self._find_project_root()
        if not project_root:
            return 0
        
        tabllm_dir = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'tabllm_like'
        if not tabllm_dir.exists():
            return 0
        
        migrated_count = 0
        
        # Migrate template files
        for template_file in tabllm_dir.glob('templates_*.yaml'):
            try:
                filename = template_file.stem
                dataset_name = filename.replace('templates_', '')
                
                if not dry_run:
                    new_config_path = self.rm.path_resolver.get_config_path('tabllm', dataset_name)
                    if new_config_path and not new_config_path.exists():
                        new_config_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(template_file, new_config_path)
                        logger.info(f"Migrated TabLLM template: {template_file} -> {new_config_path}")
                        migrated_count += 1
                else:
                    logger.info(f"[DRY RUN] Would migrate TabLLM template: {template_file}")
                    migrated_count += 1
            
            except Exception as e:
                logger.error(f"Error migrating TabLLM template {template_file}: {e}")
        
        # Migrate mapping file
        mapping_file = tabllm_dir / 'openml_task_mapping.json'
        if mapping_file.exists():
            try:
                if not dry_run:
                    new_mapping_path = self.rm.path_resolver.get_configs_dir() / 'tabllm' / 'openml_task_mapping.json'
                    new_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                    if not new_mapping_path.exists():
                        shutil.copy2(mapping_file, new_mapping_path)
                        logger.info(f"Migrated TabLLM mapping: {mapping_file} -> {new_mapping_path}")
                        migrated_count += 1
                else:
                    logger.info(f"[DRY RUN] Would migrate TabLLM mapping: {mapping_file}")
                    migrated_count += 1
            except Exception as e:
                logger.error(f"Error migrating TabLLM mapping: {e}")
        
        return migrated_count
    
    def migrate_cc18_semantic(self, dry_run: bool = False) -> int:
        """Migrate CC18 semantic data files."""
        project_root = self._find_project_root()
        if not project_root:
            return 0
        
        semantic_dir = project_root / 'data' / 'cc18_semantic'
        if not semantic_dir.exists():
            return 0
        
        migrated_count = 0
        
        for semantic_file in semantic_dir.glob('*.json'):
            try:
                dataset_id = semantic_file.stem  # The OpenML task ID
                
                if not dry_run:
                    new_config_path = self.rm.path_resolver.get_config_path('cc18_semantic', dataset_id)
                    if new_config_path and not new_config_path.exists():
                        new_config_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(semantic_file, new_config_path)
                        logger.info(f"Migrated CC18 semantic: {semantic_file} -> {new_config_path}")
                        migrated_count += 1
                else:
                    logger.info(f"[DRY RUN] Would migrate CC18 semantic: {semantic_file}")
                    migrated_count += 1
            
            except Exception as e:
                logger.error(f"Error migrating CC18 semantic {semantic_file}: {e}")
        
        return migrated_count
    
    def _find_project_root(self) -> Optional[Path]:
        """Find the project root directory."""
        # Start from this file and look for setup.py or pyproject.toml
        current = Path(__file__).resolve()
        
        for parent in [current] + list(current.parents):
            if (parent / 'setup.py').exists() or (parent / 'pyproject.toml').exists():
                return parent
        
        # Fallback: assume standard structure
        # This file is in marvis/utils/, so project root is 2 levels up
        return current.parent.parent.parent
    
    def discover_existing_datasets(self, search_dirs: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
        """Discover existing dataset files that could be registered."""
        if search_dirs is None:
            search_dirs = [
                './data',
                './cache',
                str(Path.cwd()),
                str(self.rm.path_resolver.base_dir.parent / 'data')
            ]
        
        discovered = []
        
        for search_dir_str in search_dirs:
            search_dir = Path(search_dir_str)
            if not search_dir.exists():
                continue
            
            # Look for CSV files
            for csv_file in search_dir.rglob('*.csv'):
                if '_y' in csv_file.name or '_Y' in csv_file.name:
                    continue  # Skip label files
                
                # Try to extract dataset name
                dataset_name = csv_file.stem
                if '_X' in dataset_name:
                    dataset_name = dataset_name.replace('_X', '')
                elif '_data' in dataset_name:
                    dataset_name = dataset_name.replace('_data', '')
                elif '_train' in dataset_name:
                    dataset_name = dataset_name.replace('_train', '')
                
                discovered.append((dataset_name, csv_file))
        
        return discovered
    
    def register_discovered_datasets(self, discovered: List[Tuple[str, Path]], dry_run: bool = False) -> int:
        """Register discovered datasets in the registry."""
        registered_count = 0
        
        for dataset_name, file_path in discovered:
            try:
                # Check if already registered
                existing = self.rm.dataset_registry.find_dataset_by_name(dataset_name)
                if existing:
                    continue
                
                if not dry_run:
                    # Create metadata
                    metadata = DatasetMetadata(
                        id=dataset_name,
                        name=dataset_name,
                        source_type='csv',
                        file_path=str(file_path)
                    )
                    
                    # Try to get additional info from the file
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path, nrows=1)  # Just read first row for structure
                        metadata.num_features = len(df.columns) - 1  # Assume last column is target
                    except Exception:
                        pass
                    
                    if self.rm.dataset_registry.register_dataset(metadata):
                        registered_count += 1
                        logger.info(f"Registered discovered dataset: {dataset_name}")
                else:
                    logger.info(f"[DRY RUN] Would register dataset: {dataset_name} from {file_path}")
                    registered_count += 1
            
            except Exception as e:
                logger.error(f"Error registering dataset {dataset_name}: {e}")
        
        return registered_count


def run_migration(dry_run: bool = True) -> Dict[str, int]:
    """Run the complete migration process."""
    logger.info(f"Starting migration {'(DRY RUN)' if dry_run else '(LIVE)'}")
    
    migration_manager = MigrationManager()
    
    # Run main migration
    results = migration_manager.migrate_all(dry_run)
    
    # Discover and register datasets
    discovered = migration_manager.discover_existing_datasets()
    results['discovered_datasets'] = len(discovered)
    results['registered_datasets'] = migration_manager.register_discovered_datasets(discovered, dry_run)
    
    # Log summary
    logger.info("Migration completed:")
    for key, count in results.items():
        if count > 0:
            logger.info(f"  {key}: {count}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate MARVIS resources to new management system")
    parser.add_argument("--live", action="store_true", help="Perform live migration (default is dry run)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    results = run_migration(dry_run=not args.live)
    
    print("\nMigration Summary:")
    for key, count in results.items():
        print(f"  {key}: {count}")
    
    if not args.live:
        print("\nThis was a DRY RUN. Use --live to perform actual migration.")