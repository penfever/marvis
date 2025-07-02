#!/usr/bin/env python
"""
Test script for the robust resource management system.

This script validates that the resource manager correctly:
- Resolves paths in different environments
- Finds metadata files using package-aware methods
- Manages dataset workspaces consistently
- Handles environment variable configuration
"""

import os
import tempfile
import shutil
import json
from pathlib import Path
import pytest

# Add the project root to the path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marvis.utils.resource_manager import (
    get_resource_manager, 
    reset_resource_manager,
    ResourceConfig,
    MarvisResourceManager
)

def test_resource_manager_basic_functionality():
    """Test basic resource manager functionality."""
    # Reset any existing instance
    reset_resource_manager()
    
    # Create resource manager with default config
    rm = get_resource_manager()
    
    # Test basic directory creation
    base_dir = rm.path_resolver.get_base_dir()
    assert base_dir.exists()
    assert base_dir.name == '.marvis'
    
    # Test subdirectory creation
    datasets_dir = rm.path_resolver.get_datasets_dir()
    cache_dir = rm.path_resolver.get_cache_dir()
    configs_dir = rm.path_resolver.get_configs_dir()
    
    assert datasets_dir.exists()
    assert cache_dir.exists()
    assert configs_dir.exists()
    
    # Test dataset-specific directories
    dataset_dir = rm.path_resolver.get_dataset_dir("test_dataset")
    embed_dir = rm.path_resolver.get_embedding_dir("test_dataset")
    
    assert dataset_dir.exists()
    assert embed_dir.exists()
    assert str(dataset_dir).endswith("dataset_test_dataset")
    assert str(embed_dir).endswith("dataset_test_dataset/embeddings")

def test_resource_manager_environment_config():
    """Test resource manager with environment variable configuration."""
    reset_resource_manager()
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        custom_base = Path(tmp_dir) / "custom_marvis"
        custom_cache = Path(tmp_dir) / "custom_cache"
        
        # Set environment variables
        old_base = os.environ.get('MARVIS_BASE_DIR')
        old_cache = os.environ.get('MARVIS_CACHE_DIR')
        
        os.environ['MARVIS_BASE_DIR'] = str(custom_base)
        os.environ['MARVIS_CACHE_DIR'] = str(custom_cache)
        
        try:
            # Reset to ensure we get a fresh instance
            reset_resource_manager()
            
            # Create new resource manager
            rm = get_resource_manager()
            
            # Verify custom paths are used (resolve both to handle symlinks)
            actual_base = rm.path_resolver.get_base_dir().resolve()
            actual_cache = rm.path_resolver.get_cache_dir().resolve()
            custom_base_resolved = custom_base.resolve()
            custom_cache_resolved = custom_cache.resolve()
            
            assert actual_base == custom_base_resolved
            assert actual_cache == custom_cache_resolved
            
            # Test that directories are created
            assert custom_base.exists()
            assert custom_cache.exists()
            
        finally:
            # Restore environment
            if old_base:
                os.environ['MARVIS_BASE_DIR'] = old_base
            else:
                os.environ.pop('MARVIS_BASE_DIR', None)
            
            if old_cache:
                os.environ['MARVIS_CACHE_DIR'] = old_cache
            else:
                os.environ.pop('MARVIS_CACHE_DIR', None)
            
            reset_resource_manager()

def test_config_path_resolution():
    """Test config file path resolution with fallbacks."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # Test finding existing config files (should fallback to package resources)
    jolt_mapping = rm.path_resolver.get_config_path('jolt', 'openml_task_mapping.json')
    tabllm_mapping = rm.path_resolver.get_config_path('tabllm', 'openml_task_mapping.json')
    
    # These should find the files in the package if they exist
    if jolt_mapping:
        assert jolt_mapping.exists()
        assert 'openml_task_mapping.json' in str(jolt_mapping)
    
    if tabllm_mapping:
        assert tabllm_mapping.exists()
        assert 'openml_task_mapping.json' in str(tabllm_mapping)

def test_dataset_registry():
    """Test dataset registry functionality."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # Register a test dataset
    test_metadata = {
        'name': 'test_dataset',
        'features': 10,
        'samples': 1000,
        'task_type': 'classification'
    }
    
    rm.dataset_registry.register_dataset('test_123', test_metadata)
    
    # Verify registration
    dataset_info = rm.dataset_registry.get_dataset_info('test_123')
    assert dataset_info is not None
    assert dataset_info['metadata'] == test_metadata
    
    # Verify dataset appears in list
    datasets = rm.dataset_registry.list_datasets()
    assert 'test_123' in datasets
    
    # Test unregistration
    rm.dataset_registry.unregister_dataset('test_123')
    assert rm.dataset_registry.get_dataset_info('test_123') is None

def test_csv_file_finding():
    """Test CSV file finding functionality."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # Create temporary CSV files for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test CSV file
        csv_file = tmp_path / "test_dataset.csv"
        csv_file.write_text("col1,col2,target\n1,2,0\n3,4,1\n")
        
        # Test finding the CSV file
        found_path = rm.find_csv_file("test_dataset", [str(tmp_path)])
        assert found_path is not None
        assert Path(found_path).name == "test_dataset.csv"
        
        # Test with non-existent dataset
        not_found = rm.find_csv_file("nonexistent", [str(tmp_path)])
        assert not_found is None

def test_metadata_validation():
    """Test metadata validation functionality."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # Test validation for a model type that should have configs
    # Note: This will depend on which configs actually exist in the package
    result = rm.validate_model_metadata(46, 'jolt')  # splice dataset
    
    # The result should have the expected structure
    assert 'valid' in result
    assert 'errors' in result
    assert 'warnings' in result
    assert 'dataset_name' in result
    
    # Test with non-existent task ID
    result = rm.validate_model_metadata(999999, 'jolt')
    assert not result['valid']
    assert any('No JOLT config found for OpenML task ID' in error for error in result['errors'])

def test_workspace_isolation():
    """Test that dataset workspaces are properly isolated."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # Create workspaces for different datasets
    workspace1 = rm.get_dataset_workspace("dataset_1")
    workspace2 = rm.get_dataset_workspace("dataset_2")
    
    # Verify workspaces are different
    assert workspace1 != workspace2
    assert workspace1.name == "dataset_dataset_1"
    assert workspace2.name == "dataset_dataset_2"
    
    # Verify both exist
    assert workspace1.exists()
    assert workspace2.exists()
    
    # Create files in each workspace
    file1 = workspace1 / "test_file.txt"
    file2 = workspace2 / "test_file.txt"
    
    file1.write_text("content 1")
    file2.write_text("content 2")
    
    # Verify isolation
    assert file1.read_text() == "content 1"
    assert file2.read_text() == "content 2"

def test_backward_compatibility():
    """Test that the resource manager maintains backward compatibility."""
    reset_resource_manager()
    rm = get_resource_manager()
    
    # The resource manager should work even if package resources aren't available
    # This tests the fallback mechanisms
    
    # Test config manager methods don't crash
    jolt_config = rm.config_manager.find_jolt_config("nonexistent_dataset")
    tabllm_template = rm.config_manager.find_tabllm_template("nonexistent_dataset")
    
    # These should return None gracefully, not crash
    assert jolt_config is None
    assert tabllm_template is None
    
    # Test task mapping loading
    jolt_mapping = rm.config_manager.get_openml_task_mapping('jolt')
    tabllm_mapping = rm.config_manager.get_openml_task_mapping('tabllm')
    
    # These might be None if files don't exist, but shouldn't crash
    if jolt_mapping:
        assert isinstance(jolt_mapping, dict)
    if tabllm_mapping:
        assert isinstance(tabllm_mapping, dict)

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_resource_manager_basic_functionality,
        test_resource_manager_environment_config,
        test_config_path_resolution,
        test_dataset_registry,
        test_csv_file_finding,
        test_metadata_validation,
        test_workspace_isolation,
        test_backward_compatibility,
    ]
    
    print("🧪 Running Resource Management Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"Running {test.__name__}...", end=" ")
            test()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            import traceback
            print(f"❌ FAILED: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            failed += 1
        finally:
            # Clean up after each test
            reset_resource_manager()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)