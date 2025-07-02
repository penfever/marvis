# MARVIS Tests

This directory contains test scripts for the MARVIS library.

## Current Test Structure

The tests directory has been cleaned up to remove redundancy and overlap. Current test files provide comprehensive coverage without duplication:

### ✅ Essential Tests

#### Core Functionality Tests
- **`test_install.py`**: ✅ Tests the installation of the MARVIS package and verifies all imports work correctly.
  ```bash
  python tests/test_install.py
  ```

- **`test_embedding_cache.py`**: ✅ Tests embedding cache loading functionality with limits and multi-ensemble support.
  ```bash
  python tests/test_embedding_cache.py
  ```

- **`test_mps_detection.py`**: ⚠️ Tests Metal Performance Shaders (MPS) detection on Apple Silicon. **Note: Fixed function name issue, but some platform detection features may be partially working.**
  ```bash
  python tests/test_mps_detection.py
  ```

- **`test_resource_management.py`**: ✅ Tests comprehensive resource management system functionality. **Note: Fixed error message validation.**
  ```bash
  python tests/test_resource_management.py
  ```

#### Integration Tests
- **`test_comprehensive_vlm_prompting.py`**: ⏳ Comprehensive test for VLM prompting, API integration, and model backends. **Note: Times out after 3 minutes due to model loading.**
  ```bash
  python tests/test_comprehensive_vlm_prompting.py  # May timeout
  ```

- **`test_audio_simple.py`**: ✅ Tests audio processing pipeline with Whisper and CLAP models.
  ```bash
  python tests/test_audio_simple.py
  ```

- **`test_tabllm_comprehensive.py`**: ⏳ Comprehensive test for TabLLM baseline functionality and semantic feature alignment. **Note: Times out after 3 minutes due to model loading.**
  ```bash
  python tests/test_tabllm_comprehensive.py  # May timeout
  ```

#### Specialized Tests
- **`test_metadata_validation.py`**: ✅ Tests metadata validation functionality for various benchmarks. **Note: Shows TabLLM directory not found warnings (expected).**
  ```bash
  python tests/test_metadata_validation.py
  ```

#### JOLT and Regression Tests
- **`test_jolt_integration.py`**: ✅ Tests JOLT integration functionality. **Note: Fixed import path issue.**
  ```bash
  python tests/test_jolt_integration.py
  ```

- **`test_jolt_config_loading.py`**: ⚠️ Tests JOLT config loading logic. **Note: Fixed missing 'context' field issue, but some integration tests still fail.**
  ```bash
  python tests/test_jolt_config_loading.py
  ```

- **`test_regression_jolt.py`**: ✅ Tests JOLT with regression datasets.
  ```bash
  python tests/test_regression_jolt.py
  ```

- **`test_marvis_tsne_regression_metrics.py`**: ⚠️ Tests regression metrics calculation. **Note: Fixed task type detection issue, but some metrics validation may still fail.**
  ```bash
  python tests/test_marvis_tsne_regression_metrics.py
  ```

- **`test_regression_metrics_fix.py`**: ✅ Tests the regression metrics fix implementation.
  ```bash
  python tests/test_regression_metrics_fix.py
  ```

### 🔧 Development Tools

- **`test_example_parameter_validation.py`**: ✅ Validates example script parameters for consistency.
  ```bash
  python tests/test_example_parameter_validation.py
  ```

## 🔧 Recent Fixes and Issues Resolved

### Fixed Issues
1. **MPS Detection Test**: Fixed function name error (`get_optimal_device` → `detect_optimal_device`)
2. **Resource Management Test**: Fixed error message validation (updated expected error text)
3. **JOLT Integration Test**: Fixed import path to include JOLT directory
4. **JOLT Config Loading**: Added missing 'context' field to synthesized configs
5. **Regression Metrics Test**: Fixed task type detection by providing explicit task type

### Remaining Issues
1. **Model Loading Timeouts**: Some VLM-based tests timeout after 3 minutes due to large model downloads
2. **Metrics Validation**: Some regression tests still show mixed classification/regression metrics
3. **Platform Detection**: MPS detection partially working but some platform features need verification

### Dependencies Added
- **pytest**: Added for better test framework support
  ```bash
  pip install pytest
  ```

## Test Coverage

The current test suite provides comprehensive coverage of:

1. **Package Installation**: Verify MARVIS package installs correctly and all imports work
2. **Core Infrastructure**: Resource management, caching, metadata validation
3. **Model Integration**: VLM prompting, API integration, backend selection
4. **Multi-modal Pipelines**: Tabular (TabLLM), audio (Whisper/CLAP), vision workflows
5. **Platform Support**: Apple Silicon MPS detection and optimization
6. **Semantic Features**: Feature alignment, expansion, and note generation
7. **Development Tools**: Parameter validation and code quality checks

## Removed Redundant Tests

The following test files were removed to eliminate redundancy and overlap:

- `test_api_integration.py` - Functionality covered by `test_comprehensive_vlm_prompting.py`
- `test_vllm_backend.py` - Backend testing covered by comprehensive VLM test
- `test_jolt_comprehensive.py` - Model testing covered by TabLLM comprehensive test
- `test_semantic_names_isolated.py` - Feature testing covered by comprehensive VLM test
- `test_model_save_load.py` - Basic functionality covered by other model tests
- `test_regression_workflow.py` - Workflow covered by comprehensive tabular tests

## Running Tests

### Run All Essential Tests
```bash
# Core functionality (all should pass quickly)
python tests/test_install.py
python tests/test_embedding_cache.py
python tests/test_mps_detection.py
python tests/test_resource_management.py

# Fast integration tests
python tests/test_audio_simple.py
python tests/test_metadata_validation.py
python tests/test_example_parameter_validation.py

# JOLT and regression tests
python tests/test_jolt_integration.py
python tests/test_regression_jolt.py
python tests/test_regression_metrics_fix.py

# Slow tests (may timeout at 3 minutes)
# python tests/test_comprehensive_vlm_prompting.py  # May timeout
# python tests/test_tabllm_comprehensive.py  # May timeout
# python tests/test_jolt_config_loading.py  # Some tests may fail
# python tests/test_marvis_tsne_regression_metrics.py  # Some validation issues
```

### Run with pytest
```bash
# Install pytest if not available
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_install.py tests/test_embedding_cache.py -v
```

## Dependencies

### Required for Audio Tests
```bash
pip install msclap  # For CLAP zero-shot classifier
```

### Required for VLM Tests
VLM tests can run with either:
- Local models (using transformers/VLLM)
- API models (OpenAI, Gemini) - requires API keys

### Required for All Tests
All core dependencies are installed with the main MARVIS package:
```bash
pip install -e ".[dev]"  # Includes test dependencies
```

## Writing New Tests

When adding new features to MARVIS, please add corresponding tests to this directory. Test files should:

1. Follow the naming convention `test_*.py`
2. Include docstrings explaining what is being tested
3. Use descriptive test function names
4. Handle cleanup of any temporary files/directories
5. Include proper error handling and informative error messages
6. Check for redundancy with existing comprehensive tests before creating new files

## Test Organization Principles

The current test structure follows these principles:

1. **Comprehensive over Isolated**: Prefer comprehensive integration tests over many isolated unit tests
2. **No Redundancy**: Each piece of functionality is tested in exactly one authoritative test
3. **Clear Separation**: Core tests, integration tests, and utilities are clearly separated
4. **Maintainable**: Fewer, more comprehensive tests are easier to maintain than many small tests

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running tests from the project root directory and that the MARVIS package is installed in editable mode:
```bash
cd /path/to/marvis
pip install -e .
python tests/test_name.py
```

### Missing Dependencies
Some tests require additional dependencies:
- Audio tests require `msclap`: `pip install msclap`
- VLM tests may require API keys for external models

### Timeout Issues
Some tests may take time due to model loading. The audio and VLM tests download models on first run.

### Model Loading Issues
If model loading fails:
- Check available memory (some models require significant RAM/VRAM)
- Verify internet connection for model downloads
- Check API keys for external model services