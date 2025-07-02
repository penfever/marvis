"""
Utility functions for MARVIS.

This module provides various utilities for MARVIS model work:
- Logging setup and configuration
- Model loading with path resolution
- Checkpoint finding and management
- System utilities like timeout context managers
- Feature selection utilities
- GPU monitoring for WandB integration
- JSON serialization utilities
- LLM evaluation utilities
- Training argument parsing utilities
- Weights & Biases data extraction utilities
- Unified metrics logging with standardized naming
- VLM utilities for response parsing and conversation formatting
- Visualization utilities for saving and processing plots
- Platform compatibility utilities for device detection and configuration
- Results management with standardized storage and organization
"""

from .logging import setup_logging, setup_notebook_logging
from .model_utils import load_pretrained_model, find_best_checkpoint
from .system import timeout_context
from .feature_selection_utils import (
    select_features_for_token_limit, 
    create_reduced_dataset, 
    test_feature_selection
)
from .gpu_monitoring import (
    GPUMonitor, 
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring
)
from .json_utils import (
    safe_json_dump, 
    safe_json_dumps, 
    convert_for_json_serialization, 
    save_results
)
from .llm_evaluation_utils import (
    drop_feature_for_oom,
    is_oom_error,
    create_tabllm_note,
    apply_feature_reduction,
    predict_with_jolt_logprobs,
    predict_with_simple_logprobs,
    predict_with_generation,
    unified_llm_predict,
    _test_prediction_methods
)
from .training_args import (
    create_common_parser,
    create_single_dataset_parser,
    create_multi_dataset_parser
)
from .evaluation_args import (
    create_evaluation_parser,
    create_dataset_evaluation_parser
)
from .wandb_extractor import (
    fetch_wandb_data,
    fetch_wandb_train_data,
    extract_task_id_from_dataset_name,
    extract_split_idx_from_name,
    extract_task_idx_from_name,
    is_numeric,
    safe_float_convert,
    should_exclude_failed_run,
    extract_model_metrics_from_summary,
    extract_results_from_wandb,
    detect_run_types
)
from .unified_metrics import (
    MetricsLogger,
    log_metrics_for_result,
    get_standard_metric_names
)
from .vlm_utils import (
    parse_vlm_response,
    create_vlm_conversation
)
# Visualization utilities moved to marvis.viz.utils.common
# from .visualization_utils import (...)  # Now in marvis.viz.utils.common
# Platform utilities have been removed - use device_utils instead
# from .device_utils import detect_optimal_device
from .seeding import (
    set_seed,
    set_seed_with_args,
    create_random_state,
    set_random_seed
)
from .resource_manager import (
    get_resource_manager,
    ResourceConfig,
    DatasetMetadata,
    MarvisResourceManager,
    reset_resource_manager,
    prepare_cifar_dataset
)
from .results_manager import (
    get_results_manager,
    ResultsManager,
    ExperimentMetadata,
    EvaluationResults,
    ResultsArtifacts,
    reset_results_manager,
    save_results_unified
)
from .results_migration import (
    migrate_legacy_results,
    validate_result_file,
    ResultsFormatDetector,
    ResultsMigrator,
    create_migration_adapters
)

__all__ = [
    # Logging utilities
    "setup_logging", 
    "setup_notebook_logging",
    
    # Model utilities
    "load_pretrained_model",
    "find_best_checkpoint",
    
    # System utilities
    "timeout_context",
    
    # Feature selection utilities
    "select_features_for_token_limit",
    "create_reduced_dataset", 
    "test_feature_selection",
    
    # GPU monitoring utilities
    "GPUMonitor",
    "init_wandb_with_gpu_monitoring", 
    "cleanup_gpu_monitoring",
    
    # JSON utilities
    "safe_json_dump", 
    "safe_json_dumps", 
    "convert_for_json_serialization", 
    "save_results",
    
    # LLM evaluation utilities
    "drop_feature_for_oom",
    "is_oom_error",
    "create_tabllm_note",
    "apply_feature_reduction",
    "predict_with_jolt_logprobs",
    "predict_with_simple_logprobs",
    "predict_with_generation",
    "unified_llm_predict",
    "_test_prediction_methods",
    
    # Training argument parsing utilities
    "create_common_parser",
    "create_single_dataset_parser", 
    "create_multi_dataset_parser",
    
    # Evaluation argument parsing utilities
    "create_evaluation_parser",
    "create_dataset_evaluation_parser",
    
    # Weights & Biases data extraction utilities
    "fetch_wandb_data",
    "fetch_wandb_train_data",
    "extract_task_id_from_dataset_name",
    "extract_split_idx_from_name",
    "extract_task_idx_from_name",
    "is_numeric",
    "safe_float_convert",
    "should_exclude_failed_run",
    "extract_model_metrics_from_summary",
    "extract_results_from_wandb",
    "detect_run_types",
    
    # Unified metrics utilities
    "MetricsLogger",
    "log_metrics_for_result",
    "get_standard_metric_names",
    
    # VLM utilities
    "parse_vlm_response",
    "create_vlm_conversation",
    
    # Visualization utilities moved to marvis.viz.utils.common
    
    # Platform utilities removed - use device_utils instead
    
    # Seeding utilities
    "set_seed",
    "set_seed_with_args", 
    "create_random_state",
    "set_random_seed",
    
    # Resource management utilities
    "get_resource_manager",
    "ResourceConfig",
    "DatasetMetadata",
    "MarvisResourceManager", 
    "reset_resource_manager",
    "prepare_cifar_dataset",
    
    # Results management utilities
    "get_results_manager",
    "ResultsManager",
    "ExperimentMetadata",
    "EvaluationResults",
    "ResultsArtifacts",
    "reset_results_manager",
    "save_results_unified",
    
    # Results migration utilities
    "migrate_legacy_results",
    "validate_result_file",
    "ResultsFormatDetector",
    "ResultsMigrator",
    "create_migration_adapters"
]