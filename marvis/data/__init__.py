"""
Data loading, processing, and preparation utilities.
"""

from .dataset import (
    load_dataset,
    load_datasets,
    analyze_dataset,
    create_llm_dataset,
    list_available_datasets,
    get_dataset_info
)

from .embeddings import (
    get_tabpfn_embeddings,
    prepare_tabpfn_embeddings_for_prefix
)

from .csv_utils import (
    is_csv_dataset,
    find_csv_file,
    load_csv_dataset,
    load_dataset_with_metadata,
    find_csv_with_fallbacks
)

from .evaluation_utils import (
    load_datasets_for_evaluation,
    preprocess_datasets_for_evaluation,
    validate_dataset_for_evaluation
)

from .dataset_tabular import (
    preprocess_features,
    process_tabular_dataset_for_training,
    compute_frequency_distribution,
    compute_label_frequency_mapping,
    apply_label_mapping,
    compute_baseline_probabilities
)

from .time_series import (
    TimeSeriesDataset,
    load_gift_eval_dataset,
    load_multiple_gift_eval_datasets,
    create_time_series_train_test_split,
    prepare_time_series_for_visualization,
    validate_gift_eval_environment,
    get_available_datasets as get_available_time_series_datasets,
    get_dataset_properties as get_time_series_dataset_properties
)

__all__ = [
    "load_dataset",
    "load_datasets",
    "analyze_dataset",
    "create_llm_dataset",
    "get_tabpfn_embeddings",
    "prepare_tabpfn_embeddings_for_prefix",
    "list_available_datasets",
    "get_dataset_info",
    "is_csv_dataset",
    "find_csv_file",
    "load_csv_dataset",
    "load_dataset_with_metadata",
    "find_csv_with_fallbacks",
    "load_datasets_for_evaluation",
    "preprocess_datasets_for_evaluation",
    "validate_dataset_for_evaluation",
    "preprocess_features",
    "process_tabular_dataset_for_training",
    "compute_frequency_distribution",
    "compute_label_frequency_mapping",
    "apply_label_mapping",
    "compute_baseline_probabilities",
    # Time series functionality
    "TimeSeriesDataset",
    "load_gift_eval_dataset",
    "load_multiple_gift_eval_datasets",
    "create_time_series_train_test_split",
    "prepare_time_series_for_visualization",
    "validate_gift_eval_environment",
    "get_available_time_series_datasets",
    "get_time_series_dataset_properties"
]