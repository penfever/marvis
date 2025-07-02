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
    "compute_baseline_probabilities"
]