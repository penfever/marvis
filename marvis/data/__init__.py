"""
Data loading, processing, and preparation utilities.
"""

from .csv_utils import (
    find_csv_file,
    find_csv_with_fallbacks,
    is_csv_dataset,
    load_csv_dataset,
    load_dataset_with_metadata,
)
from .dataset import (
    analyze_dataset,
    create_llm_dataset,
    get_dataset_info,
    list_available_datasets,
    load_dataset,
    load_datasets,
)
from .dataset_tabular import (
    apply_label_mapping,
    compute_baseline_probabilities,
    compute_frequency_distribution,
    compute_label_frequency_mapping,
    preprocess_features,
    process_tabular_dataset_for_training,
)
from .embeddings import get_tabpfn_embeddings, prepare_tabpfn_embeddings_for_prefix
from .evaluation_utils import (
    load_datasets_for_evaluation,
    preprocess_datasets_for_evaluation,
    validate_dataset_for_evaluation,
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
]
