"""
Metadata loading and management for MARVIS datasets.

This module provides utilities for loading and managing dataset metadata
to enhance VLM prompts with rich contextual information.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ColumnMetadata:
    """Metadata for a dataset column/feature."""
    name: str
    semantic_description: str
    data_type: str


@dataclass
class TargetClassMetadata:
    """Metadata for a target class."""
    name: str
    meaning: str


@dataclass
class DatasetMetadata:
    """Complete metadata for a dataset."""
    dataset_name: str
    description: str
    columns: List[ColumnMetadata]
    target_classes: List[TargetClassMetadata]
    dataset_history: Optional[str] = None
    inference_notes: Optional[str] = None
    original_source: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create DatasetMetadata from dictionary."""
        # Convert columns
        columns = []
        if 'columns' in data:
            for col_data in data['columns']:
                columns.append(ColumnMetadata(
                    name=col_data['name'],
                    semantic_description=col_data['semantic_description'],
                    data_type=col_data['data_type']
                ))
        
        # Convert target classes
        target_classes = []
        if 'target_classes' in data:
            for class_data in data['target_classes']:
                target_classes.append(TargetClassMetadata(
                    name=class_data['name'],
                    meaning=class_data['meaning']
                ))
        
        return cls(
            dataset_name=data.get('dataset_name', ''),
            description=data.get('description', ''),
            columns=columns,
            target_classes=target_classes,
            dataset_history=data.get('dataset_history'),
            inference_notes=data.get('inference_notes'),
            original_source=data.get('original_source')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetadataLoader:
    """Loader for dataset metadata with caching and automatic detection."""
    
    def __init__(self, metadata_base_dir: Optional[str] = None):
        """
        Initialize metadata loader.
        
        Args:
            metadata_base_dir: Base directory for metadata files. If None, uses default.
        """
        if metadata_base_dir is None:
            # Default to data directory relative to project root for recursive search
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from marvis/utils/ to project root
            metadata_base_dir = project_root / "data"
        
        self.metadata_base_dir = Path(metadata_base_dir)
        self._cache = {}  # Cache loaded metadata
        
        logger.info(f"MetadataLoader initialized with base directory: {self.metadata_base_dir}")
    
    def detect_metadata_file(self, dataset_id: Union[str, int]) -> Optional[Path]:
        """
        Detect metadata file for a dataset ID using recursive search.
        
        Args:
            dataset_id: Dataset identifier (e.g., task ID, dataset name)
            
        Returns:
            Path to metadata file if found, None otherwise
        """
        # Convert to string
        dataset_str = str(dataset_id)
        
        # First try direct patterns in base directory for backward compatibility
        patterns = [
            f"{dataset_str}.json",           # Direct match
            f"{dataset_str.lower()}.json",   # Lowercase
            f"task_{dataset_str}.json",      # With task prefix
            f"dataset_{dataset_str}.json"    # With dataset prefix
        ]
        
        for pattern in patterns:
            candidate_path = self.metadata_base_dir / pattern
            if candidate_path.exists():
                logger.info(f"Found metadata file for dataset '{dataset_id}': {candidate_path}")
                return candidate_path
        
        # If not found in base directory, do recursive search
        if self.metadata_base_dir.exists():
            logger.debug(f"Performing recursive search for dataset '{dataset_id}' in {self.metadata_base_dir}")
            
            # Search for JSON files recursively
            for json_file in self.metadata_base_dir.rglob("*.json"):
                filename = json_file.name
                stem = json_file.stem  # filename without extension
                
                # Check if filename matches the dataset_id (prioritize exact matches)
                if (dataset_str == stem or 
                    dataset_str.lower() == stem.lower() or
                    f"task_{dataset_str}" == stem or
                    f"dataset_{dataset_str}" == stem):
                    logger.info(f"Found metadata file for dataset '{dataset_id}': {json_file}")
                    return json_file
        
        logger.debug(f"No metadata file found for dataset '{dataset_id}' in {self.metadata_base_dir}")
        return None
    
    def load_metadata(
        self, 
        dataset_id: Union[str, int, Path, Dict[str, Any]], 
        use_cache: bool = True
    ) -> Optional[DatasetMetadata]:
        """
        Load dataset metadata from various sources.
        
        Args:
            dataset_id: Can be:
                - String/int: Dataset ID to auto-detect file
                - Path: Direct path to metadata file
                - Dict: Metadata dictionary to convert
            use_cache: Whether to use cached results
            
        Returns:
            DatasetMetadata object if successful, None otherwise
        """
        # Handle different input types
        if isinstance(dataset_id, dict):
            # Direct dictionary input
            try:
                return DatasetMetadata.from_dict(dataset_id)
            except Exception as e:
                logger.error(f"Failed to parse metadata dictionary: {e}")
                return None
        
        elif isinstance(dataset_id, Path):
            # Direct path input
            metadata_path = dataset_id
            cache_key = str(metadata_path)
        
        else:
            # String/int dataset ID - auto-detect
            dataset_str = str(dataset_id)
            cache_key = f"auto:{dataset_str}"
            
            # Check cache first
            if use_cache and cache_key in self._cache:
                logger.debug(f"Using cached metadata for dataset '{dataset_id}'")
                return self._cache[cache_key]
            
            # Auto-detect metadata file
            metadata_path = self.detect_metadata_file(dataset_id)
            if metadata_path is None:
                return None
        
        # Check cache for file path
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load from file
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            metadata = DatasetMetadata.from_dict(metadata_dict)
            
            # Cache the result
            if use_cache:
                self._cache[cache_key] = metadata
            
            logger.info(f"Successfully loaded metadata for dataset from {metadata_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def get_available_datasets(self) -> List[str]:
        """
        Get list of datasets with available metadata (recursive search).
        
        Returns:
            List of dataset IDs that have metadata files
        """
        if not self.metadata_base_dir.exists():
            return []
        
        datasets = []
        for file_path in self.metadata_base_dir.rglob("*.json"):
            # Extract dataset ID from filename (remove .json extension)
            dataset_id = file_path.stem
            
            # Remove common prefixes to get clean dataset ID
            if dataset_id.startswith("task_"):
                dataset_id = dataset_id[5:]
            elif dataset_id.startswith("dataset_"):
                dataset_id = dataset_id[8:]
            
            datasets.append(dataset_id)
        
        return sorted(list(set(datasets)))  # Remove duplicates and sort
    
    def clear_cache(self):
        """Clear the metadata cache."""
        self._cache.clear()
        logger.info("Metadata cache cleared")


# Global metadata loader instance
_global_loader = None


def get_metadata_loader(metadata_base_dir: Optional[str] = None) -> MetadataLoader:
    """
    Get the global metadata loader instance.
    
    Args:
        metadata_base_dir: Base directory for metadata files
        
    Returns:
        MetadataLoader instance
    """
    global _global_loader
    
    if _global_loader is None or metadata_base_dir is not None:
        _global_loader = MetadataLoader(metadata_base_dir)
    
    return _global_loader


def load_dataset_metadata(
    dataset_id: Union[str, int, Path, Dict[str, Any]],
    metadata_base_dir: Optional[str] = None
) -> Optional[DatasetMetadata]:
    """
    Convenience function to load dataset metadata.
    
    Args:
        dataset_id: Dataset identifier or metadata source
        metadata_base_dir: Base directory for metadata files
        
    Returns:
        DatasetMetadata object if successful, None otherwise
    """
    loader = get_metadata_loader(metadata_base_dir)
    return loader.load_metadata(dataset_id)


def create_metadata_summary(metadata: DatasetMetadata, max_columns: int = 10) -> str:
    """
    Create a concise summary of dataset metadata for use in prompts.
    
    Args:
        metadata: DatasetMetadata object
        max_columns: Maximum number of columns to include in summary
        
    Returns:
        Formatted metadata summary string
    """
    summary = []
    
    # Dataset name and description
    if metadata.dataset_name:
        summary.append(f"**Dataset**: {metadata.dataset_name}")
    
    if metadata.description:
        # Truncate very long descriptions
        desc = metadata.description[:300] + "..." if len(metadata.description) > 300 else metadata.description
        summary.append(f"**Description**: {desc}")
    
    # Target classes
    if metadata.target_classes:
        summary.append("**Target Classes**:")
        for target_class in metadata.target_classes:
            meaning = target_class.meaning[:100] + "..." if len(target_class.meaning) > 100 else target_class.meaning
            summary.append(f"  • {target_class.name}: {meaning}")
    
    # Column information (limited)
    if metadata.columns:
        num_columns = len(metadata.columns)
        summary.append(f"**Features** ({num_columns} total):")
        
        # Show first max_columns columns
        columns_to_show = metadata.columns[:max_columns]
        for col in columns_to_show:
            desc = col.semantic_description[:80] + "..." if len(col.semantic_description) > 80 else col.semantic_description
            summary.append(f"  • {col.name}: {desc}")
        
        if num_columns > max_columns:
            summary.append(f"  • ... and {num_columns - max_columns} more features")
    
    # Additional context
    if metadata.inference_notes:
        notes = metadata.inference_notes[:200] + "..." if len(metadata.inference_notes) > 200 else metadata.inference_notes
        summary.append(f"**Inference Notes**: {notes}")
    
    return "\n".join(summary)


def detect_dataset_id_from_path(file_path: Union[str, Path]) -> Optional[str]:
    """
    Try to extract a dataset ID from a file path.
    
    Args:
        file_path: Path to a dataset file
        
    Returns:
        Detected dataset ID if possible, None otherwise
    """
    path = Path(file_path)
    
    # Try to extract from filename
    filename = path.stem  # Remove extension
    
    # Common patterns for dataset files
    patterns = [
        r'task_(\d+)',           # task_123
        r'dataset_(\w+)',        # dataset_adult
        r'(\d+)_',               # 123_something
        r'([^_]+)_data',         # adult_data
        r'([^_]+)_train',        # adult_train
        r'([^_]+)_test'          # adult_test
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # If no pattern matches, try the whole filename
    if filename and not filename.startswith('.'):
        return filename
    
    return None