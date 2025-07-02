"""
Unified VLM prompting utilities for MARVIS.

This module provides consistent prompting strategies across different modalities
(tabular, audio, image) for Vision Language Model classification tasks.
"""

import logging
import re
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def create_metadata_summary(metadata: 'DatasetMetadata') -> str:
    """
    Create a concise metadata summary for VLM prompts.
    
    Args:
        metadata: DatasetMetadata object with rich dataset information
        
    Returns:
        Formatted metadata summary for inclusion in prompts
    """
    if not metadata:
        return ""
        
    summary_parts = []
    
    # Dataset description
    if metadata.description:
        summary_parts.append(f"Dataset: {metadata.description}")
    
    # Key features (top 3-5 most informative)
    if metadata.columns:
        feature_descriptions = []
        for col in metadata.columns[:5]:  # Limit to avoid prompt bloat
            if col.semantic_description and len(col.semantic_description) < 80:
                feature_descriptions.append(f"{col.name}: {col.semantic_description}")
            else:
                feature_descriptions.append(col.name)
        
        if feature_descriptions:
            summary_parts.append(f"Key features: {'; '.join(feature_descriptions)}")
    
    # Target classes with meanings
    if metadata.target_classes:
        class_meanings = []
        for target_class in metadata.target_classes[:10]:  # Limit to avoid bloat
            if target_class.meaning and len(target_class.meaning) < 60:
                class_meanings.append(f'"{target_class.name}": {target_class.meaning}')
            else:
                class_meanings.append(f'"{target_class.name}"')
        
        if class_meanings:
            summary_parts.append(f"Classes: {'; '.join(class_meanings)}")
    
    # Domain context from inference notes
    if metadata.inference_notes and len(metadata.inference_notes) < 200:
        summary_parts.append(f"Context: {metadata.inference_notes}")
    
    return " | ".join(summary_parts)


def validate_and_clean_class_names(class_names: List[str]) -> List[str]:
    """
    Validate and clean class names for semantic naming.
    
    Requirements:
    1. Unique names
    2. Only ASCII characters  
    3. Less than 30 characters per name
    4. No whitespace (replace with underscores)
    
    Args:
        class_names: List of class names to validate
        
    Returns:
        List of cleaned and validated class names
        
    Raises:
        ValueError: If validation fails
    """
    if not class_names:
        return class_names
        
    cleaned_names = []
    seen_names = set()
    
    for i, name in enumerate(class_names):
        # Convert to string if not already
        name_str = str(name)
        
        # Replace whitespace with underscores and remove/replace special characters
        cleaned_name = re.sub(r'\s+', '_', name_str)
        # Replace common special characters with underscores
        cleaned_name = re.sub(r'[/\\|,.;:!@#$%^&*()+=\[\]{}"`~<>?]', '_', cleaned_name)
        # Remove multiple consecutive underscores
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # Remove leading/trailing underscores
        cleaned_name = cleaned_name.strip('_')
        
        # Check ASCII only
        if not cleaned_name.isascii():
            raise ValueError(f"Class name at index {i} contains non-ASCII characters: '{name_str}' -> '{cleaned_name}'")
        
        # Check length and truncate if necessary
        if len(cleaned_name) > 30:
            original_name = cleaned_name
            cleaned_name = cleaned_name[:27] + "..."
            logger.warning(f"Class name at index {i} too long ({len(original_name)} chars), truncated: '{original_name}' -> '{cleaned_name}'")
        
        # Ensure not empty after cleaning
        if not cleaned_name or cleaned_name == '_':
            cleaned_name = f"class_{i}"
            logger.warning(f"Empty class name at index {i}, using fallback: '{cleaned_name}'")
        
        # Check uniqueness
        if cleaned_name in seen_names:
            # Make unique by appending counter, ensuring we stay under 30 chars
            original_cleaned = cleaned_name
            counter = 1
            while cleaned_name in seen_names:
                suffix = f"_{counter}"
                if len(original_cleaned) + len(suffix) > 30:
                    # Truncate base name to fit suffix
                    base_name = original_cleaned[:30 - len(suffix)]
                    cleaned_name = f"{base_name}{suffix}"
                else:
                    cleaned_name = f"{original_cleaned}{suffix}"
                counter += 1
            logger.warning(f"Duplicate class name '{original_cleaned}' at index {i}, using '{cleaned_name}'")
        
        seen_names.add(cleaned_name)
        cleaned_names.append(cleaned_name)
    
    return cleaned_names


def create_classification_prompt(
    class_names: List[str],
    modality: str = "tabular",
    use_knn: bool = False,
    use_3d: bool = False,
    nn_k: Optional[int] = None,
    legend_text: Optional[str] = None,
    include_spectrogram: bool = False,
    dataset_description: Optional[str] = None,
    use_semantic_names: bool = False,
    multi_viz_info: Optional[List[Dict[str, Any]]] = None,
    dataset_metadata: Optional[str] = None
) -> str:
    """
    Create a classification prompt for VLM based on modality and visualization type.
    
    Args:
        class_names: List of class names/labels
        modality: Type of data ("tabular", "audio", "image")
        use_knn: Whether KNN connections are shown
        use_3d: Whether 3D visualization is used
        nn_k: Number of nearest neighbors (if use_knn=True)
        legend_text: Legend text from the visualization
        include_spectrogram: Whether spectrogram is included (for audio)
        dataset_description: Optional description of the dataset/task
        use_semantic_names: Whether to use semantic class names in prompts (default: False uses "Class X")
        multi_viz_info: Optional list of visualization method information for multi-viz prompts
        dataset_metadata: Optional structured metadata summary for domain context
        
    Returns:
        Formatted prompt string
    """
    # Format class list consistently - class names are already validated at source
    if use_semantic_names:
        # Use semantic class names when use_semantic_names=True
        class_list_str = ", ".join([f'"{name}"' for name in class_names])
        class_format_example = f'"{class_names[0]}", "{class_names[1]}", etc.' if len(class_names) >= 2 else f'"{class_names[0]}"'
    else:
        # Default: Always use "Class_X" format for consistency with legends
        from .class_name_utils import normalize_class_names_to_class_num
        fallback_names = normalize_class_names_to_class_num(len(class_names))
        class_list_str = ", ".join([f'"{name}"' for name in fallback_names])
        class_format_example = '"Class_0", "Class_1", "Class_2"'
    
    # Create modality-specific description
    if modality == "audio":
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of audio classification data, you can see:

1. Colored points representing training audio samples, where each color corresponds to a different class
2. {'Gray square points representing test audio samples' if not use_knn else 'Test points (if any) shown as gray squares'}
3. One red star point which is the query audio sample I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and nn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {nn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"
            
        if include_spectrogram:
            data_description += f"\n{7 if use_knn and use_3d else 6 if use_knn or use_3d else 5}. Audio spectrogram of the query sample shown below the t-SNE plot"
            
    elif modality == "tabular":
        if multi_viz_info:
            # Multi-visualization description
            viz_names = [viz['method'].upper() for viz in multi_viz_info]
            data_description = f"""Looking at these {len(multi_viz_info)} different visualizations ({', '.join(viz_names)}) of the same tabular classification data:

Each visualization shows:
1. Colored points representing training data, where each color corresponds to a different class
2. Gray square points representing test data  
3. One red star point which is the query point I want you to classify

The multiple visualizations provide different perspectives on the same underlying data structure:"""
            
            # Add method-specific descriptions
            for i, viz in enumerate(multi_viz_info, 1):
                method = viz['method'].upper()
                if method == 'PCA':
                    data_description += f"\n- **{method}**: Shows linear relationships and directions of maximum variance"
                elif method == 'TSNE':
                    data_description += f"\n- **{method}**: Preserves local neighborhood structures, excellent for revealing clusters"
                elif method == 'UMAP':
                    data_description += f"\n- **{method}**: Preserves both local and global structure with clearer cluster separation"
                elif 'SPECTRAL' in method:
                    data_description += f"\n- **{method}**: Reveals manifold structure using graph-based relationships"
                elif method == 'ISOMAP':
                    data_description += f"\n- **{method}**: Preserves geodesic distances along the data manifold"
                elif 'LLE' in method or 'LOCALLY' in method:
                    data_description += f"\n- **{method}**: Reconstructs local geometry of the data manifold"
                elif method == 'MDS':
                    data_description += f"\n- **{method}**: Preserves pairwise distances between data points"
                else:
                    data_description += f"\n- **{method}**: {viz.get('description', 'Alternative perspective on data structure')}"
        else:
            # Single visualization description
            data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of tabular data, you can see:

1. Colored points representing training data, where each color corresponds to a different class
2. Gray square points representing test data  
3. One red star point which is the query point I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and nn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {nn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"
            
    else:  # image or other
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of {modality} data, you can see:

1. Colored points representing training samples, where each color corresponds to a different class
2. Gray square points representing test samples  
3. One red star point which is the query sample I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and nn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {nn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"

    # Add legend text if provided
    if legend_text:
        data_description += f"\n\n{legend_text}"

    # Add dataset description and metadata if provided
    dataset_context = ""
    if dataset_description or dataset_metadata:
        context_parts = []
        if dataset_description:
            context_parts.append(dataset_description)
        if dataset_metadata:
            context_parts.append(f"\n{dataset_metadata}")
        dataset_context = f"\n\nDataset Context: {' '.join(context_parts)}"

    # Create modality-specific important notes
    if use_knn:
        important_note = f"\nIMPORTANT: The pie chart shows the class distribution of the {nn_k} nearest neighbors found in the original {'Whisper ' if modality == 'audio' else ''}{'high-dimensional ' if modality == 'tabular' else ''}embedding space, NOT just based on the {'3D' if use_3d else '2D'} visualization space. Smaller average distances indicate higher similarity."
    else:
        important_note = ""

    # Create analysis instructions
    if multi_viz_info:
        # Multi-visualization analysis
        viz_list = ', '.join([viz['method'].upper() for viz in multi_viz_info])
        analysis_prompt = f"Based on the position of the red star (query {'audio sample' if modality == 'audio' else 'point'}) across ALL {len(multi_viz_info)} visualization methods ({viz_list}), which class should this query {'audio sample' if modality == 'audio' else 'point'} belong to?"
        
        considerations = [
            f"The spatial relationships across all {len(multi_viz_info)} visualization methods",
            "Which colored class clusters the red star is consistently closest to across multiple methods",
            "Patterns that appear in multiple visualizations (these are more reliable than method-specific patterns)",
            "How different visualization methods agree or disagree about the query point's classification"
        ]
        
        # Add method-specific considerations
        linear_methods = [v for v in multi_viz_info if v['method'].lower() in ['pca']]
        nonlinear_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'umap', 'isomap', 'lle']]
        
        if linear_methods and nonlinear_methods:
            considerations.append("Whether linear methods (PCA) and nonlinear methods show consistent or different cluster assignments")
        
        local_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'lle']]
        global_methods = [v for v in multi_viz_info if v['method'].lower() in ['umap', 'isomap', 'mds']]
        
        if local_methods and global_methods:
            considerations.append("How local structure methods (t-SNE, LLE) compare with global structure methods (UMAP, Isomap)")
            
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")
            
    elif use_knn:
        analysis_prompt = f"Based on BOTH the spatial position in the t-SNE visualization AND the explicit nearest neighbor connections, which class should this query {'audio sample' if modality == 'audio' else 'point'} belong to?"
        
        considerations = ["The spatial clustering patterns" + (" across all four 3D views" if use_3d else " in the t-SNE visualization")]
        considerations.append("Which classes the nearest neighbors (connected by red lines) belong to")
        considerations.append("The relative importance of close neighbors (thicker lines)")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")
    else:
        analysis_prompt = f"Based on the position of the red star (query {'audio sample' if modality == 'audio' else 'point'}) relative to the colored training points{' across ALL viewing angles' if use_3d else ''}, which class should this query {'audio sample' if modality == 'audio' else 'point'} belong to?"
        
        considerations = [f"The spatial relationships in {'3D space by examining all four views' if use_3d else 'the t-SNE visualization'}"]
        considerations.append("Which colored class clusters the red star is closest to or embedded within")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")

    consider_text = "\n".join([f"- {consideration}" for consideration in considerations])

    # Create response format instruction
    if multi_viz_info:
        viz_list = ', '.join([viz['method'].upper() for viz in multi_viz_info])
        analysis_type = f"multi-visualization analysis across {len(multi_viz_info)} methods ({viz_list})"
    elif use_knn:
        analysis_type = "spatial clustering AND the pie chart neighbor analysis"
    elif use_3d:
        analysis_type = "3D spatial clustering patterns you observe across the multiple views"
    else:
        analysis_type = "spatial clustering patterns you observe"
    
    spectrogram_text = " and spectrogram analysis" if include_spectrogram and modality == "audio" else ""
    response_format = f'Please respond with just the class label (e.g., {class_format_example}) followed by a brief explanation of your reasoning based on the {analysis_type}{spectrogram_text}.'

    # Combine all parts
    prompt = f"""{data_description}{dataset_context}{important_note}

{analysis_prompt}

Consider:
{consider_text}

{response_format}

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """

    return prompt


def create_regression_prompt(
    target_stats: Dict[str, Any],
    modality: str = "tabular",
    use_knn: bool = False,
    use_3d: bool = False,
    nn_k: Optional[int] = None,
    legend_text: Optional[str] = None,
    include_spectrogram: bool = False,
    dataset_description: Optional[str] = None,
    multi_viz_info: Optional[List[Dict[str, Any]]] = None,
    dataset_metadata: Optional[str] = None
) -> str:
    """
    Create a regression prompt for VLM based on modality and visualization type.
    
    Args:
        target_stats: Statistics about the target variable (min, max, mean, std, etc.)
        modality: Type of data ("tabular", "audio", "image")
        use_knn: Whether KNN connections are shown
        use_3d: Whether 3D visualization is used
        nn_k: Number of nearest neighbors (if use_knn=True)
        legend_text: Legend text from the visualization
        include_spectrogram: Whether spectrogram is included (for audio)
        dataset_description: Optional description of the dataset/task
        multi_viz_info: Optional list of visualization method information for multi-viz prompts
        
    Returns:
        Formatted prompt string
    """
    # Extract target range and statistics
    target_min = target_stats.get('min', 0.0)
    target_max = target_stats.get('max', 1.0)
    target_mean = target_stats.get('mean', (target_min + target_max) / 2)
    target_std = target_stats.get('std', (target_max - target_min) / 6)
    
    # Format target range description
    range_desc = f"between {target_min:.3g} and {target_max:.3g}"
    if target_stats.get('dtype', '').startswith('int'):
        range_desc = f"between {int(target_min)} and {int(target_max)}"
    
    stats_desc = f"(mean: {target_mean:.3g}, std: {target_std:.3g})"
    
    # Create modality-specific description
    if modality == "audio":
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of audio regression data, you can see:

1. Colored points representing training audio samples, where the color intensity/gradient corresponds to different target values
2. {'Gray square points representing test audio samples' if not use_knn else 'Test points (if any) shown as gray squares'}
3. One red star point which is the query audio sample I want you to predict a value for"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and nn_k:
            data_description += f"\n{5 if use_3d else 4}. A summary showing the {nn_k} nearest neighbors with their target values and distances"
            
        if include_spectrogram:
            data_description += f"\n{6 if use_knn and use_3d else 5 if use_knn or use_3d else 4}. Audio spectrogram of the query sample shown below the t-SNE plot"
            
    elif modality == "tabular":
        if multi_viz_info:
            # Multi-visualization description for regression
            viz_names = [viz['method'].upper() for viz in multi_viz_info]
            data_description = f"""Looking at these {len(multi_viz_info)} different visualizations ({', '.join(viz_names)}) of the same tabular regression data:

Each visualization shows:
1. Colored points representing training data, where the color intensity/gradient corresponds to different target values
2. Gray square points representing test data  
3. One red star point which is the query point I want you to predict a value for

The multiple visualizations provide different perspectives on how the target values are distributed in the data structure:"""
            
            # Add method-specific descriptions for regression
            for i, viz in enumerate(multi_viz_info, 1):
                method = viz['method'].upper()
                if method == 'PCA':
                    data_description += f"\n- **{method}**: Shows linear relationships between features and target values"
                elif method == 'TSNE':
                    data_description += f"\n- **{method}**: Reveals local patterns in target value distribution"
                elif method == 'UMAP':
                    data_description += f"\n- **{method}**: Preserves both local and global target value structure"
                elif 'SPECTRAL' in method:
                    data_description += f"\n- **{method}**: Shows manifold-based target value relationships"
                elif method == 'ISOMAP':
                    data_description += f"\n- **{method}**: Preserves geodesic relationships in target space"
                elif 'LLE' in method or 'LOCALLY' in method:
                    data_description += f"\n- **{method}**: Reconstructs local target value geometry"
                elif method == 'MDS':
                    data_description += f"\n- **{method}**: Preserves target value distances between points"
                else:
                    data_description += f"\n- **{method}**: {viz.get('description', 'Alternative perspective on target distribution')}"
        else:
            # Single visualization description for regression
            data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of tabular regression data, you can see:

1. Colored points representing training data, where the color intensity/gradient corresponds to different target values
2. Gray square points representing test data  
3. One red star point which is the query point I want you to predict a value for"""
            
            if use_3d:
                data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
                
            if use_knn and nn_k:
                data_description += f"\n{5 if use_3d else 4}. A summary showing the {nn_k} nearest neighbors with their target values and distances"
            
    else:  # image or other
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of {modality} regression data, you can see:

1. Colored points representing training samples, where the color intensity/gradient corresponds to different target values
2. Gray square points representing test samples  
3. One red star point which is the query sample I want you to predict a value for"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and nn_k:
            data_description += f"\n{5 if use_3d else 4}. A summary showing the {nn_k} nearest neighbors with their target values and distances"

    # Add legend text if provided
    if legend_text:
        data_description += f"\n\n{legend_text}"

    # Add dataset description and metadata if provided
    dataset_context = ""
    if dataset_description:
        dataset_context = f"\n\nDataset Context: {dataset_description}"

    # Create modality-specific important notes
    if use_knn:
        important_note = f"\nIMPORTANT: The neighbor analysis shows the target values of the {nn_k} nearest neighbors found in the original {'Whisper ' if modality == 'audio' else ''}{'high-dimensional ' if modality == 'tabular' else ''}embedding space, NOT just based on the {'3D' if use_3d else '2D'} visualization space. Smaller distances indicate higher similarity."
    else:
        important_note = f"\nIMPORTANT: The color gradient in the visualization represents the target values, with the colormap typically ranging from low values (cooler colors) to high values (warmer colors)."

    # Create analysis instructions
    if multi_viz_info:
        # Multi-visualization analysis for regression
        viz_list = ', '.join([viz['method'].upper() for viz in multi_viz_info])
        analysis_prompt = f"Based on the position of the red star (query {'audio sample' if modality == 'audio' else 'point'}) across ALL {len(multi_viz_info)} visualization methods ({viz_list}), what value should I predict? The target values in this dataset range {range_desc} {stats_desc}."
        
        considerations = [
            f"The spatial relationships and color patterns across all {len(multi_viz_info)} visualization methods",
            "The color intensity/gradient patterns that are consistent across multiple visualizations",
            "How different visualization methods represent the target value distribution in their respective spaces",
            "Target value trends that appear reliable across multiple methods (these are more trustworthy than method-specific patterns)"
        ]
        
        # Add method-specific considerations for regression
        linear_methods = [v for v in multi_viz_info if v['method'].lower() in ['pca']]
        nonlinear_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'umap', 'isomap', 'lle']]
        
        if linear_methods and nonlinear_methods:
            considerations.append("Whether linear methods (PCA) and nonlinear methods show similar target value predictions")
        
        local_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'lle']]
        global_methods = [v for v in multi_viz_info if v['method'].lower() in ['umap', 'isomap', 'mds']]
        
        if local_methods and global_methods:
            considerations.append("How local structure methods (t-SNE, LLE) compare with global methods (UMAP, Isomap) for target value estimation")
            
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")
            
    elif use_knn:
        analysis_prompt = f"Based on BOTH the spatial position in the t-SNE visualization AND the target values of the nearest neighbors, what value should I predict for this query {'audio sample' if modality == 'audio' else 'point'}? The target values in this dataset range {range_desc} {stats_desc}."
        
        considerations = ["The spatial clustering patterns" + (" across all four 3D views" if use_3d else " in the t-SNE visualization")]
        considerations.append("The target values of the nearest neighbors (connected by red lines)")
        considerations.append("The distance-weighted average of the nearest neighbor values")
        considerations.append("The color intensity/gradient of nearby training points")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")
    else:
        analysis_prompt = f"Based on the position of the red star (query {'audio sample' if modality == 'audio' else 'point'}) relative to the colored training points{' across ALL viewing angles' if use_3d else ''}, what value should I predict? The target values in this dataset range {range_desc} {stats_desc}."
        
        considerations = [f"The spatial relationships in {'3D space by examining all four views' if use_3d else 'the t-SNE visualization'}"]
        considerations.append("The color intensity/gradient of the nearby training points")
        considerations.append("Which areas of the colormap the red star is positioned within or closest to")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")

    consider_text = "\n".join([f"- {consideration}" for consideration in considerations])

    # Create response format instruction
    if multi_viz_info:
        viz_list = ', '.join([viz['method'].upper() for viz in multi_viz_info])
        analysis_type = f"multi-visualization analysis across {len(multi_viz_info)} methods ({viz_list})"
    elif use_knn:
        analysis_type = "spatial clustering AND the neighbor value analysis"
    elif use_3d:
        analysis_type = "3D spatial clustering and color patterns you observe across the multiple views"
    else:
        analysis_type = "spatial clustering and color gradient patterns you observe"
    
    spectrogram_text = " and spectrogram analysis" if include_spectrogram and modality == "audio" else ""
    response_format = f'Please respond with just the predicted numerical value followed by a brief explanation of your reasoning based on the {analysis_type}{spectrogram_text}.'

    # Combine all parts
    prompt = f"""{data_description}{dataset_context}{important_note}

{analysis_prompt}

Consider:
{consider_text}

{response_format}

Format your response as: "Value: [predicted_value] | Reasoning: [brief explanation]" """

    return prompt


def normalize_class_name(name: str) -> str:
    """
    Normalize a class name for fuzzy matching by converting to lowercase
    and standardizing separators.
    
    Args:
        name: Class name to normalize
        
    Returns:
        Normalized class name
    """
    if not name:
        return ""
    
    # Convert to lowercase and replace various separators with underscores
    normalized = str(name).lower()
    # Replace spaces, hyphens, dots, and multiple underscores with single underscore
    import re
    normalized = re.sub(r'[\s\-\.]+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized


def find_best_class_match(class_part: str, unique_classes: List, logger_instance: logging.Logger) -> Any:
    """
    Find the best matching class using various fuzzy matching strategies.
    
    Args:
        class_part: Extracted class part from VLM response
        unique_classes: List of valid class labels
        logger_instance: Logger for debugging
        
    Returns:
        Best matching class or None if no good match found
    """
    if not class_part or not unique_classes:
        return None
    
    class_part_clean = class_part.strip('"\'').strip()
    class_part_lower = class_part_clean.lower()
    class_part_norm = normalize_class_name(class_part_clean)
    
    logger_instance.debug(f"Finding match for: '{class_part_clean}' (normalized: '{class_part_norm}')")
    
    # Strategy 1: Exact match (case-insensitive)
    for cls in unique_classes:
        if str(cls).lower() == class_part_lower:
            logger_instance.debug(f"Exact match: '{class_part_clean}' -> {cls}")
            return cls
    
    # Strategy 2: Normalized match (handles spaces/underscores/hyphens)
    for cls in unique_classes:
        cls_norm = normalize_class_name(str(cls))
        if cls_norm == class_part_norm and cls_norm:  # Ensure not empty
            logger_instance.debug(f"Normalized match: '{class_part_clean}' -> {cls} (both normalize to '{cls_norm}')")
            return cls
    
    # Strategy 3: Partial match - class name appears at start of response part
    for cls in unique_classes:
        cls_str = str(cls).lower()
        if class_part_lower.startswith(cls_str) and len(cls_str) > 2:  # Avoid very short matches
            logger_instance.debug(f"Partial match (starts with): '{class_part_clean}' -> {cls}")
            return cls
    
    # Strategy 4: Partial match - normalized version starts with class name
    for cls in unique_classes:
        cls_norm = normalize_class_name(str(cls))
        if class_part_norm.startswith(cls_norm) and len(cls_norm) > 2:  # Avoid very short matches
            logger_instance.debug(f"Partial normalized match: '{class_part_clean}' -> {cls}")
            return cls
    
    # Strategy 5: Substring match with word boundaries (more strict)
    import re
    for cls in unique_classes:
        cls_str = str(cls).lower()
        # Create pattern that matches whole words and handles common separators
        pattern = r'\b' + re.escape(cls_str).replace(r'\ ', r'[\s_\-]*').replace(r'\-', r'[\s_\-]*') + r'\b'
        if re.search(pattern, class_part_lower):
            logger_instance.debug(f"Word boundary match: '{class_part_clean}' -> {cls}")
            return cls
    
    # Strategy 6: Flexible substring matching for complex class names
    for cls in unique_classes:
        cls_norm = normalize_class_name(str(cls))
        if cls_norm in class_part_norm and len(cls_norm) > 3:  # Avoid very short matches
            logger_instance.debug(f"Substring normalized match: '{class_part_clean}' -> {cls}")
            return cls
    
    logger_instance.debug(f"No match found for: '{class_part_clean}'")
    return None


def parse_vlm_response(response: str, unique_classes: List = None, logger_instance: Optional[logging.Logger] = None, use_semantic_names: bool = False, task_type: str = "classification", target_stats: Optional[Dict] = None, color_to_class_map: Optional[Dict] = None) -> Any:
    """
    Parse VLM response to extract the predicted class or value.
    
    Args:
        response: Raw VLM response string
        unique_classes: List of valid class labels (for classification)
        logger_instance: Logger for debugging
        use_semantic_names: Whether semantic names were used in the prompt
        task_type: "classification" or "regression"
        target_stats: Statistics about target variable (for regression)
        color_to_class_map: Optional mapping from color names to class labels (e.g., {"Blue": 0, "Color_111": 5})
        
    Returns:
        Predicted class (for classification) or numerical value (for regression)
    """
    if logger_instance is None:
        logger_instance = logger
        
    response_lower = response.lower().strip()
    logger_instance.debug(f"Parsing VLM response: '{response}' (task_type={task_type}, use_semantic_names={use_semantic_names})")
    
    # Handle regression tasks
    if task_type == "regression":
        # Try to parse structured response format first
        if "value:" in response_lower:
            try:
                # Extract text after "value:" - handle various separators
                response_parts = response.split(":", 1)
                if len(response_parts) > 1:
                    # Split on common separators like |, \n, or just take first part
                    after_value = response_parts[1]
                    # Split on | but handle cases where there's no | 
                    if "|" in after_value:
                        value_part = after_value.split("|")[0].strip()
                    else:
                        # Take everything until newline or reasoning keywords
                        import re
                        # Split on common reasoning indicators
                        reasoning_split = re.split(r'\s*(?:\||reasoning|because|since|explanation|rationale|the\s|this\s)', after_value, flags=re.IGNORECASE)
                        value_part = reasoning_split[0].strip()
                    
                    # Try to parse as float
                    try:
                        parsed_value = float(value_part.strip())
                        logger_instance.debug(f"Parsed structured value: '{value_part}' -> {parsed_value}")
                        return parsed_value
                    except ValueError:
                        pass
                        
            except Exception as e:
                logger_instance.warning(f"Error parsing structured value response: {e}")
        
        # Fallback: Use the regression prediction parser
        try:
            from .llm_evaluation_utils import parse_regression_prediction
            parsed_value = parse_regression_prediction(response, target_stats)
            logger_instance.debug(f"Fallback regression parsing: '{response}' -> {parsed_value}")
            return parsed_value
        except ImportError:
            # If import fails, use a simplified numeric extraction
            import re
            numeric_patterns = [
                r'[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?',  # General numeric pattern
                r'(\d+\.?\d*|\d*\.?\d+)',  # Simple decimal numbers
                r'[-+]?\d+',  # Integers
            ]
            
            for pattern in numeric_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    try:
                        parsed_value = float(matches[0])
                        logger_instance.debug(f"Fallback numeric extraction: '{response}' -> {parsed_value}")
                        return parsed_value
                    except ValueError:
                        continue
            
            # Final fallback for regression
            if target_stats and 'mean' in target_stats:
                logger_instance.warning(f"Could not parse value from response: '{response}'. Using target mean: {target_stats['mean']}")
                return float(target_stats['mean'])
            else:
                logger_instance.warning(f"Could not parse value from response: '{response}'. Using fallback: 0.0")
                return 0.0
    
    # Classification parsing starts here
    
    # Try to parse structured response format first
    if "class:" in response_lower:
        try:
            # Extract text after "class:" - handle various separators
            response_parts = response.split(":", 1)
            if len(response_parts) > 1:
                # Split on common separators like |, \n, or just take first part
                after_class = response_parts[1]
                # Split on | but handle cases where there's no | 
                if "|" in after_class:
                    class_part = after_class.split("|")[0].strip()
                else:
                    # Take everything until newline or reasoning keywords
                    import re
                    # Split on common reasoning indicators (more robust)
                    reasoning_split = re.split(r'\s*(?:\||reasoning|because|since|explanation|rationale|the\s|this\s)', after_class, flags=re.IGNORECASE)
                    class_part = reasoning_split[0].strip()
                
                # Remove quotes if present
                class_part = class_part.strip('"\'').strip()
                
                # First, try color name matching if color mapping is provided
                if color_to_class_map:
                    for color_name, class_label in color_to_class_map.items():
                        if class_part.lower() == color_name.lower():
                            logger_instance.debug(f"Color name match found: {class_part} -> {class_label}")
                            return class_label
                
                # Then, try direct exact matching (handles cases like "Class_562")
                for unique_class in unique_classes:
                    if class_part.strip().lower() == str(unique_class).strip().lower():
                        logger_instance.debug(f"Direct match found: {class_part} -> {unique_class}")
                        return unique_class
                
                # If using "Class X" format, extract the number (handle both "class 6" and "class_6") 
                if not use_semantic_names:
                    import re
                    # Try to extract number from various class formats
                    class_match = re.search(r'class[_\s]*(\d+)', class_part.lower())
                    if class_match:
                        try:
                            class_num = int(class_match.group(1))
                            if 0 <= class_num < len(unique_classes):
                                logger_instance.debug(f"Parsed Class {class_num} -> {unique_classes[class_num]}")
                                return unique_classes[class_num]
                        except (ValueError, IndexError):
                            pass
                
                # Try to match with available classes using improved fuzzy matching
                # (This now works for both semantic and non-semantic names)
                best_match = find_best_class_match(class_part, unique_classes, logger_instance)
                if best_match is not None:
                    return best_match
                        
        except Exception as e:
            logger_instance.warning(f"Error parsing structured response: {e}")
    
    # Fallback: Look for "Class X" pattern anywhere in response
    if not use_semantic_names:
        import re
        # Enhanced regex to handle both "class 6", "class_6", "Class 6", "Class_6" patterns
        class_patterns = [
            r'\bclass[_\s]+(\d+)\b',  # matches "class 6", "class_6", etc.
            r'\bclass(\d+)\b',        # matches "class6" (no separator)
            r'class[_\s]*:?[_\s]*(\d+)', # matches "class: 6", "class_: 6", etc.
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    class_num = int(match.group(1))
                    if 0 <= class_num < len(unique_classes):
                        logger_instance.debug(f"Found Class {class_num} pattern -> {unique_classes[class_num]}")
                        return unique_classes[class_num]
                except (ValueError, IndexError):
                    continue
    
    # Final fallback: Return appropriate default based on naming convention
    if use_semantic_names:
        # For semantic names, return the first class name
        logger_instance.warning(f"Could not parse class from response: '{response}'. Using fallback: {unique_classes[0]}")
        return unique_classes[0]
    else:
        # For "Class X" format, return index 0 (which should map to unique_classes[0])
        # But log it in the expected "Class X" format for consistency
        logger_instance.warning(f"Could not parse class from response: '{response}'. Using fallback: Class_0")
        return unique_classes[0]


def create_direct_classification_prompt(
    class_names: List[str],
    dataset_description: Optional[str] = None,
    use_semantic_names: bool = False
) -> str:
    """
    Create a direct image classification prompt for VLM (not t-SNE visualization).
    
    Args:
        class_names: List of class names/labels
        dataset_description: Optional description of the dataset/task
        use_semantic_names: Whether semantic names were used (for consistency)
        
    Returns:
        Formatted prompt string for direct image classification
    """
    class_list_str = ", ".join([f'"{name}"' for name in class_names])
    
    dataset_context = ""
    if dataset_description:
        dataset_context = f"\n\nDataset Context: {dataset_description}"
    
    prompt_text = f"""Please classify this image into one of the available categories.

Available classes: {class_list_str}{dataset_context}

Look at the image carefully and determine which class it belongs to based on the visual content you can observe.

Please respond with just the class label followed by a brief explanation of your reasoning based on what you see in the image.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """
    
    return prompt_text


def create_direct_regression_prompt(
    target_stats: Dict[str, Any],
    dataset_description: Optional[str] = None
) -> str:
    """
    Create a direct image regression prompt for VLM (not t-SNE visualization).
    
    Args:
        target_stats: Statistics about the target variable (min, max, mean, std, etc.)
        dataset_description: Optional description of the dataset/task
        
    Returns:
        Formatted prompt string for direct image regression
    """
    # Extract target range and statistics
    target_min = target_stats.get('min', 0.0)
    target_max = target_stats.get('max', 1.0)
    target_mean = target_stats.get('mean', (target_min + target_max) / 2)
    target_std = target_stats.get('std', (target_max - target_min) / 6)
    
    # Format target range description
    range_desc = f"between {target_min:.3g} and {target_max:.3g}"
    if target_stats.get('dtype', '').startswith('int'):
        range_desc = f"between {int(target_min)} and {int(target_max)}"
    
    stats_desc = f"(mean: {target_mean:.3g}, std: {target_std:.3g})"
    
    dataset_context = ""
    if dataset_description:
        dataset_context = f"\n\nDataset Context: {dataset_description}"
    
    prompt_text = f"""Please predict a numerical value for this image based on its visual content.

Target value range: {range_desc} {stats_desc}{dataset_context}

Look at the image carefully and predict what numerical value it should have based on the visual patterns, features, or characteristics you can observe.

Please respond with just the predicted numerical value followed by a brief explanation of your reasoning based on what you see in the image.

Format your response as: "Value: [predicted_value] | Reasoning: [brief explanation]" """
    
    return prompt_text


def generate_multi_viz_reasoning_guidance(
    multi_viz_info: List[Dict[str, Any]], 
    reasoning_focus: str = "comparison"
) -> str:
    """
    Generate advanced reasoning guidance for multi-visualization contexts.
    
    Args:
        multi_viz_info: List of visualization method information
        reasoning_focus: Type of reasoning to emphasize ("comparison", "consensus", "divergence")
        
    Returns:
        Formatted reasoning guidance string
    """
    if len(multi_viz_info) < 2:
        return ""
    
    guidance_parts = []
    
    # Method-specific descriptions
    guidance_parts.append("**Visualization Method Details:**")
    for i, viz in enumerate(multi_viz_info, 1):
        method = viz['method'].lower()
        desc = f"{i}. **{viz['method'].upper()}**: "
        
        if method == 'tsne':
            desc += "t-SNE preserves local neighborhood structures and is excellent for revealing clusters and local patterns."
        elif method == 'umap':
            desc += "UMAP preserves both local and global structure, often showing clearer cluster separation than t-SNE."
        elif method == 'pca':
            desc += "PCA shows linear relationships and the directions of maximum variance in the data."
        elif 'spectral' in method:
            desc += "Spectral embedding reveals the manifold structure using graph-based relationships."
        elif 'lle' in method or 'locally' in method:
            desc += "Locally Linear Embedding reconstructs the local geometry of the data manifold."
        elif method == 'isomap':
            desc += "Isomap preserves geodesic distances along the data manifold."
        elif method == 'mds':
            desc += "MDS preserves pairwise distances between data points in the reduced space."
        else:
            desc += f"Visualization using {viz['method']} method."
            
        guidance_parts.append(desc)
    
    # Cross-visualization analysis
    guidance_parts.append("\n**Cross-Visualization Analysis:**")
    
    # Method comparison analysis
    linear_methods = [v for v in multi_viz_info if v['method'].lower() in ['pca']]
    nonlinear_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'umap', 'isomap', 'lle']]
    
    if linear_methods and nonlinear_methods:
        guidance_parts.append("- Compare the linear perspective (PCA) with nonlinear methods to understand if the data has inherent nonlinear structure.")
    
    local_methods = [v for v in multi_viz_info if v['method'].lower() in ['tsne', 'lle']]
    global_methods = [v for v in multi_viz_info if v['method'].lower() in ['umap', 'isomap', 'mds']]
    
    if local_methods and global_methods:
        guidance_parts.append("- Local structure preserving methods (t-SNE, LLE) may show different cluster arrangements than global methods (UMAP, Isomap, MDS).")
    
    # General cross-method guidance
    guidance_parts.extend([
        "- Look for consistent cluster patterns across methods - clusters that appear in multiple visualizations are likely genuine data structures.",
        "- Points that appear as outliers in multiple visualizations are likely true outliers in the data."
    ])
    
    # Method-specific comparisons
    methods_present = [v['method'].lower() for v in multi_viz_info]
    method_pairs = [
        ('tsne', 'umap', "t-SNE may show tighter clusters while UMAP preserves more global structure"),
        ('pca', 'tsne', "PCA shows linear separability while t-SNE reveals nonlinear cluster structure"),
        ('isomap', 'lle', "Isomap preserves geodesic distances while LLE focuses on local linearity"),
    ]
    
    for method1, method2, insight in method_pairs:
        if method1 in methods_present and method2 in methods_present:
            guidance_parts.append(f"- {insight}.")
    
    # Reasoning focus guidance
    guidance_parts.append(f"\n**Reasoning Focus - {reasoning_focus.title()}:**")
    
    if reasoning_focus == "comparison":
        guidance_parts.extend([
            "Compare and contrast the patterns shown in each visualization:",
            "- Which methods show similar cluster structures?",
            "- Where do the methods disagree, and what might this indicate?",
            "- Are there patterns visible in some methods but not others?",
            "- How do the relative positions of clusters change across methods?"
        ])
    elif reasoning_focus == "consensus":
        guidance_parts.extend([
            "Look for patterns that are consistent across multiple visualizations:",
            "- Which clusters appear in most or all visualizations?",
            "- What data structures are reliably preserved across methods?",
            "- Which relationships between data points are method-independent?",
            "- What can you confidently conclude about the data structure?"
        ])
    elif reasoning_focus == "divergence":
        guidance_parts.extend([
            "Focus on where the visualizations show different patterns:",
            "- Which methods reveal unique perspectives on the data?",
            "- Where do visualizations disagree about cluster boundaries or outliers?",
            "- What might cause these differences (method assumptions, parameter settings)?",
            "- How can these differences inform our understanding of the data complexity?"
        ])
    
    return "\n".join(guidance_parts)


def create_comprehensive_multi_viz_prompt(
    class_names: List[str],
    multi_viz_info: List[Dict[str, Any]],
    modality: str = "tabular",
    dataset_description: Optional[str] = None,
    reasoning_focus: str = "comparison",
    data_shape: Optional[tuple] = None,
    use_semantic_names: bool = False
) -> str:
    """
    Create a comprehensive prompt for multi-visualization analysis.
    
    This function integrates the advanced prompting logic from the removed PromptGenerator
    into the unified VLM prompting system.
    
    Args:
        class_names: List of class names
        multi_viz_info: List of visualization method information
        modality: Data modality
        dataset_description: Description of the dataset
        reasoning_focus: Type of reasoning emphasis
        data_shape: Shape of the original data
        use_semantic_names: Whether to use semantic class names
        
    Returns:
        Comprehensive multi-visualization prompt
    """
    # Use the existing create_classification_prompt as the base
    base_prompt = create_classification_prompt(
        class_names=class_names,
        modality=modality,
        dataset_description=dataset_description,
        use_semantic_names=use_semantic_names,
        multi_viz_info=multi_viz_info
    )
    
    # Add advanced multi-viz reasoning guidance
    if len(multi_viz_info) >= 2:
        reasoning_guidance = generate_multi_viz_reasoning_guidance(multi_viz_info, reasoning_focus)
        
        # Insert the advanced guidance before the final instructions
        if "Format your response as:" in base_prompt:
            parts = base_prompt.split("Format your response as:")
            enhanced_prompt = parts[0] + "\n\n" + reasoning_guidance + "\n\nFormat your response as:" + parts[1]
        else:
            enhanced_prompt = base_prompt + "\n\n" + reasoning_guidance
            
        return enhanced_prompt
    
    return base_prompt


def create_vlm_conversation(image, prompt: str) -> List[Dict]:
    """
    Create a conversation structure for VLM input.
    
    Args:
        image: PIL Image object
        prompt: Text prompt string
        
    Returns:
        Conversation structure for VLM
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]