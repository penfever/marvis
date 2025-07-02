"""
Process one sample for VLM prediction.

This module contains the process_one_sample function that was extracted from
MarvisTsneClassifier to handle the prediction of a single test sample.
"""

import io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from marvis.utils.vlm_prompting import (
    create_classification_prompt, 
    create_regression_prompt,
    parse_vlm_response, 
    create_vlm_conversation
)
from marvis.utils.model_loader import GenerationConfig
from marvis.viz.context.layouts import LayoutStrategy


def _create_multi_viz_info(context_composer, modality):
    """Create multi-visualization info for prompts."""
    multi_viz_info = []
    for viz in context_composer.visualizations:
        multi_viz_info.append({
            'method': viz.method_name,
            'description': f"{viz.method_name} visualization of {modality} data"
        })
    return multi_viz_info


def _get_visible_classes_from_metadata(metadata):
    """Extract visible classes from visualization metadata."""
    return metadata.get('visible_classes', [])


def _create_enhanced_legend(legend_text, classifier_instance):
    """Create enhanced legend with semantic axes if enabled."""
    enhanced_legend = legend_text
    if classifier_instance.semantic_axes and classifier_instance.train_embeddings is not None:
        semantic_axes_legend = classifier_instance._get_semantic_axes_legend(
            classifier_instance.train_embeddings, 
            classifier_instance.train_tsne, 
            classifier_instance.y_train_sample,
            feature_names=getattr(classifier_instance, 'feature_names', None)
        )
        if semantic_axes_legend:
            enhanced_legend = f"{legend_text}\n\n{semantic_axes_legend}"
    return enhanced_legend


def _generate_vlm_response(classifier_instance, image, prompt):
    """Generate VLM response with consistent configuration."""
    conversation = create_vlm_conversation(image, prompt)
    
    gen_config = GenerationConfig(
        max_new_tokens=16384,
        temperature=0.1,
        do_sample=True,
        enable_thinking=classifier_instance.enable_thinking and classifier_instance.is_api_model,
        thinking_summary=False
    )
    
    return classifier_instance.vlm_wrapper.generate_from_conversation(conversation, gen_config)


def _parse_prediction(response, classifier_instance, all_classes):
    """Parse VLM response into prediction using complete class list."""
    if classifier_instance.task_type == 'regression':
        return parse_vlm_response(
            response, 
            unique_classes=None, 
            logger_instance=classifier_instance.logger, 
            use_semantic_names=False,
            task_type='regression',
            target_stats=classifier_instance.target_stats
        )
    else:
        # Convert all_classes to the format expected by the parser
        if classifier_instance.use_semantic_names:
            all_class_names = [classifier_instance.class_to_semantic.get(cls, str(cls)) for cls in all_classes]
        else:
            all_class_names = [str(cls) for cls in all_classes]
        
        prediction = parse_vlm_response(
            response, 
            unique_classes=all_class_names,
            logger_instance=classifier_instance.logger, 
            use_semantic_names=classifier_instance.use_semantic_names,
            task_type='classification',
            color_to_class_map=getattr(classifier_instance, 'color_to_class_map', None)
        )
        
        # Map back to numeric label if needed
        if prediction in all_class_names and classifier_instance.use_semantic_names:
            semantic_to_numeric = {
                name: cls for cls, name in classifier_instance.class_to_semantic.items() 
                if cls in all_classes
            }
            prediction = semantic_to_numeric.get(prediction, prediction)
        
        return prediction


def _create_multi_visualization(classifier_instance, i, save_outputs, visualization_save_cadence):
    """Create multi-visualization image."""
    highlight_indices = None  # No training points to highlight
    
    # Create composed visualization
    composed_image = classifier_instance.context_composer.compose_layout(
        highlight_indices=highlight_indices,
        highlight_test_indices=[i],  # Highlight the current test point
        layout_strategy=LayoutStrategy[classifier_instance.layout_strategy.upper()],
        class_names=classifier_instance.class_names,
        use_semantic_names=classifier_instance.use_semantic_names
    )
    
    # Save multi-visualization if requested (respecting cadence)
    if save_outputs and classifier_instance.temp_dir and (i % visualization_save_cadence == 0):
        viz_filename = f"multi_visualization_test_{i:03d}.png"
        viz_path = os.path.join(classifier_instance.temp_dir, viz_filename)
        composed_image.save(viz_path)
    
    legend_text = f"Multi-visualization analysis ({len(classifier_instance.visualization_methods)} methods)"
    return composed_image, legend_text


def _create_single_visualization(classifier_instance, i, viz_methods, viewing_angles, save_outputs, visualization_save_cadence):
    """Create single visualization image."""
    # Create visualization highlighting current test point based on task type
    if classifier_instance.task_type == 'regression':
        # Use regression visualization methods
        if classifier_instance.use_knn_connections:
            # Create visualization with KNN connections for regression
            if classifier_instance.use_3d:
                fig, legend_text, metadata = viz_methods['create_regression_tsne_3d_plot_with_knn'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    classifier_instance.train_embeddings, classifier_instance.test_embeddings,
                    highlight_test_idx=i,
                    knn_k=classifier_instance.knn_k,
                    figsize=(12, 9),
                    zoom_factor=classifier_instance.zoom_factor
                )
            else:
                fig, legend_text, metadata = viz_methods['create_regression_tsne_plot_with_knn'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    classifier_instance.train_embeddings, classifier_instance.test_embeddings,
                    highlight_test_idx=i,
                    knn_k=classifier_instance.knn_k,
                    figsize=(10, 8),
                    zoom_factor=classifier_instance.zoom_factor
                )
        else:
            # Create standard regression visualization
            if classifier_instance.use_3d:
                fig, legend_text, metadata = viz_methods['create_combined_regression_tsne_3d_plot'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    highlight_test_idx=i,
                    figsize=(12, 9),
                    zoom_factor=classifier_instance.zoom_factor
                )
            else:
                fig, legend_text, metadata = viz_methods['create_combined_regression_tsne_plot'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    highlight_test_idx=i,
                    figsize=(8, 6),
                    zoom_factor=classifier_instance.zoom_factor
                )
    else:
        # Use classification visualization methods
        if classifier_instance.use_knn_connections:
            # Create visualization with KNN connections
            if classifier_instance.use_3d:
                fig, legend_text, metadata = viz_methods['create_tsne_3d_plot_with_knn'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    classifier_instance.train_embeddings, classifier_instance.test_embeddings,
                    highlight_test_idx=i,
                    knn_k=classifier_instance.knn_k,
                    figsize=(12, 9),
                    zoom_factor=classifier_instance.zoom_factor,
                    class_names=classifier_instance.class_names,
                    use_semantic_names=classifier_instance.use_semantic_names,
                    semantic_axes_labels=getattr(classifier_instance, 'semantic_axes_labels', None),
                    all_classes=classifier_instance.unique_classes
                )
            else:
                fig, legend_text, metadata = viz_methods['create_tsne_plot_with_knn'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    classifier_instance.train_embeddings, classifier_instance.test_embeddings,
                    highlight_test_idx=i,
                    knn_k=classifier_instance.knn_k,
                    figsize=(10, 8),
                    zoom_factor=classifier_instance.zoom_factor,
                    class_names=classifier_instance.class_names,
                    use_semantic_names=classifier_instance.use_semantic_names,
                    all_classes=classifier_instance.unique_classes
                )
        else:
            # Create standard visualization
            if classifier_instance.use_3d:
                fig, legend_text, metadata = viz_methods['create_combined_tsne_3d_plot'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    highlight_test_idx=i,
                    figsize=(12, 9),
                    zoom_factor=classifier_instance.zoom_factor,
                    class_names=classifier_instance.class_names,
                    use_semantic_names=classifier_instance.use_semantic_names,
                    semantic_axes_labels=getattr(classifier_instance, 'semantic_axes_labels', None),
                    all_classes=classifier_instance.unique_classes
                )
            else:
                fig, legend_text, metadata = viz_methods['create_combined_tsne_plot'](
                    classifier_instance.train_tsne, classifier_instance.test_tsne, classifier_instance.y_train_sample,
                    highlight_test_idx=i,
                    figsize=(8, 6),
                    zoom_factor=classifier_instance.zoom_factor,
                    class_names=classifier_instance.class_names,
                    use_semantic_names=classifier_instance.use_semantic_names,
                    semantic_axes_labels=getattr(classifier_instance, 'semantic_axes_labels', None),
                    all_classes=classifier_instance.unique_classes
                )

    # Convert plot to image
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=classifier_instance.image_dpi, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    image = Image.open(img_buffer)
    
    # Save visualization if requested (respecting cadence)
    if save_outputs and classifier_instance.temp_dir and (i % visualization_save_cadence == 0):
        viz_filename = f"visualization_test_{i:03d}.png"
        viz_path = os.path.join(classifier_instance.temp_dir, viz_filename)
        fig.savefig(viz_path, dpi=classifier_instance.image_dpi, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    return image, legend_text, metadata


def _process_image(classifier_instance, image):
    """Process image: convert to RGB and resize if needed."""
    # Convert to RGB if needed
    if classifier_instance.force_rgb_mode and image.mode != 'RGB':
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
            image = rgb_image
        else:
            image = image.convert('RGB')
    
    # Resize if needed
    if image.width > classifier_instance.max_vlm_image_size or image.height > classifier_instance.max_vlm_image_size:
        ratio = min(classifier_instance.max_vlm_image_size / image.width, classifier_instance.max_vlm_image_size / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def _save_outputs(save_outputs, temp_dir, i, visualization_save_cadence, prompt, response):
    """Save prompt and response outputs if requested."""
    if save_outputs and temp_dir:
        # Save prompt only for the first test sample (prompts are generally identical)
        if i == 0:
            prompt_filename = f"prompt_test_{i:03d}.txt"
            prompt_path = os.path.join(temp_dir, prompt_filename)
            with open(prompt_path, 'w') as f:
                f.write(prompt)
        
        # Save response following visualization cadence pattern
        if i % visualization_save_cadence == 0:
            response_filename = f"response_test_{i:03d}.txt"
            response_path = os.path.join(temp_dir, response_filename)
            with open(response_path, 'w') as f:
                f.write(response)


def _store_prediction_details(return_detailed, y_test, prediction_details, i, response, prediction, test_tsne, image, visible_classes_list):
    """Store detailed prediction information if requested."""
    if return_detailed and y_test is not None and prediction_details is not None:
        true_label = y_test[i] if hasattr(y_test, '__getitem__') else y_test.iloc[i]
        # Get tsne coordinates safely
        tsne_coords = (test_tsne[i].tolist() if test_tsne is not None and i < len(test_tsne) 
                      else [0.0, 0.0])  # Default coordinates if not available
        prediction_details.append({
            'test_point_idx': i,
            'vlm_response': response,
            'parsed_prediction': prediction,
            'true_label': true_label,
            'tsne_coords': tsne_coords,
            'image_size': f"{image.width}x{image.height}",
            'visible_classes': visible_classes_list
        })


def process_one_sample(
    # Core classifier attributes
    classifier_instance,
    sample_index,
    
    # Visualization parameters
    viz_methods,
    viewing_angles,
    save_outputs=False,
    visualization_save_cadence=10,
    
    # Return parameters
    return_detailed=False,
    y_test=None,
    
    # Additional context
    prediction_details=None,
    
    # Class information
    all_classes=None
):
    """
    Process a single test sample for VLM prediction.
    
    Args:
        classifier_instance: The MarvisTsneClassifier instance containing all necessary attributes
        sample_index: Index of the test sample to process
        viz_methods: Dictionary of visualization methods
        viewing_angles: Viewing angles for 3D plots
        save_outputs: Whether to save visualization outputs
        visualization_save_cadence: How often to save visualizations
        return_detailed: Whether to return detailed prediction information
        y_test: True test labels (optional)
        prediction_details: List to append prediction details to (optional)
        all_classes: Complete list of classes from dataset (required for classification)
        
    Returns:
        tuple: (prediction, response) where prediction is the parsed prediction
               and response is the raw VLM response
    """
    i = sample_index
    
    # Ensure all_classes is provided for classification tasks
    if classifier_instance.task_type == 'classification' and all_classes is None:
        all_classes = classifier_instance.unique_classes
    
    # Create visualization and get image
    if classifier_instance.enable_multi_viz and classifier_instance.context_composer is not None:
        # Multi-visualization approach
        image, legend_text = _create_multi_visualization(
            classifier_instance, i, save_outputs, visualization_save_cadence
        )
        
        # Get visible classes from visualization metadata
        visible_classes = []
        for viz in classifier_instance.context_composer.visualizations:
            if hasattr(viz, 'metadata'):
                viz_visible = _get_visible_classes_from_metadata(viz.metadata)
                visible_classes.extend(viz_visible)
        visible_classes = list(set(visible_classes))  # Remove duplicates
        
        # Explicit error check: visible_classes should never be empty for classification tasks
        if classifier_instance.task_type == 'classification' and not visible_classes:
            viz_methods = [viz.method_name for viz in classifier_instance.context_composer.visualizations]
            viz_metadata = {viz.method_name: getattr(viz, 'metadata', {}) for viz in classifier_instance.context_composer.visualizations}
            raise ValueError(
                f"Empty visible_classes detected in multi-visualization context. "
                f"This indicates a bug in visualization metadata generation. "
                f"Visualization methods: {viz_methods}, "
                f"All classes: {all_classes}, "
                f"Visualization metadata: {viz_metadata}"
            )
        
        # Create prompt for multi-viz
        if classifier_instance.task_type == 'regression':
            multi_viz_info = _create_multi_viz_info(classifier_instance.context_composer, classifier_instance.modality)
            
            prompt = create_regression_prompt(
                target_stats=classifier_instance.target_stats,
                modality=classifier_instance.modality,
                dataset_description=f"{classifier_instance.modality.title()} data with highlighted test point",
                multi_viz_info=multi_viz_info,
                dataset_metadata=classifier_instance._get_metadata_for_prompt()
            )
        else:
            multi_viz_info = _create_multi_viz_info(classifier_instance.context_composer, classifier_instance.modality)
            
            class_count = len(all_classes) if all_classes is not None else 0
            prompt = create_classification_prompt(
                class_names=visible_classes,
                modality=classifier_instance.modality,
                dataset_description=f"{classifier_instance.modality.title()} data with {class_count} classes and highlighted test point",
                use_semantic_names=classifier_instance.use_semantic_names,
                multi_viz_info=multi_viz_info,
                dataset_metadata=classifier_instance._get_metadata_for_prompt()
            )
    else:
        # Single visualization approach
        image, legend_text, metadata = _create_single_visualization(
            classifier_instance, i, viz_methods, viewing_angles, save_outputs, visualization_save_cadence
        )
        
        # Get visible classes from visualization metadata
        visible_classes = _get_visible_classes_from_metadata(metadata)
        
        # Explicit error check: visible_classes should never be empty for classification tasks
        if classifier_instance.task_type == 'classification' and not visible_classes:
            raise ValueError(
                f"Empty visible_classes detected in single visualization context. "
                f"This indicates a bug in visualization metadata generation. "
                f"All classes: {all_classes}, "
                f"Visualization metadata: {metadata}"
            )
        
        # Create prompt for single viz
        if classifier_instance.task_type == 'regression':
            enhanced_legend = _create_enhanced_legend(legend_text, classifier_instance)
            
            prompt = create_regression_prompt(
                target_stats=classifier_instance.target_stats,
                modality=classifier_instance.modality,
                use_knn=classifier_instance.use_knn_connections,
                use_3d=classifier_instance.use_3d,
                nn_k=classifier_instance.knn_k if classifier_instance.use_knn_connections else None,
                legend_text=enhanced_legend,
                dataset_description=f"{classifier_instance.modality.title()} data embedded using appropriate features",
                dataset_metadata=classifier_instance._get_metadata_for_prompt()
            )
        else:
            enhanced_legend = _create_enhanced_legend(legend_text, classifier_instance)
            
            prompt = create_classification_prompt(
                class_names=visible_classes,
                modality=classifier_instance.modality,
                use_knn=classifier_instance.use_knn_connections,
                use_3d=classifier_instance.use_3d,
                nn_k=classifier_instance.knn_k if classifier_instance.use_knn_connections else None,
                legend_text=enhanced_legend,
                dataset_description=f"{classifier_instance.modality.title()} data embedded using appropriate features",
                use_semantic_names=classifier_instance.use_semantic_names,
                dataset_metadata=classifier_instance._get_metadata_for_prompt()
            )
    
    # Process image and generate VLM response
    image = _process_image(classifier_instance, image)
    response = _generate_vlm_response(classifier_instance, image, prompt)
    prediction = _parse_prediction(response, classifier_instance, all_classes)
    
    # Save outputs if requested
    _save_outputs(save_outputs, classifier_instance.temp_dir, i, visualization_save_cadence, prompt, response)
    
    # Store details if requested
    _store_prediction_details(
        return_detailed, y_test, prediction_details, i, response, prediction, 
        classifier_instance.test_tsne, image, visible_classes
    )
    
    return prediction, response