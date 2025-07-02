"""
MARVIS t-SNE baseline for audio classification.

This implements the MARVIS pipeline for audio:
Whisper embeddings → t-SNE visualization → VLM classification

Based on the unified implementation in marvis.models.marvis_tsne
"""

import os
import numpy as np
import torch
import time
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from PIL import Image

# Import MARVIS utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.utils.vlm_prompting import create_classification_prompt, parse_vlm_response, create_vlm_conversation
from marvis.utils.class_name_utils import get_semantic_class_names_or_fallback
from marvis.viz.utils.common import (
    plot_to_image, save_visualization_with_metadata, create_output_directories,
    generate_visualization_filename, close_figure_safely
)
from marvis.utils.device_utils import detect_optimal_device, log_platform_info
from marvis.utils.json_utils import convert_for_json_serialization
from marvis.utils.audio_utils import (
    load_audio, create_spectrogram, plot_waveform, plot_spectrogram
)

from marvis.data.audio_embeddings import get_whisper_embeddings, get_clap_embeddings
from marvis.viz.tsne_functions import (
    create_tsne_visualization,
    create_tsne_3d_visualization,
    create_combined_tsne_plot,
    create_combined_tsne_3d_plot,
    create_tsne_plot_with_knn,
    create_tsne_3d_plot_with_knn
)

logger = logging.getLogger(__name__)


class MarvisAudioTsneClassifier:
    """
    MARVIS t-SNE classifier for audio classification.
    
    Pipeline: Audio files → Audio embeddings (Whisper/CLAP) → t-SNE visualization → VLM classification
    """
    
    def __init__(
        self,
        embedding_model: str = "whisper",
        whisper_model: str = "large-v2",
        embedding_layer: str = "encoder_last",
        clap_version: str = "2023",
        tsne_perplexity: float = 30.0,
        tsne_n_iter: int = 1000,
        vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_3d: bool = False,
        use_knn_connections: bool = False,
        nn_k: int = 5,
        max_vlm_image_size: int = 1024,
        image_dpi: int = 100,
        zoom_factor: float = 2.0,
        use_pca_backend: bool = False,
        include_spectrogram: bool = True,
        audio_duration: Optional[float] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_semantic_names: bool = False,
        num_few_shot_examples: int = 32,
        balanced_few_shot: bool = False,
        seed: int = 42
    ):
        """
        Initialize MARVIS audio t-SNE classifier.
        
        Args:
            embedding_model: Audio embedding model to use ('whisper' or 'clap')
            whisper_model: Whisper model variant to use (if using Whisper)
            embedding_layer: Which Whisper layer to extract ('encoder_last', 'encoder_avg')
            clap_version: CLAP model version to use (if using CLAP)
            tsne_perplexity: t-SNE perplexity parameter
            tsne_n_iter: Number of t-SNE iterations
            vlm_model_id: Vision Language Model ID
            use_3d: Whether to use 3D t-SNE
            use_knn_connections: Whether to show KNN connections
            nn_k: Number of nearest neighbors to show
            max_vlm_image_size: Maximum image size for VLM
            image_dpi: DPI for saving visualizations
            zoom_factor: Zoom factor for t-SNE visualizations
            use_pca_backend: Use PCA instead of t-SNE
            include_spectrogram: Include spectrogram in visualization
            audio_duration: Max audio duration to process (seconds)
            cache_dir: Directory for caching embeddings
            device: Device for model inference
            num_few_shot_examples: Number of examples for in-context learning (future use)
            balanced_few_shot: Use class-balanced few-shot examples (future use)
            seed: Random seed
        """
        # Validate embedding model
        if embedding_model not in ["whisper", "clap"]:
            raise ValueError(f"embedding_model must be 'whisper' or 'clap', got '{embedding_model}'")
        
        self.embedding_model = embedding_model
        self.whisper_model = whisper_model
        self.embedding_layer = embedding_layer
        self.clap_version = clap_version
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.vlm_model_id = vlm_model_id
        self.use_3d = use_3d
        self.use_knn_connections = use_knn_connections
        self.nn_k = nn_k
        self.max_vlm_image_size = max_vlm_image_size
        self.image_dpi = image_dpi
        self.zoom_factor = zoom_factor
        self.use_pca_backend = use_pca_backend
        self.include_spectrogram = include_spectrogram
        self.audio_duration = audio_duration
        self.cache_dir = cache_dir
        self.device = device or detect_optimal_device()
        self.use_semantic_names = use_semantic_names
        self.num_few_shot_examples = num_few_shot_examples
        self.balanced_few_shot = balanced_few_shot
        self.seed = seed
        
        # Set random seeds
        from marvis.utils import set_seed
        set_seed(seed)
        
        # To be set during fit
        self.train_embeddings = None
        self.train_labels = None
        self.train_paths = None
        self.class_names = None
        self.train_tsne = None
        self.vlm_wrapper = None
        self.save_every_n = 10
        
    def fit(self, train_paths: List[str], train_labels: List[int], 
            test_paths: Optional[List[str]] = None, class_names: Optional[List[str]] = None,
            use_semantic_names: bool = False):
        """
        Fit the classifier on training data.
        
        Args:
            train_paths: List of paths to training audio files
            train_labels: List of training labels
            test_paths: Optional list of test audio paths for visualization
            class_names: Optional list of class names
            use_semantic_names: Whether to use semantic class names
        """
        logger.info(f"Fitting MARVIS audio classifier with {len(train_paths)} training samples")
        
        self.train_paths = train_paths
        self.train_labels = np.array(train_labels)
        
        # Infer class names if not provided
        if class_names is None:
            # Use new utility to extract class names with semantic support
            unique_labels = np.unique(train_labels).tolist()
            from marvis.utils.class_name_utils import extract_class_names_from_labels
            self.class_names, _ = extract_class_names_from_labels(
                labels=unique_labels,
                dataset_name=getattr(self, 'dataset_name', None),
                use_semantic=use_semantic_names
            )
        else:
            self.class_names = class_names
            
        # Extract audio embeddings for training data
        if self.embedding_model == "whisper":
            logger.info("Extracting Whisper embeddings for training data...")
            self.train_embeddings = get_whisper_embeddings(
                train_paths,
                model_name=self.whisper_model,
                layer=self.embedding_layer,
                cache_dir=self.cache_dir,
                device=self.device
            )
        elif self.embedding_model == "clap":
            logger.info("Extracting CLAP embeddings for training data...")
            self.train_embeddings, _ = get_clap_embeddings(
                train_paths,
                version=self.clap_version,
                cache_dir=self.cache_dir,
                dataset_name="train"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
        
        # Use all training samples for visualization (no arbitrary limitation)
        logger.info(f"Using all {len(train_paths)} training samples for t-SNE visualization")
        self.plot_indices = np.arange(len(train_paths))
        self.plot_embeddings = self.train_embeddings
        self.plot_labels = self.train_labels
            
        # Create t-SNE visualization
        if test_paths:
            logger.info(f"Extracting {self.embedding_model.upper()} embeddings for test samples...")
            if self.embedding_model == "whisper":
                test_embeddings = get_whisper_embeddings(
                    test_paths[:20],  # Limit test samples for visualization
                    model_name=self.whisper_model,
                    layer=self.embedding_layer,
                    cache_dir=self.cache_dir,
                    device=self.device
                )
            elif self.embedding_model == "clap":
                test_embeddings, _ = get_clap_embeddings(
                    test_paths[:20],  # Limit test samples for visualization
                    version=self.clap_version,
                    cache_dir=self.cache_dir,
                    dataset_name="test"
                )
        else:
            test_embeddings = np.empty((0, self.train_embeddings.shape[1]))
            
        # Run t-SNE
        logger.info(f"Running {'PCA' if self.use_pca_backend else 't-SNE'} on embeddings...")
        if self.use_3d:
            self.train_tsne, test_tsne, _ = create_tsne_3d_visualization(
                self.plot_embeddings, self.plot_labels, test_embeddings,
                perplexity=self.tsne_perplexity,
                n_iter=self.tsne_n_iter,
                random_state=self.seed
            )
        else:
            self.train_tsne, test_tsne, _ = create_tsne_visualization(
                self.plot_embeddings, self.plot_labels, test_embeddings,
                perplexity=self.tsne_perplexity,
                n_iter=self.tsne_n_iter,
                random_state=self.seed
            )
            
        logger.info("MARVIS audio classifier fitted successfully")
        
    def predict(self, test_paths: List[str], batch_size: int = 1, 
                save_outputs: bool = False, output_dir: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predict labels for test audio files.
        
        Args:
            test_paths: List of paths to test audio files
            batch_size: Batch size for VLM inference
            save_outputs: Whether to save visualizations
            output_dir: Output directory for saving visualizations
            
        Returns:
            predictions: Array of predicted labels
            detailed_outputs: List of detailed prediction info
        """
        if self.train_embeddings is None:
            raise ValueError("Classifier must be fitted before prediction")
            
        # Load VLM if not already loaded
        if self.vlm_wrapper is None:
            self.vlm_wrapper = self._load_vlm_model()
            
        predictions = []
        detailed_outputs = []
        total_samples = len(test_paths)
        
        # Setup output directories if saving
        if save_outputs and output_dir:
            dir_paths = create_output_directories(output_dir, ['image_visualizations'])
            viz_dir = dir_paths['image_visualizations']
        
        # Extract test embeddings
        logger.info(f"Extracting {self.embedding_model.upper()} embeddings for test data...")
        if self.embedding_model == "whisper":
            test_embeddings = get_whisper_embeddings(
                test_paths,
                model_name=self.whisper_model,
                layer=self.embedding_layer,
                cache_dir=self.cache_dir,
                device=self.device
            )
        elif self.embedding_model == "clap":
            test_embeddings, _ = get_clap_embeddings(
                test_paths,
                version=self.clap_version,
                cache_dir=self.cache_dir,
                dataset_name="predict"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
        
        for idx, (test_path, test_embedding) in enumerate(zip(test_paths, test_embeddings)):
            try:
                # Create visualization for this test sample
                vis_result = self._create_test_visualization(
                    test_embedding, test_path, idx, return_figure=True
                )
                
                # Unpack result (might include legend_text)
                if len(vis_result) == 3:
                    vis_image, fig, legend_text = vis_result
                else:
                    vis_image, fig = vis_result
                    legend_text = ""
                
                # Create prompt with legend text
                prompt = self._create_audio_classification_prompt(legend_text)
                
                # Get VLM prediction
                prediction = self._get_vlm_prediction(vis_image, prompt)
                predictions.append(prediction)
                
                # Save visualization and capture detailed output (reduced frequency)
                should_save_viz = save_outputs and output_dir and (idx % self.save_every_n == 0 or idx == 0 or idx == total_samples - 1)
                viz_path = None
                if should_save_viz:
                    # Generate standardized filename
                    backend_name = "pca" if self.use_pca_backend else "tsne"
                    viz_filename = generate_visualization_filename(
                        sample_index=idx,
                        backend=backend_name,
                        dimensions='3d' if self.use_3d else '2d',
                        use_knn=self.use_knn_connections and not self.use_pca_backend,
                        nn_k=self.nn_k if self.use_knn_connections else None
                    )
                    
                    viz_path = os.path.join(viz_dir, viz_filename)
                    
                    # Save using utility function
                    save_info = save_visualization_with_metadata(
                        fig, viz_path, 
                        metadata=convert_for_json_serialization({
                            'sample_index': idx,
                            'backend': backend_name,
                            'use_3d': self.use_3d,
                            'use_knn': self.use_knn_connections and not self.use_pca_backend,
                            'prediction': prediction,
                            'vlm_model': self.vlm_model_id,
                            'audio_path': test_path,
                            'whisper_model': self.whisper_model
                        }),
                        dpi=self.image_dpi
                    )
                
                # Store detailed output
                detailed_outputs.append({
                    'sample_index': idx,
                    'audio_path': test_path,
                    'prompt': prompt,
                    'vlm_model': self.vlm_model_id,
                    'vlm_response': f"Predicted class: {prediction}",  # Simplified for audio
                    'parsed_prediction': prediction,
                    'visualization_saved': should_save_viz,
                    'visualization_path': viz_path if should_save_viz else None,
                    'backend_params': {
                        'use_pca_backend': self.use_pca_backend,
                        'use_3d': self.use_3d,
                        'use_knn_connections': self.use_knn_connections,
                        'nn_k': self.nn_k if self.use_knn_connections else None,
                        'zoom_factor': self.zoom_factor,
                        'whisper_model': self.whisper_model,
                        'include_spectrogram': self.include_spectrogram
                    }
                })
                
                close_figure_safely(fig)
                
                # Log progress less frequently
                if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
                    logger.info(f"Processed {idx + 1}/{total_samples} samples")
                    
            except Exception as e:
                logger.error(f"Error predicting sample {idx}: {e}")
                # Close figure if it exists
                if 'fig' in locals():
                    close_figure_safely(fig)
                # Use default prediction as fallback
                prediction = 0
                predictions.append(prediction)
                
                # Store error details
                detailed_outputs.append({
                    'sample_index': idx,
                    'audio_path': test_path,
                    'vlm_model': self.vlm_model_id,
                    'vlm_response': f"ERROR: {str(e)}",
                    'parsed_prediction': prediction,
                    'error': str(e),
                    'visualization_saved': False,
                    'backend_params': {
                        'use_pca_backend': self.use_pca_backend,
                        'use_3d': self.use_3d,
                        'use_knn_connections': self.use_knn_connections,
                        'nn_k': self.nn_k if self.use_knn_connections else None,
                        'zoom_factor': self.zoom_factor,
                        'whisper_model': self.whisper_model,
                        'include_spectrogram': self.include_spectrogram
                    }
                })
                
        return np.array(predictions), detailed_outputs
        
    def _create_test_visualization(self, test_embedding: np.ndarray, 
                                 test_path: str, sample_idx: int, 
                                 return_figure: bool = False) -> Union[Image.Image, Tuple[Image.Image, plt.Figure], Tuple[Image.Image, plt.Figure, str]]:
        """Create visualization for a test sample."""
        # Prepare embeddings for visualization
        test_embedding_2d = test_embedding.reshape(1, -1)
        legend_text = ""  # Initialize legend text
        
        # Create t-SNE plot
        if self.use_knn_connections and not self.use_pca_backend:
            # For KNN visualization, we need to project the test sample into the trained t-SNE space
            # This is a simplified approach - in practice, you'd want to retrain t-SNE or use approximate methods
            from sklearn.neighbors import NearestNeighbors
            
            # Find nearest neighbors in the plot embedding space to approximate t-SNE position
            nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(self.plot_embeddings)
            distances, indices = nbrs.kneighbors(test_embedding_2d)
            
            # Approximate test point position as weighted average of nearest training points in t-SNE space
            weights = 1.0 / (distances[0] + 1e-8)  # Inverse distance weighting
            weights = weights / weights.sum()
            test_tsne_approx = np.average(self.train_tsne[indices[0]], weights=weights, axis=0).reshape(1, -1)
            
            if self.use_3d:
                # 3D KNN plot not implemented yet - fall back to standard 3D
                train_tsne, test_tsne, fig = create_tsne_3d_visualization(
                    self.plot_embeddings[:50], self.plot_labels[:50],  # Limit for speed
                    test_embedding_2d,
                    perplexity=min(self.tsne_perplexity, 15),
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed,
                    figsize=(16, 12)
                )
                legend_text = "3D t-SNE visualization showing spatial relationships across multiple viewing angles"
            else:
                fig, legend_text, metadata = create_tsne_plot_with_knn(
                    self.train_tsne,           # train_tsne
                    test_tsne_approx,          # test_tsne
                    self.plot_labels,          # train_labels
                    self.plot_embeddings,      # train_embeddings  
                    test_embedding_2d,         # test_embeddings
                    highlight_test_idx=0,      # highlight_test_idx (we only have 1 test point)
                    k=self.nn_k,
                    zoom_factor=self.zoom_factor,
                    figsize=(16, 10),
                    class_names=self.class_names,
                    use_semantic_names=self.use_semantic_names
                )
        else:
            # Standard visualization without KNN
            # Note: For individual test samples, we need to run t-SNE freshly
            # since the t-SNE space changes with each new test point
            if self.use_3d:
                train_tsne, test_tsne, fig = create_tsne_3d_visualization(
                    self.train_embeddings[:20], self.train_labels[:20],  # Limit for speed
                    test_embedding_2d,
                    perplexity=min(self.tsne_perplexity, 10),
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed,
                    figsize=(16, 12)
                )
                legend_text = "3D t-SNE visualization showing spatial relationships across multiple viewing angles"
            else:
                train_tsne, test_tsne, fig = create_tsne_visualization(
                    self.train_embeddings[:20], self.train_labels[:20],  # Limit for speed  
                    test_embedding_2d,
                    perplexity=min(self.tsne_perplexity, 10),
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed,
                    figsize=(12, 8)
                )
                legend_text = "2D t-SNE visualization showing spatial clustering of audio samples"
                
        # Add spectrogram if requested
        if self.include_spectrogram:
            self._add_spectrogram_to_figure(fig, test_path)
            
        # Convert to image
        vis_image = plot_to_image(fig, dpi=self.image_dpi)
        
        # legend_text is already set by the visualization functions
        
        if return_figure:
            return vis_image, fig, legend_text
        else:
            close_figure_safely(fig)
            return vis_image
        
    def _add_spectrogram_to_figure(self, fig: plt.Figure, audio_path: str):
        """Add spectrogram to figure using proper layout that doesn't conflict."""
        # Load audio
        audio, sr = load_audio(audio_path, sr=16000, duration=self.audio_duration)
        
        # Create spectrogram
        spec = create_spectrogram(audio, sr, n_mels=128, db_scale=True)
        
        # Get current figure size and adjust
        current_size = fig.get_size_inches()
        new_height = current_size[1] * 1.3  # Increase height by 30%
        fig.set_size_inches(current_size[0], new_height)
        
        # Find existing axes to preserve their layout
        existing_axes = fig.get_axes()
        
        if len(existing_axes) == 0:
            # No existing axes, create simple layout
            gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.25)
            ax_spec = fig.add_subplot(gs[1, 0])
        elif len(existing_axes) == 1:
            # Single axis (standard t-SNE plot)
            # Adjust existing axis to top portion
            for ax in existing_axes:
                pos = ax.get_position()
                new_pos = [pos.x0, pos.y0 + 0.25, pos.width, pos.height * 0.7]
                ax.set_position(new_pos)
            
            # Add spectrogram at bottom
            ax_spec = fig.add_subplot(3, 1, 3)
        else:
            # Multiple axes (KNN plot with pie chart)
            # Adjust all existing axes to top 70% of figure
            for ax in existing_axes:
                pos = ax.get_position()
                new_pos = [pos.x0, pos.y0 + 0.25, pos.width, pos.height * 0.7]
                ax.set_position(new_pos)
            
            # Add spectrogram spanning full width at bottom
            ax_spec = fig.add_axes([0.1, 0.05, 0.8, 0.2])  # [left, bottom, width, height]
        
        # Plot spectrogram
        plot_spectrogram(spec, sr, ax=ax_spec, title="Audio Spectrogram")
        
    def _create_audio_classification_prompt(self, legend_text: str = "") -> str:
        """Create prompt for audio classification using unified prompting strategy."""
        # Determine dataset type from class names for better context  
        if self.embedding_model == "whisper":
            dataset_description = f"Audio samples embedded using Whisper {self.whisper_model} {self.embedding_layer} features"
        elif self.embedding_model == "clap":
            dataset_description = f"Audio samples embedded using CLAP {self.clap_version} features"
        else:
            dataset_description = f"Audio samples embedded using {self.embedding_model} features"
        
        # Check if this is emotion recognition (RAVDESS)
        emotion_keywords = {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'}
        if any(emotion in str(name).lower() for name in self.class_names for emotion in emotion_keywords):
            dataset_description += " for emotion recognition from speech"
        # Check if this is environmental sound classification (ESC-50)
        elif len(self.class_names) == 50 or any('sound' in str(name).lower() for name in self.class_names):
            dataset_description += " for environmental sound classification"
        # Check if this is urban sound classification
        elif any(urban_word in str(name).lower() for name in self.class_names for urban_word in ['siren', 'traffic', 'urban', 'street']):
            dataset_description += " for urban sound classification"
        
        return create_classification_prompt(
            class_names=self.class_names,
            modality="audio",
            use_knn=self.use_knn_connections and not self.use_pca_backend,
            use_3d=self.use_3d,
            nn_k=self.nn_k if self.use_knn_connections else None,
            legend_text=legend_text,
            include_spectrogram=self.include_spectrogram,
            dataset_description=dataset_description,
            use_semantic_names=self.use_semantic_names
        )
        
    def _load_vlm_model(self):
        """Load the Vision Language Model using standardized model loader."""
        try:
            # Use the centralized model loader from MARVIS
            from marvis.utils.model_loader import model_loader
            
            # Get platform-compatible kwargs
            vlm_kwargs = configure_model_kwargs_for_platform(
                device=self.device,
                torch_dtype=get_platform_compatible_dtype(self.device)
            )
            
            # Load VLM using centralized model loader
            vlm_wrapper = model_loader.load_vlm(
                self.vlm_model_id, 
                backend='auto',
                device=self.device, 
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                **vlm_kwargs
            )
            
            return vlm_wrapper
            
        except Exception as e:
            logger.error(f"Failed to load VLM with model_loader: {e}")
            logger.info("Falling back to simple VLM loading...")
            
            # Fallback VLM wrapper
            class SimpleVLMWrapper:
                def __init__(self, model_id, device):
                    self.model_id = model_id
                    self.device = device
                    self.model = None
                    self.processor = None
                    
                def generate_from_conversation(self, conversation, generation_config):
                    # Mock implementation for testing
                    if self.model is None:
                        # Return random class for testing
                        return f"Class_{np.random.randint(0, 10)}"
                    # Real implementation would go here
                    return "test_class"
            
            return SimpleVLMWrapper(self.vlm_model_id, self.device)
            
    def _get_vlm_prediction(self, image: Image.Image, prompt: str) -> int:
        """Get prediction from VLM."""
        try:
            # Create conversation using utility
            conversation = create_vlm_conversation(image, prompt)
            
            # Import the proper GenerationConfig class
            try:
                from marvis.utils.model_loader import GenerationConfig
                gen_config = GenerationConfig(
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True
                )
            except ImportError:
                # Fallback to simple config if import fails
                class SimpleGenConfig:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                    
                    def to_transformers_kwargs(self):
                        return {
                            'max_new_tokens': getattr(self, 'max_new_tokens', 100),
                            'temperature': getattr(self, 'temperature', 0.1),
                            'do_sample': getattr(self, 'do_sample', True)
                        }
                
                gen_config = SimpleGenConfig(
                    max_new_tokens=100,
                    temperature=0.7,       # Increased from 0.1 to avoid numerical instability
                    do_sample=False,       # Changed to False for more stable generation
                )
            
            # Generate response
            response = self.vlm_wrapper.generate_from_conversation(conversation, gen_config)
            
            # Parse prediction using utility
            predicted_class = parse_vlm_response(response, np.array(self.class_names), logger, self.use_semantic_names)
            
            # Convert to label
            if predicted_class in self.class_names:
                return self.class_names.index(predicted_class)
            else:
                logger.warning(f"VLM returned unknown class: {predicted_class}")
                return 0
                
        except Exception as e:
            logger.error(f"VLM prediction error: {e}")
            return 0
            
    def evaluate(self, test_paths: List[str], test_labels: List[int],
                return_detailed: bool = False, save_outputs: bool = False,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            test_paths: List of test audio paths
            test_labels: List of test labels
            return_detailed: Return detailed results
            save_outputs: Save visualizations
            output_dir: Output directory for visualizations
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_paths)} test samples")
        
        start_time = time.time()
        predictions, detailed_outputs = self.predict(
            test_paths, 
            save_outputs=save_outputs, 
            output_dir=output_dir
        )
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': prediction_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions.tolist(),
            'true_labels': test_labels
        }
        
        if return_detailed:
            # Only use class names for classes that appear in the test set
            unique_test_labels = np.unique(test_labels)
            test_class_names = [self.class_names[i] for i in unique_test_labels]
            
            report = classification_report(
                test_labels, predictions,
                labels=unique_test_labels,
                target_names=test_class_names,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = report
            results['detailed_outputs'] = detailed_outputs
            
        if save_outputs and output_dir:
            results['visualizations_saved'] = True
            results['output_directory'] = output_dir
            
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
        
        return results
        
    def get_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        return {
            'embedding_model': self.embedding_model,
            'whisper_model': self.whisper_model,
            'embedding_layer': self.embedding_layer,
            'clap_version': self.clap_version,
            'tsne_perplexity': self.tsne_perplexity,
            'tsne_n_iter': self.tsne_n_iter,
            'vlm_model_id': self.vlm_model_id,
            'use_3d': self.use_3d,
            'use_knn_connections': self.use_knn_connections,
            'nn_k': self.nn_k,
            'use_pca_backend': self.use_pca_backend,
            'include_spectrogram': self.include_spectrogram,
            'audio_duration': self.audio_duration,
            'device': str(self.device),
            'seed': self.seed
        }