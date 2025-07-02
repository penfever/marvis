#!/usr/bin/env python
"""
MARVIS t-SNE classifier - A unified implementation for tabular, audio, and vision modalities.

This module provides a centralized MARVIS t-SNE classifier that can work across different
data modalities by using appropriate embedding methods and t-SNE visualizations with
Vision Language Model classification.
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

# Import unified model loader for VLM
from marvis.utils.model_loader import model_loader, GenerationConfig

# Import VLM prompting utilities
from marvis.utils.vlm_prompting import create_classification_prompt, parse_vlm_response, create_vlm_conversation, create_metadata_summary
from marvis.utils.class_name_utils import extract_class_names_from_labels
from marvis.models.process_one_sample import process_one_sample

# Import metadata utilities
from marvis.utils.metadata_loader import load_dataset_metadata, detect_dataset_id_from_path

# Import semantic axes utilities
from marvis.utils.semantic_axes import enhance_visualization_with_semantic_axes

# Import new multi-visualization framework
from marvis.viz import ContextComposer, VisualizationConfig
from marvis.viz.context.layouts import LayoutStrategy
from marvis.viz.context.composer import CompositionConfig


def convert_numpy_types(obj):
    """Convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class BioClip2EmbeddingExtractor:
    """Extract embeddings using BioCLIP-2 model."""
    
    def __init__(self, model_name: str = "hf-hub:imageomics/bioclip-2", device: str = "auto"):
        from marvis.utils.device_utils import detect_optimal_device
        self.model_name = model_name
        self.device = detect_optimal_device()
        self.model = None
        self.preprocess = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load BioCLIP model using OpenCLIP."""
        try:
            import open_clip
            
            self.logger.info(f"Loading BioCLIP model: {self.model_name}")
            
            # Load BioCLIP model with OpenCLIP
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.model_name)
            
            # Move to appropriate device
            self.model = model.to(self.device)
            
            # Use validation preprocessing for inference
            self.preprocess = preprocess_val
            
            # Set to eval mode
            self.model.eval()
            
            self.logger.info(f"BioCLIP model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise RuntimeError(f"BioCLIP requires open_clip library: {e}. Install with: pip install open-clip-torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load BioCLIP model: {e}")
    
    def extract_embeddings(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from images with efficient batching."""
        if self.model is None:
            self.load_model()
        
        embeddings = []
        self.logger.info(f"Extracting BioClip2 embeddings for {len(image_paths)} images with batch_size={batch_size}")
        
        # Process images in batches for efficiency
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            if batch_start % (batch_size * 10) == 0:  # Log every 10 batches
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
            
            try:
                batch_embeddings = self._extract_batch_embeddings(batch_paths)
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Fallback to individual processing for this batch
                for image_path in batch_paths:
                    try:
                        embedding = self._extract_single_embedding(image_path)
                        embeddings.append(embedding)
                    except Exception as individual_e:
                        self.logger.error(f"Error processing image {image_path}: {individual_e}")
                        raise RuntimeError(f"Failed to extract embedding for {image_path}: {individual_e}")
        
        return np.array(embeddings)
    
    def _extract_batch_embeddings(self, image_paths: list) -> list:
        """Extract embeddings from a batch of images efficiently."""
        from PIL import Image
        
        batch_images = []
        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image)
            batch_images.append(image_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(self.device)
        
        # Extract features in batch
        with torch.no_grad():
            batch_features = self.model.encode_image(batch_tensor)
            # Normalize features (standard for CLIP models)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        
        # Convert to list of numpy arrays
        embeddings = []
        for i in range(batch_features.shape[0]):
            embedding = batch_features[i].cpu().numpy().flatten()
            embeddings.append(embedding)
        
        return embeddings
    
    def _extract_single_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from single image."""
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize features (standard for CLIP models)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Convert to numpy
        embedding = image_features.cpu().numpy().flatten()
        return embedding


def get_bioclip2_embeddings(
    image_paths: list,
    model_name: str = "hf-hub:imageomics/bioclip-2", 
    cache_dir: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 128
) -> np.ndarray:
    """Get BioCLIP2 embeddings for a list of images with efficient batching."""
    extractor = BioClip2EmbeddingExtractor(model_name=model_name, device=device)
    return extractor.extract_embeddings(image_paths, batch_size=batch_size)


class MarvisTsneClassifier:
    """
    Unified MARVIS t-SNE classifier for tabular, audio, and vision data.
    
    This classifier:
    1. Generates embeddings using modality-specific methods (TabPFN, Whisper, CLAP, etc.)
    2. Creates t-SNE visualizations with training and test points
    3. Uses a Vision Language Model to classify test points based on their position
    """
    
    def __init__(
        self,
        modality: str = "tabular",
        vlm_model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        embedding_size: int = 1000,
        tsne_perplexity: int = 30,
        tsne_max_iter: int = 1000,
        use_3d: bool = False,  # Unified 3D parameter (backward compatibility: use_3d_tsne is deprecated)
        use_knn_connections: bool = False,
        nn_k: int = 5,
        show_test_points: bool = False,
        max_vlm_image_size: int = 2048,
        image_dpi: int = 100,
        force_rgb_mode: bool = True,
        zoom_factor: float = 2.0,
        max_tabpfn_samples: int = 3000,
        cache_dir: Optional[str] = None,
        use_semantic_names: bool = False,
        device: str = "auto",
        backend: str = "auto",
        vlm_backend: Optional[str] = None,  # Alias for backend (for backward compatibility)
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = True,
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        api_model: Optional[str] = None,
        seed: int = 42,
        # VLM engine parameters
        max_model_len: Optional[int] = None,
        # New multi-visualization parameters
        enable_multi_viz: bool = False,
        visualization_methods: Optional[List[str]] = None,
        layout_strategy: str = "adaptive_grid",
        reasoning_focus: str = "classification",
        multi_viz_config: Optional[Dict[str, Any]] = None,
        # Audio-specific parameters
        embedding_model: str = "whisper",
        whisper_model: str = "large-v2",
        embedding_layer: str = "encoder_last",
        clap_version: str = "2023",
        include_spectrogram: bool = True,
        audio_duration: Optional[float] = None,
        num_few_shot_examples: int = 32,
        balanced_few_shot: bool = False,
        # Vision-specific parameters
        dinov2_model: str = "dinov2_vitb14",
        embedding_backend: str = "dinov2",  # For vision: "dinov2" or "bioclip2"
        bioclip2_model: str = "hf-hub:imageomics/bioclip-2",
        bioclip2_batch_size: int = 128,  # Batch size for BioCLIP2 embedding extraction
        use_pca_backend: bool = False,
        # Metadata parameters
        dataset_metadata: Optional[Union[str, Dict[str, Any], Path]] = None,
        auto_load_metadata: bool = True,
        metadata_base_dir: Optional[str] = None,
        use_metadata: bool = False,
        semantic_axes: bool = False,
        semantic_axes_method: str = "pca_loadings",
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize MARVIS t-SNE classifier.
        
        Args:
            modality: Data modality ("tabular", "audio", "vision")
            vlm_model_id: Vision Language Model identifier (for local models)
            embedding_size: Size of embeddings for TabPFN
            tsne_perplexity: t-SNE perplexity parameter
            tsne_n_iter: Number of t-SNE iterations
            use_3d_tsne: Whether to use 3D t-SNE visualization
            use_knn_connections: Whether to show KNN connections
            knn_k: Number of nearest neighbors for KNN
            max_vlm_image_size: Maximum image size for VLM
            image_dpi: DPI for generated images
            force_rgb_mode: Whether to force RGB mode for images
            zoom_factor: Zoom factor for all visualizations
            max_tabpfn_samples: Maximum samples for TabPFN (tabular only)
            cache_dir: Directory for caching embeddings
            use_semantic_names: Whether to use semantic class names
            device: Device for computation
            backend: Backend for VLM loading
            vlm_backend: Alias for backend (for backward compatibility)
            tensor_parallel_size: Tensor parallel size for distributed inference
            gpu_memory_utilization: GPU memory utilization factor
            enable_thinking: Enable thinking mode for compatible API models
            openai_model: OpenAI model identifier (e.g., 'gpt-4o', 'gpt-4.1')
            gemini_model: Gemini model identifier (e.g., 'gemini-2.5-pro', 'gemini-2.5-flash')
            api_model: Generic API model identifier (auto-detects provider)
            seed: Random seed
            max_model_len: Maximum model length for VLM engine
            enable_multi_viz: Whether to use multi-visualization framework (default: False for backward compatibility)
            visualization_methods: List of visualization methods to use (e.g., ['tsne', 'pca', 'umap'])
            layout_strategy: Layout strategy for multi-visualization composition
            reasoning_focus: Focus for multi-visualization reasoning (classification, comparison, etc.)
            multi_viz_config: Additional configuration for multi-visualization
            
            Audio-specific parameters:
                embedding_model: Audio embedding model ('whisper' or 'clap')
                whisper_model: Whisper model variant (if using Whisper)
                embedding_layer: Which Whisper layer to extract
                clap_version: CLAP model version (if using CLAP)
                include_spectrogram: Whether to include spectrogram in prompts
                audio_duration: Maximum audio duration to process
                num_few_shot_examples: Number of few-shot examples for audio
                balanced_few_shot: Whether to balance few-shot examples
                
            Vision-specific parameters:
                dinov2_model: DINOV2 model variant (if embedding_backend="dinov2")
                embedding_backend: Vision embedding backend ("dinov2" or "bioclip2")
                bioclip2_model: BioCLIP2 model variant (if embedding_backend="bioclip2")
                bioclip2_batch_size: Batch size for BioCLIP2 embedding extraction (default: 128)
                use_pca_backend: Use PCA instead of t-SNE
                
            Metadata and enhancement parameters:
                dataset_metadata: Path or dict with dataset metadata
                auto_load_metadata: Automatically load metadata if available  
                metadata_base_dir: Base directory for metadata files
                use_metadata: Incorporate semantic feature names and domain context into prompts
                semantic_axes: Compute factor weighting of named features to improve visualization legends
                
            **kwargs: Additional modality-specific arguments
        """
        self.modality = modality.lower()
        self.vlm_model_id = vlm_model_id
        self.embedding_size = embedding_size
        self.tsne_perplexity = tsne_perplexity
        self.tsne_max_iter = tsne_max_iter
        # Handle backward compatibility for use_3d_tsne -> use_3d
        if 'use_3d_tsne' in kwargs:
            # Use a temporary logger for the deprecation warning
            temp_logger = logging.getLogger(__name__)
            temp_logger.warning("Parameter 'use_3d_tsne' is deprecated. Use 'use_3d' instead.")
            use_3d = kwargs.pop('use_3d_tsne', use_3d)
        self.use_3d = use_3d
        self.use_knn_connections = use_knn_connections
        self.knn_k = nn_k
        self.show_test_points = show_test_points
        self.max_vlm_image_size = max_vlm_image_size
        self.image_dpi = image_dpi
        self.force_rgb_mode = force_rgb_mode
        self.zoom_factor = zoom_factor
        self.max_tabpfn_samples = max_tabpfn_samples
        self.cache_dir = cache_dir
        self.use_semantic_names = use_semantic_names
        self.device = device if device is not None else "auto"
        # Handle vlm_backend alias for backward compatibility
        if vlm_backend is not None:
            backend = vlm_backend
        self.backend = backend
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_thinking = enable_thinking
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.api_model = api_model
        self.seed = seed
        self.max_model_len = max_model_len
        
        # Metadata parameters
        self.dataset_metadata = dataset_metadata
        self.auto_load_metadata = auto_load_metadata
        self.metadata_base_dir = metadata_base_dir
        self.use_metadata = use_metadata
        self.semantic_axes = semantic_axes
        self.semantic_axes_method = semantic_axes_method
        self._loaded_metadata = None  # Cached loaded metadata
        self._embedding_model = None  # For perturbation method (tabular modality)
        
        # New multi-visualization parameters
        self.enable_multi_viz = enable_multi_viz
        self.visualization_methods = visualization_methods or ['tsne']
        self.layout_strategy = layout_strategy
        self.reasoning_focus = reasoning_focus
        self.multi_viz_config = multi_viz_config or {}
        
        # Determine the actual model to use (API models take precedence)
        self.effective_model_id = self._determine_effective_model()
        self.is_api_model = self._is_api_model(self.effective_model_id)
        
        # Store modality-specific parameters
        self.modality_kwargs = kwargs
        
        # Audio-specific parameters
        if self.modality == "audio":
            self.modality_kwargs.update({
                'embedding_model': embedding_model,
                'whisper_model': whisper_model,
                'embedding_layer': embedding_layer,
                'clap_version': clap_version,
                'include_spectrogram': include_spectrogram,
                'audio_duration': audio_duration,
                'num_few_shot_examples': num_few_shot_examples,
                'balanced_few_shot': balanced_few_shot
            })
        
        # Vision-specific parameters
        elif self.modality == "vision":
            self.modality_kwargs.update({
                'dinov2_model': dinov2_model,
                'embedding_backend': embedding_backend,
                'bioclip2_model': bioclip2_model,
                'bioclip2_batch_size': bioclip2_batch_size,
                'use_pca_backend': use_pca_backend,
            })
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize VLM wrapper (loaded lazily)
        self.vlm_wrapper = None
        
        # Store fitted data
        self.train_embeddings = None
        self.test_embeddings = None
        
        # Store feature names for semantic axes computation
        self.feature_names = feature_names
        self.train_tsne = None
        self.test_tsne = None
        self.y_train_sample = None
        self.class_names = None
        self.unique_classes = None
        self.class_to_semantic = None
        self.semantic_axes_labels = None
        self.class_color_name_map = None  # Maps class labels to color names (e.g., {0: "Blue", 5: "Color_111"})
        self.color_to_class_map = None    # Maps color names to class labels (e.g., {"Blue": 0, "Color_111": 5})
        
        # Multi-visualization context composer
        self.context_composer = None
        
        # Embedding model for perturbation-based semantic axes (tabular only)
        self._embedding_model = None
    
    def _determine_effective_model(self) -> str:
        """Determine the effective model ID to use based on API model parameters."""
        # Priority order: api_model > openai_model > gemini_model > vlm_model_id
        if self.api_model:
            return self.api_model
        elif self.openai_model:
            return self.openai_model
        elif self.gemini_model:
            return self.gemini_model
        else:
            return self.vlm_model_id
    
    def _is_api_model(self, model_id: str) -> bool:
        """Check if the model ID corresponds to an API model."""
        api_model_patterns = [
            # OpenAI models
            'gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4.1',
            # Gemini models
            'gemini-', 'gemini-2.', 'gemini-2.5', 'gemini-2.0'
        ]
        return any(pattern in model_id.lower() for pattern in api_model_patterns)
        
    def _get_embedding_method(self):
        """Get the appropriate embedding method for the modality."""
        if self.modality == "tabular":
            from marvis.data.embeddings import get_tabpfn_embeddings
            return get_tabpfn_embeddings
        elif self.modality == "audio":
            # Get audio-specific embedding method
            embedding_model = self.modality_kwargs.get('embedding_model', 'whisper')
            if embedding_model == 'whisper':
                from marvis.data.audio_embeddings import get_whisper_embeddings
                return get_whisper_embeddings
            elif embedding_model == 'clap':
                from marvis.data.audio_embeddings import get_clap_embeddings
                return get_clap_embeddings
            else:
                raise ValueError(f"Unsupported audio embedding model: {embedding_model}")
        elif self.modality == "vision":
            # Get vision-specific embedding method based on backend
            embedding_backend = self.modality_kwargs.get('embedding_backend', 'dinov2')
            if embedding_backend == 'bioclip2':
                # Use BioCLIP2 embedding method (defined in this file)
                return get_bioclip2_embeddings
            elif embedding_backend == 'dinov2':
                from marvis.data.embeddings import get_dinov2_embeddings
                return get_dinov2_embeddings
            else:
                raise ValueError(f"Unsupported vision embedding backend: {embedding_backend}. Supported: 'dinov2', 'bioclip2'")
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
    
    def _apply_dimensionality_reduction(self, embeddings, n_components):
        """
        Apply dimensionality reduction (t-SNE) to embeddings.
        
        This method is used by the perturbation-based semantic axes computation
        to apply the same dimensionality reduction as used in the main visualization.
        
        Args:
            embeddings: High-dimensional embeddings to reduce
            n_components: Number of components for the reduced representation
            
        Returns:
            Reduced coordinates with shape [n_samples, n_components]
        """
        from sklearn.manifold import TSNE
        
        # Use same parameters as the main t-SNE computation, with fallbacks
        if hasattr(self, '_tsne_params'):
            perplexity = self._tsne_params['perplexity']
            max_iter = self._tsne_params.get('max_iter', self._tsne_params.get('n_iter', 1000))  # Support both old and new parameter names
            random_state = self._tsne_params['random_state']
        else:
            # Fallback to instance parameters
            perplexity = self.tsne_perplexity
            max_iter = self.tsne_max_iter
            random_state = self.seed
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(embeddings) // 4),
            max_iter=max_iter,
            random_state=random_state
        )
        
        return tsne.fit_transform(embeddings)
    
    def _get_tsne_visualization_methods(self):
        """Get t-SNE visualization methods."""
        from marvis.viz.tsne_functions import (
            create_tsne_visualization,
            create_tsne_3d_visualization,
            create_combined_tsne_plot,
            create_combined_tsne_3d_plot,
            create_tsne_plot_with_knn,
            create_tsne_3d_plot_with_knn,
            create_regression_tsne_visualization,
            create_regression_tsne_3d_visualization,
            create_combined_regression_tsne_plot,
            create_combined_regression_tsne_3d_plot,
            create_regression_tsne_plot_with_knn,
            create_regression_tsne_3d_plot_with_knn
        )
        return {
            'create_tsne_visualization': create_tsne_visualization,
            'create_tsne_3d_visualization': create_tsne_3d_visualization,
            'create_combined_tsne_plot': create_combined_tsne_plot,
            'create_combined_tsne_3d_plot': create_combined_tsne_3d_plot,
            'create_tsne_plot_with_knn': create_tsne_plot_with_knn,
            'create_tsne_3d_plot_with_knn': create_tsne_3d_plot_with_knn,
            'create_regression_tsne_visualization': create_regression_tsne_visualization,
            'create_regression_tsne_3d_visualization': create_regression_tsne_3d_visualization,
            'create_combined_regression_tsne_plot': create_combined_regression_tsne_plot,
            'create_combined_regression_tsne_3d_plot': create_combined_regression_tsne_3d_plot,
            'create_regression_tsne_plot_with_knn': create_regression_tsne_plot_with_knn,
            'create_regression_tsne_3d_plot_with_knn': create_regression_tsne_3d_plot_with_knn
        }
    
    def _load_vlm(self):
        """Load the Vision Language Model (local or API-based)."""
        if self.vlm_wrapper is not None:
            return self.vlm_wrapper
            
        model_to_load = self.effective_model_id
        self.logger.info(f"Loading Vision Language Model: {model_to_load}")
        
        if self.is_api_model:
            # API model - minimal configuration needed
            self.logger.info("Using API-based VLM (OpenAI/Gemini)")
            vlm_kwargs = {}
            
            # For API models, backend is auto-detected by model_loader
            backend = "auto"
        else:
            # Local model - configure hardware parameters
            self.logger.info("Using local VLM")
            vlm_kwargs = {}
            # Import device utilities for proper device detection
            from marvis.utils.device_utils import detect_optimal_device
            
            # Resolve device if set to auto
            self.logger.info(f"Initial VLM device value: {self.device}")
            actual_device = self.device
            if self.device == "auto":
                actual_device = detect_optimal_device(prefer_mps=True)
                self.logger.info(f"Auto-detected device for VLM: {actual_device}")
                # Update self.device so it's not "auto" anymore
                self.device = actual_device
            else:
                self.logger.info(f"Using configured device for VLM: {actual_device}")
            
            if actual_device == "cuda" and torch.cuda.is_available():
                vlm_kwargs.update({
                    'torch_dtype': torch.bfloat16,
                    'device_map': "auto",
                    'low_cpu_mem_usage': True
                })
                self.logger.info("Configured VLM for CUDA")
            elif actual_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                vlm_kwargs.update({
                    'torch_dtype': torch.float16,
                    'device_map': actual_device,
                    'low_cpu_mem_usage': True
                })
                self.logger.info("Configured VLM for MPS (Metal Performance Shaders)")
            else:
                vlm_kwargs.update({
                    'low_cpu_mem_usage': True
                })
                self.logger.info(f"Configured VLM for {actual_device}")
            
            backend = self.backend
        
        # Add max_model_len if specified
        if self.max_model_len is not None:
            vlm_kwargs['max_model_len'] = self.max_model_len
        
        # Load VLM using centralized model loader
        try:
            self.logger.info(f"Loading VLM with parameters: model={model_to_load}, backend={backend}, device={actual_device}")
            
            self.vlm_wrapper = model_loader.load_vlm(
                model_to_load,
                backend=backend,
                device=actual_device,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                **vlm_kwargs
            )
            
            if self.vlm_wrapper is None:
                self.logger.error(f"Model loader returned None for model {model_to_load}")
                raise RuntimeError(f"Model loader returned None for model {model_to_load}")
            
            self.logger.info("VLM loaded successfully")
            return self.vlm_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load VLM {model_to_load}: {e}")
            self.logger.error(f"VLM loading failed with exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Set vlm_wrapper to None explicitly in case of error
            self.vlm_wrapper = None
            raise RuntimeError(f"Failed to load VLM {model_to_load}: {e}") from e
    
    def _initialize_multi_viz_composer(self, X_train, y_train, X_test=None):
        """Initialize the multi-visualization context composer."""
        if not self.enable_multi_viz:
            return
            
        self.logger.info("Initializing multi-visualization context composer...")
        
        # Create composition configuration
        config = CompositionConfig(
            layout_strategy=LayoutStrategy[self.layout_strategy.upper()],
            reasoning_focus=self.reasoning_focus,
            optimize_for_vlm=True
        )
        
        # Initialize context composer
        self.context_composer = ContextComposer(config)
        
        # Add visualizations based on specified methods
        for viz_method in self.visualization_methods:
            viz_config = VisualizationConfig(
                use_3d=self.use_3d,
                title=f"{viz_method.upper()} - {self.modality.title()} Data",
                random_state=self.seed,
                figsize=(8, 6),
                point_size=50,
                use_knn_connections=self.use_knn_connections,
                nn_k=self.knn_k,
                show_test_points=self.show_test_points,
                zoom_factor=self.zoom_factor
            )
            
            # Get method-specific configuration
            method_config = self.multi_viz_config.get(viz_method, {})
            
            # Add method-specific parameters based on visualization type
            if viz_method == 'tsne':
                method_config.update({
                    'perplexity': min(self.tsne_perplexity, len(X_train) // 4),
                    'max_iter': self.tsne_max_iter
                })
            elif viz_method == 'umap':
                method_config.setdefault('n_neighbors', 15)
                method_config.setdefault('min_dist', 0.1)
            elif viz_method == 'spectral':
                method_config.update({
                    'n_neighbors': max(2, len(X_train) // 20),
                    'affinity': 'nearest_neighbors'
                })
            elif viz_method == 'isomap':
                method_config.setdefault('n_neighbors', 10)
            elif viz_method == 'decision_regions':
                # Pass through any decision_classifier config from multi_viz_config
                method_config.setdefault('decision_classifier', 'svm')
                method_config.setdefault('embedding_method', 'pca')
            elif viz_method == 'frequent_patterns':
                # Configure frequent patterns analysis
                method_config.setdefault('min_support', 0.1)
                method_config.setdefault('min_confidence', 0.6)
                method_config.setdefault('n_bins', 5)
                
            self.context_composer.add_visualization(
                viz_method,
                config=method_config,
                viz_config=viz_config
            )
        
        self.logger.info(f"Added {len(self.context_composer.visualizations)} visualization methods")
        
        # Fit all visualizations
        self.context_composer.fit(X_train, y_train, X_test)
    
    def _load_dataset_metadata(self, dataset_info=None):
        """Load dataset metadata for enhanced VLM prompts."""
        try:
            from marvis.utils.metadata_loader import load_dataset_metadata, detect_dataset_id_from_path
            
            metadata = None
            
            # Try to load from explicit metadata parameter
            if self.dataset_metadata is not None:
                metadata = load_dataset_metadata(
                    self.dataset_metadata, 
                    metadata_base_dir=self.metadata_base_dir
                )
                if metadata:
                    self.logger.info(f"Loaded metadata from explicit parameter: {metadata.dataset_name}")
            
            # Try auto-detection if no explicit metadata and auto_load is enabled
            if metadata is None and self.auto_load_metadata:
                dataset_id = None
                
                # Try to detect from dataset_info if available
                if dataset_info and isinstance(dataset_info, dict):
                    dataset_id = dataset_info.get('dataset_id') or dataset_info.get('task_id')
                    self.logger.info(f"Trying to load metadata for dataset_id: {dataset_id} from dataset_info: {dataset_info}")
                
                # Try to load metadata using detected ID
                if dataset_id:
                    try:
                        metadata = load_dataset_metadata(
                            dataset_id, 
                            metadata_base_dir=self.metadata_base_dir
                        )
                        if metadata:
                            self.logger.info(f"Auto-loaded metadata for dataset ID '{dataset_id}': {metadata.dataset_name}")
                        else:
                            self.logger.info(f"No metadata found for dataset ID '{dataset_id}'")
                    except Exception as e:
                        self.logger.warning(f"Failed to load metadata for dataset ID '{dataset_id}': {e}")
                
            # If no metadata was loaded but we have dataset info and feature names, create basic metadata
            if metadata is None and dataset_info and self.use_metadata:
                metadata = self._create_basic_metadata_from_dataset_info(dataset_info)
                if metadata:
                    self.logger.info(f"Created basic metadata from dataset info: {metadata.dataset_name}")
            
            # Store loaded metadata
            self._loaded_metadata = metadata
            
            if metadata:
                self.logger.info(f"Metadata loaded successfully: {len(metadata.columns)} columns, {len(metadata.target_classes)} target classes")
            else:
                self.logger.debug("No metadata loaded - will use basic prompts")
                
        except ImportError:
            self.logger.warning("Metadata loading not available - install metadata utilities")
            self._loaded_metadata = None
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")
            self._loaded_metadata = None
    
    def _create_basic_metadata_from_dataset_info(self, dataset_info):
        """Create basic metadata from dataset_info when no metadata file exists."""
        try:
            from marvis.utils.metadata_loader import DatasetMetadata, ColumnMetadata, TargetClassMetadata
            
            # Extract basic info
            dataset_name = dataset_info.get('name', 'Unknown Dataset')
            feature_names = dataset_info.get('feature_names', [])
            n_features = dataset_info.get('n_features', 0)
            n_classes = dataset_info.get('n_classes', 0)
            
            # Create basic column metadata
            columns = []
            for i, feature_name in enumerate(feature_names):
                if feature_name:
                    columns.append(ColumnMetadata(
                        name=feature_name,
                        semantic_description=f"Feature {i}: {feature_name}",
                        data_type="numeric"
                    ))
            
            # Create basic target class metadata
            target_classes = []
            if hasattr(self, 'class_names') and self.class_names:
                for i, class_name in enumerate(self.class_names):
                    target_classes.append(TargetClassMetadata(
                        name=class_name,
                        meaning=f"Class {i}: {class_name}"
                    ))
            else:
                for i in range(n_classes):
                    target_classes.append(TargetClassMetadata(
                        name=f"Class_{i}",
                        meaning=f"Target class {i}"
                    ))
            
            # Create basic description
            description = f"Dataset with {n_features} features and {n_classes} classes"
            if dataset_info.get('data_source') == 'openml':
                description += f" from OpenML task {dataset_info.get('task_id', 'unknown')}"
            
            return DatasetMetadata(
                dataset_name=dataset_name,
                description=description,
                columns=columns,
                target_classes=target_classes
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create basic metadata: {e}")
            return None
    
    def _get_metadata_for_prompt(self):
        """Get metadata summary for use in VLM prompts."""
        if not self.use_metadata or self._loaded_metadata is None:
            return None
            
        try:
            summary = create_metadata_summary(self._loaded_metadata)
            return summary
        except Exception as e:
            self.logger.warning(f"Failed to create metadata summary: {e}")
            return None
    
    def _get_semantic_axes_legend(self, embeddings, reduced_coords, labels, feature_names=None):
        """Get semantic axes legend for visualization enhancement."""
        if not self.semantic_axes:
            return ""
            
        try:
            return enhance_visualization_with_semantic_axes(
                embeddings=embeddings,
                reduced_coords=reduced_coords,
                labels=labels,
                metadata=self._loaded_metadata,
                feature_names=feature_names,
                method="pca_loadings"
            )
        except Exception as e:
            self.logger.warning(f"Failed to create semantic axes legend: {e}")
            return ""
    
    def _compute_semantic_axes_labels(self, embeddings, reduced_coords, labels, feature_names=None):
        """Compute semantic axes labels for plot axes."""
        if not self.semantic_axes:
            return None
            
        try:
            from marvis.utils.semantic_axes import SemanticAxesComputer
            computer = SemanticAxesComputer(method=self.semantic_axes_method)
            
            # For perturbation method, we need additional parameters
            if self.semantic_axes_method == "perturbation":
                # Check if we have the necessary components for perturbation
                if (hasattr(self, '_original_features') and 
                    hasattr(self, '_embedding_model') and 
                    self._original_features is not None and
                    self._embedding_model is not None):
                    
                    # Create reduction function
                    def reduction_func(emb):
                        return self._apply_dimensionality_reduction(emb, reduced_coords.shape[1])
                    
                    semantic_axes = computer.compute_semantic_axes(
                        embeddings, reduced_coords, labels, feature_names, self._loaded_metadata,
                        original_features=self._original_features,
                        embedding_model=self._embedding_model,
                        reduction_func=reduction_func
                    )
                else:
                    self.logger.warning("Perturbation method requires original features and embedding model, falling back to PCA")
                    # Fallback to PCA method
                    computer = SemanticAxesComputer(method="pca_loadings")
                    semantic_axes = computer.compute_semantic_axes(
                        embeddings, reduced_coords, labels, feature_names, self._loaded_metadata
                    )
            else:
                # Standard methods (pca_loadings, feature_importance)
                semantic_axes = computer.compute_semantic_axes(
                    embeddings, reduced_coords, labels, feature_names, self._loaded_metadata
                )
            return semantic_axes
        except Exception as e:
            self.logger.warning(f"Failed to compute semantic axes labels: {e}")
            return None
    
    def fit(self, X_train, y_train, X_test=None, class_names=None, task_type=None, **kwargs):
        """
        Fit the MARVIS t-SNE model for both classification and regression.
        
        Args:
            X_train: Training features
            y_train: Training labels/targets
            X_test: Test features (optional, for creating visualizations)
            class_names: Class names (optional, for classification)
            task_type: 'classification' or 'regression' (auto-detected if None)
            **kwargs: Additional arguments
        """
        self.logger.info(f"Fitting MARVIS t-SNE model for {self.modality} data...")
        
        # Handle different input formats
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)
            
        if hasattr(y_train, 'values'):
            y_train_array = y_train.values
        else:
            y_train_array = np.array(y_train)
        
        # Detect task type
        try:
            from marvis.utils.task_detection import (
                detect_task_type, get_target_statistics,
                VISION_CLASSIFICATION_TASK_ID, AUDIO_CLASSIFICATION_TASK_ID
            )
            dataset_info = kwargs.get('dataset_info')
            task_id = dataset_info.get('task_id') if dataset_info else None
            
            # For non-tabular modalities, use special task IDs if task_id not provided
            if task_id is None and self.modality == "vision":
                task_id = VISION_CLASSIFICATION_TASK_ID
                self.logger.debug(f"Using special vision classification task_id: {task_id}")
            elif task_id is None and self.modality == "audio":
                task_id = AUDIO_CLASSIFICATION_TASK_ID
                self.logger.debug(f"Using special audio classification task_id: {task_id}")
            
            self.task_type, detection_method = detect_task_type(
                y=y_train_array, 
                manual_override=task_type,
                task_id=task_id,
                dataset_info=dataset_info
            )
            self.logger.info(f"Task type: {self.task_type} (detected via: {detection_method})")
            
            # Get target statistics for regression or class info for classification
            if self.task_type == 'regression':
                self.target_stats = get_target_statistics(y_train_array)
                self.logger.info(f"Target statistics: {self.target_stats}")
            else:
                self.unique_classes = np.unique(y_train_array)
                self.target_stats = None
        except ImportError:
            # Fallback if task detection is not available
            self.logger.warning("Task detection not available, assuming classification")
            self.task_type = 'classification'
            self.unique_classes = np.unique(y_train_array)
            self.target_stats = None
        
        # Load metadata if enabled
        if self.auto_load_metadata or self.dataset_metadata is not None:
            self._load_dataset_metadata(kwargs.get('dataset_info'))
        
        # Apply feature reduction for tabular data if needed
        if self.modality == "tabular":
            from marvis.utils import apply_feature_reduction
            # Create a mock dataset dict for feature reduction
            mock_dataset = {'name': 'training_data'}
            mock_args = type('Args', (), {
                'feature_selection_threshold': getattr(self, 'feature_selection_threshold', 500),
                'seed': self.seed
            })()
            
            X_train_array, X_test_array, _, self.selected_feature_indices = apply_feature_reduction(
                X_train_array, y_train_array, 
                X_test.values if X_test is not None and hasattr(X_test, 'values') else X_test,
                mock_dataset, mock_args, self.logger
            )
            
            if X_test is not None and self.selected_feature_indices is not None:
                X_test = X_test_array
        
        # Use all training data (no internal validation split)
        X_train_fit, y_train_fit = X_train_array, y_train_array
        
        # Store original features for semantic axes computation (before any transformation)
        if self.semantic_axes and self.modality == "tabular":
            self._original_features = X_train_fit.copy()
        
        # Generate embeddings using modality-specific method
        embedding_method = self._get_embedding_method()
        
        if self.modality == "tabular":
            # Prepare test data for embedding
            if X_test is not None:
                X_test_for_embedding = X_test
            else:
                # Use a small subset of training data as test for visualization
                X_test_for_embedding = X_train_fit[:5]
                
            self.train_embeddings, self.val_embeddings, self.test_embeddings, self.tabpfn, self.y_train_sample = embedding_method(
                X_train_fit, y_train_fit, X_test_for_embedding,
                max_samples=self.max_tabpfn_samples,
                embedding_size=self.embedding_size,
                cache_dir=self.cache_dir,
                dataset_name='training_data',
                force_recompute=getattr(self, 'force_recompute_embeddings', False),
                task_type=self.task_type,
                seed=self.seed
            )
            
            # Store the TabPFN model in _embedding_model for perturbation method
            if self.tabpfn is not None:
                # Create a wrapper that provides a transform method for TabPFN
                class TabPFNWrapper:
                    def __init__(self, tabpfn_model, parent_logger):
                        self.tabpfn = tabpfn_model
                        self.logger = parent_logger
                    
                    def transform(self, X):
                        """Wrapper method to provide sklearn-style transform interface."""
                        # Ensure X is 2D (fix for single sample case)
                        if X.ndim == 1:
                            X = X.reshape(1, -1)
                            self.logger.debug(f"TabPFNWrapper: Reshaped 1D input to 2D: {X.shape}")
                        
                        embeddings = self.tabpfn.get_embeddings(X)
                        # Handle ensemble embeddings by averaging if needed
                        if len(embeddings.shape) == 3 and embeddings.shape[0] > 1:
                            embeddings = np.mean(embeddings, axis=0)
                        elif len(embeddings.shape) == 3:
                            embeddings = embeddings[0]
                        return embeddings
                
                self._embedding_model = TabPFNWrapper(self.tabpfn, self.logger)
            else:
                self._embedding_model = None
            
        elif self.modality == "audio":
            # Audio embedding generation
            self.logger.info(f"Generating {self.modality_kwargs.get('embedding_model', 'whisper')} embeddings for audio...")
            
            # Prepare test data
            if X_test is not None:
                X_test_for_embedding = X_test
            else:
                X_test_for_embedding = X_train_fit[:5]
            
            # Get embeddings based on embedding model type
            if self.modality_kwargs.get('embedding_model') == 'whisper':
                embeddings_dict = embedding_method(
                    X_train_fit,  # audio files
                    whisper_model=self.modality_kwargs.get('whisper_model', 'large-v2'),
                    embedding_layer=self.modality_kwargs.get('embedding_layer', 'encoder_last'),
                    max_duration=self.modality_kwargs.get('audio_duration'),
                    cache_dir=self.cache_dir,
                    device=self.device
                )
            else:  # CLAP
                embeddings_dict = embedding_method(
                    X_train_fit,  # audio files
                    model_version=self.modality_kwargs.get('clap_version', '2023'),
                    cache_dir=self.cache_dir,
                    device=self.device
                )
            
            self.train_embeddings = embeddings_dict['embeddings']
            self.val_embeddings = None  # Audio doesn't use validation split for embeddings
            
            # Get test embeddings if available
            if X_test_for_embedding is not None and len(X_test_for_embedding) > 0:
                if self.modality_kwargs.get('embedding_model') == 'whisper':
                    test_embeddings_dict = embedding_method(
                        X_test_for_embedding,
                        whisper_model=self.modality_kwargs.get('whisper_model', 'large-v2'),
                        embedding_layer=self.modality_kwargs.get('embedding_layer', 'encoder_last'),
                        max_duration=self.modality_kwargs.get('audio_duration'),
                        cache_dir=self.cache_dir,
                        device=self.device
                    )
                else:
                    test_embeddings_dict = embedding_method(
                        X_test_for_embedding,
                        model_version=self.modality_kwargs.get('clap_version', '2023'),
                        cache_dir=self.cache_dir,
                        device=self.device
                    )
                self.test_embeddings = test_embeddings_dict['embeddings']
            else:
                self.test_embeddings = None
                
            self.y_train_sample = y_train_fit
            self.tabpfn = None  # Not used for audio
            self._embedding_model = None  # Perturbation method not available for audio
            
        elif self.modality == "vision":
            # Vision embedding generation
            embedding_backend = self.modality_kwargs.get('embedding_backend', 'dinov2')
            self.logger.info(f"Generating {embedding_backend.upper()} embeddings for images...")
            
            # Prepare test data
            if X_test is not None:
                X_test_for_embedding = X_test
            else:
                X_test_for_embedding = X_train_fit[:5]
            
            # Get embeddings based on backend
            if embedding_backend == 'bioclip2':
                # Use BioCLIP2 embeddings with efficient batching
                batch_size = self.modality_kwargs.get('bioclip2_batch_size', 128)
                train_embeddings = embedding_method(
                    X_train_fit,  # image files
                    model_name=self.modality_kwargs.get('bioclip2_model', 'hf-hub:imageomics/bioclip-2'),
                    cache_dir=self.cache_dir,
                    device=self.device,
                    batch_size=batch_size
                )
                
                # Get test embeddings if available
                if X_test_for_embedding is not None and len(X_test_for_embedding) > 0:
                    self.test_embeddings = embedding_method(
                        X_test_for_embedding,
                        model_name=self.modality_kwargs.get('bioclip2_model', 'hf-hub:imageomics/bioclip-2'),
                        cache_dir=self.cache_dir,
                        device=self.device,
                        batch_size=batch_size
                    )
                else:
                    self.test_embeddings = None
            else:
                # Use DINOV2 embeddings (default)
                train_embeddings = embedding_method(
                    X_train_fit,  # image files or arrays
                    model_name=self.modality_kwargs.get('dinov2_model', 'dinov2_vitb14'),
                    batch_size=32,
                    cache_dir=self.cache_dir,
                    device=self.device
                )
                
                # Get test embeddings if available
                if X_test_for_embedding is not None and len(X_test_for_embedding) > 0:
                    self.test_embeddings = embedding_method(
                        X_test_for_embedding,
                        model_name=self.modality_kwargs.get('dinov2_model', 'dinov2_vitb14'),
                        batch_size=32,
                        cache_dir=self.cache_dir,
                        device=self.device
                    )
                else:
                    self.test_embeddings = None
            
            self.train_embeddings = train_embeddings
            self.val_embeddings = None  # Vision doesn't use validation split for embeddings
            self.y_train_sample = y_train_fit
            self.tabpfn = None  # Not used for vision
            self._embedding_model = None  # Perturbation method not available for vision
            
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        
        # Store t-SNE parameters for potential reuse in perturbation method
        self._tsne_params = {
            'perplexity': self.tsne_perplexity,
            'max_iter': self.tsne_max_iter,
            'random_state': self.seed
        }
        
        # Use all test embeddings for both visualization and prediction
        test_embeddings_for_viz = self.test_embeddings
        
        # Create t-SNE visualization based on task type
        viz_methods = self._get_tsne_visualization_methods()
        
        if self.task_type == 'regression':
            # Use regression-specific visualization methods
            if self.use_3d:
                self.logger.info("Creating 3D regression t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_regression_tsne_3d_visualization'](
                    self.train_embeddings, self.y_train_sample, test_embeddings_for_viz,
                    perplexity=self.tsne_perplexity,
                    max_iter=self.tsne_max_iter,
                    random_state=self.seed,
                    zoom_factor=self.zoom_factor,
                    cached_color_mapping={'class_to_color': self.class_color_name_map, 'color_to_class': self.color_to_class_map}
                )
            else:
                dimension_str = "3D" if self.use_3d else "2D"
                self.logger.info(f"Creating {dimension_str} regression t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_regression_tsne_visualization'](
                    self.train_embeddings, self.y_train_sample, test_embeddings_for_viz,
                    perplexity=self.tsne_perplexity,
                    max_iter=self.tsne_max_iter,
                    random_state=self.seed,
                    use_3d=self.use_3d,
                    zoom_factor=self.zoom_factor,
                    cached_color_mapping={'class_to_color': self.class_color_name_map, 'color_to_class': self.color_to_class_map}
                )
        else:
            # Use classification visualization methods
            if self.use_3d:
                self.logger.info("Creating 3D classification t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_tsne_3d_visualization'](
                    self.train_embeddings, self.y_train_sample, test_embeddings_for_viz,
                    perplexity=self.tsne_perplexity,
                    max_iter=self.tsne_max_iter,
                    random_state=self.seed,
                    zoom_factor=self.zoom_factor,
                    cached_color_mapping={'class_to_color': self.class_color_name_map, 'color_to_class': self.color_to_class_map}
                )
            else:
                self.logger.info("Creating 2D classification t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_tsne_visualization'](
                    self.train_embeddings, self.y_train_sample, test_embeddings_for_viz,
                    perplexity=self.tsne_perplexity,
                    max_iter=self.tsne_max_iter,
                    random_state=self.seed,
                    zoom_factor=self.zoom_factor,
                    cached_color_mapping={'class_to_color': self.class_color_name_map, 'color_to_class': self.color_to_class_map}
                )
        
        # Close base figure to save memory
        plt.close(base_fig)
        
        # Set up class/target information based on task type
        if self.task_type == 'classification':
            # Get unique classes and set up class names
            if not hasattr(self, 'unique_classes') or self.unique_classes is None:
                self.unique_classes = np.unique(self.y_train_sample)
            
            # Use provided class_names if available, otherwise extract semantic class names
            if class_names is not None:
                # Use explicitly provided class names (highest priority)
                semantic_class_names = class_names
                self.class_names = class_names
            else:
                # Try to use class names from loaded metadata first
                if (self.use_semantic_names and self._loaded_metadata and 
                    hasattr(self._loaded_metadata, 'target_classes') and 
                    self._loaded_metadata.target_classes and
                    len(self._loaded_metadata.target_classes) == len(self.unique_classes)):
                    # Extract class names from metadata
                    semantic_class_names = [tc.name if hasattr(tc, 'name') else f'Class_{i}' 
                                           for i, tc in enumerate(self._loaded_metadata.target_classes)]
                    self.logger.info(f"Using semantic class names from metadata: {semantic_class_names}")
                else:
                    # Extract semantic class names with fallback
                    semantic_class_names, _ = extract_class_names_from_labels(
                        labels=self.unique_classes.tolist() if self.unique_classes is not None else [],
                        dataset_name=kwargs.get('dataset_name', None),
                        semantic_data_dir=kwargs.get('semantic_data_dir', None),
                        use_semantic=self.use_semantic_names
                    )
                self.class_names = semantic_class_names
            
            # Create mapping from numeric labels to semantic names
            if self.unique_classes is not None:
                self.class_to_semantic = {cls: name for cls, name in zip(sorted(self.unique_classes), semantic_class_names)}
            else:
                self.class_to_semantic = {}
        else:
            # For regression, we don't have class names
            self.unique_classes = None
            self.class_to_semantic = None
            self.class_names = None
        
        # Get cached color mappings for VLM parsing (classification only)
        if self.task_type == 'classification' and self.unique_classes is not None:
            try:
                from marvis.utils.resource_manager import get_resource_manager
                
                # Determine dataset identifier for caching
                dataset_id = kwargs.get('dataset_name', '')
                if 'dataset_info' in kwargs and kwargs['dataset_info']:
                    # Prefer task_id if available
                    dataset_id = kwargs['dataset_info'].get('task_id', dataset_id)
                
                if not dataset_id:
                    # Fallback to a generic identifier based on number of classes
                    dataset_id = f"unknown_{len(self.unique_classes)}classes"
                
                # Get cached color mapping from resource manager
                resource_manager = get_resource_manager()
                color_mappings = resource_manager.dataset_preparer.get_cached_color_mapping(
                    dataset_id=dataset_id,
                    unique_classes=self.unique_classes.tolist()
                )
                
                self.class_color_name_map = color_mappings.get('class_to_color', {})
                self.color_to_class_map = color_mappings.get('color_to_class', {})
                
                if self.class_color_name_map and self.color_to_class_map:
                    self.logger.info(f"Using cached color mappings for dataset {dataset_id} with {len(self.unique_classes)} classes")
                else:
                    self.logger.warning(f"Failed to get valid color mappings for dataset {dataset_id}")
                    
            except Exception as e:
                self.logger.warning(f"Could not get cached color mappings: {e}")
                self.class_color_name_map = None
                self.color_to_class_map = None
        else:
            self.class_color_name_map = None
            self.color_to_class_map = None
        
        # Compute semantic axes labels if enabled
        if self.semantic_axes and self.train_embeddings is not None and self.train_tsne is not None:
            # For tabular data, prefer original features over processed embeddings for semantic axes
            if self.modality == "tabular" and hasattr(self, '_original_features'):
                self.semantic_axes_labels = self._compute_semantic_axes_labels(
                    self._original_features, 
                    self.train_tsne, 
                    self.y_train_sample,
                    feature_names=self.feature_names
                )
            else:
                self.semantic_axes_labels = self._compute_semantic_axes_labels(
                    self.train_embeddings, 
                    self.train_tsne, 
                    self.y_train_sample,
                    feature_names=self.feature_names
                )
            if self.semantic_axes_labels:
                self.logger.info(f"Computed semantic axes: X={self.semantic_axes_labels.get('X', 'N/A')}, Y={self.semantic_axes_labels.get('Y', 'N/A')}")
        
        # Initialize multi-visualization framework if enabled
        if self.enable_multi_viz:
            # Limit test samples for multi-visualization as well
            X_test_for_multi_viz = X_test_for_embedding
            
            # Use the reduced training data for multi-visualization
            self._initialize_multi_viz_composer(
                X_train_fit, 
                y_train_fit, 
                X_test_for_multi_viz
            )
            
        self.logger.info(f"MARVIS t-SNE {self.task_type} model fitted successfully")
    
    def predict(self, X_test, y_test=None, return_detailed=False, save_outputs=False, output_dir=None, visualization_save_cadence=10):
        """
        Make predictions using the fitted MARVIS t-SNE classifier.
        
        Args:
            X_test: Test features (not used directly, embeddings already computed in fit)
            y_test: Test labels (for evaluation)
            return_detailed: Whether to return detailed prediction information
            save_outputs: Whether to save visualizations and outputs
            output_dir: Directory to save outputs
            visualization_save_cadence: Save visualizations for every N samples (default: 10)
            
        Returns:
            predictions or detailed results dict
        """
        if self.train_tsne is None or self.test_tsne is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Set up output directory if saving outputs
        if save_outputs and output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.temp_dir = output_dir  # Store for test script access
        elif save_outputs:
            # Create a temporary directory
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix='marvis_tsne_')
        else:
            self.temp_dir = None
        
        # Load VLM
        self._load_vlm()
        
        # Ensure VLM wrapper was loaded successfully
        if self.vlm_wrapper is None:
            raise RuntimeError(f"Failed to load VLM model {self.vlm_model_id}. Cannot proceed with classification.")

        from PIL import Image
        import io
        
        # Get visualization methods
        viz_methods = self._get_tsne_visualization_methods()
        
        # Parse custom viewing angles if provided
        viewing_angles = None
        if self.use_3d and 'viewing_angles' in self.modality_kwargs:
            try:
                # Parse format: "elev1,azim1;elev2,azim2;..."
                angle_pairs = self.modality_kwargs['viewing_angles'].split(';')
                viewing_angles = []
                for pair in angle_pairs:
                    elev, azim = map(int, pair.split(','))
                    viewing_angles.append((elev, azim))
                self.logger.info(f"Using custom viewing angles: {viewing_angles}")
            except Exception as e:
                self.logger.warning(f"Error parsing viewing angles: {e}. Using defaults.")
                viewing_angles = None
        
        # Make predictions for each test point
        predictions = []
        prediction_details = []
        completed_samples = 0
        
        self.logger.info(f"Starting VLM predictions for {len(self.test_tsne)} test points...")
        
        for i in range(len(self.test_tsne)):
            # Track figures at start of iteration to ensure cleanup
            iteration_start_figures = set(plt.get_fignums())
            
            try:
                # Process the sample using the extracted function
                prediction, response = process_one_sample(
                    classifier_instance=self,
                    sample_index=i,
                    viz_methods=viz_methods,
                    viewing_angles=viewing_angles,
                    save_outputs=save_outputs,
                    visualization_save_cadence=visualization_save_cadence,
                    return_detailed=return_detailed,
                    y_test=y_test,
                    prediction_details=prediction_details,
                    all_classes=self.unique_classes
                )
                
                predictions.append(prediction)
                completed_samples = i + 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(self.test_tsne)} predictions")
                
            except Exception as e:
                self.logger.error(f"VLM prediction failed for test point {i}: {e}")
                # Use random prediction as fallback
                if self.unique_classes is not None and len(self.unique_classes) > 0:
                    prediction = np.random.choice(self.unique_classes)
                else:
                    # For regression or when classes are not available, use a default value
                    prediction = 0 if self.task_type == 'regression' else 0
                predictions.append(prediction)
                completed_samples = i + 1
            
            finally:
                # Ensure figures created during this iteration are cleaned up
                iteration_end_figures = set(plt.get_fignums())
                new_iteration_figures = iteration_end_figures - iteration_start_figures
                if new_iteration_figures:
                    for fignum in new_iteration_figures:
                        plt.close(fignum)
        
        if return_detailed:
            return {
                'predictions': predictions,
                'prediction_details': prediction_details,
                'completed_samples': completed_samples,
                'completion_rate': completed_samples / len(self.test_tsne) if len(self.test_tsne) > 0 else 0.0
            }
        else:
            return predictions
    
    def evaluate(self, X_test, y_test, return_detailed=False, save_outputs=False, output_dir=None, visualization_save_cadence=10):
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            return_detailed: Whether to return detailed results
            save_outputs: Whether to save outputs
            output_dir: Directory to save outputs
            visualization_save_cadence: Save visualizations for every N samples (default: 10)
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        # Make predictions
        detailed_results = self.predict(X_test, y_test, return_detailed=True, save_outputs=save_outputs, output_dir=output_dir, visualization_save_cadence=visualization_save_cadence)
        predictions = detailed_results['predictions']
        completed_samples = detailed_results['completed_samples']
        
        # Calculate metrics
        if completed_samples > 0:
            # Get partial ground truth
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            # Convert predictions to same type as ground truth
            predictions_converted = []
            target_type = type(y_test_partial[0])
            
            for pred in predictions:
                try:
                    predictions_converted.append(target_type(pred))
                except (ValueError, TypeError):
                    predictions_converted.append(pred)
            
            # Calculate metrics using shared utility
            from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
            metrics = calculate_llm_metrics(
                y_test_partial, predictions_converted, self.unique_classes,
                all_class_log_probs=None, logger=self.logger, task_type=self.task_type
            )
        else:
            metrics = {
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'roc_auc': None,
                'f1_macro': None,
                'f1_micro': None,
                'f1_weighted': None,
                'precision_macro': None,
                'recall_macro': None
            }
        
        # Calculate timing
        total_time = time.time() - start_time
        
        # Build results
        results = {
            'model_name': f'MARVIS-t-SNE-{self.modality}',
            'accuracy': float(metrics['accuracy']) if metrics['accuracy'] is not None else None,
            'balanced_accuracy': float(metrics['balanced_accuracy']) if metrics['balanced_accuracy'] is not None else None,
            'prediction_time': float(total_time),
            'total_time': float(total_time),
            'num_test_samples': len(X_test) if hasattr(X_test, '__len__') else len(self.test_tsne),
            'completed_samples': completed_samples,
            'completion_rate': detailed_results['completion_rate'],
            'num_classes': len(self.unique_classes) if self.unique_classes is not None else 0,
            'predictions': predictions_converted if 'predictions_converted' in locals() else predictions,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist')
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics
            'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] is not None else None,
            'f1_macro': float(metrics['f1_macro']) if metrics['f1_macro'] is not None else None,
            'f1_micro': float(metrics['f1_micro']) if metrics['f1_micro'] is not None else None,
            'f1_weighted': float(metrics['f1_weighted']) if metrics['f1_weighted'] is not None else None,
            'precision_macro': float(metrics['precision_macro']) if metrics['precision_macro'] is not None else None,
            'recall_macro': float(metrics['recall_macro']) if metrics['recall_macro'] is not None else None,
            # Regression metrics (None for classification tasks)
            'r2_score': float(metrics['r2_score']) if metrics.get('r2_score') is not None else None,
            'mae': float(metrics['mae']) if metrics.get('mae') is not None else None,
            'mse': float(metrics['mse']) if metrics.get('mse') is not None else None,
            'rmse': float(metrics['rmse']) if metrics.get('rmse') is not None else None,
            # Task type for reference
            'task_type': metrics.get('task_type', 'unknown'),
            # Model configuration
            'config': self.get_config()
        }
        
        if return_detailed:
            results.update({
                'prediction_details': detailed_results.get('prediction_details', [])
            })
        
        return results
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'modality': self.modality,
            'vlm_model_id': self.vlm_model_id,
            'embedding_size': self.embedding_size,
            'tsne_perplexity': self.tsne_perplexity,
            'tsne_max_iter': self.tsne_max_iter,
            'use_3d': self.use_3d,
            'use_knn_connections': self.use_knn_connections,
            'nn_k': self.knn_k,
            'show_test_points': self.show_test_points,
            'max_vlm_image_size': self.max_vlm_image_size,
            'image_dpi': self.image_dpi,
            'force_rgb_mode': self.force_rgb_mode,
            'zoom_factor': self.zoom_factor,
            'max_tabpfn_samples': self.max_tabpfn_samples,
            'use_semantic_names': self.use_semantic_names,
            'device': self.device,
            'backend': self.backend,
            'enable_thinking': self.enable_thinking,
            'openai_model': self.openai_model,
            'gemini_model': self.gemini_model,
            'api_model': self.api_model,
            'effective_model_id': self.effective_model_id,
            'is_api_model': self.is_api_model,
            'seed': self.seed,
            # Multi-visualization parameters
            'enable_multi_viz': self.enable_multi_viz,
            'visualization_methods': self.visualization_methods,
            'layout_strategy': self.layout_strategy,
            'reasoning_focus': self.reasoning_focus,
            'multi_viz_config': self.multi_viz_config,
            # Vision-specific parameters (if vision modality)
            'dinov2_model': self.modality_kwargs.get('dinov2_model') if self.modality == 'vision' else None,
            'embedding_backend': self.modality_kwargs.get('embedding_backend') if self.modality == 'vision' else None,
            'bioclip2_model': self.modality_kwargs.get('bioclip2_model') if self.modality == 'vision' else None,
            'use_pca_backend': self.modality_kwargs.get('use_pca_backend') if self.modality == 'vision' else None,
        }


def evaluate_marvis_tsne(dataset, args):
    """
    Evaluate MARVIS t-SNE baseline on a dataset (legacy function for backward compatibility).
    
    This function maintains compatibility with existing tabular LLM baseline scripts.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating MARVIS t-SNE on dataset {dataset['name']}")
    
    try:
        # Import required utilities
        from marvis.utils import (
            drop_feature_for_oom,
            is_oom_error,
            apply_feature_reduction
        )
        
        # Use preprocessed data if available, otherwise split data
        # Require pre-split data - no automatic splitting
        required_keys = ["X_train", "X_test", "y_train", "y_test"]
        missing_keys = [key for key in required_keys if key not in dataset]
        
        if missing_keys:
            raise ValueError(
                f"Dataset must contain pre-split data. Missing keys: {missing_keys}. "
                f"Expected keys: {required_keys}. "
                f"Please split your data before passing it to MARVIS t-SNE."
            )
        
        # Use preprocessed data which respects balanced_few_shot sampling
        X_train = dataset["X_train"]
        X_test = dataset["X_test"]
        y_train = dataset["y_train"]
        y_test = dataset["y_test"]
        logger.info(f"Using pre-split data - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Limit test samples if specified
        if args.max_test_samples and args.max_test_samples < len(X_test):
            X_test = X_test[:args.max_test_samples]
            y_test = y_test[:args.max_test_samples]
        
        # Create classifier
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id=getattr(args, 'vlm_model_id', "Qwen/Qwen2.5-VL-32B-Instruct"),
            embedding_size=getattr(args, 'embedding_size', 1000),
            tsne_perplexity=getattr(args, 'tsne_perplexity', 30),
            tsne_max_iter=getattr(args, 'tsne_max_iter', getattr(args, 'tsne_n_iter', 1000)),
            use_3d=getattr(args, 'use_3d', False),
            use_knn_connections=getattr(args, 'use_knn_connections', False),
            nn_k=getattr(args, 'nn_k', 5),
            show_test_points=getattr(args, 'show_test_points', False),
            max_vlm_image_size=getattr(args, 'max_vlm_image_size', 2048),
            image_dpi=getattr(args, 'image_dpi', 100),
            force_rgb_mode=getattr(args, 'force_rgb_mode', True),
            zoom_factor=getattr(args, 'zoom_factor', getattr(args, 'tsne_zoom_factor', 2.0)),  # Backward compatibility
            max_tabpfn_samples=getattr(args, 'max_tabpfn_samples', 3000),
            cache_dir=getattr(args, 'cache_dir', None),
            use_semantic_names=getattr(args, 'use_semantic_names', False),
            use_metadata=getattr(args, 'use_metadata', False),
            device=args.device,
            backend=getattr(args, 'backend', 'auto'),
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.9),
            enable_thinking=getattr(args, 'enable_thinking', True),
            openai_model=getattr(args, 'openai_model', None),
            gemini_model=getattr(args, 'gemini_model', None),
            api_model=getattr(args, 'api_model', None),
            seed=args.seed,
            # Multi-visualization parameters
            enable_multi_viz=getattr(args, 'enable_multi_viz', False),
            visualization_methods=getattr(args, 'visualization_methods', ['tsne']),
            layout_strategy=getattr(args, 'layout_strategy', 'adaptive_grid'),
            reasoning_focus=getattr(args, 'reasoning_focus', 'classification'),
            multi_viz_config=getattr(args, 'multi_viz_config', {}),
            # Pass additional args as kwargs
            viewing_angles=getattr(args, 'viewing_angles', None),
            feature_selection_threshold=getattr(args, 'feature_selection_threshold', 500)
        )
        
        # Fit and evaluate
        # Resolve task_id using resource manager as per CLAUDE.md guidelines
        task_id = dataset.get('task_id')
        if task_id is None:
            # Try to resolve task_id from dataset_id using resource manager
            try:
                from marvis.utils.resource_manager import get_resource_manager
                rm = get_resource_manager()
                dataset_id = dataset.get('id')
                if dataset_id:
                    identifiers = rm.resolve_openml_identifiers(dataset_id=dataset_id)
                    task_id = identifiers.get('task_id')
            except Exception as e:
                logger.warning(f"Could not resolve task_id from dataset_id {dataset.get('id')}: {e}")
        
        dataset_info = {
            'name': dataset['name'],
            'task_id': task_id
        }
        classifier.fit(X_train, y_train, X_test, dataset_name=dataset['name'], dataset_info=dataset_info)
        results = classifier.evaluate(
            X_test, y_test, 
            return_detailed=True,
            save_outputs=getattr(args, 'save_sample_visualizations', True),
            output_dir=getattr(args, 'output_dir', None),
            visualization_save_cadence=getattr(args, 'visualization_save_cadence', 10)
        )
        
        # Add dataset information
        results.update({
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id']
        })
        
        # Log appropriate metric based on task type
        if results.get('accuracy') is not None:
            logger.info(f"MARVIS t-SNE accuracy on {dataset['name']}: {results['accuracy']:.4f}")
        elif results.get('r2_score') is not None:
            logger.info(f"MARVIS t-SNE R² score on {dataset['name']}: {results['r2_score']:.4f}")
        else:
            logger.info(f"MARVIS t-SNE evaluation completed on {dataset['name']}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating MARVIS t-SNE: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'MARVIS-t-SNE',
            'dataset_name': dataset['name'],
            'error': str(e)
        }


# Backward compatibility classes
class MarvisAudioTsneClassifier(MarvisTsneClassifier):
    """
    MARVIS t-SNE classifier for audio classification.
    
    This is a convenience wrapper that sets modality="audio" automatically.
    All functionality is provided by the unified MarvisTsneClassifier.
    """
    
    def __init__(self, **kwargs):
        # Set modality to audio
        kwargs['modality'] = 'audio'
        super().__init__(**kwargs)


class MarvisImageTsneClassifier(MarvisTsneClassifier):
    """
    MARVIS t-SNE classifier for image/vision classification.
    
    This is a convenience wrapper that sets modality="vision" automatically.
    All functionality is provided by the unified MarvisTsneClassifier.
    """
    
    def __init__(self, **kwargs):
        # Set modality to vision
        kwargs['modality'] = 'vision'
        super().__init__(**kwargs)
    
    def fit(self, train_image_paths, train_labels, test_image_paths=None, class_names=None, **kwargs):
        """
        Fit the MARVIS image t-SNE classifier.
        
        Args:
            train_image_paths: List of training image paths or training features
            train_labels: List of training labels
            test_image_paths: List of test image paths (optional)
            class_names: List of class names (optional)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Call the parent fit method with appropriate arguments
        return super().fit(
            X_train=train_image_paths,
            y_train=train_labels,
            X_test=test_image_paths,
            class_names=class_names,
            **kwargs
        )