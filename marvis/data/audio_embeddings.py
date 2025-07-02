"""
Audio embedding extraction using Whisper and other audio models.
"""

import numpy as np
import torch
import logging
import os
import hashlib
import json
import librosa
from typing import Tuple, List, Optional, Union, Dict, Any
from pathlib import Path

__all__ = [
    'get_whisper_embeddings',
    'load_whisper_model',
    'prepare_audio_for_whisper',
    'get_clap_embeddings',
    'load_clap_model',
    'prepare_audio_for_clap',
    'generate_audio_dataset_hash'
]

logger = logging.getLogger(__name__)


def load_whisper_model(model_name: str = "large-v2", device: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load Whisper model for audio embedding extraction.
    
    Args:
        model_name: Whisper model variant (tiny, base, small, medium, large, large-v2)
        device: Device to load model on (None for auto-detect)
        
    Returns:
        model: Whisper model
        processor: Whisper processor/feature extractor
    """
    try:
        import whisper
        from transformers import WhisperProcessor, WhisperModel
        
        logger.info(f"Loading Whisper model: {model_name}")
        
        # Determine device using centralized utility
        from ..utils.device_utils import configure_device_for_model, log_device_usage
        device, _ = configure_device_for_model('whisper', device)
        log_device_usage(f"Whisper {model_name}", device)
            
        # Map model names to HuggingFace IDs
        model_map = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base", 
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3"
        }
        
        model_id = model_map.get(model_name, f"openai/whisper-{model_name}")
        
        # Load model and processor
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperModel.from_pretrained(model_id).to(device)
        model.eval()
        
        logger.info(f"Whisper model loaded successfully on {device}")
        return model, processor
        
    except ImportError:
        logger.error("Whisper model requires transformers and openai-whisper libraries")
        raise ImportError("Install with: pip install transformers openai-whisper")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise


def prepare_audio_for_whisper(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    duration: Optional[float] = None,
    pad_or_trim: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and prepare audio for Whisper processing.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (Whisper uses 16kHz)
        duration: Maximum duration in seconds (None for full audio)
        pad_or_trim: Whether to pad short audio or trim long audio
        
    Returns:
        audio: Audio waveform array
        sr: Sample rate
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Handle duration
    if duration is not None:
        target_length = int(duration * sample_rate)
        
        if pad_or_trim:
            if len(audio) > target_length:
                # Trim
                audio = audio[:target_length]
            elif len(audio) < target_length:
                # Pad with zeros
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
    
    return audio, sr


def get_whisper_embeddings(
    audio_paths: List[str],
    model_name: str = "large-v2",
    layer: str = "encoder_last",
    batch_size: int = 8,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """
    Extract audio embeddings using Whisper encoder.
    
    Args:
        audio_paths: List of paths to audio files
        model_name: Whisper model variant
        layer: Which layer to extract embeddings from ('encoder_last', 'encoder_avg')
        batch_size: Batch size for processing
        cache_dir: Directory for caching embeddings
        device: Device for computation
        progress_callback: Callback function for progress updates
        
    Returns:
        embeddings: Audio embeddings [n_samples, embedding_dim]
    """
    # Generate cache key if caching is enabled
    cache_key = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Create hash from audio paths and model config
        hash_data = {
            'audio_paths': sorted(audio_paths),
            'model_name': model_name,
            'layer': layer
        }
        hash_str = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()
        cache_key = os.path.join(cache_dir, f"whisper_embeddings_{hash_str}.npy")
        
        if os.path.exists(cache_key):
            logger.info(f"Loading cached Whisper embeddings from {cache_key}")
            return np.load(cache_key)
    
    # Load model
    model, processor = load_whisper_model(model_name, device)
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_embeddings = []
        
        for audio_path in batch_paths:
            try:
                # Load and prepare audio
                audio, sr = prepare_audio_for_whisper(audio_path)
                
                # Process audio
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Extract embeddings
                with torch.no_grad():
                    outputs = model.encoder(**inputs)
                    
                    if layer == "encoder_last":
                        # Use last hidden state
                        hidden_states = outputs.last_hidden_state
                    elif layer == "encoder_avg":
                        # Average all encoder layers
                        hidden_states = torch.stack(outputs.hidden_states).mean(dim=0)
                    else:
                        raise ValueError(f"Unknown layer: {layer}")
                    
                    # Average over time dimension to get fixed-size embedding
                    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                    
                batch_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing audio {audio_path}: {e}")
                # Use zero embedding as fallback
                embedding_dim = model.config.d_model
                batch_embeddings.append(np.zeros(embedding_dim))
        
        all_embeddings.extend(batch_embeddings)
        
        # Progress callback
        if progress_callback:
            progress_callback(min(i + batch_size, len(audio_paths)), len(audio_paths))
    
    embeddings = np.array(all_embeddings)
    
    # Cache embeddings
    if cache_key:
        logger.info(f"Caching Whisper embeddings to {cache_key}")
        np.save(cache_key, embeddings)
    
    return embeddings


def generate_audio_dataset_hash(
    audio_paths: List[str],
    labels: np.ndarray,
    model_name: str,
    dataset_name: Optional[str] = None
) -> str:
    """
    Generate a unique hash for an audio dataset configuration.
    
    Args:
        audio_paths: List of audio file paths
        labels: Labels array
        model_name: Whisper model name
        dataset_name: Optional dataset name
        
    Returns:
        hash_str: Unique hash string
    """
    # Get dataset properties
    n_samples = len(audio_paths)
    n_classes = len(np.unique(labels))
    
    # Create hash data
    hash_data = {
        'n_samples': n_samples,
        'n_classes': n_classes,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'first_10_paths': sorted(audio_paths[:10]),  # Sample of paths
        'label_distribution': np.bincount(labels).tolist()
    }
    
    # Generate hash
    hash_str = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()
    
    return hash_str


def load_clap_model(version: str = "2023", use_cuda: Optional[bool] = None) -> Any:
    """
    Load CLAP model for audio embedding extraction.
    
    Args:
        version: CLAP model version ('2022', '2023', or 'clapcap')
        use_cuda: Whether to use CUDA (auto-detect if None)
        
    Returns:
        model: CLAP model
    """
    try:
        from msclap import CLAP
        
        logger.info(f"Loading CLAP model version: {version}")
        
        # Auto-detect device if not specified
        from ..utils.device_utils import detect_optimal_device, log_device_usage
        if use_cuda is None:
            device = detect_optimal_device()
            use_cuda = (device == 'cuda')
            log_device_usage(f"CLAP {version}", device)
            
        model = CLAP(version=version, use_cuda=use_cuda)
        
        logger.info("CLAP model loaded successfully")
        return model
        
    except ImportError:
        logger.error("msclap library not found. Please install with: pip install msclap")
        raise ImportError("msclap library required. Install with: pip install msclap")
    except Exception as e:
        logger.error(f"Failed to load CLAP model: {e}")
        raise


def prepare_audio_for_clap(
    audio_path: Union[str, Path],
    sample_rate: int = 48000,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load and prepare audio for CLAP processing.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (CLAP uses 48kHz by default)
        duration: Maximum duration in seconds (None for full audio)
        
    Returns:
        audio: Audio waveform array
        sr: Sample rate
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Handle duration
    if duration is not None:
        target_length = int(duration * sample_rate)
        if len(audio) > target_length:
            # Trim from center
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
    
    return audio, sr


def get_clap_embeddings(
    audio_paths: List[Union[str, Path]],
    clap_model: Optional[Any] = None,
    version: str = "2023",
    use_cuda: Optional[bool] = None,
    batch_size: int = 8,
    cache_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLAP embeddings from audio files with caching support.
    
    Args:
        audio_paths: List of paths to audio files
        clap_model: Pre-loaded CLAP model (optional, will load if None)
        version: CLAP model version if loading model
        use_cuda: Whether to use CUDA
        batch_size: Batch size for processing
        cache_dir: Directory to cache embeddings (None disables caching)
        dataset_name: Name for cache identification
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        embeddings: Audio embeddings [n_samples, embedding_dim]
        labels: Dummy labels (for API compatibility)
    """
    # Convert paths to strings
    audio_paths = [str(path) for path in audio_paths]
    
    # Setup caching
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename
        if dataset_name is None:
            dataset_name = "unknown"
            
        dataset_hash = generate_audio_dataset_hash(
            audio_paths, list(range(len(audio_paths))), 
            f"clap_{version}", dataset_name
        )
        cache_file = cache_dir / f"clap_{version}_{dataset_name}_{dataset_hash}.npz"
        
        # Try to load from cache
        if not force_recompute and cache_file.exists():
            try:
                logger.info(f"Loading CLAP embeddings from cache: {cache_file}")
                cached_data = np.load(cache_file)
                embeddings = cached_data['embeddings']
                logger.info(f"Loaded {len(embeddings)} cached CLAP embeddings")
                return embeddings, np.arange(len(embeddings))
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
    
    # Load model if not provided
    if clap_model is None:
        clap_model = load_clap_model(version=version, use_cuda=use_cuda)
    
    logger.info(f"Extracting CLAP embeddings from {len(audio_paths)} audio files...")
    
    # Extract embeddings in batches
    all_embeddings = []
    
    try:
        import torch
        
        with torch.no_grad():
            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")
                
                # Get audio embeddings using msclap
                batch_embeddings = clap_model.get_audio_embeddings(batch_paths)
                
                # Ensure embeddings are detached and on CPU
                if torch.is_tensor(batch_embeddings):
                    batch_embeddings = batch_embeddings.detach().cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Extracted CLAP embeddings: {embeddings.shape}")
        
        # Cache embeddings if requested
        if cache_dir is not None:
            try:
                logger.info(f"Caching CLAP embeddings to: {cache_file}")
                np.savez_compressed(
                    cache_file,
                    embeddings=embeddings,
                    audio_paths=audio_paths,
                    version=version,
                    batch_size=batch_size
                )
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
        
        return embeddings, np.arange(len(embeddings))
        
    except Exception as e:
        logger.error(f"Failed to extract CLAP embeddings: {e}")
        raise