"""
Audio processing and visualization utilities.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with various options.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (None to use native rate)
        mono: Convert to mono
        offset: Start time in seconds
        duration: Duration to load in seconds
        
    Returns:
        audio: Audio signal
        sample_rate: Sample rate
    """
    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio from {audio_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    audio_path: Union[str, Path],
    sr: int,
    subtype: Optional[str] = None
) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio signal
        audio_path: Output path
        sr: Sample rate
        subtype: Audio subtype (e.g., 'PCM_16', 'FLOAT')
    """
    try:
        sf.write(audio_path, audio, sr, subtype=subtype)
    except Exception as e:
        logger.error(f"Error saving audio to {audio_path}: {e}")
        raise


def create_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    db_scale: bool = True
) -> np.ndarray:
    """
    Create spectrogram from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands (None for regular spectrogram)
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Power for magnitude (1 for magnitude, 2 for power)
        db_scale: Convert to dB scale
        
    Returns:
        spectrogram: Spectrogram array
    """
    if n_mels is not None:
        # Mel spectrogram
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power
        )
    else:
        # Regular spectrogram
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** power
    
    if db_scale:
        spec = librosa.power_to_db(spec, ref=np.max)
    
    return spec


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    ax: Optional[plt.Axes] = None,
    title: str = "Waveform",
    color: str = "blue",
    alpha: float = 0.8
) -> plt.Axes:
    """
    Plot audio waveform.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        color: Waveform color
        alpha: Transparency
        
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, color=color, alpha=alpha, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_spectrogram(
    spec: np.ndarray,
    sr: int,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
    title: str = "Spectrogram",
    cmap: str = "viridis",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None
) -> plt.Axes:
    """
    Plot spectrogram.
    
    Args:
        spec: Spectrogram array (in dB)
        sr: Sample rate
        hop_length: Hop length used for spectrogram
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap
        fmin: Minimum frequency to display
        fmax: Maximum frequency to display
        
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display spectrogram
    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap=cmap,
        fmin=fmin,
        fmax=fmax
    )
    
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return ax


def augment_audio(
    audio: np.ndarray,
    sr: int,
    augmentation: str,
    **kwargs
) -> np.ndarray:
    """
    Apply audio augmentation.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        augmentation: Type of augmentation
            - 'noise': Add Gaussian noise
            - 'pitch_shift': Shift pitch
            - 'time_stretch': Stretch time
            - 'volume': Change volume
        **kwargs: Augmentation-specific parameters
        
    Returns:
        augmented_audio: Augmented audio
    """
    if augmentation == 'noise':
        noise_factor = kwargs.get('noise_factor', 0.005)
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise
        
    elif augmentation == 'pitch_shift':
        n_steps = kwargs.get('n_steps', 2)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
    elif augmentation == 'time_stretch':
        rate = kwargs.get('rate', 1.1)
        return librosa.effects.time_stretch(audio, rate=rate)
        
    elif augmentation == 'volume':
        factor = kwargs.get('factor', 1.5)
        return audio * factor
        
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def extract_audio_features(
    audio: np.ndarray,
    sr: int,
    features: List[str] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract various audio features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        features: List of features to extract
            - 'mfcc': Mel-frequency cepstral coefficients
            - 'chroma': Chromagram
            - 'spectral_centroid': Spectral centroid
            - 'spectral_rolloff': Spectral rolloff
            - 'zero_crossing_rate': Zero crossing rate
            - 'rms': Root mean square energy
            
    Returns:
        feature_dict: Dictionary of extracted features
    """
    if features is None:
        features = ['mfcc', 'spectral_centroid', 'zero_crossing_rate', 'rms']
    
    feature_dict = {}
    
    for feature in features:
        if feature == 'mfcc':
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            feature_dict['mfcc_mean'] = np.mean(mfcc, axis=1)
            feature_dict['mfcc_std'] = np.std(mfcc, axis=1)
            
        elif feature == 'chroma':
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            feature_dict['chroma_mean'] = np.mean(chroma, axis=1)
            feature_dict['chroma_std'] = np.std(chroma, axis=1)
            
        elif feature == 'spectral_centroid':
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            feature_dict['spectral_centroid_mean'] = np.mean(centroid)
            feature_dict['spectral_centroid_std'] = np.std(centroid)
            
        elif feature == 'spectral_rolloff':
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            feature_dict['spectral_rolloff_mean'] = np.mean(rolloff)
            feature_dict['spectral_rolloff_std'] = np.std(rolloff)
            
        elif feature == 'zero_crossing_rate':
            zcr = librosa.feature.zero_crossing_rate(audio)
            feature_dict['zero_crossing_rate_mean'] = np.mean(zcr)
            feature_dict['zero_crossing_rate_std'] = np.std(zcr)
            
        elif feature == 'rms':
            rms = librosa.feature.rms(y=audio)
            feature_dict['rms_mean'] = np.mean(rms)
            feature_dict['rms_std'] = np.std(rms)
    
    return feature_dict


def normalize_audio(
    audio: np.ndarray,
    method: str = 'peak'
) -> np.ndarray:
    """
    Normalize audio signal.
    
    Args:
        audio: Audio signal
        method: Normalization method
            - 'peak': Normalize to peak value
            - 'rms': Normalize by RMS
            
    Returns:
        normalized_audio: Normalized audio
    """
    if method == 'peak':
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio
        
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio / rms
        return audio
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_synthetic_audio(
    frequency: float,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    waveform: str = 'sine'
) -> np.ndarray:
    """
    Create synthetic audio signal for testing.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate
        amplitude: Amplitude (0-1)
        waveform: Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
        
    Returns:
        audio: Synthetic audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if waveform == 'sine':
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
    elif waveform == 'square':
        audio = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == 'sawtooth':
        audio = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif waveform == 'triangle':
        audio = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - amplitude
    else:
        raise ValueError(f"Unknown waveform: {waveform}")
    
    return audio.astype(np.float32)


def create_audio_thumbnail(
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    duration: float = 5.0,
    sr: int = 16000
) -> None:
    """
    Create a short audio thumbnail for preview.
    
    Args:
        audio_path: Input audio path
        output_path: Output thumbnail path
        duration: Thumbnail duration in seconds
        sr: Sample rate for thumbnail
    """
    # Load first few seconds
    audio, _ = load_audio(audio_path, sr=sr, duration=duration)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Save thumbnail
    save_audio(audio, output_path, sr)