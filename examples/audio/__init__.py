"""
Audio classification examples using MARVIS with Whisper embeddings.
"""

from .marvis_tsne_audio_baseline import MarvisAudioTsneClassifier
from .audio_datasets import ESC50Dataset, UrbanSound8KDataset, RAVDESSDataset

__all__ = [
    'MarvisAudioTsneClassifier',
    'ESC50Dataset',
    'UrbanSound8KDataset', 
    'RAVDESSDataset'
]