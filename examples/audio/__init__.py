"""
Audio classification examples using MARVIS with Whisper embeddings.
"""

from .audio_datasets import ESC50Dataset, RAVDESSDataset, UrbanSound8KDataset
from .marvis_tsne_audio_baseline import MarvisAudioTsneClassifier

__all__ = [
    "MarvisAudioTsneClassifier",
    "ESC50Dataset",
    "UrbanSound8KDataset",
    "RAVDESSDataset",
]
