"""
Audio dataset loaders for ESC-50, UrbanSound8K, and RAVDESS.
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import requests
import zipfile
import tarfile
import json
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AudioDataset:
    """Base class for audio datasets."""
    
    def __init__(self, root_dir: str, download: bool = True):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        if download and not self.check_exists():
            self.download()
            
    def check_exists(self) -> bool:
        """Check if dataset already exists."""
        raise NotImplementedError
        
    def download(self):
        """Download dataset."""
        raise NotImplementedError
        
    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        raise NotImplementedError
        
    def get_samples(self, split: str = 'all') -> Tuple[List[str], List[int], List[str]]:
        """Get audio paths, labels, and class names."""
        raise NotImplementedError
        
    def create_few_shot_split(
        self, 
        k_shot: int = 5,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Create few-shot learning splits.
        
        Args:
            k_shot: Number of examples per class for training
            val_size: Validation set size (fraction)
            test_size: Test set size (fraction)
            random_state: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        paths, labels, class_names = self.get_samples()
        
        # Check if dataset is empty
        if len(paths) == 0:
            raise ValueError(f"No audio samples found in dataset. Please check the dataset download and directory structure.")
        
        # Convert to numpy arrays
        paths = np.array(paths)
        labels = np.array(labels)
        
        logger.info(f"Dataset contains {len(paths)} samples across {len(np.unique(labels))} classes")
        
        # Split into train+val and test
        indices = np.arange(len(paths))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=random_state
        )
        
        # Further split train+val into train and val
        train_val_labels = labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size/(1-test_size), 
            stratify=train_val_labels, random_state=random_state
        )
        
        # Select k-shot samples per class for training
        train_paths_full = paths[train_idx]
        train_labels_full = labels[train_idx]
        
        train_paths_kshot = []
        train_labels_kshot = []
        
        for class_idx in np.unique(labels):
            class_mask = train_labels_full == class_idx
            class_paths = train_paths_full[class_mask]
            class_labels = train_labels_full[class_mask]
            
            # Select k samples
            n_samples = min(k_shot, len(class_paths))
            selected_idx = np.random.RandomState(random_state).choice(
                len(class_paths), n_samples, replace=False
            )
            
            train_paths_kshot.extend(class_paths[selected_idx])
            train_labels_kshot.extend(class_labels[selected_idx])
            
        return {
            'train': (train_paths_kshot, train_labels_kshot),
            'val': (paths[val_idx].tolist(), labels[val_idx].tolist()),
            'test': (paths[test_idx].tolist(), labels[test_idx].tolist()),
            'class_names': class_names
        }


class ESC50Dataset(AudioDataset):
    """ESC-50: Environmental Sound Classification dataset."""
    
    URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    
    def check_exists(self) -> bool:
        audio_dir = self.root_dir / "ESC-50-master" / "audio"
        meta_file = self.root_dir / "ESC-50-master" / "meta" / "esc50.csv"
        return audio_dir.exists() and meta_file.exists()
        
    def download(self):
        """Download ESC-50 dataset."""
        logger.info("Downloading ESC-50 dataset...")
        
        zip_path = self.root_dir / "esc50.zip"
        
        # Download
        response = requests.get(self.URL, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Extract
        logger.info("Extracting ESC-50...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
            
        # Clean up
        zip_path.unlink()
        logger.info("ESC-50 download complete")
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ESC-50 metadata."""
        meta_path = self.root_dir / "ESC-50-master" / "meta" / "esc50.csv"
        return pd.read_csv(meta_path)
        
    def get_samples(self, split: str = 'all') -> Tuple[List[str], List[int], List[str]]:
        """Get ESC-50 samples."""
        metadata = self.load_metadata()
        audio_dir = self.root_dir / "ESC-50-master" / "audio"
        
        # Get unique categories
        categories = sorted(metadata['category'].unique())
        category_to_label = {cat: i for i, cat in enumerate(categories)}
        
        paths = []
        labels = []
        
        for _, row in metadata.iterrows():
            if split != 'all' and row['fold'] != int(split):
                continue
                
            audio_path = audio_dir / row['filename']
            if audio_path.exists():
                paths.append(str(audio_path))
                labels.append(category_to_label[row['category']])
                
        return paths, labels, categories


class UrbanSound8KDataset(AudioDataset):
    """UrbanSound8K dataset with soundata download support."""
    
    def check_exists(self) -> bool:
        # First try to check using soundata API
        try:
            import soundata
            dataset = soundata.initialize('urbansound8k', data_home=str(self.root_dir))
            # Try to validate the dataset - this checks if it's properly installed
            dataset.validate()
            
            # Additional check: make sure we can actually load clips and they have audio files
            clips = dataset.load_clips()
            if len(clips) == 0:
                logger.debug("soundata validation passed but no clips found")
                return False
                
            # Check if at least some audio files exist
            valid_audio_count = 0
            for clip_id, clip in list(clips.items())[:10]:  # Check first 10 clips
                try:
                    audio_path = clip.get_path('audio')
                    if audio_path and Path(audio_path).exists():
                        valid_audio_count += 1
                except:
                    continue
            
            # If we have some valid audio files, consider it existing
            if valid_audio_count > 0:
                logger.debug(f"soundata dataset exists with {valid_audio_count}/10 sample audio files")
                return True
            else:
                logger.debug("soundata validation passed but no audio files found")
                return False
                
        except ImportError:
            logger.debug("soundata not available, checking for manual download")
        except Exception as e:
            logger.debug(f"soundata check failed: {e}, checking for manual download")
        
        # Fallback to checking file structure
        soundata_exists = (
            (self.root_dir / "UrbanSound8K").exists() and
            (self.root_dir / "UrbanSound8K" / "audio").exists()
        )
        manual_exists = (
            (self.root_dir / "UrbanSound8K" / "audio").exists() and 
            (self.root_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").exists()
        )
        return soundata_exists or manual_exists
        
    def download(self):
        """Download UrbanSound8K using soundata library."""
        try:
            import soundata
        except ImportError:
            logger.error("soundata library is required for UrbanSound8K download. Install with: pip install soundata")
            raise ImportError("Please install soundata: pip install soundata")
        
        logger.info("Downloading UrbanSound8K dataset using soundata...")
        
        # Initialize soundata dataset
        dataset = soundata.initialize('urbansound8k', data_home=str(self.root_dir))
        
        # Download the dataset
        dataset.download()
        
        logger.info("UrbanSound8K download complete")
        
    def load_metadata(self) -> pd.DataFrame:
        """Load UrbanSound8K metadata using soundata API or CSV fallback."""
        # First try to use soundata API
        try:
            import soundata
            dataset = soundata.initialize('urbansound8k', data_home=str(self.root_dir))
            
            # Make sure dataset is validated first
            dataset.validate()
            clips = dataset.load_clips()
            
            # Convert soundata clips to pandas DataFrame
            metadata_records = []
            for clip_id, clip in clips.items():
                # Use clip attributes directly from soundata
                metadata_records.append({
                    'slice_file_name': clip.slice_file_name,
                    'fold': clip.fold,
                    'classID': clip.class_id,
                    'class': clip.class_label,
                    'fsID': clip.freesound_id
                })
            
            logger.info(f"Loaded UrbanSound8K metadata using soundata API: {len(metadata_records)} clips")
            return pd.DataFrame(metadata_records)
            
        except ImportError:
            logger.warning("soundata not available, falling back to CSV files")
        except Exception as e:
            logger.warning(f"Failed to load metadata using soundata API: {e}, falling back to CSV files")
        
        # Fallback to CSV files for manual downloads
        possible_paths = [
            self.root_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv",  # standard structure
            self.root_dir / "urbansound8k" / "UrbanSound8K.csv"  # alternate structure
        ]
        
        for meta_path in possible_paths:
            if meta_path.exists():
                logger.info(f"Loading UrbanSound8K metadata from CSV: {meta_path}")
                return pd.read_csv(meta_path)
        
        raise FileNotFoundError(f"UrbanSound8K metadata not found using soundata API or CSV files in: {possible_paths}")
        
    def get_samples(self, split: str = 'all') -> Tuple[List[str], List[int], List[str]]:
        """Get UrbanSound8K samples."""
        # First try to use soundata API for both metadata and audio paths
        try:
            import soundata
            dataset = soundata.initialize('urbansound8k', data_home=str(self.root_dir))
            
            # Make sure dataset is validated first
            dataset.validate()
            clips = dataset.load_clips()
            
            # UrbanSound8K class names in order
            class_names = [
                'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                'siren', 'street_music'
            ]
            
            paths = []
            labels = []
            missing_files = 0
            
            for clip_id, clip in clips.items():
                # Filter by fold if specified
                if split != 'all' and clip.fold != int(split):
                    continue
                
                # Get audio path from soundata using get_path
                try:
                    audio_path = clip.get_path('audio')
                    if audio_path and Path(audio_path).exists():
                        paths.append(str(audio_path))
                        # Use class_id directly
                        labels.append(clip.class_id)
                    else:
                        missing_files += 1
                except Exception as e:
                    missing_files += 1
                    continue
            
            if len(paths) > 0:
                logger.info(f"Loaded UrbanSound8K samples using soundata API: {len(paths)} samples")
                if missing_files > 0:
                    logger.warning(f"Some audio files were missing: {missing_files} files not found")
                return paths, labels, class_names
            else:
                logger.warning(f"No valid audio files found using soundata API (missing: {missing_files}), falling back to CSV-based loading")
                # Force fallback to CSV
                raise Exception("No valid audio files found")
            
        except ImportError:
            logger.warning("soundata not available, falling back to CSV-based loading")
        except Exception as e:
            logger.warning(f"Failed to load samples using soundata API: {e}, falling back to CSV-based loading")
        
        # Fallback to CSV-based loading
        metadata = self.load_metadata()
        
        # Try soundata structure first, then fall back to manual structure
        possible_audio_bases = [
            self.root_dir / "UrbanSound8K" / "audio",    # standard structure
            self.root_dir / "urbansound8k" / "audio"     # alternate structure
        ]
        
        audio_base = None
        for candidate_base in possible_audio_bases:
            if candidate_base.exists():
                audio_base = candidate_base
                break
        
        if audio_base is None:
            raise FileNotFoundError(f"UrbanSound8K audio directory not found in any of: {possible_audio_bases}")
        
        # Class names
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]
        
        paths = []
        labels = []
        
        for _, row in metadata.iterrows():
            if split != 'all' and row['fold'] != int(split):
                continue
                
            fold_dir = f"fold{row['fold']}"
            audio_path = audio_base / fold_dir / row['slice_file_name']
            
            if audio_path.exists():
                paths.append(str(audio_path))
                labels.append(row['classID'])
                
        return paths, labels, class_names


class RAVDESSDataset(AudioDataset):
    """RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song."""
    
    URL_BASE = "https://zenodo.org/record/1188976/files/"
    FILES = [
        "Audio_Speech_Actors_01-24.zip",
        "Audio_Song_Actors_01-24.zip"
    ]
    
    # Emotion mapping
    EMOTIONS = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }
    
    def check_exists(self) -> bool:
        # Check multiple possible directory structures
        possible_dirs = [
            self.root_dir / "RAVDESS" / "Audio_Speech_Actors_01-24",  # Original expected structure
            self.root_dir / "RAVDESS"  # Direct extraction structure
        ]
        
        for audio_dir in possible_dirs:
            if audio_dir.exists() and len(list(audio_dir.glob("Actor_*"))) >= 24:
                return True
        return False
        
    def download(self):
        """Download RAVDESS dataset."""
        logger.info("Downloading RAVDESS dataset...")
        
        ravdess_dir = self.root_dir / "RAVDESS"
        ravdess_dir.mkdir(exist_ok=True)
        
        for filename in self.FILES:
            if "Speech" in filename and (ravdess_dir / "Audio_Speech_Actors_01-24").exists():
                continue
            if "Song" in filename and (ravdess_dir / "Audio_Song_Actors_01-24").exists():
                continue
                
            logger.info(f"Downloading {filename}...")
            url = self.URL_BASE + filename
            zip_path = ravdess_dir / filename
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Extract
            logger.info(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ravdess_dir)
                # List extracted contents for debugging
                logger.debug(f"Extracted contents: {zip_ref.namelist()[:5]}...")  # Show first 5 files
                
            zip_path.unlink()
            
        logger.info("RAVDESS download complete")
        
    def parse_filename(self, filename) -> Dict[str, int]:
        """Parse RAVDESS filename to extract metadata."""
        # Format: 03-01-06-01-02-01-12.wav
        # Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
        if hasattr(filename, 'stem'):
            # Path object
            parts = filename.stem.split('-')
        else:
            # String
            parts = Path(filename).stem.split('-')
        
        return {
            'modality': int(parts[0]),      # 1=full-AV, 2=video-only, 3=audio-only
            'vocal_channel': int(parts[1]),  # 1=speech, 2=song
            'emotion': int(parts[2]),        # 1-8 (see EMOTIONS dict)
            'intensity': int(parts[3]),      # 1=normal, 2=strong
            'statement': int(parts[4]),      # 1="Kids are talking", 2="Dogs are sitting"
            'repetition': int(parts[5]),     # 1=1st rep, 2=2nd rep
            'actor': int(parts[6])           # 1-24 (odd=male, even=female)
        }
        
    def get_samples(self, split: str = 'all', speech_only: bool = True) -> Tuple[List[str], List[int], List[str]]:
        """
        Get RAVDESS samples.
        
        Args:
            split: Data split ('all' or specific fold)
            speech_only: If True, only include speech (not song)
        """
        # Find the correct audio directory
        audio_dir = None
        possible_dirs = [
            self.root_dir / "RAVDESS" / "Audio_Speech_Actors_01-24",  # Original expected structure
            self.root_dir / "RAVDESS"  # Direct extraction structure
        ]
        
        for candidate_dir in possible_dirs:
            if candidate_dir.exists() and len(list(candidate_dir.glob("Actor_*"))) > 0:
                audio_dir = candidate_dir
                logger.info(f"Found RAVDESS audio directory: {audio_dir}")
                break
        
        if audio_dir is None:
            logger.error(f"No valid RAVDESS audio directory found. Checked: {possible_dirs}")
            # List what's actually in the RAVDESS directory for debugging
            ravdess_dir = self.root_dir / "RAVDESS"
            if ravdess_dir.exists():
                contents = list(ravdess_dir.iterdir())
                logger.debug(f"RAVDESS directory contents: {[str(c) for c in contents]}")
            return [], [], []
        
        paths = []
        labels = []
        
        logger.debug(f"Looking for RAVDESS audio files in: {audio_dir}")
        logger.debug(f"Directory exists: {audio_dir.exists()}")
        
        # Get all actor directories
        actor_dirs = sorted(audio_dir.glob("Actor_*"))
        logger.debug(f"Found {len(actor_dirs)} actor directories")
        
        if len(actor_dirs) == 0:
            logger.warning(f"No Actor_* directories found in {audio_dir}")
            # List all contents for debugging
            if audio_dir.exists():
                contents = list(audio_dir.iterdir())
                logger.debug(f"Directory contents: {[str(c) for c in contents]}")
            return [], [], []
        
        for actor_dir in actor_dirs:
            audio_files = sorted(actor_dir.glob("*.wav"))
            logger.debug(f"Actor {actor_dir.name}: found {len(audio_files)} audio files")
            
            for audio_file in audio_files:
                try:
                    metadata = self.parse_filename(audio_file)
                    
                    # Filter by vocal channel if requested
                    if speech_only and metadata['vocal_channel'] != 1:
                        continue
                        
                    # Skip neutral in song (doesn't exist)
                    if metadata['vocal_channel'] == 2 and metadata['emotion'] == 1:
                        continue
                        
                    paths.append(str(audio_file))
                    labels.append(metadata['emotion'] - 1)  # 0-indexed
                    
                except Exception as e:
                    logger.warning(f"Error parsing filename {audio_file}: {e}")
                    continue
                
        logger.info(f"RAVDESS: Found {len(paths)} audio samples")
        
        # Get emotion names
        emotion_names = [self.EMOTIONS[i] for i in sorted(self.EMOTIONS.keys())]
        
        return paths, labels, emotion_names


def download_all_datasets(root_dir: str = "./audio_datasets"):
    """Download all supported audio datasets."""
    logger.info("Downloading all audio datasets...")
    
    # ESC-50
    try:
        esc50 = ESC50Dataset(os.path.join(root_dir, "ESC50"), download=True)
        logger.info("ESC-50 download successful")
    except Exception as e:
        logger.error(f"ESC-50 download failed: {e}")
        
    # UrbanSound8K (will show manual download message)
    try:
        urbansound = UrbanSound8KDataset(os.path.join(root_dir, "UrbanSound8K"), download=True)
    except RuntimeError:
        pass  # Expected - requires manual download
        
    # RAVDESS
    try:
        ravdess = RAVDESSDataset(os.path.join(root_dir, "RAVDESS"), download=True)
        logger.info("RAVDESS download successful")
    except Exception as e:
        logger.error(f"RAVDESS download failed: {e}")


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load ESC-50 with few-shot split
    esc50 = ESC50Dataset("./test_audio_data/ESC50", download=True)
    splits = esc50.create_few_shot_split(k_shot=5)
    
    print(f"ESC-50 Few-shot splits:")
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val: {len(splits['val'][0])} samples")
    print(f"  Test: {len(splits['test'][0])} samples")
    print(f"  Classes: {len(splits['class_names'])} - {', '.join(splits['class_names'][:5])}...")