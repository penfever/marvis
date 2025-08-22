#!/usr/bin/env python3
"""
Test script for biological dataset classification using BioClip2 and MARVIS t-SNE baselines.

This script evaluates models on FishNet, AwA2, and PlantDoc datasets with:
- BioClip2 embeddings + KNN baseline
- Qwen VL baseline
- MARVIS t-SNE with BioClip2 backend

Based on BioClip2 paper: https://arxiv.org/abs/2505.23883
Datasets: FishNet (habitat classification), AwA2 (trait prediction), PlantDoc (disease detection)
"""

import argparse
import logging
import os
import sys
import time
import json
import datetime
import random
import urllib.request
import zipfile
import shutil
import subprocess
import requests
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from marvis.utils.class_name_utils import get_semantic_class_names_or_fallback
from marvis.utils.vlm_prompting import validate_and_clean_class_names
from marvis.utils import set_seed

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.utils.json_utils import convert_for_json_serialization
from marvis.utils.device_utils import log_platform_info
from marvis.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring,
    MetricsLogger
)

from marvis.models.marvis_tsne import MarvisImageTsneClassifier
from examples.vision.qwen_vl_baseline import QwenVLBaseline
from examples.vision.image_baselines import DINOV2LinearProbe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenericEmbeddingExtractor:
    """Generic interface for extracting embeddings from images."""
    
    def extract_embeddings(self, image_paths: list) -> np.ndarray:
        """Extract embeddings for a list of images."""
        raise NotImplementedError
    
    def _extract_single_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding for a single image."""
        raise NotImplementedError


class BioClip2EmbeddingExtractor(GenericEmbeddingExtractor):
    """Extract embeddings using BioCLIP-2 model."""
    
    def __init__(self, model_name: str = "hf-hub:imageomics/bioclip-2", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        
    def load_model(self):
        """Load BioCLIP model using OpenCLIP."""
        try:
            import open_clip
            
            logger.info(f"Loading BioCLIP model: {self.model_name}")
            
            # Load BioCLIP model with OpenCLIP
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.model_name)
            
            # Move to appropriate device
            self.model = model.to(self.device)
            
            # Use validation preprocessing for inference
            self.preprocess = preprocess_val
            
            # Set to eval mode
            self.model.eval()
            
            logger.info(f"BioCLIP model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise RuntimeError(f"BioCLIP requires open_clip library: {e}. Install with: pip install open-clip-torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load BioCLIP model: {e}")
    
    def extract_embeddings(self, image_paths: list) -> np.ndarray:
        """Extract embeddings from images."""
        if self.model is None:
            self.load_model()
        
        embeddings = []
        logger.info(f"Extracting BioClip2 embeddings for {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            try:
                embedding = self._extract_single_embedding(image_path)
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                raise RuntimeError(f"Failed to extract embedding for {image_path}: {e}")
        
        return np.array(embeddings)
    
    def _extract_single_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from single image."""
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


class GenericKNNBaseline:
    """Generic KNN classifier baseline that works with any embedding extractor."""
    
    def __init__(self, embedding_extractor: GenericEmbeddingExtractor, 
                 n_neighbors: int = 5, metric: str = "cosine", standardize: bool = True):
        self.embedding_extractor = embedding_extractor
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.standardize = standardize
        
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        self.scaler = StandardScaler() if standardize else None
        
        self.train_embeddings = None
        self.train_labels = None
        self.class_names = None
        self.is_fitted = False
    
    def fit(self, train_paths: list, train_labels: list, class_names: list = None):
        """Fit the classifier."""
        logger.info(f"Fitting generic KNN classifier with {len(train_paths)} training samples")
        
        self.train_labels = np.array(train_labels)
        if class_names is None:
            # Use new utility to extract class names with semantic support
            unique_labels = np.unique(train_labels).tolist()
            self.class_names = get_semantic_class_names_or_fallback(
                labels=unique_labels,
                dataset_name=getattr(self, 'dataset_name', None)
            )
        else:
            self.class_names = class_names
        
        # Extract embeddings
        self.train_embeddings = self.embedding_extractor.extract_embeddings(train_paths)
        
        # Standardize if requested
        if self.scaler is not None:
            self.train_embeddings = self.scaler.fit_transform(self.train_embeddings)
        
        # Fit KNN
        self.knn.fit(self.train_embeddings, self.train_labels)
        self.is_fitted = True
        
        logger.info("Generic KNN classifier fitted successfully")
    
    def predict(self, test_paths: list) -> np.ndarray:
        """Predict labels for test images."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Extract test embeddings
        test_embeddings = self.embedding_extractor.extract_embeddings(test_paths)
        
        # Standardize if needed
        if self.scaler is not None:
            test_embeddings = self.scaler.transform(test_embeddings)
        
        # Predict
        predictions = self.knn.predict(test_embeddings)
        return predictions
    
    def evaluate(self, test_paths: list, test_labels: list) -> dict:
        """Evaluate classifier on test data."""
        start_time = time.time()
        predictions = self.predict(test_paths)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions,
            'true_labels': test_labels
        }


class BioClip2KNNBaseline(GenericKNNBaseline):
    """BioClip2 embeddings + KNN classifier baseline."""
    
    def __init__(self, n_neighbors: int = 5, metric: str = "cosine", standardize: bool = True, device: str = "auto"):
        embedding_extractor = BioClip2EmbeddingExtractor(device=device)
        super().__init__(embedding_extractor, n_neighbors, metric, standardize)


def download_and_prepare_awa2(data_dir: str = "./awa2_data") -> tuple:
    """Download and prepare AwA2 dataset."""
    data_dir = Path(data_dir)
    
    # Check if already prepared
    if (data_dir / "images").exists() and len(list((data_dir / "images").glob("*/*.jpg"))) > 1000:
        logger.info("AwA2 already prepared, loading existing data...")
        return load_existing_awa2(data_dir)
    
    logger.info("Downloading AwA2 dataset...")
    data_dir.mkdir(exist_ok=True)
    
    # Download URLs for AwA2 dataset components
    base_url = "https://cvml.ista.ac.at/AwA2"
    urls = {
        "AwA2-base.zip": f"{base_url}/AwA2-base.zip",
        "AwA2-data.zip": f"{base_url}/AwA2-data.zip"  # This is the 13GB image data
    }
    
    # Download and extract each component
    for filename, url in urls.items():
        zip_path = data_dir / filename
        if not zip_path.exists():
            logger.info(f"Downloading {filename} from {url}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                logger.info(f"Downloaded {filename}")
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename} from {url}: {e}")
        
        # Check if extraction is needed
        extracted_indicator = data_dir / filename.replace('.zip', '')  # e.g., "AwA2-base", "AwA2-data"
        if not extracted_indicator.exists():
            # Extract the zip file
            logger.info(f"Extracting {filename}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                logger.info(f"Extracted {filename}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract {filename}: {e}")
        else:
            logger.info(f"{filename} already extracted, skipping extraction...")
    
    # Organize the extracted data (only if not already organized)
    if not (data_dir / "images").exists() or len(list((data_dir / "images").glob("*/*.jpg"))) < 1000:
        organize_awa2_data(data_dir)
    else:
        logger.info("AwA2 data already organized, skipping organization...")
    
    return load_existing_awa2(data_dir)


def download_and_prepare_plantdoc(data_dir: str = "./plantdoc_data") -> tuple:
    """Download and prepare PlantDoc dataset."""
    data_dir = Path(data_dir)
    
    # Check if already prepared
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        logger.info("PlantDoc already prepared, loading existing data...")
        return load_existing_plantdoc(data_dir)
    
    logger.info("Downloading PlantDoc dataset from GitHub...")
    data_dir.mkdir(exist_ok=True)
    
    # Clone the GitHub repository
    repo_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
    repo_dir = data_dir / "PlantDoc-Dataset"
    
    try:
        if not repo_dir.exists():
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
            logger.info("Repository cloned successfully")
        
        # Copy the train and test directories to the main data directory
        train_src = repo_dir / "train"
        test_src = repo_dir / "test"
        train_dst = data_dir / "train"
        test_dst = data_dir / "test"
        
        if train_src.exists() and not train_dst.exists():
            shutil.copytree(train_src, train_dst)
            logger.info("Copied training data")
        
        if test_src.exists() and not test_dst.exists():
            shutil.copytree(test_src, test_dst)
            logger.info("Copied test data")
            
        # Clean up the repository directory if desired
        # shutil.rmtree(repo_dir)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone PlantDoc repository: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare PlantDoc dataset: {e}")
    
    return load_existing_plantdoc(data_dir)


def download_and_prepare_fishnet(data_dir: str = "./fishnet_data") -> tuple:
    """Download and prepare FishNet dataset."""
    data_dir = Path(data_dir)
    
    # Check if already prepared
    if (data_dir / "images").exists() and len(list((data_dir / "images").glob("*/*.jpg"))) > 1000:
        logger.info("FishNet already prepared, loading existing data...")
        return load_existing_fishnet(data_dir)
    
    logger.info("Downloading FishNet dataset...")
    data_dir.mkdir(exist_ok=True)
    
    # FishNet dataset Google Drive URL (from fishnet-2023.github.io)
    gdrive_url = "https://drive.google.com/file/d/1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3/view?usp=sharing"
    file_id = "1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3"
    
    # Download from Google Drive using gdown if available, otherwise provide instructions
    zip_path = data_dir / "fishnet_dataset.zip"
    
    try:
        # Try to import and use gdown for Google Drive downloads
        import gdown
        
        if not zip_path.exists():
            logger.info(f"Downloading FishNet dataset from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False)
            logger.info("FishNet dataset downloaded")
        
        # Check if extraction is needed (look for any extracted directory besides known ones)
        extracted_dirs = [d for d in data_dir.iterdir() 
                         if d.is_dir() and d.name not in ["images", "__pycache__", "train", "test"]]
        if not extracted_dirs:
            # Extract the dataset
            logger.info("Extracting FishNet dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("FishNet dataset extracted")
        else:
            logger.info("FishNet dataset already extracted, skipping extraction...")
        
        # Organize the extracted data (only if not already organized)
        if not (data_dir / "images").exists() or len(list((data_dir / "images").glob("*/*.jpg"))) < 1000:
            organize_fishnet_data(data_dir)
        else:
            logger.info("FishNet data already organized, skipping organization...")
        
    except ImportError:
        raise RuntimeError(
            "gdown package is required to download FishNet dataset. Install with: pip install gdown\n"
            f"Alternatively, manually download from: {gdrive_url}\n"
            f"And extract to: {data_dir}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download or extract FishNet dataset: {e}")
    
    return load_existing_fishnet(data_dir)


def organize_fishnet_data(data_dir: Path):
    """Organize extracted FishNet data into train/test structure."""
    logger.info("Organizing FishNet data...")
    
    # Create images directory structure
    images_dir = data_dir / "images"
    train_dir = images_dir / "train"
    test_dir = images_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the FishNet data directory
    fishnet_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name not in ["images", "__pycache__", "train", "test"]:
            fishnet_dirs.append(item)
    
    if not fishnet_dirs:
        raise RuntimeError("No FishNet data directories found after extraction")
    
    # Assume the first directory contains the fish images
    source_dir = fishnet_dirs[0]
    logger.info(f"Processing FishNet data from: {source_dir}")
    
    # Look for subdirectories that represent fish species
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        # If no subdirectories, look for images directly
        image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
        if image_files:
            logger.warning("FishNet data appears to be flat structure. Creating single class.")
            # Create a single class directory
            fish_class_train = train_dir / "fish"
            fish_class_test = test_dir / "fish"
            fish_class_train.mkdir(exist_ok=True)
            fish_class_test.mkdir(exist_ok=True)
            
            # Split 80/20 train/test
            set_seed(42)  # Ensure reproducible splits
            random.shuffle(image_files)
            split_idx = int(len(image_files) * 0.8)
            
            for idx, img_path in enumerate(image_files):
                if idx < split_idx:
                    shutil.copy2(img_path, fish_class_train / img_path.name)
                else:
                    shutil.copy2(img_path, fish_class_test / img_path.name)
        else:
            raise RuntimeError("No image files found in FishNet data")
    else:
        # Process each fish species directory
        logger.info(f"Found {len(class_dirs)} fish species classes")
        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
                
            class_name = validate_and_clean_class_names([class_dir.name])[0]
            train_class_dir = train_dir / class_name
            test_class_dir = test_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)
            
            # Collect all images from this class
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            if image_files:
                # Split 80/20 train/test
                set_seed(42)  # Ensure reproducible splits
                random.shuffle(image_files)
                split_idx = int(len(image_files) * 0.8)
                
                for idx, img_path in enumerate(image_files):
                    if idx < split_idx:
                        shutil.copy2(img_path, train_class_dir / img_path.name)
                    else:
                        shutil.copy2(img_path, test_class_dir / img_path.name)
    
    logger.info("FishNet data organized successfully")


def organize_awa2_data(data_dir: Path):
    """Organize extracted AwA2 data into train/test structure."""
    logger.info("Organizing AwA2 data...")
    
    # AwA2 dataset structure needs to be organized
    # Look for JPEGImages directory from the extraction
    jpeg_dir = None
    for root_dir in data_dir.iterdir():
        if root_dir.is_dir():
            potential_jpeg = root_dir / "JPEGImages"
            if potential_jpeg.exists():
                jpeg_dir = potential_jpeg
                break
    
    if not jpeg_dir:
        # Look for any Images directory
        for root_dir in data_dir.iterdir():
            if root_dir.is_dir() and "images" in root_dir.name.lower():
                jpeg_dir = root_dir
                break
    
    if jpeg_dir and jpeg_dir.exists():
        # Create organized structure
        images_dir = data_dir / "images"
        train_dir = images_dir / "train"  
        test_dir = images_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class directories from the original structure
        class_dirs = [d for d in jpeg_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = validate_and_clean_class_names([class_dir.name])[0]
            images = list(class_dir.glob("*.jpg"))
            
            if images:
                # Shuffle for random split
                set_seed(42)  # Ensure reproducible splits
                random.shuffle(images)
                # Split 80/20 train/test
                split_idx = int(0.8 * len(images))
                train_images = images[:split_idx]
                test_images = images[split_idx:]
                
                # Create class directories
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(exist_ok=True)
                test_class_dir.mkdir(exist_ok=True)
                
                # Copy or symlink images
                for img in train_images:
                    shutil.copy2(img, train_class_dir / img.name)
                for img in test_images:
                    shutil.copy2(img, test_class_dir / img.name)
        
        logger.info("AwA2 data organized successfully")
    else:
        logger.warning("Could not find AwA2 image directory in extracted data")


def load_existing_fishnet(data_dir: Path) -> tuple:
    """Load existing FishNet data."""
    return load_existing_dataset(data_dir, "FishNet")


def load_existing_awa2(data_dir: Path) -> tuple:
    """Load existing AwA2 data."""
    return load_existing_dataset(data_dir, "AwA2")




def load_existing_plantdoc(data_dir: Path) -> tuple:
    """Load existing PlantDoc data."""
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    class_names = []
    
    # PlantDoc has train/ and test/ directories directly
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    if not (train_dir.exists() and test_dir.exists()):
        raise RuntimeError(f"PlantDoc data not found in {data_dir}. Expected train/ and test/ directories.")
    
    # Get class names from train directory structure and normalize them
    raw_class_names = []
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            raw_class_names.append(class_dir.name)
    
    # Normalize all class names at once using the existing utility
    class_names = validate_and_clean_class_names(raw_class_names)
    
    # Create a mapping from original directory names to normalized class names
    dir_name_to_class_idx = {}
    for i, class_dir in enumerate(sorted(train_dir.iterdir())):
        if class_dir.is_dir():
            dir_name_to_class_idx[class_dir.name] = i
    
    # Load train and test data
    for split, split_dir, (paths_list, labels_list) in [
        ("train", train_dir, (train_paths, train_labels)), 
        ("test", test_dir, (test_paths, test_labels))
    ]:
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name in dir_name_to_class_idx:
                class_idx = dir_name_to_class_idx[class_dir.name]
                # Support multiple image formats
                for pattern in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in sorted(class_dir.glob(pattern)):
                        paths_list.append(str(img_path))
                        labels_list.append(class_idx)
    
    logger.info(f"Loaded PlantDoc: {len(train_paths)} train, {len(test_paths)} test images, {len(class_names)} classes")
    return train_paths, train_labels, test_paths, test_labels, class_names


def load_existing_dataset(data_dir: Path, dataset_name: str) -> tuple:
    """Generic function to load existing dataset with images/ structure."""
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    class_names = []
    
    images_dir = data_dir / "images"
    train_dir = images_dir / "train"
    test_dir = images_dir / "test"
    
    if not (train_dir.exists() and test_dir.exists()):
        raise RuntimeError(f"{dataset_name} data not found in {data_dir}. Expected images/train/ and images/test/ directories.")
    
    # Get class names from directory structure and normalize them
    raw_class_names = []
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            raw_class_names.append(class_dir.name)
    
    # Normalize all class names at once using the existing utility
    class_names = validate_and_clean_class_names(raw_class_names)
    
    # Create a mapping from original directory names to normalized class names
    dir_name_to_class_idx = {}
    for i, class_dir in enumerate(sorted(train_dir.iterdir())):
        if class_dir.is_dir():
            dir_name_to_class_idx[class_dir.name] = i
    
    # Load train and test data
    for split, split_dir, (paths_list, labels_list) in [
        ("train", train_dir, (train_paths, train_labels)), 
        ("test", test_dir, (test_paths, test_labels))
    ]:
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name in dir_name_to_class_idx:
                class_idx = dir_name_to_class_idx[class_dir.name]
                # Support multiple image formats
                for pattern in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in sorted(class_dir.glob(pattern)):
                        paths_list.append(str(img_path))
                        labels_list.append(class_idx)
    
    logger.info(f"Loaded {dataset_name}: {len(train_paths)} train, {len(test_paths)} test images, {len(class_names)} classes")
    return train_paths, train_labels, test_paths, test_labels, class_names


def test_single_dataset(dataset_name: str, args):
    """Test a single biological dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Debug: Check models argument
    logger.info(f"Models to evaluate: {getattr(args, 'models', 'MISSING')}")
    if not hasattr(args, 'models') or not args.models:
        logger.error("No models specified for evaluation! Using default models.")
        args.models = ["bioclip2_knn", "qwen_vl", "marvis_tsne_bioclip2"]
    
    use_wandb_logging = args.use_wandb and WANDB_AVAILABLE
    results = {}
    
    # Prepare dataset with isolated workspace directories using resource manager
    try:
        from marvis.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        dataset_workspace = resource_manager.get_dataset_workspace(dataset_name)
        dataset_dir = dataset_workspace / "downloads"
        dataset_dir.mkdir(exist_ok=True)
        dataset_dir = str(dataset_dir)
        logger.info(f"Using isolated dataset workspace: {dataset_dir}")
    except Exception as e:
        logger.warning(f"Could not use resource manager, falling back to data_dir: {e}")
        # Fallback to original method with dataset-specific subdirectories
        dataset_dir = os.path.join(args.data_dir, dataset_name)
    
    if dataset_name == "fishnet":
        train_paths, train_labels, test_paths, test_labels, class_names = download_and_prepare_fishnet(
            dataset_dir
        )
    elif dataset_name == "awa2":
        train_paths, train_labels, test_paths, test_labels, class_names = download_and_prepare_awa2(
            dataset_dir
        )
    elif dataset_name == "plantdoc":
        train_paths, train_labels, test_paths, test_labels, class_names = download_and_prepare_plantdoc(
            dataset_dir
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Apply balanced few-shot sampling if requested
    if args.balanced_few_shot:
        logger.info(f"Using balanced few-shot sampling with {args.num_few_shot_examples} examples per class")
        from sklearn.model_selection import train_test_split
        from collections import Counter
        import numpy as np
        
        # Group training data by class
        class_data = {}
        for path, label in zip(train_paths, train_labels):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(path)
        
        # Sample balanced few-shot examples
        sampled_paths = []
        sampled_labels = []
        for label, paths in class_data.items():
            n_samples = min(args.num_few_shot_examples, len(paths))
            sampled_indices = np.random.choice(len(paths), n_samples, replace=False)
            for idx in sampled_indices:
                sampled_paths.append(paths[idx])
                sampled_labels.append(label)
        
        train_paths = sampled_paths
        train_labels = sampled_labels
        logger.info(f"Sampled {len(train_paths)} training examples across {len(class_names)} classes")
    
    # Apply max test samples limit if specified
    if args.max_test_samples and len(test_paths) > args.max_test_samples:
        logger.info(f"Limiting test set to {args.max_test_samples} samples")
        test_paths = test_paths[:args.max_test_samples]
        test_labels = test_labels[:args.max_test_samples]
    
    # Test BioClip2 + KNN
    if 'bioclip2_knn' in args.models:
        logger.info("Testing BioClip2 + KNN...")
        try:
            classifier = BioClip2KNNBaseline(
                n_neighbors=args.knn_neighbors,
                metric="cosine",
                standardize=True,
                device=args.device
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(test_paths, test_labels)
            eval_results['training_time'] = training_time
            
            results['bioclip2_knn'] = eval_results
            logger.info(f"BioClip2 KNN completed: {eval_results['accuracy']:.4f} accuracy")
            
            if use_wandb_logging:
                log_results_to_wandb('bioclip2_knn', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"BioClip2 KNN failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['bioclip2_knn'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test Qwen VL
    if 'qwen_vl' in args.models:
        logger.info("Testing Qwen VL...")
        try:
            classifier = QwenVLBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name=args.vlm_model_id,
                backend=args.backend,
                use_semantic_names=args.use_semantic_names
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=f"{dataset_name.lower()}_biological"
            )
            eval_results['training_time'] = training_time
            
            results['qwen_vl'] = eval_results
            logger.info(f"Qwen VL completed: {eval_results['accuracy']:.4f} accuracy")
            
            if use_wandb_logging:
                log_results_to_wandb('qwen_vl', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"Qwen VL failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['qwen_vl'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test API VLM models
    api_models_to_test = []
    if 'openai_vlm' in args.models or 'api_vlm' in args.models:
        if args.openai_model:
            api_models_to_test.append(('openai', args.openai_model))
        else:
            logger.warning("openai_vlm model selected but no --openai_model specified")
    
    if 'gemini_vlm' in args.models or 'api_vlm' in args.models:
        if args.gemini_model:
            api_models_to_test.append(('gemini', args.gemini_model))
        else:
            logger.warning("gemini_vlm model selected but no --gemini_model specified")
    
    for api_backend, model_name in api_models_to_test:
        model_key = f"{api_backend}_vlm"
        logger.info(f"Testing {api_backend.title()} VLM ({model_name})...")
        try:
            # Import API baseline classes
            if api_backend == 'openai':
                from examples.vision.openai_vlm_baseline import OpenAIVLMBaseline as APIVLMBaseline
            else:  # gemini
                from examples.vision.gemini_vlm_baseline import GeminiVLMBaseline as APIVLMBaseline
            
            classifier = APIVLMBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name=model_name,
                use_semantic_names=args.use_semantic_names
            )
            
            start_time = time.time()
            classifier.fit(class_names=class_names)  # API models don't need training data but need class names
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=f"{dataset_name.lower()}_biological"
            )
            eval_results['training_time'] = training_time
            
            results[model_key] = eval_results
            logger.info(f"{api_backend.title()} VLM completed: {eval_results['accuracy']:.4f} accuracy")
            
            if use_wandb_logging:
                log_results_to_wandb(model_key, eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"{api_backend.title()} VLM failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results[model_key] = {'error': str(e), 'traceback': traceback.format_exc()}

    # Test MARVIS t-SNE with BioClip2 backend
    if 'marvis_tsne_bioclip2' in args.models:
        logger.info("Testing MARVIS t-SNE with BioClip2 backend...")
        try:
            classifier = MarvisImageTsneClassifier(
                modality="vision",
                embedding_backend="bioclip2",
                bioclip2_model=getattr(args, 'bioclip2_model', 'hf-hub:imageomics/bioclip-2'),
                embedding_size=512,  # Not used for BioClip2 but kept for compatibility
                tsne_perplexity=min(30.0, len(train_paths) / 4),
                tsne_max_iter=1000,
                vlm_model_id=args.vlm_model_id,
                vlm_backend=args.backend,
                use_3d=args.use_3d,
                use_knn_connections=args.use_knn_connections,
                nn_k=args.nn_k,
                max_vlm_image_size=1024,
                zoom_factor=args.zoom_factor,
                use_pca_backend=args.use_pca_backend,
                cache_dir=args.cache_dir,
                device=args.device,
                use_semantic_names=args.use_semantic_names,
                seed=getattr(args, 'seed', 42)
            )
            
            # Pass save_every_n parameter
            classifier.save_every_n = args.save_every_n
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, test_paths, class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels, 
                return_detailed=True,
                save_outputs=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None
            )
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            
            results['marvis_tsne_bioclip2'] = eval_results
            logger.info(f"MARVIS t-SNE BioClip2 completed: {eval_results['accuracy']:.4f} accuracy")
            
            if use_wandb_logging:
                log_results_to_wandb('marvis_tsne_bioclip2', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"MARVIS t-SNE BioClip2 failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['marvis_tsne_bioclip2'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    return results


def run_all_biological_tests(args):
    """Run biological dataset tests on multiple datasets."""
    all_results = {}
    
    # Log platform information
    from marvis.utils.device_utils import log_platform_info
    platform_info = log_platform_info(logger)
    
    for dataset_name in args.dataset:
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING {dataset_name.upper()} DATASET")
        logger.info(f"{'='*60}")
        
        # Initialize Weights & Biases for this dataset if requested
        gpu_monitor = None
        if args.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.wandb_name is None:
                    feature_suffix = ""
                    if args.use_3d:
                        feature_suffix += "_3d"
                    if args.use_knn_connections:
                        feature_suffix += f"_knn{args.nn_k}"
                    if args.use_pca_backend:
                        feature_suffix += "_pca"
                    run_name = f"{dataset_name}_bioclip2_{timestamp}{feature_suffix}"
                else:
                    run_name = f"{args.wandb_name}_{dataset_name}"
                
                gpu_monitor = init_wandb_with_gpu_monitoring(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=vars(args),
                    output_dir=args.output_dir,
                    enable_system_monitoring=True,
                    gpu_log_interval=30.0,
                    enable_detailed_gpu_logging=True
                )
                logger.info(f"Initialized Weights & Biases run: {run_name}")
        
        # Test single dataset
        dataset_results = test_single_dataset(dataset_name, args)
        all_results[dataset_name] = dataset_results
        
        # Save results for this dataset
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Create a temporary args object with single dataset for save_results
        import argparse
        temp_args = argparse.Namespace(**vars(args))
        temp_args.dataset = dataset_name  # For backward compatibility with save_results
        save_results(dataset_results, dataset_output_dir, temp_args)
        
        # Clean up wandb for this dataset
        if gpu_monitor is not None:
            cleanup_gpu_monitoring(gpu_monitor)
    
    return all_results


def log_results_to_wandb(model_name: str, eval_results: dict, args, class_names: list, dataset_name: str = None):
    """Log evaluation results to Weights & Biases."""
    dataset = dataset_name or getattr(args, 'dataset', 'unknown')
    if 'error' in eval_results:
        wandb.log({
            f"{model_name}/status": "failed",
            f"{model_name}/error": eval_results['error'],
            "model_name": model_name,
            "dataset": dataset,
            "backend": args.backend,
            "balanced_few_shot": args.balanced_few_shot,
            "num_few_shot_examples": args.num_few_shot_examples,
            "max_test_samples": args.max_test_samples
        })
        return
    
    metrics = {
        f"{model_name}/accuracy": eval_results['accuracy'],
        f"{model_name}/training_time": eval_results.get('training_time', 0),
        f"{model_name}/prediction_time": eval_results.get('prediction_time', 0),
        f"{model_name}/num_test_samples": eval_results.get('num_test_samples', 0),
        "model_name": model_name,
        "dataset": dataset,
        "num_classes": len(class_names),
        "backend": args.backend,
        "balanced_few_shot": args.balanced_few_shot,
        "num_few_shot_examples": args.num_few_shot_examples,
        "max_test_samples": args.max_test_samples
    }
    
    wandb.log(metrics)


def save_results(results: dict, output_dir: str, args):
    """Save test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{args.dataset}_bioclip2_test_results.json")
    with open(results_file, 'w') as f:
        json_results = convert_for_json_serialization(results)
        json.dump(json_results, f, indent=2)
    
    # Create summary
    summary_data = []
    for model, model_results in results.items():
        if 'error' in model_results:
            summary_data.append({
                'model': model,
                'status': 'ERROR',
                'error': model_results['error'],
                'accuracy': None,
                'training_time': None,
                'prediction_time': None
            })
        else:
            summary_data.append({
                'model': model,
                'status': 'SUCCESS',
                'error': None,
                'accuracy': model_results['accuracy'],
                'training_time': model_results['training_time'],
                'prediction_time': model_results['prediction_time']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"{args.dataset}_bioclip2_test_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"{args.dataset.upper()} BIOCLIP2 TEST RESULTS")
    logger.info("="*60)
    
    for _, row in summary_df.iterrows():
        if row['status'] == 'SUCCESS':
            logger.info(f"{row['model']:20s}: ✓ {row['accuracy']:.4f} accuracy "
                       f"(train: {row['training_time']:.1f}s, test: {row['prediction_time']:.1f}s)")
        else:
            logger.info(f"{row['model']:20s}: ✗ ERROR - {row['error']}")
    
    logger.info(f"\nDetailed results saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test biological datasets with BioClip2 and MARVIS baselines")
    
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["fishnet"],
        choices=["fishnet", "awa2", "plantdoc"],
        help="Biological datasets to test"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./biological_data",
        help="Directory for biological dataset data"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./biological_test_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bioclip2_knn", "qwen_vl", "marvis_tsne_bioclip2"],
        choices=["bioclip2_knn", "qwen_vl", "marvis_tsne_bioclip2", "openai_vlm", "gemini_vlm", "api_vlm"],
        help="Models to test"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Backend to use for VLM models (default: auto)"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use balanced few-shot sampling for training data"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=5,
        help="Number of few-shot examples per class (default: 5)"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="Maximum number of test samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    
    # Model-specific parameters
    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN classifier"
    )
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Vision Language Model to use"
    )
    parser.add_argument(
        "--bioclip2_model",
        type=str,
        default="hf-hub:imageomics/bioclip-2",
        help="BioClip2 model to use for embedding extraction"
    )
    
    # API model support arguments
    parser.add_argument(
        "--openai_model",
        type=str,
        help="OpenAI VLM model to use (e.g., gpt-4.1, gpt-4o, gpt-4o-mini)"
    )
    parser.add_argument(
        "--gemini_model", 
        type=str,
        help="Gemini VLM model to use (e.g., gemini-2.5-pro, gemini-2.5-flash)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode for compatible API models (default: True)"
    )
    parser.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode for API models"
    )
    
    # MARVIS t-SNE parameters (matching CIFAR script)
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations (default: 4.0)"
    )
    parser.add_argument(
        "--use_pca_backend",
        action="store_true",
        help="Use PCA instead of t-SNE for dimensionality reduction"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and VLM responses (default: True)"
    )
    parser.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and VLM responses"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors in embedding space (marvis_tsne only)"
    )
    parser.add_argument(
        "--nn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show when using KNN connections (default: 5)"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles (isometric, front, side, top views) instead of 2D (marvis_tsne only)"
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save visualizations every N predictions to reduce I/O overhead (default: 10)"
    )
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="bioclip2-biological-datasets",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    from marvis.utils import set_seed_with_args
    set_seed_with_args(args)
    
    logger.info(f"Starting biological dataset classification tests...")
    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {', '.join(args.dataset)}")
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Balanced few-shot: {args.balanced_few_shot}")
    if args.balanced_few_shot:
        logger.info(f"  Few-shot examples per class: {args.num_few_shot_examples}")
    if args.max_test_samples:
        logger.info(f"  Max test samples: {args.max_test_samples}")
    logger.info(f"  Use PCA: {getattr(args, 'use_pca_backend', False)}")
    logger.info(f"  3D t-SNE: {getattr(args, 'use_3d', False)}")
    logger.info(f"  KNN connections: {getattr(args, 'use_knn_connections', False)}")
    if getattr(args, 'use_knn_connections', False):
        logger.info(f"  KNN k: {getattr(args, 'nn_k', 5)}")
    
    # Run tests on all datasets
    all_results = run_all_biological_tests(args)
    
    # Print summary for all datasets
    logger.info(f"\n{'='*80}")
    logger.info("ALL DATASETS SUMMARY")
    logger.info(f"{'='*80}")
    
    total_experiments = 0
    successful_experiments = 0
    
    for dataset_name, dataset_results in all_results.items():
        logger.info(f"\n{dataset_name.upper()}:")
        for model_name, model_result in dataset_results.items():
            total_experiments += 1
            if 'error' not in model_result:
                successful_experiments += 1
                accuracy = model_result.get('accuracy', 0)
                logger.info(f"  {model_name:20s}: ✓ {accuracy:.4f} accuracy")
            else:
                logger.info(f"  {model_name:20s}: ✗ ERROR - {model_result['error']}")
    
    logger.info(f"\nOverall: {successful_experiments}/{total_experiments} experiments successful")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Biological dataset tests completed!")


if __name__ == "__main__":
    main()