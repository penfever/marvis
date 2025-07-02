"""
Utilities for image loading, preprocessing, and dataset management for ImageNet classification.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ImageNetDataset(Dataset):
    """
    Custom dataset for loading ImageNet-style directory structure.
    
    Expected structure:
    dataset_path/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            import traceback
            error_msg = f"Failed to load image {image_path}: {e}. Using black image."
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_imagenet_subset(
    dataset_path: str, 
    num_classes: Optional[int] = None,
    samples_per_class: Optional[int] = None,
    min_samples_per_class: int = 10,
    valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'),
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load a subset of ImageNet-style dataset from directory structure.
    
    Args:
        dataset_path: Path to dataset root directory
        num_classes: Maximum number of classes to include (None for all)
        samples_per_class: Maximum samples per class (None for all)
        min_samples_per_class: Minimum samples required per class
        valid_extensions: Valid image file extensions
        random_seed: Random seed for reproducible sampling
        
    Returns:
        image_paths: List of paths to image files
        labels: List of integer labels (0-indexed)
        class_names: List of class names corresponding to label indices
    """
    np.random.seed(random_seed)
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Get all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering
    
    logger.info(f"Found {len(class_dirs)} potential class directories")
    
    # Filter classes by minimum sample count
    valid_classes = []
    for class_dir in class_dirs:
        image_files = []
        for ext in valid_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        if len(image_files) >= min_samples_per_class:
            valid_classes.append((class_dir.name, image_files))
    
    logger.info(f"Found {len(valid_classes)} classes with at least {min_samples_per_class} samples")
    
    # Limit number of classes if specified
    if num_classes is not None and num_classes < len(valid_classes):
        # Randomly sample classes to ensure diversity
        selected_indices = np.random.choice(len(valid_classes), num_classes, replace=False)
        valid_classes = [valid_classes[i] for i in sorted(selected_indices)]
        logger.info(f"Randomly selected {num_classes} classes")
    
    # Build image paths and labels
    image_paths = []
    labels = []
    class_names = []
    
    for label_idx, (class_name, image_files) in enumerate(valid_classes):
        class_names.append(class_name)
        
        # Convert to string paths
        class_image_paths = [str(p) for p in image_files]
        
        # Limit samples per class if specified
        if samples_per_class is not None and len(class_image_paths) > samples_per_class:
            selected_indices = np.random.choice(len(class_image_paths), samples_per_class, replace=False)
            class_image_paths = [class_image_paths[i] for i in selected_indices]
        
        # Add to overall lists
        image_paths.extend(class_image_paths)
        labels.extend([label_idx] * len(class_image_paths))
        
        logger.info(f"Class '{class_name}' (label {label_idx}): {len(class_image_paths)} samples")
    
    logger.info(f"Total dataset: {len(image_paths)} images across {len(class_names)} classes")
    
    return image_paths, labels, class_names


def split_dataset(
    image_paths: List[str], 
    labels: List[int], 
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split dataset into train/validation/test sets with stratification.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
    
    # First split: separate out test set
    temp_paths, test_paths, temp_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: separate train and validation from remaining data
    relative_val_size = val_size / (train_size + val_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_val_size,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    logger.info(f"Dataset split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_image_transforms(mode: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.
    
    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    if mode == 'train':
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    train_paths: List[str], train_labels: List[int],
    val_paths: List[str], val_labels: List[int],
    test_paths: List[str], test_labels: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for train/val/test sets.
    
    Args:
        train_paths, train_labels: Training data
        val_paths, val_labels: Validation data  
        test_paths, test_labels: Test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ImageNetDataset(
        train_paths, train_labels, 
        transform=get_image_transforms('train', image_size)
    )
    val_dataset = ImageNetDataset(
        val_paths, val_labels,
        transform=get_image_transforms('val', image_size)
    )
    test_dataset = ImageNetDataset(
        test_paths, test_labels,
        transform=get_image_transforms('test', image_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def extract_features_from_loader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from images using a pre-trained model.
    
    Args:
        model: Pre-trained model for feature extraction
        data_loader: DataLoader containing images
        device: Device to run inference on
        
    Returns:
        features: Extracted features as numpy array
        labels: Corresponding labels as numpy array
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_features.append(features)
            all_labels.append(labels)
    
    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels