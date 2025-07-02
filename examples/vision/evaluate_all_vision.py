#!/usr/bin/env python3
"""
Comprehensive test script for image classification datasets and models.

This script supports:
- CIFAR-10/CIFAR-100 (automatically downloaded)
- Custom ImageNet-style datasets 
- Multiple models: MARVIS t-SNE, Simple MARVIS, DINOV2 baselines, QwenVL
- Various configurations and visualization options

Examples:
    # Test CIFAR-10 with multiple models
    python evaluate_all_vision.py --dataset cifar10 --models marvis_tsne dinov2_linear qwen_vl
    
    # Test custom dataset
    python evaluate_all_vision.py --dataset custom --dataset_path /path/to/data --num_classes 10
    
    # Test with 3D visualizations and KNN connections
    python evaluate_all_vision.py --dataset cifar10 --use_3d --use_knn_connections --nn_k 10
"""

import argparse
import logging
import os
import sys
import time
import json
import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import centralized argument parser
from marvis.utils.evaluation_args import create_vision_evaluation_parser

from marvis.utils.json_utils import convert_for_json_serialization
from marvis.utils.device_utils import log_platform_info
from marvis.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring,
    MetricsLogger
)

from examples.vision.marvis_image_baseline import MarvisImageClassifier
from examples.vision.simple_marvis_baseline import SimpleMarvisImageClassifier
from marvis.models.marvis_tsne import MarvisImageTsneClassifier
from examples.vision.image_baselines import (
    DINOV2LinearProbe, DINOV2RandomForest
)
from examples.vision.qwen_vl_baseline import QwenVLBaseline
from examples.vision.image_utils import (
    ImageNetDataset, get_image_transforms, extract_features_from_loader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MacCompatibleDINOV2LinearProbe(DINOV2LinearProbe):
    """Mac-compatible version of DINOV2 Linear Probe."""
    
    def __init__(self, *args, **kwargs):
        # Force CPU on Mac for compatibility
        if sys.platform == "darwin":
            kwargs['device'] = 'cpu'
            logger.info("Mac detected: forcing CPU usage for DINOV2")
        super().__init__(*args, **kwargs)


def download_and_prepare_cifar(data_dir: str = "./cifar_data", dataset_type: str = "cifar10") -> tuple:
    """
    Download CIFAR-10 or CIFAR-100 and prepare it in ImageNet-style directory structure.
    
    Args:
        data_dir: Directory to store the dataset
        dataset_type: Either "cifar10" or "cifar100"
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # Get class names and dataset
    if dataset_type == "cifar10":
        # CIFAR-10 class names
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        min_images = 1000
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_type == "cifar100":
        # CIFAR-100 class names (fine labels)
        class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        min_images = 5000
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'cifar10' or 'cifar100'")
    
    # Check if already prepared
    if images_dir.exists() and len(list(images_dir.glob("*/*.png"))) > min_images:
        logger.info(f"{dataset_type.upper()} already prepared, loading existing data...")
        return load_existing_cifar(images_dir, class_names)
    
    logger.info(f"Downloading and preparing {dataset_type.upper()}...")
    
    # Download dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = dataset_class(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    
    # Create directory structure
    for split in ['train', 'test']:
        for class_name in class_names:
            (images_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Convert and save images
    train_paths, train_labels = save_cifar_images(
        train_dataset, images_dir / 'train', class_names, 'train'
    )
    test_paths, test_labels = save_cifar_images(
        test_dataset, images_dir / 'test', class_names, 'test'
    )
    
    logger.info(f"{dataset_type.upper()} prepared: {len(train_paths)} train, {len(test_paths)} test images")
    
    return train_paths, train_labels, test_paths, test_labels, class_names


def save_cifar_images(dataset, base_dir: Path, class_names: list, split: str) -> tuple:
    """Save CIFAR images to disk in ImageNet-style structure."""
    paths = []
    labels = []
    
    logger.info(f"Saving {split} images...")
    
    for idx, (image_tensor, label) in enumerate(dataset):
        if idx % 10000 == 0:
            logger.info(f"Processed {idx}/{len(dataset)} {split} images")
        
        # Convert tensor to PIL Image
        image = transforms.ToPILImage()(image_tensor)
        
        # Save image
        class_name = class_names[label]
        image_path = base_dir / class_name / f"{idx:05d}.png"
        image.save(image_path)
        
        paths.append(str(image_path))
        labels.append(label)
    
    return paths, labels


def load_existing_cifar(images_dir: Path, class_names: list) -> tuple:
    """Load existing CIFAR directory structure."""
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for split, (paths_list, labels_list) in [('train', (train_paths, train_labels)), 
                                            ('test', (test_paths, test_labels))]:
        for label, class_name in enumerate(class_names):
            class_dir = images_dir / split / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob("*.png")):
                    paths_list.append(str(img_path))
                    labels_list.append(label)
    
    return train_paths, train_labels, test_paths, test_labels, class_names


def load_imagenet_dataset(args) -> tuple:
    """
    Load ImageNet dataset from HuggingFace.
    
    Args:
        args: Arguments containing configuration
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    try:
        from datasets import load_dataset
        from PIL import Image
        import tempfile
        import os
    except ImportError:
        raise ImportError("Please install datasets library: pip install datasets[vision]")
    
    logger.info("Loading ImageNet-1k from HuggingFace (this may take a while on first run)...")
    
    try:
        # Try to load ImageNet-1k from HuggingFace
        train_dataset = load_dataset("imagenet-1k", split="train", trust_remote_code=True)
        val_dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load imagenet-1k: {e}")
        logger.info("Trying alternative ImageNet repository...")
        try:
            train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
            val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True)
        except Exception as e2:
            raise ValueError(f"Failed to load ImageNet from HuggingFace. You may need authentication. Error: {e2}")
    
    # Create temporary directory for images
    temp_dir = Path(tempfile.mkdtemp(prefix="imagenet_"))
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Get class names from dataset features
    class_names = train_dataset.features['label'].names
    
    def save_dataset_images(dataset, output_dir, max_samples=None):
        """Save dataset images to disk and return paths/labels."""
        paths = []
        labels = []
        
        # Limit samples if max_samples specified
        if max_samples:
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            dataset = dataset.select(indices)
        
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
                
            image = example['image']
            label = example['label']
            
            # Create class directory
            class_dir = output_dir / class_names[label]
            class_dir.mkdir(exist_ok=True)
            
            # Save image
            image_path = class_dir / f"image_{i:06d}.jpg"
            image.save(str(image_path))
            
            paths.append(str(image_path))
            labels.append(label)
            
            if i % 1000 == 0:
                logger.info(f"Processed {i} images...")
        
        return paths, labels
    
    # Save train and validation images
    max_train = getattr(args, 'max_test_samples', None)  # Use max_test_samples for train too if specified
    max_val = getattr(args, 'max_test_samples', None)
    
    logger.info("Saving training images...")
    train_paths, train_labels = save_dataset_images(train_dataset, train_dir, max_train)
    
    logger.info("Saving validation images...")
    test_paths, test_labels = save_dataset_images(val_dataset, val_dir, max_val)
    
    logger.info(f"ImageNet loaded: {len(train_paths)} train, {len(test_paths)} test images")
    
    return train_paths, train_labels, test_paths, test_labels, class_names


def load_custom_dataset(dataset_path: str, num_classes: int) -> tuple:
    """
    Load custom ImageNet-style dataset.
    
    Expected structure:
    dataset_path/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
        test/ (or val/)
            class1/
                ...
    """
    dataset_path = Path(dataset_path)
    
    # Find train directory
    train_dir = None
    for possible_train in ['train', 'training']:
        if (dataset_path / possible_train).exists():
            train_dir = dataset_path / possible_train
            break
    
    if train_dir is None:
        raise ValueError(f"No train directory found in {dataset_path}")
    
    # Find test directory
    test_dir = None
    for possible_test in ['test', 'val', 'validation']:
        if (dataset_path / possible_test).exists():
            test_dir = dataset_path / possible_test
            break
    
    if test_dir is None:
        raise ValueError(f"No test directory found in {dataset_path}")
    
    # Get class names from train directory
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    if len(class_names) != num_classes:
        logger.warning(f"Found {len(class_names)} classes, but expected {num_classes}")
    
    # Load file paths and labels
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for label, class_name in enumerate(class_names):
        # Train files
        train_class_dir = train_dir / class_name
        if train_class_dir.exists():
            for img_path in train_class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    train_paths.append(str(img_path))
                    train_labels.append(label)
        
        # Test files
        test_class_dir = test_dir / class_name
        if test_class_dir.exists():
            for img_path in test_class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    test_paths.append(str(img_path))
                    test_labels.append(label)
    
    logger.info(f"Loaded custom dataset: {len(train_paths)} train, {len(test_paths)} test images")
    logger.info(f"Classes: {class_names}")
    
    return train_paths, train_labels, test_paths, test_labels, class_names


def test_single_dataset(dataset_name: str, args):
    """Test a single dataset."""
    logger.info(f"\\n{'='*60}")
    logger.info(f"TESTING {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        # Load dataset
        if dataset_name in ['cifar10', 'cifar100']:
            # Use unified resource management for CIFAR datasets
            from marvis.utils.resource_manager import prepare_cifar_dataset
            train_paths, train_labels, test_paths, test_labels, class_names = prepare_cifar_dataset(
                dataset_type=dataset_name,
                force_redownload=False
            )
        elif dataset_name == 'imagenet':
            # Load ImageNet dataset from HuggingFace
            train_paths, train_labels, test_paths, test_labels, class_names = load_imagenet_dataset(args)
        elif dataset_name == 'custom':
            # Load custom dataset
            if not args.dataset_path:
                raise ValueError("--dataset_path is required for custom datasets")
            if not args.num_classes:
                raise ValueError("--num_classes is required for custom datasets")
            
            train_paths, train_labels, test_paths, test_labels, class_names = load_custom_dataset(
                args.dataset_path, args.num_classes
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"  Train: {len(train_paths)} samples")
        logger.info(f"  Test: {len(test_paths)} samples") 
        logger.info(f"  Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return run_models_on_dataset(dataset_name, train_paths, train_labels, test_paths, test_labels, class_names, args)
    
    except Exception as e:
        logger.error(f"{dataset_name} dataset loading failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def run_models_on_dataset(dataset_name: str, train_paths, train_labels, test_paths, test_labels, class_names, args):
    """Run all models on a single dataset."""
    
    # Debug: Check models argument
    logger.info(f"Models to evaluate: {getattr(args, 'models', 'MISSING')}")
    if not hasattr(args, 'models') or not args.models:
        logger.error("No models specified for evaluation! Using default models.")
        args.models = ["marvis_tsne", "dinov2_linear"]
    
    # Store wandb availability for logging
    use_wandb_logging = args.use_wandb and WANDB_AVAILABLE
    
    # Apply balanced few-shot sampling if requested
    if getattr(args, 'balanced_few_shot', False):
        logger.info(f"Using balanced few-shot sampling with {getattr(args, 'num_few_shot_examples', 5)} examples per class")
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
            n_samples = min(getattr(args, 'num_few_shot_examples', 5), len(paths))
            sampled_indices = np.random.choice(len(paths), n_samples, replace=False)
            for idx in sampled_indices:
                sampled_paths.append(paths[idx])
                sampled_labels.append(label)
        
        train_paths = sampled_paths
        train_labels = sampled_labels
        logger.info(f"Sampled {len(train_paths)} training examples across {len(class_names)} classes")
    
    # Apply max test samples limit if specified
    if getattr(args, 'max_test_samples', None) and len(test_paths) > args.max_test_samples:
        logger.info(f"Limiting test set to {args.max_test_samples} samples")
        test_paths = test_paths[:args.max_test_samples]
        test_labels = test_labels[:args.max_test_samples]
    
    results = {}
    
    # Test MARVIS t-SNE (DINOV2 → t-SNE/PCA → VLM)
    if 'marvis_tsne' in args.models:
        backend_name = "PCA" if args.use_pca_backend else "t-SNE"
        features = []
        if args.use_3d:
            features.append("3D")
        if args.use_knn_connections:
            features.append(f"KNN-{args.nn_k}")
        feature_str = f" ({', '.join(features)})" if features else ""
        
        # Validate feature combinations
        if args.use_knn_connections and args.use_pca_backend:
            logger.warning("KNN connections are only supported with t-SNE, not PCA. KNN will be disabled for PCA backend.")
        
        logger.info(f"Testing MARVIS {backend_name}{feature_str} (DINOV2 → {backend_name} → VLM)...")
        try:
            classifier = MarvisImageTsneClassifier(
                modality="vision",
                dinov2_model=args.dinov2_model,
                embedding_size=512,
                tsne_perplexity=min(30.0, len(train_paths) / 4),
                tsne_max_iter=1000,
                vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                vlm_backend=getattr(args, 'backend', 'auto'),
                use_3d=args.use_3d,
                use_knn_connections=args.use_knn_connections,
                nn_k=args.nn_k,
                max_vlm_image_size=1024,
                zoom_factor=args.zoom_factor,
                use_pca_backend=args.use_pca_backend,
                cache_dir=args.cache_dir,
                device='cpu' if sys.platform == "darwin" or args.device == "cpu" else args.device,
                use_semantic_names=args.use_semantic_names,
                seed=42
            )
            
            # Pass save_every_n parameter to evaluation
            classifier.save_every_n = args.save_every_n
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, test_paths, class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels, 
                return_detailed=args.return_detailed,
                save_outputs=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None
            )
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            
            results['marvis_tsne'] = eval_results
            logger.info(f"MARVIS t-SNE completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('marvis_tsne', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"MARVIS t-SNE failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['marvis_tsne'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test Simple MARVIS (DINOV2 → PCA → k-NN)
    if 'marvis_simple' in args.models:
        logger.info("Testing Simple MARVIS (DINOV2 → PCA → k-NN)...")
        try:
            classifier = SimpleMarvisImageClassifier(
                dinov2_model=args.dinov2_model,
                embedding_size=512,
                use_pca=True,
                pca_components=50,
                knn_neighbors=5,
                cache_dir=args.cache_dir,
                device='cpu' if sys.platform == "darwin" or args.device == "cpu" else args.device
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels, class_names)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(test_paths, test_labels, return_detailed=True)
            eval_results['training_time'] = training_time
            eval_results['config'] = classifier.get_config()
            
            results['marvis_simple'] = eval_results
            logger.info(f"Simple MARVIS completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('marvis_simple', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"Simple MARVIS failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['marvis_simple'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test DINOV2 Linear Probe
    if 'dinov2_linear' in args.models:
        logger.info("Testing DINOV2 Linear Probe...")
        try:
            # Create data loaders
            train_dataset = ImageNetDataset(train_paths, train_labels, get_image_transforms('train'))
            test_dataset = ImageNetDataset(test_paths, test_labels, get_image_transforms('test'))
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            classifier = MacCompatibleDINOV2LinearProbe(
                num_classes=len(class_names),
                class_names=class_names,
                dinov2_model=args.dinov2_model
            )
            
            start_time = time.time()
            classifier.fit(train_loader, test_loader)  # Using test as validation for simplicity
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(test_loader, test_labels)
            eval_results['training_time'] = training_time
            
            results['dinov2_linear'] = eval_results
            logger.info(f"DINOV2 Linear Probe completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('dinov2_linear', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"DINOV2 Linear Probe failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['dinov2_linear'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test Qwen VL
    if 'qwen_vl' in args.models:
        logger.info("Testing Qwen VL...")
        try:
            classifier = QwenVLBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                backend=getattr(args, 'backend', 'auto'),
                use_semantic_names=args.use_semantic_names
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=dataset_name
            )
            eval_results['training_time'] = training_time
            
            results['qwen_vl'] = eval_results
            logger.info(f"Qwen VL completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('qwen_vl', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"Qwen VL failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['qwen_vl'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test OpenAI VLM
    if 'openai_vlm' in args.models:
        logger.info("Testing OpenAI VLM...")
        try:
            from examples.vision.openai_vlm_baseline import OpenAIVLMBaseline
            
            if not hasattr(args, 'openai_model') or not args.openai_model:
                raise ValueError("--openai_model is required when using openai_vlm")
            
            classifier = OpenAIVLMBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name=args.openai_model,
                use_semantic_names=args.use_semantic_names,
                enable_thinking=getattr(args, 'enable_thinking', True)
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=dataset_name
            )
            eval_results['training_time'] = training_time
            
            results['openai_vlm'] = eval_results
            logger.info(f"OpenAI VLM completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('openai_vlm', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"OpenAI VLM failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['openai_vlm'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test Gemini VLM
    if 'gemini_vlm' in args.models:
        logger.info("Testing Gemini VLM...")
        try:
            from examples.vision.gemini_vlm_baseline import GeminiVLMBaseline
            
            if not hasattr(args, 'gemini_model') or not args.gemini_model:
                raise ValueError("--gemini_model is required when using gemini_vlm")
            
            classifier = GeminiVLMBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name=args.gemini_model,
                use_semantic_names=args.use_semantic_names,
                enable_thinking=getattr(args, 'enable_thinking', True)
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=dataset_name
            )
            eval_results['training_time'] = training_time
            
            results['gemini_vlm'] = eval_results
            logger.info(f"Gemini VLM completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('gemini_vlm', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"Gemini VLM failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['gemini_vlm'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test API VLM (generic API baseline)
    if 'api_vlm' in args.models:
        logger.info("Testing API VLM...")
        try:
            from examples.vision.api_vlm_baseline import APIVLMBaseline
            
            # Determine which API model to use
            if hasattr(args, 'openai_model') and args.openai_model:
                api_model = args.openai_model
                api_type = 'openai'
            elif hasattr(args, 'gemini_model') and args.gemini_model:
                api_model = args.gemini_model
                api_type = 'gemini'
            else:
                raise ValueError("Either --openai_model or --gemini_model is required when using api_vlm")
            
            classifier = APIVLMBaseline(
                num_classes=len(class_names),
                class_names=class_names,
                model_name=api_model,
                api_type=api_type,
                use_semantic_names=args.use_semantic_names,
                enable_thinking=getattr(args, 'enable_thinking', True)
            )
            
            start_time = time.time()
            classifier.fit(train_paths, train_labels)
            training_time = time.time() - start_time
            
            eval_results = classifier.evaluate(
                test_paths, test_labels,
                save_raw_responses=args.save_outputs,
                output_dir=args.output_dir if args.save_outputs else None,
                benchmark_name=dataset_name
            )
            eval_results['training_time'] = training_time
            
            results['api_vlm'] = eval_results
            logger.info(f"API VLM completed: {eval_results['accuracy']:.4f} accuracy")
            
            # Log to wandb
            if use_wandb_logging:
                log_results_to_wandb('api_vlm', eval_results, args, class_names, dataset_name)
            
        except Exception as e:
            import traceback
            error_msg = f"API VLM failed: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            results['api_vlm'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    return results


def log_results_to_wandb(model_name: str, eval_results: dict, args, class_names: list, dataset_name: str = None):
    """Log evaluation results to Weights & Biases."""
    if 'error' in eval_results:
        # Log failed runs
        wandb.log({
            f"{model_name}/status": "failed",
            f"{model_name}/error": eval_results['error'],
            "model_name": model_name,
            "dataset": (dataset_name or getattr(args, 'dataset', 'unknown')).upper(),
            "backend": getattr(args, 'backend', 'auto'),
            "balanced_few_shot": getattr(args, 'balanced_few_shot', False),
            "num_few_shot_examples": getattr(args, 'num_few_shot_examples', 5),
            "max_test_samples": getattr(args, 'max_test_samples', None)
        })
        return
    
    # Create base metrics
    metrics = {
        f"{model_name}/accuracy": eval_results['accuracy'],
        f"{model_name}/training_time": eval_results.get('training_time', 0),
        f"{model_name}/prediction_time": eval_results.get('prediction_time', 0),
        f"{model_name}/num_test_samples": eval_results.get('num_test_samples', 0),
        "model_name": model_name,
        "dataset": (dataset_name or getattr(args, 'dataset', 'unknown')).upper(),
        "num_classes": len(class_names),
        "backend": getattr(args, 'backend', 'auto'),
        "balanced_few_shot": getattr(args, 'balanced_few_shot', False),
        "num_few_shot_examples": getattr(args, 'num_few_shot_examples', 5),
        "max_test_samples": getattr(args, 'max_test_samples', None)
    }
    
    # Add MARVIS t-SNE specific metrics
    if model_name == 'marvis_tsne':
        config = eval_results.get('config', {})
        metrics.update({
            f"{model_name}/use_3d": config.get('use_3d', False),
            f"{model_name}/use_knn_connections": config.get('use_knn_connections', False),
            f"{model_name}/nn_k": config.get('nn_k', 0),
            f"{model_name}/use_pca_backend": config.get('use_pca_backend', False),
            f"{model_name}/zoom_factor": config.get('zoom_factor', 1.0),
            f"{model_name}/vlm_model": config.get('vlm_model_id', 'unknown'),
            f"{model_name}/dinov2_model": config.get('dinov2_model', 'unknown'),
            f"{model_name}/embedding_size": config.get('embedding_size', 0),
        })
        
        # Add visualization info if available
        if eval_results.get('visualizations_saved', False):
            metrics[f"{model_name}/visualizations_saved"] = True
            metrics[f"{model_name}/output_directory"] = eval_results.get('output_directory', 'unknown')
    
    # Add Simple MARVIS specific metrics
    elif model_name == 'marvis_simple':
        config = eval_results.get('config', {})
        metrics.update({
            f"{model_name}/use_pca": config.get('use_pca', False),
            f"{model_name}/pca_components": config.get('pca_components', 0),
            f"{model_name}/knn_neighbors": config.get('knn_neighbors', 0),
            f"{model_name}/dinov2_model": config.get('dinov2_model', 'unknown'),
            f"{model_name}/embedding_size": config.get('embedding_size', 0),
        })
    
    # Add classification report metrics if available
    if 'classification_report' in eval_results:
        class_report = eval_results['classification_report']
        if isinstance(class_report, dict):
            # Log macro/weighted averages
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in class_report:
                    avg_metrics = class_report[avg_type]
                    avg_prefix = avg_type.replace(' ', '_')
                    metrics.update({
                        f"{model_name}/precision_{avg_prefix}": avg_metrics.get('precision', 0),
                        f"{model_name}/recall_{avg_prefix}": avg_metrics.get('recall', 0),
                        f"{model_name}/f1_{avg_prefix}": avg_metrics.get('f1-score', 0),
                    })
    
    # Log all metrics to wandb
    wandb.log(metrics)


def save_results(results: dict, output_dir: str, args):
    """Save test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results  
    results_file = os.path.join(output_dir, f"{args.dataset}_test_results.json")
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
    summary_file = os.path.join(output_dir, f"{args.dataset}_test_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    logger.info("\\n" + "="*60)
    logger.info(f"{args.dataset.upper()} TEST RESULTS")
    logger.info("="*60)
    
    for _, row in summary_df.iterrows():
        if row['status'] == 'SUCCESS':
            logger.info(f"{row['model']:15s}: ✓ {row['accuracy']:.4f} accuracy "
                       f"(train: {row['training_time']:.1f}s, test: {row['prediction_time']:.1f}s)")
        else:
            logger.info(f"{row['model']:15s}: ✗ ERROR - {row['error']}")
    
    logger.info(f"\\nDetailed results saved to: {output_dir}")


def parse_args():
    """Parse command line arguments using centralized vision evaluation parser."""
    parser = create_vision_evaluation_parser("Test image classification datasets with MARVIS and baselines")
    
    # Add missing cache_dir argument
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    
    # Add missing save_outputs argument
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and VLM responses"
    )
    parser.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and VLM responses"
    )
    
    # Add return_detailed parameter
    parser.add_argument(
        "--no_detailed_outputs",
        dest="return_detailed",
        action="store_false",
        default=True,
        help="Disable detailed prediction outputs (VLM responses, coordinates, etc.)"
    )
    
    # Add new parameters
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
    
    # Set vision-specific defaults
    parser.set_defaults(
        output_dir="./all_vision_test_results",
        datasets="cifar10"
    )
    
    args = parser.parse_args()
    
    # Convert comma-separated strings to lists for compatibility
    if hasattr(args, 'datasets') and isinstance(args.datasets, str):
        args.datasets = args.datasets.split(',')
    if hasattr(args, 'vision_datasets') and args.vision_datasets:
        args.datasets = args.vision_datasets  # Use vision_datasets if specified
    
    # models is already a list from nargs="+" so no conversion needed
    
    return args


def parse_args_old():
    """Legacy argument parser - replaced by centralized parser."""
    parser = argparse.ArgumentParser(description="Test image classification datasets with MARVIS and baselines")
    
    # Dataset options
    dataset_group = parser.add_argument_group('dataset options')
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10"],
        choices=["cifar10", "cifar100", "imagenet", "custom"],
        help="Datasets to test (default: cifar10). Can specify multiple datasets."
    )
    dataset_group.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset (required for custom/imagenet datasets)"
    )
    dataset_group.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes (required for custom datasets, optional for imagenet)"
    )
    dataset_group.add_argument(
        "--imagenet_subset",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="ImageNet subset to use (default: val)"
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        default="./image_data",
        help="Directory for dataset storage"
    )
    
    # Model options
    model_group = parser.add_argument_group('model options')
    model_group.add_argument(
        "--models",
        nargs="+",
        default=["marvis_tsne", "dinov2_linear"],
        choices=["marvis_tsne", "marvis_simple", "dinov2_linear", "qwen_vl"],
        help="Models to test"
    )
    model_group.add_argument(
        "--dinov2_model",
        type=str,
        default="dinov2_vits14",
        choices=[
            # Standard models
            "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
            # Models with registers  
            "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
            # Linear classifier variants
            "dinov2_vits14_lc", "dinov2_vitb14_lc", "dinov2_vitl14_lc", "dinov2_vitg14_lc"
        ],
        help="DINOv2 model variant"
    )
    
    # MARVIS t-SNE options
    tsne_group = parser.add_argument_group('MARVIS t-SNE options')
    tsne_group.add_argument(
        "--use_pca_backend",
        action="store_true",
        help="Use PCA instead of t-SNE for dimensionality reduction"
    )
    tsne_group.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles"
    )
    tsne_group.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors"
    )
    tsne_group.add_argument(
        "--nn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors for KNN connections"
    )
    tsne_group.add_argument(
        "--zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations"
    )
    tsne_group.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts"
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./image_test_results", 
        help="Directory for test results"
    )
    output_group.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and outputs"
    )
    output_group.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and outputs"
    )
    output_group.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save visualizations every N predictions"
    )
    
    # Runtime options
    runtime_group = parser.add_argument_group('runtime options')
    # Removed --quick_test parameter - use --balanced_few_shot and --max_test_samples instead
    runtime_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )
    
    # Weights & Biases logging
    wandb_group = parser.add_argument_group('Weights & Biases options')
    wandb_group.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default="image-marvis",
        help="Weights & Biases project name"
    )
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    wandb_group.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    return parser.parse_args()


def run_all_image_tests(args):
    """Run image classification tests on multiple datasets."""
    all_results = {}
    
    # Log platform information using utility
    platform_info = log_platform_info(logger)
    
    for dataset_name in args.datasets:
        logger.info(f"\\n{'='*60}")
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
                    run_name = f"{dataset_name}_marvis_{timestamp}{feature_suffix}"
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
        temp_args = argparse.Namespace(**vars(args))
        temp_args.dataset = dataset_name  # For backward compatibility with save_results
        save_results(dataset_results, dataset_output_dir, temp_args)
        
        # Clean up wandb for this dataset
        if gpu_monitor is not None:
            cleanup_gpu_monitoring(gpu_monitor)
        
        logger.info(f"{dataset_name.upper()} test completed!")
    
    return all_results


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    from marvis.utils import set_seed_with_args
    set_seed_with_args(args)
    
    logger.info(f"Starting image classification tests...")
    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {', '.join(args.datasets)}")
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  DINOV2 model: {args.dinov2_model}")
    logger.info(f"  Backend: {getattr(args, 'backend', 'auto')}")
    logger.info(f"  Balanced few-shot: {getattr(args, 'balanced_few_shot', False)}")
    if getattr(args, 'balanced_few_shot', False):
        logger.info(f"  Few-shot examples per class: {getattr(args, 'num_few_shot_examples', 5)}")
    if getattr(args, 'max_test_samples', None):
        logger.info(f"  Max test samples: {args.max_test_samples}")
    logger.info(f"  Use PCA: {getattr(args, 'use_pca_backend', False)}")
    logger.info(f"  3D t-SNE: {getattr(args, 'use_3d', False)}")
    logger.info(f"  KNN connections: {getattr(args, 'use_knn_connections', False)}")
    if getattr(args, 'use_knn_connections', False):
        logger.info(f"  KNN k: {getattr(args, 'nn_k', 5)}")
    
    # Run tests on all datasets
    all_results = run_all_image_tests(args)
    
    # Print summary for all datasets
    logger.info(f"\\n{'='*80}")
    logger.info("ALL DATASETS SUMMARY")
    logger.info(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        logger.info(f"\\n{dataset_name.upper()}:")
        if 'status' in results and results['status'] == 'error':
            logger.info(f"  ERROR: {results['error']}")
        else:
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    logger.info(f"  {model_name:15s}: ✗ ERROR - {model_results['error']}")
                else:
                    accuracy = model_results.get('accuracy', 0)
                    train_time = model_results.get('training_time', 0)
                    test_time = model_results.get('prediction_time', 0)
                    logger.info(f"  {model_name:15s}: ✓ {accuracy:.4f} accuracy "
                               f"(train: {train_time:.1f}s, test: {test_time:.1f}s)")
    
    logger.info(f"\\nAll tests completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()