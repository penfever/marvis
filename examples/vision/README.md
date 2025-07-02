# ImageNet Classification with MARVIS t-SNE and Computer Vision Baselines

This directory contains examples for image classification using DINOV2 embeddings with MARVIS t-SNE baseline and standard computer vision approaches.

## Overview

The goal is to demonstrate that for certain image classification tasks, especially those with long-tail distributions or limited training data, MARVIS t-SNE applied to pre-trained image embeddings (DINOV2) can compete with or outperform traditional fine-tuned computer vision models.

## Structure

- `run_imagenet_classification.py`: Main script for running experiments
- `image_baselines.py`: Standard computer vision baseline implementations
- `image_utils.py`: Utilities for image loading and preprocessing

## Models Included

### MARVIS t-SNE Baseline
- Uses DINOV2 embeddings as input features
- Applies MARVIS t-SNE methodology for classification

### Computer Vision Baselines
- ResNet-50 (fine-tuned)
- EfficientNet-B0 (fine-tuned)
- Vision Transformer (ViT-B/16) (fine-tuned)
- Linear probe on DINOV2 features

## Usage

```bash
# Run full comparison on ImageNet subset
python run_imagenet_classification.py --dataset_path /path/to/imagenet --num_classes 100 --samples_per_class 100

# Use specific models only
python run_imagenet_classification.py --models marvis_tsne resnet50 --dataset_path /path/to/imagenet

# Specify DINOV2 model variant
python run_imagenet_classification.py --dinov2_model dinov2_vitl14 --dataset_path /path/to/imagenet
```

## Expected Use Cases

MARVIS t-SNE with DINOV2 embeddings should perform particularly well on:
- Long-tail image classification datasets
- Few-shot learning scenarios
- Datasets with limited training data per class
- Domain-specific image classification where fine-tuning data is scarce

# Image Classification with MARVIS

This guide shows how to evaluate MARVIS t-SNE on various image datasets and compare it with computer vision baselines.

## Installation

First, install the vision dependencies:

```bash
# Install vision dependencies
pip install -e ".[vision]"

# Optional: Install VLM dependencies for Qwen baseline
# On Mac (CPU-only):
pip install -e ".[vlm]"

# On Linux/Windows with CUDA GPU:
pip install -e ".[vlm,vlm_cuda]"
```

## Quick Start

### 1. CIFAR-10 Basic Test

```bash
# Quick test with CIFAR-10 (recommended for first run)
python examples/vision/evaluate_all_vision.py \
    --dataset cifar10 \
    --models marvis_tsne \
    --quick_test \
    --output_dir ./cifar10_results
```

### 2. Multiple Datasets

```bash
# Test multiple datasets at once
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 cifar100 \
    --models marvis_tsne dinov2_linear \
    --output_dir ./multi_dataset_results
```

### 3. CIFAR-100 Evaluation

```bash
# Full CIFAR-100 evaluation
python examples/vision/evaluate_all_vision.py \
    --datasets cifar100 \
    --models marvis_tsne dinov2_linear qwen_vl \
    --output_dir ./cifar100_results
```

### 4. ImageNet from HuggingFace

```bash
# ImageNet-1k from HuggingFace (requires authentication)
python examples/vision/evaluate_all_vision.py \
    --datasets imagenet \
    --models marvis_tsne dinov2_linear \
    --quick_test \
    --output_dir ./imagenet_results
```

### 5. Custom Dataset

```bash
# Your own ImageNet-style dataset
python examples/vision/evaluate_all_vision.py \
    --datasets custom \
    --dataset_path /path/to/your/dataset \
    --num_classes 50 \
    --models marvis_tsne dinov2_linear \
    --output_dir ./custom_results
```

## Supported Datasets

### Automatic Downloads
- **CIFAR-10**: `--datasets cifar10` (10 classes, auto-downloaded)
- **CIFAR-100**: `--datasets cifar100` (100 classes, auto-downloaded)
- **ImageNet-1k**: `--datasets imagenet` (1000 classes, from HuggingFace)

### Custom Datasets
Use `--datasets custom` with ImageNet-style structure:
```
dataset_path/
    train/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    test/ (or val/)
        class1/
            image1.jpg
        class2/
            image1.jpg
```

## Available Models

```bash
--models MODEL [MODEL ...]
```

Choose from:
- **marvis_tsne**: DINOV2 → t-SNE → VLM (main method)
- **marvis_simple**: DINOV2 → PCA → k-NN (simpler baseline)
- **dinov2_linear**: DINOV2 → Linear Probe (traditional CV)
- **qwen_vl**: Qwen Vision-Language Model (zero-shot)

## Advanced MARVIS t-SNE Configuration

### 3D Visualizations

```bash
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne \
    --use_3d_tsne \
    --output_dir ./3d_results
```

### KNN Connections

```bash
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne \
    --use_knn_connections \
    --nn_k 10 \
    --output_dir ./knn_results
```

### PCA Backend (faster)

```bash
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne \
    --use_pca_backend \
    --output_dir ./pca_results
```

### Combined Features

```bash
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne \
    --use_3d_tsne \
    --use_knn_connections \
    --nn_k 20 \
    --zoom_factor 7.0 \
    --use_semantic_names \
    --output_dir ./full_features_results
```

## DINOV2 Model Variants

```bash
--dinov2_model MODEL
```

Available models:
- **Standard**: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`
- **With Registers**: `dinov2_vits14_reg`, `dinov2_vitb14_reg`, etc.
- **Linear Classifier**: `dinov2_vits14_lc`, `dinov2_vitb14_lc`, etc.

Recommendation: Start with `dinov2_vits14` (fastest) or `dinov2_vitb14` (good balance).

## Weights & Biases Integration

```bash
python examples/vision/evaluate_all_vision.py \
    --dataset cifar10 \
    --models marvis_tsne dinov2_linear qwen_vl \
    --use_wandb \
    --wandb_project my-image-experiments \
    --output_dir ./wandb_results
```

## Performance Optimization

### For Mac/CPU Systems

```bash
python examples/vision/evaluate_all_vision.py \
    --dataset cifar10 \
    --models marvis_tsne \
    --dinov2_model dinov2_vits14 \
    --quick_test \
    --device cpu \
    --max_train_plot_samples 500
```

### For GPU Systems

```bash
python examples/vision/evaluate_all_vision.py \
    --dataset cifar100 \
    --models marvis_tsne dinov2_linear \
    --dinov2_model dinov2_vitl14 \
    --device cuda \
    --max_train_plot_samples 2000
```

## Output Files

The script generates:

1. **Results JSON**: `{dataset}_test_results.json`
   - Complete evaluation metrics
   - Model configurations
   - Timing information

2. **Summary CSV**: `{dataset}_test_summary.csv`
   - Quick comparison table
   - Success/error status

3. **QwenVL Raw Responses**: `{dataset}_qwenvl_raw_responses.json` (if QwenVL used with `--save_outputs`)
   - Raw VLM responses for analysis
   - Prompts and parsed results

4. **Visualizations**: `image_visualizations/` (if `--save_outputs`)
   - t-SNE plots with query points
   - Metadata for each visualization

## Sample Output

```
CIFAR10 TEST RESULTS
============================================================
marvis_tsne     : ✓ 0.6420 accuracy (train: 45.2s, test: 8.1s)
marvis_simple   : ✓ 0.5890 accuracy (train: 12.1s, test: 3.2s)
dinov2_linear  : ✓ 0.7150 accuracy (train: 18.3s, test: 2.4s)
qwen_vl        : ✓ 0.4230 accuracy (train: 0.1s, test: 125.6s)

Detailed results saved to: ./cifar10_results
```

## Biological Datasets

For specialized biological image classification, use the separate script:

```bash
python examples/vision/test_bioclip2_biological_datasets.py \
    --datasets fishnet awa2 plantdoc \
    --models marvis_tsne_bioclip2 qwen_vl \
    --output_dir ./biological_results
```

## Troubleshooting

### Memory Issues
- Use `--quick_test` for initial testing
- Reduce `--max_train_plot_samples` (default: 1000)
- Use smaller DINOV2 model: `--dinov2_model dinov2_vits14`

### GPU Issues
- Framework automatically detects and uses available hardware
- Force CPU with `--device cpu`
- Mac MPS is supported with fallbacks

### Dataset Issues
- CIFAR datasets are automatically downloaded
- For custom datasets, ensure proper directory structure
- Check that `--num_classes` matches actual number of class directories

This framework provides a comprehensive evaluation suite for comparing MARVIS t-SNE with traditional computer vision approaches across various image classification tasks.