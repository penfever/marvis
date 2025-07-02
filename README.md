# MARVIS: Modality Adaptive Reasoning over VISualizations

**MARVIS** is a powerful framework for multi-modal classification that leverages Vision Language Models (VLMs) to perform classification on tabular, audio, and vision data through intelligent visualization and embedding techniques.

## 🚀 Quick Install

```bash
pip install -e ".[vision,audio,api]"
```

## 🌟 Key Features

* **Multi-modal Support**: Tabular, audio, and vision data classification
* **Vision Language Models**: Leverages state-of-the-art VLMs for intelligent reasoning  
* **Advanced Visualizations**: t-SNE, PCA, UMAP, and multi-visualization frameworks
* **API Integration**: Support for OpenAI, Google Gemini, and local models
* **Rich Embeddings**: TabPFN, Whisper, DINOV2, and more

## 💡 Quick Start

### Tabular Data (30 seconds)
```python
from marvis.models.marvis_tsne import MarvisTsneClassifier
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=100, n_features=10, n_classes=3)

# Create and train classifier
classifier = MarvisTsneClassifier(modality="tabular")
classifier.fit(X, y)

# Make predictions
predictions = classifier.predict(X)
print(f"Accuracy: {(predictions == y).mean():.2f}")
```

### Vision Classification
```bash
# Test CIFAR-10 with advanced features
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne \
    --use_3d \
    --use_knn_connections
```

### Audio Classification
```bash
# Test ESC-50 and RAVDESS datasets
python examples/audio/evaluate_all_audio.py \
    --datasets esc50 ravdess \
    --models marvis_tsne
```

### API Models
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

from marvis.models.marvis_tsne import MarvisTsneClassifier

# OpenAI GPT-4.1
classifier = MarvisTsneClassifier(
    modality="tabular",
    vlm_model_id="openai/gpt-4o-2024-08-06"
)

# Local Qwen2.5-VL model
classifier = MarvisTsneClassifier(
    modality="vision", 
    vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"
)
```

## 🏗️ Architecture

MARVIS follows a **modality-agnostic VLM classification pipeline**:

1. **Embedding Generation**: Convert raw data to feature vectors using modality-specific encoders
2. **Dimensionality Reduction**: Apply t-SNE, PCA, UMAP for visualization  
3. **VLM Classification**: Feed visualizations to Vision Language Models for reasoning-based classification

### Core Components

- **Multi-modal Embeddings**: TabPFN (tabular), Whisper/CLAP (audio), DINOV2/BioCLIP (vision)
- **Visualization Framework**: Modular system supporting t-SNE, PCA, UMAP, multi-viz compositions
- **VLM Integration**: Local models (Qwen2.5-VL) and API models (GPT-4.1, Gemini 2.5)
- **Resource Management**: Intelligent caching, device detection, memory optimization

## 📊 Supported Data Types

### Tabular Data
- **Embeddings**: TabPFN (in-context learning), scikit-learn features
- **Datasets**: OpenML CC18, regression benchmarks, custom CSV
- **Features**: Semantic class names, metadata integration, few-shot learning

### Vision Data  
- **Embeddings**: DINOV2 (natural images), BioCLIP2 (biological data)
- **Datasets**: CIFAR-10/100, ImageNet subsets, biological datasets
- **Features**: 3D visualizations, KNN connections, multi-panel layouts

### Audio Data
- **Embeddings**: Whisper encoder, CLAP for zero-shot classification
- **Datasets**: ESC-50, RAVDESS, custom audio files  
- **Features**: Spectrogram-like visualizations, time-series analysis

## 🛠️ Installation

### Basic Installation
```bash
git clone https://github.com/penfever/marvis.git
cd marvis
pip install -e .
```

### With Vision Support
```bash
pip install -e ".[vision]"
```

### With Audio Support  
```bash
pip install -e ".[audio]"
```

### With API Models
```bash
pip install -e ".[api]"
```

### Full Installation
```bash
pip install -e ".[vision,audio,api,dev]"
```

## 🚀 Advanced Usage

### Multi-Visualization Framework
```python
from marvis.models.marvis_tsne import MarvisTsneClassifier

classifier = MarvisTsneClassifier(
    modality="tabular",
    enable_multi_viz=True,
    viz_methods=["tsne", "pca", "umap"],
    layout_strategy="grid"
)
```

### Custom VLM Configuration
```python
classifier = MarvisTsneClassifier(
    modality="vision",
    vlm_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_backend="transformers",  # or "vllm" for faster inference
    generation_config={
        "max_new_tokens": 512,
        "temperature": 0.1
    }
)
```

### Semantic Enhancement
```python
classifier = MarvisTsneClassifier(
    modality="tabular",
    use_semantic_class_names=True,
    use_metadata=True,
    semantic_axes=True
)
```

## 📈 Performance

MARVIS achieves competitive performance across modalities:

- **Tabular**: Comparable to TabPFN/CatBoost on OpenML benchmarks
- **Vision**: Strong performance on CIFAR-10/100, biological datasets
- **Audio**: Effective classification on ESC-50, RAVDESS

### Benchmarks
```bash
# Run tabular benchmarks
python examples/tabular/evaluate_llm_baselines_tabular.py \
    --task_ids 23,31,3918 \
    --models marvis_tsne

# Run vision benchmarks  
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models marvis_tsne
```

## 🔧 Configuration

### Environment Variables
```bash
export MARVIS_CACHE_DIR="/path/to/cache"
export MARVIS_BASE_DIR="/path/to/data"
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-gemini-key"
```

### Device Configuration
```python
# Automatic device detection
classifier = MarvisTsneClassifier(device="auto")

# Force specific device
classifier = MarvisTsneClassifier(device="cuda")  # or "mps", "cpu"
```

## 📚 Examples

### Tabular Classification
```bash
# Single dataset evaluation
python examples/tabular/evaluate_on_dataset_tabular.py \
    --models marvis_tsne \
    --task_ids 23 \
    --max_test_samples 100

# Multi-dataset comparison
python examples/tabular/evaluate_llm_baselines_tabular.py \
    --models marvis_tsne tabpfn_v2 catboost \
    --task_ids 23,31,3918
```

### Vision with BioCLIP
```bash
python examples/vision/evaluate_bioclip2_biological_datasets.py \
    --dataset fishnet \
    --models marvis_tsne_bioclip2
```

### Audio Analysis
```bash
python examples/audio/evaluate_all_audio.py \
    --datasets esc50 \
    --models marvis_tsne \
    --max_test_samples 200
```

## 🧪 Testing

```bash
# Run core tests
python tests/test_install.py
python tests/test_comprehensive_vlm_prompting.py

# Test specific modalities
python tests/test_audio_simple.py
python tests/test_resource_management.py

# Run with pytest
python -m pytest tests/ -v
```

## 📖 Documentation

Full documentation is available at: [docs/](docs/)

Key guides:
- [Getting Started](docs/getting-started/quick-start.rst)
- [API Reference](docs/api-reference/)
- [Technical Guides](docs/technical-guides/)
- [User Guide](docs/user-guide/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run tests: `python -m pytest tests/`
6. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) for details.

## 🙏 Citation

If you use MARVIS in your research, please cite:

```bibtex
@misc{marvis2025,
  title={MARVIS: Modality Adaptive Reasoning over VISualizations},
  author={Benjamin Feuer and Lennart Purucker and Oussama Elachqar and Chinmay Hegde},
  year={2025},
  url={https://github.com/penfever/marvis}
}
```

## 🔗 Related Work

- [TabPFN](https://github.com/automl/TabPFN): Transformer for tabular data
- [DINOV2](https://github.com/facebookresearch/dinov2): Self-supervised vision features  
- [Whisper](https://github.com/openai/whisper): Speech recognition and audio features
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct): Vision-language model
