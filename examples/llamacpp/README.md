# LlamaCPP/GGUF Integration with MARVIS

This directory contains examples and documentation for using MARVIS with GGUF quantized models via LlamaCPP.

## Overview

MARVIS now supports GGUF (GPT-Generated Unified Format) quantized models through the LlamaCPP backend. This enables:

- **Memory Efficiency**: Significant reduction in VRAM/RAM usage
- **Hardware Flexibility**: Support for CPU, CUDA, and Apple Silicon (Metal)
- **Local Inference**: No API dependencies or costs
- **Multiple Quantizations**: Q4_K_M, Q5_K_M, Q8_0, F16, and more

## Installation

Install the LlamaCPP dependencies:

```bash
pip install "marvis[llamacpp]"
```

Or manually:

```bash
pip install llama-cpp-python
```

## Supported GGUF URLs

MARVIS automatically detects and handles various GGUF URL formats:

```python
# HuggingFace file browser URL
"https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf"

# Direct resolve URL
"https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/model.gguf"

# Local GGUF file
"/path/to/your/model.gguf"
```

## Basic Usage

### Auto-Detection

MARVIS automatically detects GGUF models and selects the LlamaCPP backend:

```python
from marvis.models.marvis_tsne import MarvisTsneClassifier

# GGUF URL will auto-select llamacpp backend
classifier = MarvisTsneClassifier(
    modality="tabular",
    vlm_model_id="https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=model.gguf",
    backend="auto",  # Will detect "llamacpp"
    device="auto"    # Will detect optimal device
)
```

### Explicit Configuration

You can also explicitly specify the LlamaCPP backend:

```python
classifier = MarvisTsneClassifier(
    modality="tabular", 
    vlm_model_id="/path/to/model.gguf",
    backend="llamacpp",
    device="cuda",       # or "mps", "cpu"
    n_ctx=2048,         # Context window
    n_gpu_layers=-1,    # Use all GPU layers
    verbose=True
)
```

## Configuration Options

### Memory Management
- `n_ctx`: Context window size (512, 1024, 2048, 4096, etc.)
- `use_mlock`: Lock memory pages for better performance
- `use_mmap`: Use memory mapping for efficient loading

### Hardware Acceleration
- `n_gpu_layers`: Number of layers to run on GPU (-1 for all, 0 for CPU only)
- `device`: Target device ("auto", "cuda", "mps", "cpu")
- `n_threads`: CPU threads (None for auto-detect)

### Model Loading
- `verbose`: Show detailed loading information
- `clip_model_path`: Path to CLIP model for vision-language models

## Hardware Recommendations

### CPU Only
```python
classifier = MarvisTsneClassifier(
    vlm_model_id="model.gguf",
    backend="llamacpp",
    device="cpu",
    n_ctx=1024,        # Smaller context for CPU
    n_gpu_layers=0,    # CPU only
    n_threads=None     # Auto-detect CPU threads
)
```

### NVIDIA GPU
```python
classifier = MarvisTsneClassifier(
    vlm_model_id="model.gguf", 
    backend="llamacpp",
    device="cuda",
    n_ctx=4096,        # Larger context with GPU
    n_gpu_layers=-1,   # Use all GPU layers
)
```

### Apple Silicon (M1/M2/M3/M4)
```python
classifier = MarvisTsneClassifier(
    vlm_model_id="model.gguf",
    backend="llamacpp", 
    device="mps",      # Metal Performance Shaders
    n_ctx=2048,
    n_gpu_layers=-1,   # Use Metal acceleration
)
```

## Model Recommendations

### Small Models (3B-7B parameters)
- **Qwen2.5-VL-3B-Instruct**: Excellent vision-language capabilities
- **Llama-3.2-3B-Vision**: Good general performance
- Quantization: Q4_K_M or Q5_K_M for best quality/size balance

### Medium Models (8B-13B parameters)  
- **Qwen2.5-VL-7B-Instruct**: Enhanced reasoning capabilities
- **Llava-Next-7B**: Strong multimodal understanding
- Quantization: Q4_K_M recommended

### Large Models (30B+ parameters)
- **Qwen2.5-VL-32B-Instruct**: State-of-the-art performance
- Quantization: Q4_K_M or Q8_0 if VRAM allows

## Quantization Guide

| Format | Size Reduction | Quality | Use Case |
|--------|----------------|---------|----------|
| Q4_K_M | ~75% | Good | Best balance |
| Q5_K_M | ~70% | Better | Higher quality |
| Q8_0   | ~50% | Excellent | Large VRAM |
| F16    | ~50% | Perfect | Maximum quality |

## Examples

- `gguf_example.py`: Basic GGUF usage demonstration
- See main MARVIS examples for integration with real datasets

## Troubleshooting

### Installation Issues
```bash
# If compilation fails, try pre-built wheels
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For Metal support on Mac
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### Memory Issues
- Reduce `n_ctx` (context window size)
- Use more aggressive quantization (Q4_K_M vs Q8_0)
- Reduce `n_gpu_layers` for hybrid CPU/GPU inference

### Performance Issues
- Enable `use_mlock=True` for memory locking
- Enable `use_mmap=True` for memory mapping
- Adjust `n_threads` for CPU inference
- Monitor GPU utilization and adjust `n_gpu_layers`

## Integration with MARVIS Workflows

GGUF models integrate seamlessly with all MARVIS features:

- **Tabular Data**: Classification and regression tasks
- **Image Data**: Vision-language model inference  
- **Audio Data**: Multimodal audio understanding
- **Multi-Visualization**: Enhanced reasoning with quantized models
- **Few-Shot Learning**: Efficient inference for small datasets

The quantized models often provide 90-95% of full precision performance while using significantly less memory and offering faster inference.