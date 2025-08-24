#!/usr/bin/env python
"""
Example: Using MARVIS with GGUF quantized models via LlamaCPP

This example demonstrates how to use MARVIS with GGUF quantized models,
which offer significant memory savings and efficient inference.

Install LlamaCPP dependencies:
    pip install "marvis[llamacpp]"

Or manually:
    pip install llama-cpp-python
"""

import sys
import os
from pathlib import Path

# Add MARVIS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from marvis.models.marvis_tsne import MarvisTsneClassifier
from marvis.utils.gguf_utils import is_gguf_url, suggest_gguf_files


def main():
    """Demonstrate GGUF model usage with MARVIS."""

    print("🧠 MARVIS GGUF/LlamaCPP Example")
    print("=" * 50)

    # Example GGUF URLs
    example_gguf_urls = [
        # Qwen2.5-VL GGUF variants
        "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF?show_file_info=Qwen2.5-VL-3B-Instruct-q4_k_m.gguf",
        "https://huggingface.co/QuantFactory/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
        # Local GGUF file (if you have one)
        # "/path/to/your/model.gguf"
    ]

    print("\n📋 Example GGUF URLs:")
    for i, url in enumerate(example_gguf_urls, 1):
        print(f"   {i}. {url}")
        print(f"      GGUF detected: {is_gguf_url(url)}")

    # Demonstrate GGUF file suggestions
    print(f"\n🔍 Suggested GGUF files for repository:")
    repo_url = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
    suggestions = suggest_gguf_files(repo_url)
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"   {i}. {Path(suggestion).name}")

    # Create synthetic dataset
    print(f"\n📊 Creating synthetic tabular dataset...")
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_classes=3,
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    class_names = ["Category A", "Category B", "Category C"]

    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Test samples: {len(X_test)}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Classes: {len(class_names)}")

    # Example 1: Auto-detect GGUF backend
    print(f"\n🔧 Example 1: Auto-detection of GGUF backend")
    gguf_url = example_gguf_urls[0]  # Use first example URL

    print(f"   Model URL: {gguf_url}")
    print(f"   Backend: auto (will detect 'llamacpp' for GGUF)")

    try:
        classifier_auto = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id=gguf_url,
            backend="auto",  # Will auto-detect llamacpp
            use_3d=False,
            use_knn_connections=True,
            nn_k=3,
            device="auto",  # Will auto-detect best device
            # LlamaCPP-specific parameters
            n_ctx=2048,  # Context window
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False,
            seed=42,
        )

        print(f"   ✅ Classifier initialized with auto backend detection")
        print(f"   Backend selected: {classifier_auto.backend}")

        # Note: We don't actually fit/predict here to avoid downloading the model
        # In a real scenario, you would continue with:
        # classifier_auto.fit(X_train, y_train, X_test, class_names=class_names)
        # results = classifier_auto.evaluate(X_test, y_test)

    except ImportError as e:
        print(f"   ⚠️ LlamaCPP not available: {e}")
        print(f"   Install with: pip install 'marvis[llamacpp]'")

    # Example 2: Explicit GGUF backend
    print(f"\n🔧 Example 2: Explicit LlamaCPP backend")

    try:
        classifier_explicit = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id=gguf_url,
            backend="llamacpp",  # Explicitly use llamacpp
            device="cpu",  # Force CPU for demonstration
            n_ctx=1024,  # Smaller context window for CPU
            n_gpu_layers=0,  # CPU only
            use_mlock=True,  # Lock memory for better performance
            use_mmap=True,  # Use memory mapping
            verbose=True,  # Show loading details
        )

        print(f"   ✅ Classifier initialized with explicit llamacpp backend")

    except ImportError as e:
        print(f"   ⚠️ LlamaCPP not available: {e}")

    # Example 3: Configuration options
    print(f"\n⚙️ GGUF/LlamaCPP Configuration Options:")

    config_options = {
        "Memory Management": [
            "n_ctx: Context window size (512, 1024, 2048, 4096, etc.)",
            "use_mlock: Lock memory pages (True for better performance)",
            "use_mmap: Use memory mapping (True for efficient loading)",
        ],
        "Hardware Acceleration": [
            "n_gpu_layers: GPU layers (-1 for all, 0 for CPU only)",
            "device: 'auto', 'cuda', 'mps', 'cpu'",
            "n_threads: CPU threads (None for auto-detect)",
        ],
        "Model Loading": [
            "verbose: Show loading details (True/False)",
            "clip_model_path: Path to CLIP model for VLMs",
        ],
        "Generation": [
            "max_new_tokens: Maximum tokens to generate",
            "temperature: Sampling temperature (0.1-2.0)",
            "top_p: Nucleus sampling probability",
            "top_k: Top-k sampling limit",
        ],
    }

    for category, options in config_options.items():
        print(f"\n   {category}:")
        for option in options:
            print(f"     • {option}")

    # Benefits summary
    print(f"\n🎯 GGUF Benefits:")
    benefits = [
        "Memory Efficient: Quantized models use significantly less VRAM/RAM",
        "Fast Loading: GGUF format optimized for quick model loading",
        "Hardware Agnostic: Works on CPU, CUDA, and Apple Silicon (Metal)",
        "Quality Preserved: Minimal accuracy loss with proper quantization",
        "Multiple Formats: Support for Q4_K_M, Q5_K_M, Q8_0, F16, etc.",
        "Local Inference: No API dependencies or costs",
    ]

    for benefit in benefits:
        print(f"   ✅ {benefit}")

    print(f"\n💡 Next Steps:")
    print(f"   1. Install LlamaCPP: pip install 'marvis[llamacpp]'")
    print(f"   2. Choose a GGUF model from HuggingFace")
    print(f"   3. Use with MARVIS as shown above")
    print(f"   4. Adjust n_ctx and n_gpu_layers for your hardware")
    print(f"   5. Monitor GPU/CPU usage during inference")

    print(f"\n🚀 Ready to use GGUF models with MARVIS!")


if __name__ == "__main__":
    main()
