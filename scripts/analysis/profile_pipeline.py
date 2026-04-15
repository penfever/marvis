#!/usr/bin/env python3
"""
MARVIS Pipeline Latency Profiler

Profiles each stage of the MARVIS pipeline using torch synchronization
barriers and time.perf_counter() for accurate GPU-aware wall-clock timing.
Runs multiple samples to smooth variance and reports aggregate statistics.

Usage:
    python profile_pipeline.py [n_samples]   # default: 10

Author: MARVIS Team
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Add MARVIS to Python path
script_dir = Path(__file__).parent
marvis_root = script_dir.parent.parent
sys.path.insert(0, str(marvis_root))

warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def sync_device():
    """Synchronize GPU/MPS to ensure all queued operations complete."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(label=None):
    """Context manager for synchronized timing."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            sync_device()
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            sync_device()
            self.elapsed = time.perf_counter() - self.start
    return Timer()


def profile_pipeline(n_samples=10, seed=42):
    """Profile the full MARVIS pipeline, breaking down latency by stage."""

    print("MARVIS Pipeline Latency Profiler")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple MPS"
    else:
        device = "cpu"
        device_name = "CPU"
    print(f"Device: {device_name}")
    print(f"Samples: {n_samples}, seed: {seed}\n")

    # ── Data creation (not timed) ─────────────────────────────────
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=8,
        n_redundant=1, n_clusters_per_class=1, class_sep=1.2, random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )
    class_names = ["Class A", "Class B", "Class C"]

    # ── Stage 1: VLM model loading ────────────────────────────────
    print("[1/4] Loading VLM model...")
    from marvis.models.marvis_tsne import MarvisTsneClassifier

    classifier = MarvisTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        tsne_perplexity=15,
        use_3d=False,
        use_knn_connections=True,
        nn_k=5,
        use_semantic_names=True,
        seed=seed,
    )

    with timed() as t_vlm_load:
        classifier._load_vlm()
    print(f"       VLM load: {t_vlm_load.elapsed:.2f}s\n")

    # ── Stage 2: Embedding generation + t-SNE ─────────────────────
    # Monkey-patch to capture sub-timings within fit()
    _sub = {}

    original_get_embedding = classifier._get_embedding_method
    def patched_get_embedding():
        method = original_get_embedding()
        def wrapped(*args, **kwargs):
            sync_device()
            t = time.perf_counter()
            result = method(*args, **kwargs)
            sync_device()
            _sub["embedding"] = time.perf_counter() - t
            return result
        return wrapped
    classifier._get_embedding_method = patched_get_embedding

    original_get_viz = classifier._get_tsne_visualization_methods
    def patched_get_viz():
        methods = original_get_viz()
        original_create = methods["create_tsne_visualization"]
        def wrapped(*args, **kwargs):
            sync_device()
            t = time.perf_counter()
            result = original_create(*args, **kwargs)
            sync_device()
            _sub["tsne_fit"] = time.perf_counter() - t
            return result
        methods["create_tsne_visualization"] = wrapped
        return methods
    classifier._get_tsne_visualization_methods = patched_get_viz

    print("[2/4] Fitting (embedding generation + t-SNE)...")
    with timed() as t_fit:
        classifier.fit(
            X_train, y_train, X_test,
            class_names=class_names, task_type="classification",
        )

    embed_time = _sub.get("embedding", 0)
    tsne_time = _sub.get("tsne_fit", 0)
    fit_overhead = t_fit.elapsed - embed_time - tsne_time

    print(f"       Embedding generation: {embed_time:.3f}s")
    print(f"       t-SNE fitting:        {tsne_time:.3f}s")
    print(f"       Other fit overhead:   {fit_overhead:.3f}s")
    print(f"       Total fit:            {t_fit.elapsed:.3f}s\n")

    # ── Stage 3-4: Per-sample profiling ───────────────────────────
    print(f"[3/4] Profiling {n_samples} per-sample predictions...")

    from marvis.models.process_one_sample import (
        _create_single_visualization,
        _generate_vlm_response,
        _parse_prediction,
        _process_image,
    )
    from marvis.utils.vlm_prompting import create_classification_prompt
    import matplotlib.pyplot as plt

    viz_methods = original_get_viz()  # use original, un-patched methods

    n_to_profile = min(n_samples, len(X_test))

    viz_times = []
    img_proc_times = []
    prompt_times = []
    vlm_times = []
    parse_times = []
    total_times = []

    for idx in range(n_to_profile):
        sync_device()
        sample_t0 = time.perf_counter()

        # (a) Visualization generation (matplotlib rendering)
        with timed() as t_viz:
            image, legend_text, metadata = _create_single_visualization(
                classifier, idx, viz_methods, viewing_angles=None,
                save_outputs=False, visualization_save_cadence=10,
            )

        # (b) Image processing (resize, RGB conversion)
        with timed() as t_img:
            image = _process_image(classifier, image)

        # (c) Prompt construction
        with timed() as t_prompt:
            visible_classes = metadata.get("visible_classes", [])
            prompt = create_classification_prompt(
                class_names=visible_classes,
                modality=classifier.modality,
                use_knn=classifier.use_knn_connections,
                use_3d=classifier.use_3d,
                nn_k=classifier.knn_k if classifier.use_knn_connections else None,
                legend_text=legend_text,
                dataset_description=(
                    f"{classifier.modality.title()} data embedded "
                    f"using appropriate features"
                ),
                use_semantic_names=classifier.use_semantic_names,
            )

        # (d) VLM inference
        with timed() as t_vlm:
            response = _generate_vlm_response(classifier, image, prompt)

        # (e) Response parsing
        with timed() as t_parse:
            prediction = _parse_prediction(
                response, classifier, classifier.unique_classes,
            )

        sync_device()
        sample_total = time.perf_counter() - sample_t0

        viz_times.append(t_viz.elapsed)
        img_proc_times.append(t_img.elapsed)
        prompt_times.append(t_prompt.elapsed)
        vlm_times.append(t_vlm.elapsed)
        parse_times.append(t_parse.elapsed)
        total_times.append(sample_total)

        plt.close("all")

        print(
            f"  Sample {idx+1:2d}/{n_to_profile}: "
            f"viz={t_viz.elapsed*1000:.0f}ms  "
            f"img={t_img.elapsed*1000:.0f}ms  "
            f"vlm={t_vlm.elapsed*1000:.0f}ms  "
            f"parse={t_parse.elapsed*1000:.1f}ms  "
            f"total={sample_total*1000:.0f}ms"
        )

    # ── Aggregate results ─────────────────────────────────────────
    print(f"\n[4/4] Results")
    print("=" * 60)

    def report(label, times_s):
        arr = np.array(times_s) * 1000  # → ms
        print(
            f"  {label:<28s}  "
            f"mean={arr.mean():7.1f}ms  "
            f"std={arr.std():6.1f}ms  "
            f"min={arr.min():7.1f}ms  "
            f"max={arr.max():7.1f}ms"
        )
        return arr.mean()

    print(f"\nOne-time costs (amortized over test set):")
    print(f"  VLM loading                  {t_vlm_load.elapsed*1000:10.0f}ms")
    print(f"  Embedding generation         {embed_time*1000:10.0f}ms")
    print(f"  t-SNE fitting                {tsne_time*1000:10.0f}ms")
    print(f"  Total fit                    {t_fit.elapsed*1000:10.0f}ms")

    print(f"\nPer-sample costs (N={n_to_profile}):")
    m_viz = report("Visualization generation", viz_times)
    m_img = report("Image processing", img_proc_times)
    m_prompt = report("Prompt construction", prompt_times)
    m_vlm = report("VLM inference", vlm_times)
    m_parse = report("Response parsing", parse_times)
    m_total = report("End-to-end per sample", total_times)

    components = m_viz + m_img + m_prompt + m_vlm + m_parse
    print(f"\n  Breakdown (% of per-sample mean):")
    for name, val in [
        ("Visualization generation", m_viz),
        ("Image processing", m_img),
        ("Prompt construction", m_prompt),
        ("VLM inference", m_vlm),
        ("Response parsing", m_parse),
    ]:
        print(f"    {name:<28s} {val/components*100:5.1f}%")

    print(f"\n  Throughput: {1000/m_total:.2f} samples/sec (single-stream)")

    amort_embed = embed_time * 1000 / n_to_profile
    amort_tsne = tsne_time * 1000 / n_to_profile
    print(f"\n  With amortized one-time costs (N={n_to_profile}):")
    print(f"    + embedding:  {amort_embed:.1f}ms/sample")
    print(f"    + t-SNE:      {amort_tsne:.1f}ms/sample")
    print(f"    = total:      {m_total + amort_embed + amort_tsne:.1f}ms/sample")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    profile_pipeline(n_samples=n)
