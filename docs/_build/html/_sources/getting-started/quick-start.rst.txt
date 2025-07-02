Quick Start Guide
=================

This guide gets you up and running with MARVIS in minutes across all supported modalities.

30-Second Example: Tabular Data
-------------------------------

.. code-block:: python

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

Vision Classification
---------------------

Image Classification with CIFAR-10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Quick test with CIFAR-10
   python examples/vision/evaluate_all_vision.py \\
       --datasets cifar10 \\
       --models marvis_tsne \\
       --quick_test

   # Advanced: 3D visualization with KNN connections
   python examples/vision/evaluate_all_vision.py \\
       --datasets cifar10 \\
       --models marvis_tsne \\
       --use_3d \\
       --use_knn_connections \\
       --knn_k 5

Custom Image Dataset
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   import torch
   from pathlib import Path

   # Prepare image paths and labels
   image_paths = [
       "path/to/cat1.jpg", "path/to/cat2.jpg",
       "path/to/dog1.jpg", "path/to/dog2.jpg"
   ]
   labels = ["cat", "cat", "dog", "dog"]

   # Create vision classifier
   classifier = MarvisTsneClassifier(
       modality="vision",
       vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
       use_3d=False
   )

   # Fit and predict
   classifier.fit(image_paths, labels)
   predictions = classifier.predict(image_paths)

Audio Classification
--------------------

Quick Audio Test
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Test ESC-50 and RAVDESS datasets
   python examples/audio/evaluate_all_audio.py \\
       --datasets esc50 ravdess \\
       --models marvis_tsne

   # Test with CLAP embeddings
   python examples/audio/evaluate_all_audio.py \\
       --datasets esc50 \\
       --embedding_type clap \\
       --models marvis_tsne

Custom Audio Dataset
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   # Prepare audio file paths and labels
   audio_paths = [
       "path/to/speech1.wav", "path/to/speech2.wav",
       "path/to/music1.wav", "path/to/music2.wav"
   ]
   labels = ["speech", "speech", "music", "music"]

   # Create audio classifier
   classifier = MarvisTsneClassifier(
       modality="audio",
       embedding_model="whisper",  # or "clap"
       whisper_model="large-v2",
       include_spectrogram=True
   )

   # Fit and predict
   classifier.fit(audio_paths, labels)
   predictions = classifier.predict(audio_paths)

Advanced Features
-----------------

Multi-Visualization Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   # Create multi-visualization classifier
   classifier = MarvisTsneClassifier(
       modality="tabular",
       enable_multi_viz=True,
       visualization_methods=["pca", "tsne", "umap"],
       layout_strategy="adaptive_grid",
       reasoning_focus="comparison"
   )

   # Fit with multiple visualization perspectives
   classifier.fit(X_train, y_train, X_test)
   
   # Get detailed analysis
   results = classifier.evaluate(X_test, y_test, return_detailed=True)

API Model Integration
~~~~~~~~~~~~~~~~~~~~~

OpenAI GPT-4V
^^^^^^^^^^^^^^

.. code-block:: python

   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key"

   classifier = MarvisTsneClassifier(
       modality="vision",
       api_model="gpt-4o",  # Auto-detects as OpenAI
       enable_thinking=True
   )

Google Gemini
^^^^^^^^^^^^^

.. code-block:: python

   import os
   os.environ["GOOGLE_API_KEY"] = "your-api-key"

   classifier = MarvisTsneClassifier(
       modality="vision", 
       gemini_model="gemini-2.0-flash-exp",
       enable_thinking=True
   )

Resource Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   # Optimize for limited resources
   classifier = MarvisTsneClassifier(
       modality="tabular",
       max_vlm_image_size=1024,  # Reduce image size
       gpu_memory_utilization=0.7,  # Conservative GPU usage
       cache_dir="./marvis_cache",  # Enable caching
       max_tabpfn_samples=1000  # Limit embedding samples
   )

Evaluation Workflows
--------------------

OpenML CC18 Benchmark
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run on OpenML CC18 suite
   python examples/tabular/openml_cc18/run_openml_cc18_tabular.py \\
       --models marvis_tsne \\
       --task_ids 3 6 11 12 14 \\
       --use_3d \\
       --use_knn_connections

Custom Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.metrics import classification_report

   # Create classifier
   classifier = MarvisTsneClassifier(modality="tabular")

   # Cross-validation
   scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
   print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

   # Detailed evaluation
   classifier.fit(X_train, y_train, X_test)
   results = classifier.evaluate(X_test, y_test, return_detailed=True)
   
   print("Classification Report:")
   print(classification_report(y_test, results['predictions']))

Configuration Examples
----------------------

Configuration Files
~~~~~~~~~~~~~~~~~~~

Create ``config.yaml``:

.. code-block:: yaml

   # MARVIS Configuration
   modality: "tabular"
   vlm_model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
   
   # Visualization settings
   use_3d: false
   use_knn_connections: true
   knn_k: 5
   
   # Performance settings
   max_vlm_image_size: 2048
   gpu_memory_utilization: 0.8
   
   # Cache settings
   cache_dir: "./cache"
   
   # Multi-viz settings
   enable_multi_viz: false
   visualization_methods: ["tsne", "pca"]

Load configuration:

.. code-block:: python

   import yaml
   from marvis.models.marvis_tsne import MarvisTsneClassifier

   with open("config.yaml") as f:
       config = yaml.safe_load(f)
   
   classifier = MarvisTsneClassifier(**config)

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # API Keys
   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   
   # Cache directory
   export MARVIS_CACHE_DIR="./cache"
   
   # CUDA settings
   export CUDA_VISIBLE_DEVICES="0"

Interactive Notebooks
----------------------

Jupyter Examples
~~~~~~~~~~~~~~~~

Check out our interactive notebooks:

* ``notebooks/Getting_Started.ipynb`` - Complete walkthrough
* ``examples/unified_marvis_example.py`` - Multi-modal example
* ``examples/tabular/pfn_knn_expts.ipynb`` - TabPFN experiments

Run with:

.. code-block:: bash

   jupyter notebook notebooks/Getting_Started.ipynb

Performance Tips
----------------

Speed Optimization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fast configuration for development
   classifier = MarvisTsneClassifier(
       modality="tabular",
       tsne_n_iter=250,  # Reduce iterations
       max_vlm_image_size=512,  # Smaller images
       vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"  # Smaller model
   )

Memory Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory-efficient configuration
   classifier = MarvisTsneClassifier(
       modality="tabular",
       max_tabpfn_samples=500,  # Limit samples
       gpu_memory_utilization=0.6,  # Conservative GPU usage
       tensor_parallel_size=1  # Single GPU
   )

Next Steps
----------

Now that you're familiar with the basics:

1. **Explore Modalities**: Dive deeper into :doc:`../user-guide/vision/index`, :doc:`../user-guide/audio/index`, or :doc:`../user-guide/tabular/index`

2. **Advanced Features**: Learn about :doc:`../technical-guides/resource-management` and :doc:`../technical-guides/caching-system`

3. **Tutorials**: Follow comprehensive :doc:`../tutorials/basic-classification` tutorials

4. **API Reference**: Browse the complete :doc:`../api-reference/marvis.models` documentation

Need Help?
----------

* **Troubleshooting**: :doc:`../troubleshooting`
* **Examples**: :doc:`../examples/index`
* **GitHub Issues**: https://github.com/penfever/marvis/issues