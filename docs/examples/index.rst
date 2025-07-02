Examples
========

Comprehensive examples for all MARVIS modalities and use cases.

Vision Examples
---------------

CIFAR Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic CIFAR-10 classification
   python examples/vision/evaluate_all_vision.py --datasets cifar10 --models marvis_tsne

   # Advanced features with 3D visualization
   python examples/vision/evaluate_all_vision.py \\
       --datasets cifar10 cifar100 \\
       --models marvis_tsne \\
       --use_3d \\
       --use_knn_connections \\
       --nn_k 5

API Model Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # OpenAI GPT-4V
   python examples/vision/openai_vlm_baseline.py --dataset cifar10

   # Google Gemini
   python examples/vision/gemini_vlm_baseline.py --dataset cifar10

Audio Examples
--------------

ESC-50 Classification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Whisper embeddings
   python examples/audio/evaluate_all_audio.py \\
       --datasets esc50 \\
       --models marvis_tsne \\
       --embedding_type whisper

   # CLAP embeddings
   python examples/audio/evaluate_all_audio.py \\
       --datasets esc50 \\
       --embedding_type clap

Tabular Examples
----------------

OpenML Datasets
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Single dataset evaluation
   python examples/tabular/evaluate_llm_baselines_tabular.py \\
       --dataset_name adult \\
       --models marvis_tsne

   # OpenML CC18 benchmark suite
   python examples/tabular/openml_cc18/run_openml_cc18_tabular.py \\
       --models marvis_tsne \\
       --task_ids 3 6 11 12 14

Multi-Modal Examples
--------------------

Unified Interface
~~~~~~~~~~~~~~~~~

See ``examples/unified_marvis_example.py`` for examples using MARVIS across all modalities.

Interactive Notebooks
----------------------

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

* ``notebooks/Getting_Started.ipynb`` - Complete MARVIS walkthrough
* ``examples/tabular/pfn_knn_expts.ipynb`` - TabPFN experiments

Example Files Overview
----------------------

Vision Examples (``examples/vision/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``evaluate_all_vision.py`` - Comprehensive vision evaluation
* ``openai_vlm_baseline.py`` - OpenAI API integration
* ``gemini_vlm_baseline.py`` - Google Gemini integration
* ``qwen_vl_baseline.py`` - Qwen VL local model

Audio Examples (``examples/audio/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``evaluate_all_audio.py`` - Comprehensive audio evaluation
* ``marvis_tsne_audio_baseline.py`` - MARVIS t-SNE baseline
* ``audio_baselines.py`` - Traditional audio baselines

Tabular Examples (``examples/tabular/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``evaluate_llm_baselines_tabular.py`` - LLM baseline evaluation
* ``openml_cc18/run_openml_cc18_tabular.py`` - OpenML CC18 runner
* ``train_tabular_dataset_tabular.py`` - Custom dataset training

Configuration Examples
----------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   # Simple configuration
   classifier = MarvisTsneClassifier(
       modality="tabular",
       vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced multi-visualization setup
   classifier = MarvisTsneClassifier(
       modality="tabular",
       enable_multi_viz=True,
       visualization_methods=["pca", "tsne", "spectral"],
       layout_strategy="adaptive_grid",
       reasoning_focus="comparison",
       use_3d=True,
       use_knn_connections=True,
       cache_dir="./cache"
   )

Performance Examples
--------------------

Resource Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory-efficient configuration
   classifier = MarvisTsneClassifier(
       modality="tabular",
       max_vlm_image_size=512,
       max_tabpfn_samples=1000,
       gpu_memory_utilization=0.6
   )

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple datasets
   datasets = ["adult", "credit-g", "diabetes"]
   
   for dataset_name in datasets:
       classifier = MarvisTsneClassifier(modality="tabular")
       # Load and process dataset
       # Save results

Complete Workflow Examples
--------------------------

End-to-End Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report

   # 1. Load data
   # X, y = load_your_data()

   # 2. Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # 3. Create and fit classifier
   classifier = MarvisTsneClassifier(modality="tabular")
   classifier.fit(X_train, y_train, X_test)

   # 4. Make predictions
   predictions = classifier.predict(X_test)

   # 5. Evaluate results
   print(classification_report(y_test, predictions))

   # 6. Get detailed results
   results = classifier.evaluate(X_test, y_test, return_detailed=True)

Running Examples
----------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have MARVIS installed with the appropriate dependencies:

.. code-block:: bash

   pip install -e ".[vision,audio,api]"

Environment Setup
~~~~~~~~~~~~~~~~~

For API models, set up your API keys:

.. code-block:: bash

   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"

Example Execution
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Navigate to MARVIS directory
   cd /path/to/marvis

   # Run vision example
   python examples/vision/evaluate_all_vision.py --datasets cifar10 --quick_test

   # Run audio example  
   python examples/audio/evaluate_all_audio.py --datasets esc50

   # Run tabular example
   python examples/tabular/evaluate_llm_baselines_tabular.py --dataset_name adult

Getting Help
------------

If you encounter issues with examples:

1. Check the :doc:`../troubleshooting` guide
2. Ensure all dependencies are installed
3. Verify your environment setup
4. Check GitHub issues for similar problems

Next Steps
----------

* **API Reference**: :doc:`../api-reference/marvis.models`
* **Configuration Guide**: :doc:`../getting-started/configuration`
* **Tutorials**: :doc:`../tutorials/basic-classification`