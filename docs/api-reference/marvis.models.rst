marvis.models
=============

Core classification models and utilities for multi-modal data.

.. currentmodule:: marvis.models

Main Classifier
---------------

.. autoclass:: marvis.models.marvis_tsne.MarvisTsneClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~MarvisTsneClassifier.fit
      ~MarvisTsneClassifier.predict
      ~MarvisTsneClassifier.predict_proba
      ~MarvisTsneClassifier.score
      ~MarvisTsneClassifier.evaluate

Utility Functions
-----------------

KNN Utilities
~~~~~~~~~~~~~

.. automodule:: marvis.models.knn_utils
   :members:
   :undoc-members:

Vector Quantization
~~~~~~~~~~~~~~~~~~~

.. automodule:: marvis.models.vq
   :members:
   :undoc-members:

   .. autoclass:: marvis.models.vq.vector_quantizer.VectorQuantizer
      :members:
      :undoc-members:
      :show-inheritance:

   .. note::
      Vector quantization modules are available for advanced use cases.

Model Configuration
-------------------

The main classifier accepts the following key parameters:

Core Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``modality``
     - str
     - Data modality: ``"tabular"``, ``"audio"``, or ``"vision"``
   * - ``vlm_model_id``
     - str
     - Vision Language Model identifier (e.g., ``"Qwen/Qwen2.5-VL-3B-Instruct"``)
   * - ``use_3d``
     - bool
     - Whether to use 3D visualizations (default: False)
   * - ``use_knn_connections``
     - bool
     - Whether to show KNN connections in visualizations (default: False)

Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``tsne_perplexity``
     - int
     - t-SNE perplexity parameter (default: 30)
   * - ``tsne_n_iter``
     - int
     - Number of t-SNE iterations (default: 1000)
   * - ``enable_multi_viz``
     - bool
     - Enable multi-visualization framework (default: False)
   * - ``visualization_methods``
     - List[str]
     - Visualization methods to use (e.g., ``["pca", "tsne", "umap"]``)

API Model Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``api_model``
     - str
     - Generic API model (auto-detects provider)
   * - ``openai_model``
     - str
     - OpenAI model (e.g., ``"gpt-4o"``)
   * - ``gemini_model``
     - str
     - Google Gemini model (e.g., ``"gemini-2.0-flash-exp"``)
   * - ``enable_thinking``
     - bool
     - Enable thinking mode for API models (default: True)

Resource Management
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``max_vlm_image_size``
     - int
     - Maximum image size for VLM (default: 2048)
   * - ``gpu_memory_utilization``
     - float
     - GPU memory utilization factor (default: 0.9)
   * - ``cache_dir``
     - str
     - Directory for caching embeddings
   * - ``max_tabpfn_samples``
     - int
     - Maximum samples for TabPFN (default: 3000)

Modality-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audio Parameters
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``embedding_model``
     - str
     - Audio embedding model: ``"whisper"`` or ``"clap"``
   * - ``whisper_model``
     - str
     - Whisper model variant (default: ``"large-v2"``)
   * - ``include_spectrogram``
     - bool
     - Include spectrogram in prompts (default: True)

Vision Parameters
^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``dinov2_model``
     - str
     - DINOV2 model variant (default: ``"dinov2_vitb14"``)
   * - ``use_pca_backend``
     - bool
     - Use PCA instead of t-SNE (default: False)

Usage Examples
--------------

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.datasets import make_classification

   # Create sample data
   X, y = make_classification(n_samples=100, n_features=10, n_classes=3)

   # Basic classifier
   classifier = MarvisTsneClassifier(modality="tabular")
   classifier.fit(X, y)
   predictions = classifier.predict(X)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced classifier with 3D visualization
   classifier = MarvisTsneClassifier(
       modality="tabular",
       vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
       use_3d=True,
       use_knn_connections=True,
       knn_k=5,
       tsne_perplexity=25,
       max_vlm_image_size=1024,
       cache_dir="./cache"
   )

Multi-Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-visualization framework
   classifier = MarvisTsneClassifier(
       modality="tabular",
       enable_multi_viz=True,
       visualization_methods=["pca", "tsne", "spectral"],
       layout_strategy="adaptive_grid",
       reasoning_focus="comparison"
   )

API Model Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   # OpenAI GPT-4V
   classifier = MarvisTsneClassifier(
       modality="vision",
       openai_model="gpt-4o",
       enable_thinking=True
   )

   # Google Gemini
   classifier = MarvisTsneClassifier(
       modality="vision",
       gemini_model="gemini-2.0-flash-exp"
   )

Audio Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Whisper embeddings
   classifier = MarvisTsneClassifier(
       modality="audio",
       embedding_model="whisper",
       whisper_model="large-v2",
       include_spectrogram=True
   )

   # CLAP embeddings
   classifier = MarvisTsneClassifier(
       modality="audio",
       embedding_model="clap",
       clap_version="2023"
   )

Vision Classification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # DINOV2 embeddings
   classifier = MarvisTsneClassifier(
       modality="vision",
       dinov2_model="dinov2_vitb14",
       use_3d=False
   )

Error Handling
--------------

Common exceptions and how to handle them:

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   import logging

   try:
       classifier = MarvisTsneClassifier(
           modality="tabular",
           vlm_model_id="invalid-model"
       )
       classifier.fit(X, y)
   except ValueError as e:
       logging.error(f"Configuration error: {e}")
   except RuntimeError as e:
       logging.error(f"Runtime error: {e}")
   except Exception as e:
       logging.error(f"Unexpected error: {e}")

Best Practices
--------------

1. **Start Simple**: Begin with basic configuration and add complexity gradually
2. **Cache Embeddings**: Use ``cache_dir`` to avoid recomputing embeddings
3. **Monitor Resources**: Adjust ``gpu_memory_utilization`` based on your hardware
4. **Use Appropriate Models**: Smaller models for development, larger for production
5. **Validate Data**: Ensure your data format matches the expected modality

See Also
--------

* :doc:`../user-guide/tabular/index` - Tabular data guide
* :doc:`../user-guide/vision/index` - Vision data guide  
* :doc:`../user-guide/audio/index` - Audio data guide
* :doc:`../technical-guides/resource-management` - Resource optimization