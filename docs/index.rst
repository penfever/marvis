MARVIS: Classification using Language Model embeddings
======================================================

**MARVIS** (Classification using Language Model embeddings) is a powerful framework for multi-modal classification that leverages Vision Language Models (VLMs) to perform classification on tabular, audio, and vision data through intelligent visualization and embedding techniques.

🚀 **Quick Start**
------------------

.. code-block:: bash

   pip install -e ".[vision,audio,api]"

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.datasets import make_classification
   
   # Create classifier
   classifier = MarvisTsneClassifier(modality="tabular")
   
   # Generate sample data
   X, y = make_classification(n_samples=100, n_features=10, n_classes=3)
   
   # Fit and predict
   classifier.fit(X, y)
   predictions = classifier.predict(X)

🌟 **Key Features**
-------------------

* **Multi-modal Support**: Tabular, audio, and vision data classification
* **Vision Language Models**: Leverages state-of-the-art VLMs for intelligent reasoning
* **Advanced Visualizations**: t-SNE, PCA, UMAP, and multi-visualization frameworks
* **API Integration**: Support for OpenAI, Google Gemini, and local models
* **Rich Embeddings**: TabPFN, Whisper, DINOV2, and more
* **Resource Management**: Intelligent caching and memory optimization

📋 **Documentation Contents**
-----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quick-start
   getting-started/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/vision/index
   user-guide/audio/index
   user-guide/tabular/index
   user-guide/api-models/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic-classification
   tutorials/multi-modal-pipeline
   tutorials/custom-datasets

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api-reference/marvis.models
   api-reference/marvis.data
   api-reference/marvis.utils
   api-reference/marvis.viz

.. toctree::
   :maxdepth: 2
   :caption: Technical Guides

   technical-guides/resource-management
   technical-guides/caching-system
   technical-guides/evaluation-frameworks

.. toctree::
   :maxdepth: 1
   :caption: Examples & Troubleshooting

   examples/index
   troubleshooting
   changelog
   contributing

🔗 **Quick Links**
------------------

* **GitHub Repository**: https://github.com/penfever/marvis
* **Issue Tracker**: https://github.com/penfever/marvis/issues
* **PyPI Package**: https://pypi.org/project/marvis/

🏗️ **Supported Modalities**
----------------------------

Vision
~~~~~~
* CIFAR-10/100, ImageNet, custom image datasets
* DINOV2 embeddings with t-SNE visualization
* API model integration (GPT-4V, Gemini Vision)

Audio
~~~~~
* ESC-50, RAVDESS, AudioSet, custom audio datasets  
* Whisper and CLAP embeddings
* Spectrogram visualization support

Tabular
~~~~~~~
* OpenML datasets, UCI repository, custom CSV data
* TabPFN embeddings with advanced preprocessing
* Feature selection and dimensionality reduction

💡 **Getting Help**
-------------------

If you encounter issues or have questions:

1. Check the :doc:`troubleshooting` guide
2. Browse existing `GitHub Issues <https://github.com/penfever/marvis/issues>`_
3. Create a new issue with a minimal reproducible example
4. Join our community discussions

**License**: MIT License

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`