Installation Guide
==================

This guide covers the installation of MARVIS and its dependencies for different use cases.

Requirements
------------

**Python Version**: 3.8 or higher (3.9+ recommended)

**System Requirements**:

* **GPU**: 
  * NVIDIA GPU with CUDA support (recommended for VLM models)
  * Apple Silicon Mac (M1/M2/M3/M4) with Metal Performance Shaders support
* **Memory**: 8GB+ RAM (16GB+ recommended for large models)
* **Storage**: 5GB+ free space for models and cache

Basic Installation
------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/penfever/marvis.git
   cd marvis

   # Install in development mode
   pip install -e .

This installs the core MARVIS package with basic dependencies.

Modality-Specific Installation
------------------------------

Vision Models
~~~~~~~~~~~~~

For image classification with DINOV2 embeddings and vision models:

.. code-block:: bash

   pip install -e ".[vision]"

This includes:

* ``torch`` and ``torchvision`` for PyTorch models
* ``timm`` for pre-trained vision models  
* ``PIL`` for image processing
* ``opencv-python`` for advanced image operations

Audio Models
~~~~~~~~~~~~

For audio classification with Whisper and CLAP embeddings:

.. code-block:: bash

   pip install -e ".[audio]"

This includes:

* ``librosa`` for audio processing
* ``soundfile`` for audio I/O
* ``whisper`` for speech embeddings
* ``transformers`` for CLAP models

API Models  
~~~~~~~~~~

For integration with commercial API models (OpenAI, Google):

.. code-block:: bash

   pip install -e ".[api]"

This includes:

* ``openai`` client library
* ``google-generativeai`` for Gemini models
* ``anthropic`` for Claude models (future support)

Complete Installation
~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

   pip install -e ".[vision,audio,api]"

Advanced Installation
---------------------

VLLM Backend (Optional)
~~~~~~~~~~~~~~~~~~~~~~~

For faster local VLM inference with VLLM:

.. code-block:: bash

   pip install vllm

.. note::
   VLLM requires CUDA and is not compatible with Apple Silicon (MPS).
   See `VLLM installation guide <https://docs.vllm.ai/en/latest/getting_started/installation.html>`_ for details.

Apple Silicon Support
~~~~~~~~~~~~~~~~~~~~~

MARVIS automatically detects and uses Metal Performance Shaders (MPS) on Apple Silicon Macs:

.. code-block:: bash

   # Force transformers backend for MPS support
   export VLLM_AVAILABLE=false
   
   # Run with automatic MPS detection
   python your_script.py

To verify MPS is being used:

.. code-block:: bash

   python tests/test_mps_detection.py

.. note::
   The transformers backend provides full MPS support for GPU acceleration on Mac.
   Monitor GPU usage in Activity Monitor to verify MPS utilization.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributing to MARVIS development:

.. code-block:: bash

   # Install with development dependencies
   pip install -e ".[vision,audio,api,dev]"

   # Install pre-commit hooks
   pre-commit install

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

To build documentation locally:

.. code-block:: bash

   pip install -r docs/requirements.txt

Verification
------------

Test Your Installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test basic import
   import marvis
   print(f"MARVIS version: {marvis.__version__}")

   # Test core functionality
   from marvis.models.marvis_tsne import MarvisTsneClassifier
   classifier = MarvisTsneClassifier(modality="tabular")
   print("✓ Core MARVIS functionality available")

   # Test vision support (if installed)
   try:
       import torch
       import torchvision
       print("✓ Vision dependencies available")
   except ImportError:
       print("✗ Vision dependencies not installed")

   # Test audio support (if installed)
   try:
       import librosa
       import whisper
       print("✓ Audio dependencies available")
   except ImportError:
       print("✗ Audio dependencies not installed")

Quick Functionality Test
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Test with sample data
   python -c "
   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.datasets import make_classification
   import numpy as np

   # Create sample data
   X, y = make_classification(n_samples=50, n_features=10, n_classes=3, random_state=42)
   
   # Test tabular classification
   classifier = MarvisTsneClassifier(
       modality='tabular',
       vlm_model_id='Qwen/Qwen2.5-VL-3B-Instruct'
   )
   
   # This should complete without errors
   print('✓ MARVIS installation verified successfully')
   "

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'marvis'**
   Make sure you installed with ``pip install -e .`` and are in the correct directory.

**CUDA out of memory**
   Try using smaller VLM models or reducing batch sizes. See :doc:`../technical-guides/resource-management`.

**Model download failures**
   Check your internet connection. Models are downloaded from HuggingFace Hub on first use.

**Permission errors during installation**
   Try using ``pip install --user`` or create a virtual environment.

Virtual Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a virtual environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv marvis-env
   
   # Activate (Linux/Mac)
   source marvis-env/bin/activate
   
   # Activate (Windows)
   marvis-env\\Scripts\\activate
   
   # Install MARVIS
   pip install -e ".[vision,audio,api]"

Docker Installation
~~~~~~~~~~~~~~~~~~~

For a containerized environment:

.. code-block:: bash

   # Build Docker image (Dockerfile needed)
   docker build -t marvis:latest .
   
   # Run with GPU support
   docker run --gpus all -it marvis:latest

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**macOS**
   Some audio dependencies may require Homebrew: ``brew install ffmpeg``

**Windows**
   Make sure to install Visual Studio Build Tools for compiling native extensions.

**Linux**
   CUDA toolkit installation may be required for GPU support.

Getting Help
------------

If you encounter installation issues:

1. Check our `GitHub Issues <https://github.com/penfever/marvis/issues>`_
2. Review the :doc:`../troubleshooting` guide
3. Create a new issue with your system information and error messages

Next Steps
----------

After successful installation, proceed to:

* :doc:`quick-start` - Learn basic MARVIS usage
* :doc:`configuration` - Configure MARVIS for your needs
* :doc:`../tutorials/basic-classification` - Follow detailed tutorials