Troubleshooting Guide
====================

Common issues and their solutions when using MARVIS.

Installation Issues
-------------------

ImportError: No module named 'marvis'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: MARVIS module not found after installation.

**Solutions**:

1. **Verify installation**:

   .. code-block:: bash

      pip list | grep marvis

2. **Reinstall in editable mode**:

   .. code-block:: bash

      cd /path/to/marvis
      pip install -e .

3. **Check Python path**:

   .. code-block:: python

      import sys
      print(sys.path)

CUDA/GPU Issues
~~~~~~~~~~~~~~~

**Problem**: CUDA out of memory errors.

**Solutions**:

1. **Reduce GPU memory utilization**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          modality="tabular",
          gpu_memory_utilization=0.6  # Reduce from default 0.9
      )

2. **Use smaller images**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          max_vlm_image_size=512  # Reduce from default 2048
      )

3. **Enable CPU fallback**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          device="cpu"  # Force CPU usage
      )

Model Loading Issues
--------------------

HuggingFace Model Download Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Models fail to download from HuggingFace Hub.

**Solutions**:

1. **Check internet connection**:

   .. code-block:: bash

      ping huggingface.co

2. **Clear HuggingFace cache**:

   .. code-block:: bash

      rm -rf ~/.cache/huggingface/

3. **Set HuggingFace token** (for gated models):

   .. code-block:: bash

      export HUGGINGFACE_HUB_TOKEN="your-token"

Model Too Large for Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: VLM models are too large for available memory.

**Solutions**:

1. **Use smaller models**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"  # Instead of 32B
      )

2. **Enable tensor parallelism** (multiple GPUs):

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          tensor_parallel_size=2  # Use 2 GPUs
      )

API Integration Issues
----------------------

OpenAI API Errors
~~~~~~~~~~~~~~~~~

**Problem**: OpenAI API calls failing.

**Solutions**:

1. **Check API key**:

   .. code-block:: bash

      echo $OPENAI_API_KEY

2. **Verify account balance**: Check your OpenAI account billing.

3. **Handle rate limits**:

   .. code-block:: python

      import time
      import openai

      try:
          classifier = MarvisTsneClassifier(openai_model="gpt-4o")
          classifier.fit(X, y)
      except openai.RateLimitError:
          time.sleep(60)  # Wait and retry

Google Gemini API Errors
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Gemini API calls failing.

**Solutions**:

1. **Check API key**:

   .. code-block:: bash

      echo $GOOGLE_API_KEY

2. **Verify API access**: Ensure Gemini API is enabled in Google Cloud Console.

3. **Handle safety filters**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          gemini_model="gemini-2.0-flash-exp",
          # Adjust safety settings if needed
      )

Data Processing Issues
----------------------

TabPFN Embedding Errors
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: TabPFN fails to generate embeddings.

**Solutions**:

1. **Check data size**:

   .. code-block:: python

      print(f"Dataset size: {X.shape}")
      # TabPFN works best with < 3000 samples

2. **Reduce sample size**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          max_tabpfn_samples=1000  # Reduce from default 3000
      )

3. **Handle categorical data**:

   .. code-block:: python

      from sklearn.preprocessing import LabelEncoder
      
      # Encode categorical columns
      for col in categorical_columns:
          le = LabelEncoder()
          X[col] = le.fit_transform(X[col])

Audio Processing Errors
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Audio files fail to load or process.

**Solutions**:

1. **Check audio format**:

   .. code-block:: bash

      file audio_file.wav

2. **Convert audio format**:

   .. code-block:: bash

      ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

3. **Handle missing audio files**:

   .. code-block:: python

      import os
      audio_paths = [p for p in audio_paths if os.path.exists(p)]

Image Processing Errors
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Images fail to load or are corrupt.

**Solutions**:

1. **Validate image files**:

   .. code-block:: python

      from PIL import Image
      
      def validate_image(path):
          try:
              Image.open(path).verify()
              return True
          except:
              return False

2. **Handle corrupt images**:

   .. code-block:: python

      valid_images = [p for p in image_paths if validate_image(p)]

Performance Issues
------------------

Slow Training/Inference
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: MARVIS is running slower than expected.

**Solutions**:

1. **Enable caching**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          cache_dir="./marvis_cache"  # Enable caching
      )

2. **Reduce t-SNE iterations**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          tsne_n_iter=250  # Reduce from default 1000
      )

3. **Use smaller perplexity**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          tsne_perplexity=15  # Reduce from default 30
      )

High Memory Usage
~~~~~~~~~~~~~~~~

**Problem**: MARVIS uses too much memory.

**Solutions**:

1. **Limit embedding samples**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          max_tabpfn_samples=500,
          max_train_plot_samples=500
      )

2. **Disable multi-visualization**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          enable_multi_viz=False
      )

3. **Use smaller image sizes**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(
          max_vlm_image_size=256
      )

Configuration Issues
--------------------

Invalid Configuration
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Configuration parameters are rejected.

**Solutions**:

1. **Validate parameters**:

   .. code-block:: python

      try:
          classifier = MarvisTsneClassifier(**config)
      except ValueError as e:
          print(f"Configuration error: {e}")

2. **Check parameter types**:

   .. code-block:: python

      # Ensure correct types
      config = {
          "modality": "tabular",        # str
          "use_3d": False,              # bool
          "tsne_perplexity": 30,        # int
          "gpu_memory_utilization": 0.8 # float
      }

Environment Variable Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Environment variables not being recognized.

**Solutions**:

1. **Check variable names**:

   .. code-block:: bash

      env | grep MARVIS
      env | grep OPENAI
      env | grep GOOGLE

2. **Set in Python**:

   .. code-block:: python

      import os
      os.environ["OPENAI_API_KEY"] = "your-key"

Visualization Issues
--------------------

Empty or Corrupted Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: t-SNE plots are empty or corrupted.

**Solutions**:

1. **Check data dimensions**:

   .. code-block:: python

      print(f"Embeddings shape: {embeddings.shape}")
      print(f"Labels shape: {labels.shape}")

2. **Validate perplexity**:

   .. code-block:: python

      # Perplexity should be < n_samples / 3
      n_samples = len(embeddings)
      max_perplexity = n_samples // 3
      perplexity = min(30, max_perplexity)

3. **Check for NaN values**:

   .. code-block:: python

      import numpy as np
      print(f"NaN in embeddings: {np.isnan(embeddings).any()}")

3D Visualization Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: 3D visualizations fail to render.

**Solutions**:

1. **Check matplotlib backend**:

   .. code-block:: python

      import matplotlib
      print(matplotlib.get_backend())

2. **Update dependencies**:

   .. code-block:: bash

      pip install --upgrade matplotlib

3. **Use 2D fallback**:

   .. code-block:: python

      classifier = MarvisTsneClassifier(use_3d=False)

Debugging Tips
--------------

Enable Detailed Logging
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

   # MARVIS will now show detailed debug information

Check System Resources
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import torch

   print(f"CPU usage: {psutil.cpu_percent()}%")
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   
   if torch.cuda.is_available():
       print(f"GPU memory: {torch.cuda.memory_allocated()} / {torch.cuda.max_memory_allocated()}")

Test with Minimal Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Minimal test case
   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.datasets import make_classification

   X, y = make_classification(n_samples=20, n_features=5, n_classes=2)
   
   classifier = MarvisTsneClassifier(
       modality="tabular",
       tsne_n_iter=100,  # Minimal iterations
       max_vlm_image_size=256  # Small images
   )
   
   try:
       classifier.fit(X, y)
       print("✓ Basic functionality working")
   except Exception as e:
       print(f"✗ Error: {e}")

Getting Additional Help
-----------------------

Community Resources
~~~~~~~~~~~~~~~~~~

1. **GitHub Issues**: https://github.com/penfever/marvis/issues
2. **Documentation**: https://marvis.readthedocs.io
3. **Examples**: Check ``examples/`` directory

Creating Bug Reports
~~~~~~~~~~~~~~~~~~~~

When reporting bugs, please include:

1. **System information**:

   .. code-block:: python

      import platform
      import sys
      import torch
      
      print(f"Python: {sys.version}")
      print(f"Platform: {platform.platform()}")
      print(f"PyTorch: {torch.__version__}")
      print(f"CUDA available: {torch.cuda.is_available()}")

2. **Minimal reproducible example**
3. **Full error traceback**
4. **Configuration used**
5. **Steps to reproduce**

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import psutil

   def monitor_performance():
       start_time = time.time()
       start_memory = psutil.virtual_memory().used
       
       # Your MARVIS code here
       classifier.fit(X, y)
       
       end_time = time.time()
       end_memory = psutil.virtual_memory().used
       
       print(f"Execution time: {end_time - start_time:.2f} seconds")
       print(f"Memory used: {(end_memory - start_memory) / 1024**2:.2f} MB")

Common Error Messages
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Error Message
     - Solution
   * - ``CUDA out of memory``
     - Reduce ``gpu_memory_utilization`` or ``max_vlm_image_size``
   * - ``No module named 'marvis'``
     - Reinstall with ``pip install -e .``
   * - ``API key not found``
     - Set environment variables for API keys
   * - ``Model not found``
     - Check model ID spelling and HuggingFace availability
   * - ``Invalid perplexity``
     - Reduce perplexity or increase dataset size
   * - ``TabPFN embedding failed``
     - Reduce ``max_tabpfn_samples`` or check data format

This troubleshooting guide covers the most common issues. For additional help, please check our GitHub issues or create a new issue with detailed information about your problem.