Configuration Guide
===================

This guide covers how to configure MARVIS for optimal performance across different use cases.

Configuration Methods
---------------------

MARVIS can be configured through multiple methods:

1. **Constructor Parameters** - Direct parameter passing
2. **Configuration Files** - YAML/JSON configuration files  
3. **Environment Variables** - System environment variables
4. **Runtime Settings** - Dynamic configuration changes

Constructor Parameters
----------------------

The most direct way to configure MARVIS:

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   classifier = MarvisTsneClassifier(
       modality="tabular",
       vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
       use_3d=True,
       use_knn_connections=True,
       nn_k=5,
       max_vlm_image_size=2048,
       gpu_memory_utilization=0.8,
       cache_dir="./cache"
   )

Configuration Files
-------------------

YAML Configuration
~~~~~~~~~~~~~~~~~~

Create ``marvis_config.yaml``:

.. code-block:: yaml

   # Core settings
   modality: "tabular"
   vlm_model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
   
   # Visualization settings
   use_3d: false
   use_knn_connections: true
   nn_k: 5
   tsne_perplexity: 30
   tsne_n_iter: 1000
   
   # Multi-visualization
   enable_multi_viz: false
   visualization_methods: ["tsne", "pca"]
   layout_strategy: "adaptive_grid"
   reasoning_focus: "classification"
   
   # Performance settings
   max_vlm_image_size: 2048
   image_dpi: 100
   gpu_memory_utilization: 0.8
   tensor_parallel_size: 1
   
   # Resource management
   cache_dir: "./cache"
   max_tabpfn_samples: 3000
   
   # API settings
   enable_thinking: true

Load configuration:

.. code-block:: python

   import yaml
   from marvis.models.marvis_tsne import MarvisTsneClassifier

   with open("marvis_config.yaml") as f:
       config = yaml.safe_load(f)
   
   classifier = MarvisTsneClassifier(**config)

JSON Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "modality": "vision",
     "vlm_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
     "dinov2_model": "dinov2_vitb14",
     "use_3d": true,
     "max_vlm_image_size": 1024,
     "cache_dir": "./vision_cache"
   }

Environment Variables
---------------------

System Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # API Keys
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   
   # Cache and data directories
   export MARVIS_CACHE_DIR="/path/to/cache"
   export MARVIS_DATA_DIR="/path/to/data"
   
   # CUDA settings
   export CUDA_VISIBLE_DEVICES="0,1"
   export CUDA_DEVICE_ORDER="PCI_BUS_ID"
   
   # Model settings
   export MARVIS_DEFAULT_VLM="Qwen/Qwen2.5-VL-3B-Instruct"
   export MARVIS_MAX_IMAGE_SIZE="2048"

Python Environment
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   # Set environment variables in Python
   os.environ["MARVIS_CACHE_DIR"] = "./cache"
   os.environ["MARVIS_MAX_IMAGE_SIZE"] = "1024"

   # MARVIS will automatically use these settings
   from marvis.models.marvis_tsne import MarvisTsneClassifier
   classifier = MarvisTsneClassifier(modality="tabular")

Modality-Specific Configuration
-------------------------------

Tabular Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Optimal for tabular data
   modality: "tabular"
   embedding_size: 1000
   max_tabpfn_samples: 3000
   use_semantic_names: true
   
   # TabPFN-specific settings
   tabular_config:
     feature_selection: true
     normalize_features: true
     handle_categorical: true

Audio Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Optimal for audio data
   modality: "audio"
   embedding_model: "whisper"
   whisper_model: "large-v2"
   include_spectrogram: true
   audio_duration: 30.0
   
   # Audio-specific settings
   audio_config:
     sample_rate: 16000
     n_mels: 80
     hop_length: 512

Vision Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Optimal for vision data
   modality: "vision"
   dinov2_model: "dinov2_vitb14"
   max_train_plot_samples: 1000
   
   # Vision-specific settings
   vision_config:
     image_size: [224, 224]
     normalize: true
     augment: false

Performance Configurations
--------------------------

Development/Fast Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick testing and development:

.. code-block:: yaml

   # Fast configuration for development
   modality: "tabular"
   vlm_model_id: "Qwen/Qwen2.5-VL-3B-Instruct"  # Smaller model
   tsne_n_iter: 250  # Fewer iterations
   max_vlm_image_size: 512  # Smaller images
   max_tabpfn_samples: 500  # Fewer samples
   gpu_memory_utilization: 0.6  # Conservative

Production/High-Quality Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production use with best quality:

.. code-block:: yaml

   # Production configuration
   modality: "tabular"
   vlm_model_id: "Qwen/Qwen2.5-VL-32B-Instruct"  # Larger model
   tsne_n_iter: 1000  # More iterations
   max_vlm_image_size: 4096  # High resolution
   max_tabpfn_samples: 5000  # More samples
   gpu_memory_utilization: 0.9  # Aggressive
   
   # Quality settings
   enable_multi_viz: true
   visualization_methods: ["pca", "tsne", "umap", "spectral"]
   reasoning_focus: "consensus"

Memory-Constrained Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For limited memory environments:

.. code-block:: yaml

   # Memory-efficient configuration
   modality: "tabular"
   vlm_model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
   max_vlm_image_size: 256  # Very small images
   max_tabpfn_samples: 200  # Limited samples
   gpu_memory_utilization: 0.4  # Very conservative
   tensor_parallel_size: 1  # Single GPU
   
   # Disable memory-intensive features
   enable_multi_viz: false
   use_3d: false

Hardware and Platform Configuration
-----------------------------------

Device Selection
~~~~~~~~~~~~~~~~

MARVIS automatically detects the optimal device for your hardware:

.. code-block:: yaml

   # Automatic device detection (default)
   device: "auto"  # Automatically selects MPS on Mac, CUDA on Linux/Windows, CPU otherwise
   
   # Explicit device selection
   device: "mps"   # Force Metal Performance Shaders (Apple Silicon)
   device: "cuda"  # Force CUDA (NVIDIA GPUs)  
   device: "cpu"   # Force CPU-only execution

Backend Selection
~~~~~~~~~~~~~~~~~

Choose between VLLM (fast) and transformers (compatible) backends:

.. code-block:: yaml

   # Backend configuration
   backend: "auto"         # Automatically choose best backend
   backend: "vllm"         # Force VLLM (CUDA only, fastest)
   backend: "transformers" # Force transformers (MPS/CUDA/CPU compatible)

Apple Silicon Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimized settings for M1/M2/M3/M4 Macs:

.. code-block:: yaml

   # Apple Silicon optimized
   device: "mps"
   backend: "transformers"  # VLLM doesn't support MPS
   torch_dtype: "float32"   # MPS performs better with float32
   low_cpu_mem_usage: true
   
   # Force transformers backend via environment
   # export VLLM_AVAILABLE=false

NVIDIA GPU Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Settings for CUDA-enabled GPUs:

.. code-block:: yaml

   # NVIDIA GPU optimized
   device: "cuda"
   backend: "vllm"  # Use VLLM for fastest inference
   torch_dtype: "bfloat16"  # Better numerical stability than float16
   tensor_parallel_size: 1  # Multi-GPU if available
   gpu_memory_utilization: 0.9

API Model Configurations
------------------------

OpenAI Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # OpenAI GPT-4V configuration
   modality: "vision"
   openai_model: "gpt-4o"
   enable_thinking: true
   max_vlm_image_size: 2048
   
   # API-specific settings
   api_config:
     max_tokens: 4096
     temperature: 0.1
     timeout: 60

Google Gemini Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Google Gemini configuration
   modality: "vision"
   gemini_model: "gemini-2.0-flash-exp"
   enable_thinking: true
   
   # Gemini-specific settings
   gemini_config:
     safety_settings: "low"
     candidate_count: 1

Configuration Validation
------------------------

Validate Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   import yaml

   # Load and validate configuration
   with open("config.yaml") as f:
       config = yaml.safe_load(f)

   try:
       classifier = MarvisTsneClassifier(**config)
       print("✓ Configuration valid")
   except ValueError as e:
       print(f"✗ Configuration error: {e}")
   except Exception as e:
       print(f"✗ Unexpected error: {e}")

Configuration Profiles
----------------------

Create reusable configuration profiles:

Development Profile
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # dev_config.py
   DEV_CONFIG = {
       "modality": "tabular",
       "vlm_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
       "tsne_n_iter": 250,
       "max_vlm_image_size": 512,
       "gpu_memory_utilization": 0.6
   }

Production Profile
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # prod_config.py
   PROD_CONFIG = {
       "modality": "tabular",
       "vlm_model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
       "tsne_n_iter": 1000,
       "max_vlm_image_size": 2048,
       "gpu_memory_utilization": 0.9,
       "enable_multi_viz": True,
       "visualization_methods": ["pca", "tsne", "umap"]
   }

Use profiles:

.. code-block:: python

   from dev_config import DEV_CONFIG
   from prod_config import PROD_CONFIG
   
   # Choose profile based on environment
   import os
   if os.getenv("MARVIS_ENV") == "production":
       config = PROD_CONFIG
   else:
       config = DEV_CONFIG
   
   classifier = MarvisTsneClassifier(**config)

Best Practices
--------------

1. **Start with Defaults**: Begin with default configurations and adjust as needed
2. **Profile by Use Case**: Create specific profiles for development, testing, and production
3. **Monitor Resources**: Adjust memory and GPU utilization based on your hardware
4. **Cache Settings**: Always configure caching for repeated experiments
5. **Validate Early**: Test configurations with small datasets before scaling up
6. **Document Changes**: Keep track of configuration changes and their impacts

Troubleshooting
---------------

Common Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Invalid Model ID**
   Check that the model ID exists on HuggingFace Hub or your local system.

**Memory Errors**
   Reduce ``max_vlm_image_size``, ``max_tabpfn_samples``, or ``gpu_memory_utilization``.

**API Key Issues**
   Ensure API keys are set correctly and have the necessary permissions.

**Cache Permission Errors**
   Check that the cache directory is writable and has sufficient space.

Next Steps
----------

* **Performance Tuning**: :doc:`../technical-guides/resource-management`
* **Caching Setup**: :doc:`../technical-guides/caching-system`
* **Troubleshooting**: :doc:`../troubleshooting`