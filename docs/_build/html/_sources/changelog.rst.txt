Changelog
=========

All notable changes to MARVIS will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
* Sphinx documentation infrastructure
* Unified 2D/3D visualization system
* Metadata integration for enhanced VLM prompts
* Multi-visualization framework with adaptive layouts
* Comprehensive troubleshooting guide
* **Apple Silicon (MPS) Support**: Automatic detection and use of Metal Performance Shaders for GPU acceleration on M1/M2/M3/M4 Macs
* Platform detection utilities for optimal device selection
* MPS-specific configuration and optimization

Changed
~~~~~~~
* Streamlined README from 606 to ~150 lines
* Unified ``use_3d`` parameter (deprecated ``use_3d_tsne``)
* Enhanced VLM prompting with metadata context
* **Model Loading**: Improved device detection to support MPS alongside CUDA and CPU
* **Platform Optimization**: Automatic dtype selection (float32 for MPS, bfloat16 for CUDA)
* Added MPS detection test (`test_mps_detection.py`)

Fixed
~~~~~
* Sequential layout for multi-visualization now ensures consistent sizing
* Semantic names properly loaded from CC18 data
* KNN connections now supported across all visualization types

[1.0.0] - 2024-XX-XX
---------------------

Added
~~~~~
* Initial release of MARVIS framework
* Multi-modal classification support (tabular, audio, vision)
* Vision Language Model integration
* t-SNE visualization with KNN connections
* TabPFN embeddings for tabular data
* Whisper and CLAP embeddings for audio
* DINOV2 embeddings for vision
* OpenAI and Google Gemini API integration
* Resource management and caching system
* OpenML CC18 benchmark support

Technical Implementation
~~~~~~~~~~~~~~~~~~~~~~~~
* Core ``MarvisTsneClassifier`` with modality dispatch
* VLM prompting utilities with thinking mode
* Embedding caching for performance optimization
* Multi-GPU and distributed inference support
* Comprehensive evaluation frameworks

Examples and Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~
* Complete examples for all modalities
* Jupyter notebook tutorials
* API integration examples
* Performance optimization guides

[0.9.0] - Development
---------------------

This represents the pre-release development phase where core functionality was established.

Migration Guide
---------------

From Pre-Documentation to v1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main changes for users:

1. **Parameter Updates**:

   .. code-block:: python

      # Old (deprecated but still works)
      classifier = MarvisTsneClassifier(use_3d_tsne=True)
      
      # New (recommended)
      classifier = MarvisTsneClassifier(use_3d=True)

2. **Documentation Structure**:

   * README simplified - full docs now in Sphinx
   * Examples organized by modality
   * Comprehensive API reference available

3. **Enhanced Features**:

   * Metadata integration for richer VLM prompts
   * Multi-visualization framework
   * Improved resource management

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

All existing code should continue to work with deprecation warnings for:

* ``use_3d_tsne`` parameter (use ``use_3d`` instead)
* Some internal API changes (external API unchanged)

Future Releases
---------------

Planned features for future releases:

**v1.1.0**
* Advanced multi-modal fusion techniques
* Improved API model integration
* Enhanced caching strategies

**v1.2.0**
* Scikit-learn pipeline compatibility
* Automated hyperparameter optimization
* Extended evaluation metrics

**v2.0.0**
* Unified classifier interface across modalities
* Breaking changes for cleaner API
* Advanced visualization techniques

Contributing to Changelog
-------------------------

When contributing to MARVIS:

1. Add entries to the ``[Unreleased]`` section
2. Use consistent formatting and categories
3. Include migration notes for breaking changes
4. Reference relevant issues/PRs when available