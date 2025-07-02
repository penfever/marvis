Audio Classification Guide
===========================

Comprehensive guide for audio classification with MARVIS.

.. note::
   This section is under development. Please refer to the examples in the meantime.

Overview
--------

MARVIS supports audio classification through:

* Whisper and CLAP embeddings for feature extraction
* t-SNE/PCA visualization of audio embeddings
* Vision Language Model classification with spectrograms
* Support for ESC-50, RAVDESS, and custom audio datasets

Quick Start
-----------

.. code-block:: bash

   # Test ESC-50 classification
   python examples/audio/evaluate_all_audio.py \\
       --datasets esc50 \\
       --models marvis_tsne

Examples
--------

See ``examples/audio/`` directory for complete examples.

API Reference
-------------

See :doc:`../../api-reference/marvis.models` for detailed API documentation.