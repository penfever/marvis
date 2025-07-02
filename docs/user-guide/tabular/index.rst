Tabular Data Classification Guide
==================================

Comprehensive guide for tabular data classification with MARVIS.

.. note::
   This section is under development. Please refer to the examples in the meantime.

Overview
--------

MARVIS supports tabular data classification through:

* TabPFN embeddings for feature extraction
* t-SNE/PCA visualization of tabular embeddings
* Vision Language Model classification
* Support for OpenML datasets, UCI repository, and custom CSV data

Quick Start
-----------

.. code-block:: python

   from marvis.models.marvis_tsne import MarvisTsneClassifier
   from sklearn.datasets import make_classification

   # Create sample data
   X, y = make_classification(n_samples=100, n_features=10, n_classes=3)

   # Create and train classifier
   classifier = MarvisTsneClassifier(modality="tabular")
   classifier.fit(X, y)
   predictions = classifier.predict(X)

Examples
--------

See ``examples/tabular/`` directory for complete examples.

API Reference
-------------

See :doc:`../../api-reference/marvis.models` for detailed API documentation.