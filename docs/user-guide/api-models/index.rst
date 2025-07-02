API Models Integration Guide
===============================

Guide for integrating MARVIS with commercial API models.

.. note::
   This section is under development. Please refer to the examples in the meantime.

Overview
--------

MARVIS supports integration with:

* OpenAI GPT-4V and GPT-4o models
* Google Gemini Vision models
* Anthropic Claude models (future support)

Quick Start
-----------

OpenAI Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key"

   from marvis.models.marvis_tsne import MarvisTsneClassifier

   classifier = MarvisTsneClassifier(
       modality="vision",
       openai_model="gpt-4o",
       enable_thinking=True
   )

Google Gemini Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   os.environ["GOOGLE_API_KEY"] = "your-api-key"

   classifier = MarvisTsneClassifier(
       modality="vision",
       gemini_model="gemini-2.0-flash-exp"
   )

Examples
--------

See ``examples/vision/`` directory for API model examples.

API Reference
-------------

See :doc:`../../api-reference/marvis.models` for detailed API documentation.