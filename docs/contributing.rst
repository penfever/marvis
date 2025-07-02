Contributing to MARVIS
====================

We welcome contributions to MARVIS! This guide will help you get started.

.. note::
   This contributing guide is under development. Basic guidelines are provided below.

Getting Started
---------------

1. **Fork the Repository**

   .. code-block:: bash

      git clone https://github.com/your-username/marvis.git
      cd marvis

2. **Create Development Environment**

   .. code-block:: bash

      python -m venv marvis-dev
      source marvis-dev/bin/activate  # On Windows: marvis-dev\\Scripts\\activate
      pip install -e ".[vision,audio,api,dev]"

3. **Create Feature Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Guidelines
---------------------

Code Style
~~~~~~~~~~

* Follow PEP 8 style guidelines
* Use type hints where possible
* Add docstrings for all public functions
* Keep line length under 100 characters

Testing
~~~~~~~

* Add tests for new functionality
* Ensure all existing tests pass
* Use pytest for testing framework

Documentation
~~~~~~~~~~~~~

* Update docstrings for API changes
* Add examples for new features
* Update this documentation as needed

Submitting Changes
------------------

1. **Commit Your Changes**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of changes"

2. **Push to Your Fork**

   .. code-block:: bash

      git push origin feature/your-feature-name

3. **Create Pull Request**

   * Describe your changes clearly
   * Include tests and documentation updates
   * Reference any related issues

Areas for Contribution
----------------------

We especially welcome contributions in:

* New modality support
* Performance optimizations  
* Documentation improvements
* Example scripts and tutorials
* Bug fixes and testing

Code of Conduct
---------------

Please be respectful and constructive in all interactions.

Questions?
----------

* Open an issue for bug reports
* Start a discussion for feature requests
* Check existing issues before creating new ones

Thank you for contributing to MARVIS!