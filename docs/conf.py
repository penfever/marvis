# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Force CPU-only mode for documentation builds
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['VLLM_AVAILABLE'] = 'false'
os.environ['MARVIS_DOCS_BUILD'] = 'true'

# Add the project root to Python path for autodoc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock modules that might not be available during docs build
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'torch',
    'torch.nn',
    'torch.nn.functional', 
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    'torchaudio',
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'datasets', 
    'tabpfn',
    'tabpfn.scripts.transformer_prediction_interface',
    'accelerate',
    'openai',
    'google.generativeai',
    'librosa',
    'soundfile',
    'msclap',
    'msclap.clap',
    'optimum',
    'vllm',
    'llama_cpp',
    'umap',
    'umap.umap_',
    'sklearn.manifold',
    'sklearn.decomposition', 
    'sklearn.preprocessing',
    'sklearn.metrics',
    'sklearn.metrics._scorer',
    'sklearn.metrics.pairwise',
    'sklearn.neighbors',
    'sklearn.ensemble',
    'sklearn.linear_model',
    'sklearn.model_selection',
    'sklearn.base',
    'openml',
    'openml.datasets',
    'openml.tasks',
    'wandb',
    'PIL',
    'PIL.Image',
    'cv2',
    'albumentations',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MARVIS'
copyright = '2024, MARVIS Development Team'
author = 'MARVIS Development Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# MyST Markdown support
myst_enable_extensions = [
    "deflist",
    "fieldlist", 
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = [
    'custom.css',
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Handle import errors gracefully - comprehensive mock list
autodoc_mock_imports = [
    # PyTorch ecosystem
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    'torchaudio',
    'torchaudio.transforms',
    'torchaudio.functional',
    
    # Transformers and HuggingFace
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'datasets',
    'accelerate',
    'optimum',
    'peft',
    
    # TabPFN and ML libraries
    'tabpfn',
    'tabpfn.scripts',
    'tabpfn.scripts.transformer_prediction_interface',
    
    # API clients
    'openai',
    'google',
    'google.generativeai',
    
    # Audio processing
    'librosa',
    'soundfile',
    'msclap',
    'msclap.clap',
    'whisper',
    
    # Vision processing  
    'cv2',
    'albumentations',
    'timm',
    'open_clip_torch',
    
    # VLM backends
    'vllm',
    'llama_cpp',
    
    # Dimensionality reduction and ML
    'umap',
    'umap.umap_',
    'sklearn',
    'sklearn.manifold',
    'sklearn.decomposition', 
    'sklearn.preprocessing',
    'sklearn.metrics',
    'sklearn.metrics._scorer',
    'sklearn.metrics.pairwise',
    'sklearn.neighbors',
    'sklearn.ensemble',
    'sklearn.linear_model',
    'sklearn.model_selection',
    'sklearn.base',
    
    # Data sources
    'openml',
    'openml.datasets',
    'openml.tasks',
    
    # Logging and monitoring
    'wandb',
    
    # Image processing
    'PIL',
    'PIL.Image',
]

# Suppress warnings for missing imports
suppress_warnings = [
    'autodoc.import_object',
    'autosummary.import_cycle',
    'config.cache',
]

# Additional autodoc configuration
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

# Don't fail on import errors
autodoc_inherit_docstrings = False

# Autosummary settings
autosummary_generate = True

# Intersphinx configuration for cross-referencing external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/', None),
}

# -- Custom configuration ---------------------------------------------------

# Add version info
version = release

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# Language for content autogenerated by Sphinx
language = 'en'

# List of patterns to ignore when looking for source files
exclude_patterns.extend([
    '**.ipynb_checkpoints',
    'build/*',
    'examples/**/cache/*',
    'examples/**/results/*',
    '**/test_*',
])

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    
    # Additional stuff for the LaTeX preamble.
    'preamble': '',
    
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'MARVIS.tex', 'MARVIS Documentation',
     'MARVIS Development Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'marvis', 'MARVIS Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'MARVIS', 'MARVIS Documentation',
     author, 'MARVIS', 'Multimodal Analysis and Reasoning with VISion language models.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
epub_identifier = 'https://github.com/anon/marvis'

# A unique identification for the text.
epub_uid = 'MARVIS'

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']