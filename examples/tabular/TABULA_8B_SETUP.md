# Tabula-8B Environment Setup

This document provides instructions for setting up the environment to use Tabula-8B specific baselines and dependencies.

## Overview

Tabula-8B related dependencies have been separated from the main MARVIS installation to avoid dependency conflicts, particularly with torch version requirements. These dependencies include:

- `rtdl` (requires torch<2.0, conflicts with MARVIS's torch>=2.0 requirement)
- `tableshift` (depends on rtdl)
- Various other research-specific packages

## Environment Setup

### Option 1: Separate Virtual Environment (Recommended)

Create a separate Python environment specifically for Tabula-8B work:

```bash
# Create a new virtual environment
python -m venv tabula8b_env
source tabula8b_env/bin/activate  # On Windows: tabula8b_env\Scripts\activate

# Install specific torch version required by rtdl/tableshift
pip install "torch>=1.6,<2.0"

# Install Tabula-8B specific dependencies
pip install rtdl
pip install git+https://github.com/mlfoundations/tableshift.git
pip install git+https://github.com/jpgard/llama-recipes.git
pip install git+https://github.com/penfever/tabliblib.git
pip install git+https://github.com/penfever/rtfm.git

# Install additional dependencies as needed
pip install uv>=0.6.0
pip install black==23.11.0
pip install bitsandbytes>=0.45.0
pip install fire>=0.7.0
pip install groovy>=0.1.2
pip install ninja>=1.11.0
pip install optimum>=1.24.0
pip install ruff>=0.11.0
pip install texttable>=1.7.0
pip install torchmetrics>=1.7.0
pip install uvicorn>=0.34.0
pip install fastapi>=0.115.0
pip install gradio>=5.22.0
pip install ray
pip install catboost>=1.2.5
pip install fairlearn
pip install folktables>=0.0.12
pip install ucimlrepo>=0.0.7
pip install xgboost
pip install nltk

# Finally, install MARVIS without torch version conflicts
pip install --no-deps marvis  # Install MARVIS without dependencies
```

### Option 2: Conda Environment

```bash
# Create conda environment with specific torch version
conda create -n tabula8b python=3.10
conda activate tabula8b

# Install torch version compatible with rtdl
conda install pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies via pip
pip install rtdl
pip install git+https://github.com/mlfoundations/tableshift.git
# ... (other dependencies as above)
```

## Usage

When using Tabula-8B related code:

1. Activate the dedicated environment:
   ```bash
   source tabula8b_env/bin/activate  # or conda activate tabula8b
   ```

2. Run the Tabula-8B baseline scripts:
   ```bash
   python llm_baselines/tabula_8b_baseline.py --help
   ```

## Files Using Tabula-8B Dependencies

The following files in this directory may require the Tabula-8B environment:

- `llm_baselines/tabula_8b_baseline.py` - Tabula-8B specific baseline
- Files in `llm_baselines/jolt/` - JOLT (part of RTFM) configurations
- Any scripts importing from `tableshift`, `rtdl`, or related packages

## Troubleshooting

### Torch Version Conflicts

If you encounter torch version conflicts:
- Ensure you're using the correct virtual environment
- Check torch version: `python -c "import torch; print(torch.__version__)"`
- The Tabula-8B environment should have torch<2.0
- The main MARVIS environment should have torch>=2.0

### Missing Dependencies

If packages are missing:
- Check you're in the correct environment
- Install missing packages with pip
- For git dependencies, ensure you have git access to the repositories

### Performance Issues

- Use CUDA if available for better performance
- Consider using smaller batch sizes if running out of memory
- Some Tabula-8B experiments may require significant computational resources