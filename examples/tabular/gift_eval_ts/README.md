# MARVIS Time Series Gift-Eval Batch Evaluation

This directory contains tools for running MARVIS's novel time series distribution classification approach on the complete gift-eval benchmark.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Ensure you have gift-eval installed
git clone https://github.com/SalesforceAIResearch/gift-eval.git
cd gift-eval
pip install -r requirements.txt

# Set environment variable
export GIFT_EVAL=/path/to/gift-eval

# Download datasets (this may take a while)
python -m gift_eval.data.download_datasets
```

### 2. Run Evaluation

```bash
# Navigate to this directory
cd marvis/examples/tabular/gift_eval_ts

# Basic usage - short-term evaluation only (fastest)
python batch_evaluate_gift_eval.py --mode short

# Complete benchmark - short, medium, and long-term
python batch_evaluate_gift_eval.py --mode full

# Parallel execution across 4 GPUs (fastest for complete benchmark)
python batch_evaluate_gift_eval.py --mode parallel --num_gpus 4

# Fast evaluation with reduced settings
python batch_evaluate_gift_eval.py --mode short --max_series 10 --n_distributions 3

# High-quality evaluation with larger model
python batch_evaluate_gift_eval.py --mode full --vlm_model Qwen/Qwen2.5-VL-7B-Instruct --max_series 100
```

## 📊 Evaluation Modes

| Mode | Description | Datasets | Terms | Est. Runtime |
|------|-------------|----------|-------|-------------|
| `short` | Short-term forecasting only | All 50+ datasets | short | 8-16 hours |
| `full` | Complete benchmark | All datasets | short, medium, long | 18-36 hours |
| `parallel` | Parallel execution | All datasets | short | 6-12 hours |

## ⚙️ Configuration Options

### Model Selection
```bash
# Smaller model (lower GPU memory requirements)
--vlm_model Qwen/Qwen2.5-VL-3B-Instruct

# Larger model (potentially better performance)
--vlm_model Qwen/Qwen2.5-VL-7B-Instruct
```

### Performance Tuning
```bash
# Fast evaluation (lower accuracy)
--max_series 10 --n_distributions 3

# Default (balanced)
--max_series 50 --n_distributions 5

# High quality (slower but more thorough)
--max_series 100 --n_distributions 8
```

### Distribution Fitting
```bash
# Keypoint selection strategies
--keypoint_strategy uniform      # Evenly spaced points (default)
--keypoint_strategy extrema      # Local minima/maxima
--keypoint_strategy changepoints # Detected change points
```

### Visualization Options
```bash
# Save visualization images (requires more storage)
--save_visualizations

# Show confidence bands in plots
--show_confidence_bands
```

## 🔄 Parallel Execution

For fastest evaluation with multiple GPUs:

```bash
# Automatic load balancing across 4 GPUs
python batch_evaluate_gift_eval.py --mode parallel --num_gpus 4

# Manual GPU assignment (alternative approach)
CUDA_VISIBLE_DEVICES=0 python batch_evaluate_gift_eval.py --mode short --output_dir ./results_gpu0 &
CUDA_VISIBLE_DEVICES=1 python batch_evaluate_gift_eval.py --mode short --output_dir ./results_gpu1 &
```

## 📁 Output Structure

```
gift_eval_complete_results/
├── evaluation_summary_YYYYMMDD_HHMMSS.json    # Overall results summary
├── batch_evaluation_YYYYMMDD_HHMMSS.log       # Detailed logs
├── gift_eval_short_complete/                   # Short-term results
│   ├── time_series_results_YYYYMMDD_HHMMSS.json
│   └── time_series_summary_YYYYMMDD_HHMMSS.json
└── gift_eval_medlong_complete/                 # Medium/long-term results
    ├── time_series_results_YYYYMMDD_HHMMSS.json
    └── time_series_summary_YYYYMMDD_HHMMSS.json
```

## 📈 Results Analysis

The evaluation produces results in gift-eval compatible format for easy comparison with baselines:

- **Chronos/Chronos-Bolt**: Foundation models for time series
- **TabPFN-TS**: Tabular foundation model adapted for time series  
- **Toto**: Large-scale time series foundation model

Key metrics include:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MASE (Mean Absolute Scaled Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric MAPE)

## 🔧 Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   ```bash
   # Use smaller model
   --vlm_model Qwen/Qwen2.5-VL-3B-Instruct
   
   # Reduce series count
   --max_series 10
   
   # Reduce distributions
   --n_distributions 3
   ```

2. **gift-eval Import Error**
   ```bash
   # Ensure GIFT_EVAL environment variable is set
   export GIFT_EVAL=/path/to/gift-eval
   
   # Add to Python path if needed
   export PYTHONPATH=$PYTHONPATH:/path/to/gift-eval
   ```

3. **Long Runtime**
   ```bash
   # Use fast settings for initial testing
   --max_series 5 --n_distributions 3 --mode short
   
   # Monitor progress with verbose logging
   --verbose
   ```

### Hardware Requirements

| Setting | GPU Memory | System RAM | Storage | Runtime |
|---------|------------|------------|---------|---------|
| Fast | 8GB+ | 16GB+ | 50GB | 4-8 hours |
| Default | 16GB+ | 32GB+ | 100GB | 8-16 hours |
| High Quality | 24GB+ | 64GB+ | 200GB | 24-48 hours |

## 🧪 Novel Approach

MARVIS introduces a groundbreaking approach to time series forecasting:

1. **Distribution Fitting**: Fits multiple Student's T distributions to keypoints in training data
2. **Visual Classification**: Creates visualizations showing candidate forecast paths
3. **VLM Reasoning**: Uses Vision Language Model to select the most appropriate pattern
4. **Probabilistic Sampling**: Samples predictions from the selected distribution

This combines statistical rigor with modern VLM reasoning capabilities, providing interpretable and uncertainty-aware forecasts.

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{marvis_time_series,
  title={MARVIS Time Series Distribution Classification},
  author={MARVIS Team},
  year={2024},
  url={https://github.com/your-repo/marvis}
}
```

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed logs in the output directory
3. Ensure all dependencies are properly installed
4. Verify gift-eval environment setup

The batch evaluation script provides comprehensive logging and error reporting to help diagnose any issues.