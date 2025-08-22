#!/bin/bash
# MARVIS Time Series Gift-Eval Evaluation Launcher
# 
# This script provides convenient shortcuts for common evaluation scenarios.
# Make sure gift-eval is installed and GIFT_EVAL environment variable is set.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if conda environment is activated
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        print_warning "No conda environment detected. Make sure to activate the marvis environment."
    else
        print_status "Conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    # Check GIFT_EVAL environment variable
    if [[ -z "$GIFT_EVAL" ]]; then
        print_error "GIFT_EVAL environment variable not set!"
        print_error "Please set it with: export GIFT_EVAL=/path/to/gift-eval"
        exit 1
    else
        print_success "GIFT_EVAL set to: $GIFT_EVAL"
    fi
    
    # Check if gift-eval directory exists
    if [[ ! -d "$GIFT_EVAL" ]]; then
        print_error "GIFT_EVAL directory does not exist: $GIFT_EVAL"
        exit 1
    fi
    
    # Check if Python can import gift_eval
    if ! python -c "import gift_eval" 2>/dev/null; then
        print_error "Cannot import gift_eval. Please ensure it's properly installed."
        exit 1
    fi
    
    print_success "All prerequisites satisfied!"
}

# Function to estimate runtime and storage
estimate_resources() {
    local mode=$1
    local max_series=$2
    local n_distributions=$3
    
    print_status "Resource estimation for mode: $mode"
    
    case $mode in
        "fast")
            echo "  Estimated runtime: 4-8 hours"
            echo "  Estimated storage: 20-50 GB"
            echo "  GPU memory required: 8+ GB"
            ;;
        "short")
            echo "  Estimated runtime: 8-16 hours"
            echo "  Estimated storage: 50-100 GB"
            echo "  GPU memory required: 16+ GB"
            ;;
        "full")
            echo "  Estimated runtime: 18-36 hours"
            echo "  Estimated storage: 100-200 GB"
            echo "  GPU memory required: 16+ GB"
            ;;
        "parallel")
            echo "  Estimated runtime: 6-12 hours (with 4 GPUs)"
            echo "  Estimated storage: 100-200 GB"
            echo "  GPU memory required: 16+ GB per GPU"
            ;;
    esac
    
    echo "  Max series per dataset: $max_series"
    echo "  Number of distributions: $n_distributions"
}

# Main evaluation function
run_evaluation() {
    local mode=$1
    local extra_args="${@:2}"
    
    print_status "Starting MARVIS time series evaluation in $mode mode..."
    
    case $mode in
        "fast")
            print_status "Running fast evaluation (short-term only, reduced settings)..."
            python batch_evaluate_gift_eval.py \
                --mode short \
                --max_series 10 \
                --n_distributions 3 \
                --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
                --output_dir "./results_fast" \
                $extra_args
            ;;
        "test")
            print_status "Running test evaluation (very fast, minimal datasets)..."
            python batch_evaluate_gift_eval.py \
                --mode short \
                --max_series 2 \
                --n_distributions 2 \
                --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
                --output_dir "./results_test" \
                --timeout 2 \
                $extra_args
            ;;
        "short")
            print_status "Running short-term evaluation (all datasets, short-term only)..."
            python batch_evaluate_gift_eval.py \
                --mode short \
                --max_series 50 \
                --n_distributions 5 \
                --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
                --output_dir "./results_short" \
                $extra_args
            ;;
        "full")
            print_status "Running full evaluation (all datasets, all terms)..."
            python batch_evaluate_gift_eval.py \
                --mode full \
                --max_series 50 \
                --n_distributions 5 \
                --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
                --output_dir "./results_full" \
                $extra_args
            ;;
        "parallel")
            print_status "Running parallel evaluation (4 GPUs)..."
            python batch_evaluate_gift_eval.py \
                --mode parallel \
                --num_gpus 4 \
                --max_series 50 \
                --n_distributions 5 \
                --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
                --output_dir "./results_parallel" \
                $extra_args
            ;;
        "high-quality")
            print_status "Running high-quality evaluation (larger model, more comprehensive)..."
            python batch_evaluate_gift_eval.py \
                --mode full \
                --max_series 100 \
                --n_distributions 8 \
                --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct" \
                --keypoint_strategy "changepoints" \
                --save_visualizations \
                --show_confidence_bands \
                --output_dir "./results_high_quality" \
                $extra_args
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
}

# Help function
show_help() {
    echo "MARVIS Time Series Gift-Eval Evaluation Launcher"
    echo ""
    echo "Usage: $0 <mode> [additional_args...]"
    echo ""
    echo "Available modes:"
    echo "  test          - Very fast test run (2 series, 2 distributions, 2 hour timeout)"
    echo "  fast          - Fast evaluation (short-term only, reduced settings)"
    echo "  short         - Short-term evaluation only (all datasets)"
    echo "  full          - Complete evaluation (short + medium + long term)"
    echo "  parallel      - Parallel evaluation across 4 GPUs"
    echo "  high-quality  - High-quality evaluation (larger model, comprehensive settings)"
    echo ""
    echo "Examples:"
    echo "  $0 test                              # Quick test"
    echo "  $0 fast                              # Fast evaluation"
    echo "  $0 short --verbose                   # Short with verbose logging"
    echo "  $0 full --save_visualizations        # Full with saved visualizations"
    echo "  $0 parallel --num_gpus 8             # Parallel with 8 GPUs"
    echo ""
    echo "Additional arguments are passed directly to batch_evaluate_gift_eval.py"
    echo "See 'python batch_evaluate_gift_eval.py --help' for all options."
}

# Main script logic
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    local mode=$1
    shift  # Remove mode from arguments
    
    case $mode in
        "-h"|"--help"|"help")
            show_help
            exit 0
            ;;
    esac
    
    # Check prerequisites
    check_prerequisites
    
    # Show resource estimates
    case $mode in
        "test")
            estimate_resources "fast" 2 2
            ;;
        "fast")
            estimate_resources "fast" 10 3
            ;;
        "short")
            estimate_resources "short" 50 5
            ;;
        "full")
            estimate_resources "full" 50 5
            ;;
        "parallel")
            estimate_resources "parallel" 50 5
            ;;
        "high-quality")
            estimate_resources "full" 100 8
            ;;
    esac
    
    echo ""
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Evaluation cancelled."
        exit 0
    fi
    
    # Run evaluation
    print_status "Starting evaluation..."
    start_time=$(date +%s)
    
    if run_evaluation "$mode" "$@"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        
        print_success "Evaluation completed successfully!"
        print_success "Total runtime: ${hours}h ${minutes}m"
        print_status "Results are available in the output directory."
    else
        print_error "Evaluation failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"