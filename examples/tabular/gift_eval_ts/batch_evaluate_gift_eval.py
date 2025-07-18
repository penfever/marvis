#!/usr/bin/env python
"""
Batch evaluation script for complete gift-eval benchmark using MARVIS time series distribution classification.

This script runs the novel MARVIS time series approach on the complete gift-eval benchmark,
which includes all datasets and evaluation terms used in the original gift-eval paper.

The approach frames time series forecasting as classification over Student's T distribution
patterns, providing a unique combination of statistical rigor and VLM reasoning.

Usage:
    # Run complete benchmark
    python batch_evaluate_gift_eval.py --mode full

    # Run only short-term evaluations (faster)
    python batch_evaluate_gift_eval.py --mode short

    # Run with custom settings
    python batch_evaluate_gift_eval.py --mode full --vlm_model Qwen/Qwen2.5-VL-7B-Instruct --max_series 100

    # Parallel execution across GPUs
    python batch_evaluate_gift_eval.py --mode parallel --num_gpus 4
"""

import os
import sys
import argparse
import subprocess
import datetime
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add parent directory to path for imports
current_dir = Path(__file__).parent
tabular_dir = current_dir.parent
sys.path.insert(0, str(tabular_dir))

logger = logging.getLogger(__name__)


# Complete dataset configurations from gift-eval benchmark
GIFT_EVAL_DATASETS = {
    "short_only": [
        # M4 Competition datasets
        "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly",
        
        # Energy datasets
        "electricity/15T", "electricity/H", "electricity/D", "electricity/W",
        "solar/10T", "solar/H", "solar/D", "solar/W",
        
        # Health and demographics
        "hospital", "covid_deaths",
        "us_births/D", "us_births/M", "us_births/W",
        
        # Environmental
        "saugeenday/D", "saugeenday/M", "saugeenday/W",
        "temperature_rain_with_missing",
        
        # Industrial and web
        "kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D",
        "car_parts_with_missing", "restaurant",
        "hierarchical_sales/D", "hierarchical_sales/W",
        
        # Transportation and urban
        "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
        "SZ_TAXI/15T", "SZ_TAXI/H",
        
        # Sensor and IoT
        "M_DENSE/H", "M_DENSE/D",
        "ett1/15T", "ett1/H", "ett1/D", "ett1/W",
        "ett2/15T", "ett2/H", "ett2/D", "ett2/W",
        "jena_weather/10T", "jena_weather/H", "jena_weather/D",
        
        # Cloud and infrastructure
        "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
        "bitbrains_rnd/5T", "bitbrains_rnd/H",
        "bizitobs_application", "bizitobs_service",
        "bizitobs_l2c/5T", "bizitobs_l2c/H"
    ],
    
    "med_long_capable": [
        # Datasets that support medium and long-term forecasting
        "electricity/15T", "electricity/H", "solar/10T", "solar/H",
        "kdd_cup_2018_with_missing/H", "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H",
        "SZ_TAXI/15T", "M_DENSE/H", "ett1/15T", "ett1/H",
        "ett2/15T", "ett2/H", "jena_weather/10T", "jena_weather/H",
        "bitbrains_fast_storage/5T", "bitbrains_rnd/5T",
        "bizitobs_application", "bizitobs_service",
        "bizitobs_l2c/5T", "bizitobs_l2c/H"
    ]
}

# Dataset groupings for parallel execution
DATASET_GROUPS = {
    "m4": ["m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly"],
    "energy": ["electricity/15T", "electricity/H", "electricity/D", "electricity/W", 
               "solar/10T", "solar/H", "solar/D", "solar/W"],
    "health_demo": ["hospital", "covid_deaths", "us_births/D", "us_births/M", "us_births/W"],
    "environment": ["saugeenday/D", "saugeenday/M", "saugeenday/W", 
                    "temperature_rain_with_missing", "jena_weather/10T", "jena_weather/H", "jena_weather/D"],
    "industrial": ["kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D", 
                   "car_parts_with_missing", "restaurant", "hierarchical_sales/D", "hierarchical_sales/W"],
    "transport": ["LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D", "SZ_TAXI/15T", "SZ_TAXI/H"],
    "sensors": ["M_DENSE/H", "M_DENSE/D", "ett1/15T", "ett1/H", "ett1/D", "ett1/W",
                "ett2/15T", "ett2/H", "ett2/D", "ett2/W"],
    "cloud": ["bitbrains_fast_storage/5T", "bitbrains_fast_storage/H", "bitbrains_rnd/5T", "bitbrains_rnd/H",
              "bizitobs_application", "bizitobs_service", "bizitobs_l2c/5T", "bizitobs_l2c/H"]
}


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Set up logging for the batch evaluation."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"batch_evaluation_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return root_logger


def validate_environment() -> bool:
    """Validate that all required components are available."""
    logger.info("Validating environment...")
    
    # Check if gift-eval is available
    gift_eval_path = os.environ.get('GIFT_EVAL')
    if not gift_eval_path:
        logger.error("GIFT_EVAL environment variable not set. Please set it to point to your gift-eval installation.")
        return False
    
    if not os.path.exists(gift_eval_path):
        logger.error(f"GIFT_EVAL path does not exist: {gift_eval_path}")
        return False
    
    # Check if evaluate_time_series.py exists
    eval_script = current_dir.parent / "evaluate_time_series.py"
    if not eval_script.exists():
        logger.error(f"evaluate_time_series.py not found at {eval_script}")
        return False
    
    logger.info("Environment validation passed")
    return True


def run_evaluation_batch(
    datasets: List[str],
    terms: str,
    output_suffix: str,
    args,
    gpu_id: Optional[int] = None
) -> Tuple[bool, Dict]:
    """Run evaluation on a batch of datasets."""
    
    # Prepare command
    cmd = [
        "python", str(current_dir.parent / "evaluate_time_series.py"),
        "--datasets", ",".join(datasets),
        "--terms", terms,
        "--vlm_model_id", args.vlm_model,
        "--output_dir", str(args.output_dir / f"gift_eval_{output_suffix}"),
        "--max_series_per_dataset", str(args.max_series),
        "--experiment_name", f"gift_eval_{output_suffix}",
        "--seed", str(args.seed),
        "--n_distributions", str(args.n_distributions),
        "--keypoint_strategy", args.keypoint_strategy
    ]
    
    # Add optional arguments
    if args.forecast_horizon:
        cmd.extend(["--forecast_horizon", str(args.forecast_horizon)])
    
    if args.save_visualizations:
        cmd.append("--save_visualizations")
    
    if args.show_confidence_bands:
        cmd.append("--show_confidence_bands")
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Set GPU if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Running on GPU {gpu_id}")
    
    # Log command
    logger.info(f"Running batch {output_suffix}: {' '.join(cmd)}")
    
    # Run evaluation
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            timeout=args.timeout * 3600  # Convert hours to seconds
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            logger.info(f"✓ Completed batch {output_suffix} in {duration/3600:.2f} hours")
        else:
            logger.error(f"✗ Failed batch {output_suffix}: {result.stderr}")
        
        return success, {
            "batch": output_suffix,
            "success": success,
            "duration_hours": duration / 3600,
            "datasets": datasets,
            "terms": terms,
            "stdout": result.stdout,
            "stderr": result.stderr if not success else "",
            "command": " ".join(cmd)
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Batch {output_suffix} timed out after {args.timeout} hours")
        return False, {
            "batch": output_suffix,
            "success": False,
            "error": "Timeout",
            "duration_hours": args.timeout,
            "datasets": datasets,
            "terms": terms
        }
    except Exception as e:
        logger.error(f"✗ Batch {output_suffix} failed with exception: {e}")
        return False, {
            "batch": output_suffix,
            "success": False,
            "error": str(e),
            "datasets": datasets,
            "terms": terms
        }


def run_sequential_evaluation(args) -> List[Dict]:
    """Run evaluation sequentially on all datasets."""
    logger.info("Starting sequential evaluation...")
    
    results = []
    
    if args.mode in ["full", "short"]:
        # Short-term evaluation
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Short-term forecasting")
        logger.info("="*60)
        
        success, result = run_evaluation_batch(
            GIFT_EVAL_DATASETS["short_only"],
            "short",
            "short_complete",
            args
        )
        results.append(result)
    
    if args.mode == "full":
        # Medium and long-term evaluation
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Medium and long-term forecasting")
        logger.info("="*60)
        
        success, result = run_evaluation_batch(
            GIFT_EVAL_DATASETS["med_long_capable"],
            "medium,long",
            "medlong_complete",
            args
        )
        results.append(result)
    
    return results


def run_parallel_evaluation(args) -> List[Dict]:
    """Run evaluation in parallel across multiple GPUs."""
    import concurrent.futures
    
    logger.info(f"Starting parallel evaluation across {args.num_gpus} GPUs...")
    
    # Distribute dataset groups across GPUs
    group_names = list(DATASET_GROUPS.keys())
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = []
        
        # Submit short-term jobs
        if args.mode in ["full", "short", "parallel"]:
            for i, (group_name, datasets) in enumerate(DATASET_GROUPS.items()):
                gpu_id = i % args.num_gpus
                future = executor.submit(
                    run_evaluation_batch,
                    datasets,
                    "short",
                    f"short_{group_name}",
                    args,
                    gpu_id
                )
                futures.append(future)
        
        # Submit medium/long-term jobs for capable datasets
        if args.mode == "full":
            for i, (group_name, datasets) in enumerate(DATASET_GROUPS.items()):
                # Only use datasets that support med/long terms
                capable_datasets = [d for d in datasets if d in GIFT_EVAL_DATASETS["med_long_capable"]]
                if capable_datasets:
                    gpu_id = i % args.num_gpus
                    future = executor.submit(
                        run_evaluation_batch,
                        capable_datasets,
                        "medium,long",
                        f"medlong_{group_name}",
                        args,
                        gpu_id
                    )
                    futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                success, result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel job failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e)
                })
    
    return results


def generate_summary_report(results: List[Dict], args) -> Dict:
    """Generate a comprehensive summary report."""
    
    total_batches = len(results)
    successful_batches = sum(1 for r in results if r.get("success", False))
    failed_batches = total_batches - successful_batches
    
    total_duration = sum(r.get("duration_hours", 0) for r in results)
    
    # Count datasets
    all_datasets = set()
    for result in results:
        if "datasets" in result:
            all_datasets.update(result["datasets"])
    
    summary = {
        "evaluation_summary": {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": args.mode,
            "vlm_model": args.vlm_model,
            "total_batches": total_batches,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "success_rate": successful_batches / total_batches if total_batches > 0 else 0,
            "total_duration_hours": total_duration,
            "total_datasets_evaluated": len(all_datasets),
            "max_series_per_dataset": args.max_series,
            "n_distributions": args.n_distributions
        },
        "batch_results": results,
        "configuration": vars(args)
    }
    
    return summary


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation of MARVIS on complete gift-eval benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        choices=["short", "full", "parallel"],
        default="full",
        help="Evaluation mode: 'short' (short-term only), 'full' (short + med/long), 'parallel' (parallel execution)"
    )
    
    # Model configuration
    parser.add_argument(
        "--vlm_model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="VLM model to use for classification"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--max_series",
        type=int,
        default=50,
        help="Maximum number of series to evaluate per dataset"
    )
    
    # Time series parameters
    parser.add_argument(
        "--n_distributions",
        type=int,
        default=5,
        help="Number of Student's T distributions to fit"
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=None,
        help="Override forecast horizon (uses dataset default if None)"
    )
    parser.add_argument(
        "--keypoint_strategy",
        choices=["uniform", "extrema", "changepoints"],
        default="uniform",
        help="Strategy for selecting keypoints"
    )
    
    # Visualization options
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--show_confidence_bands",
        action="store_true",
        help="Show confidence bands in visualizations"
    )
    
    # Parallel execution
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel execution"
    )
    
    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./gift_eval_complete_results"),
        help="Output directory for all results"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=24,
        help="Timeout per batch in hours"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(args.output_dir, args.verbose)
    
    # Log configuration
    logger.info("="*60)
    logger.info("MARVIS Time Series Gift-Eval Batch Evaluation")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"VLM Model: {args.vlm_model}")
    logger.info(f"Max series per dataset: {args.max_series}")
    logger.info(f"Number of distributions: {args.n_distributions}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return 1
    
    # Run evaluation
    start_time = time.time()
    
    try:
        if args.mode == "parallel":
            results = run_parallel_evaluation(args)
        else:
            results = run_sequential_evaluation(args)
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = generate_summary_report(results, args)
        summary["evaluation_summary"]["wall_clock_duration_hours"] = total_duration / 3600
        
        # Save summary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = args.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total duration: {total_duration/3600:.2f} hours")
        logger.info(f"Successful batches: {summary['evaluation_summary']['successful_batches']}/{summary['evaluation_summary']['total_batches']}")
        logger.info(f"Success rate: {summary['evaluation_summary']['success_rate']:.2%}")
        logger.info(f"Total datasets evaluated: {summary['evaluation_summary']['total_datasets_evaluated']}")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Check for failures
        failed_batches = [r for r in results if not r.get("success", False)]
        if failed_batches:
            logger.warning(f"\n{len(failed_batches)} batches failed:")
            for batch in failed_batches:
                logger.warning(f"  - {batch.get('batch', 'Unknown')}: {batch.get('error', 'Unknown error')}")
            return 1
        else:
            logger.info("\n🎉 All evaluations completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nEvaluation failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())