#!/usr/bin/env python3
"""
Standalone script to check metadata coverage for TabLLM and JOLT baselines.

This script helps users understand which OpenML CC18 tasks have complete metadata
for running TabLLM and JOLT baselines.
"""

import argparse
import sys
import os
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from marvis.utils.metadata_validation import (
    generate_metadata_coverage_report, 
    print_metadata_coverage_report,
    validate_metadata_for_models
)

def main():
    parser = argparse.ArgumentParser(description="Check metadata coverage for TabLLM and JOLT baselines")
    parser.add_argument(
        "--models",
        type=str,
        default="tabllm,jolt",
        help="Comma-separated list of models to check (default: tabllm,jolt)"
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated list of OpenML task IDs to check (default: all available)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save detailed report to JSON file"
    )
    parser.add_argument(
        "--check_single",
        type=int,
        default=None,
        help="Check a single task ID and show detailed validation results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Parse models
    models = [model.strip() for model in args.models.split(',')]
    
    # Handle single task check
    if args.check_single:
        print(f"Checking metadata for task {args.check_single}...")
        print("="*60)
        
        results = validate_metadata_for_models(args.check_single, models)
        
        for model, result in results.items():
            print(f"\n{model.upper()} VALIDATION:")
            print(f"  Valid: {result['valid']}")
            if result['dataset_name']:
                print(f"  Dataset: {result['dataset_name']}")
            
            if result['errors']:
                print("  Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
            
            if result['warnings']:
                print("  Warnings:")
                for warning in result['warnings']:
                    print(f"    - {warning}")
            
            if result['missing_files']:
                print("  Missing files:")
                for missing_file in result['missing_files']:
                    print(f"    - {missing_file}")
        
        return
    
    # Parse task IDs if provided
    task_ids = None
    if args.task_ids:
        try:
            task_ids = [int(task_id.strip()) for task_id in args.task_ids.split(',')]
        except ValueError as e:
            print(f"Error parsing task IDs: {e}")
            sys.exit(1)
    
    print("Generating metadata coverage report...")
    print("This may take a moment if checking all available tasks...")
    
    # Generate report
    report = generate_metadata_coverage_report(task_ids, models)
    
    # Print report
    print_metadata_coverage_report(report)
    
    # Save detailed report if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.output_file}")
    
    # Show actionable suggestions
    print("\nACTIONABLE SUGGESTIONS:")
    print("="*60)
    
    summary = report['summary']
    if summary['tasks_with_no_coverage'] > 0:
        print(f"• {summary['tasks_with_no_coverage']} tasks have no metadata coverage")
        print("  → Run metadata generation scripts to create missing metadata")
        print("  → TabLLM: python examples/tabular/llm_baselines/tabllm_like/synthesize_tabllm_real_data.py")
        print("  → JOLT: python examples/tabular/llm_baselines/jolt/synthesize_jolt_data.py")
    
    if summary['tasks_with_partial_coverage'] > 0:
        print(f"• {summary['tasks_with_partial_coverage']} tasks have partial metadata coverage")
        print("  → Some models will work, others won't")
        print("  → Use --skip_missing_metadata flag when running evaluations")
    
    if summary['tasks_with_full_coverage'] > 0:
        print(f"• {summary['tasks_with_full_coverage']} tasks have full metadata coverage")
        print("  → These tasks are ready for evaluation with all models")
    
    # Show coverage percentages
    print("\nMODEL-SPECIFIC COVERAGE:")
    for model, stats in summary['coverage_by_model'].items():
        coverage = stats['coverage_percentage']
        if coverage < 50:
            status = "❌ LOW"
        elif coverage < 80:
            status = "⚠️  MEDIUM"
        else:
            status = "✅ HIGH"
        print(f"  {model.upper()}: {coverage:.1f}% {status}")

if __name__ == "__main__":
    main()