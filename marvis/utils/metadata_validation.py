"""
Metadata validation utilities for TabLLM and JOLT baselines.

This module provides functions to validate that required metadata files exist
and contain valid information for running TabLLM and JOLT baselines.

This module now uses the robust resource manager for path resolution.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


def validate_tabllm_metadata(openml_task_id: int, feature_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate TabLLM metadata for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary with validation results:
        - 'valid': bool - whether metadata is valid
        - 'missing_files': list - list of missing required files
        - 'errors': list - list of validation errors
        - 'warnings': list - list of warnings
        - 'dataset_name': str - mapped dataset name if found
    """
    resource_manager = get_resource_manager()
    return resource_manager.validate_model_metadata(openml_task_id, 'tabllm')


def validate_jolt_metadata(openml_task_id: int, feature_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate JOLT metadata for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary with validation results:
        - 'valid': bool - whether metadata is valid
        - 'missing_files': list - list of missing required files
        - 'errors': list - list of validation errors
        - 'warnings': list - list of warnings
        - 'dataset_name': str - mapped dataset name if found
    """
    resource_manager = get_resource_manager()
    return resource_manager.validate_model_metadata(openml_task_id, 'jolt')


def validate_metadata_for_models(openml_task_id: int, models: List[str], feature_count: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate metadata for multiple models for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        models: List of model names to validate ('tabllm', 'jolt')
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary mapping model names to validation results
    """
    resource_manager = get_resource_manager()
    results = {}
    
    for model in models:
        model_lower = model.lower()
        if model_lower in ['tabllm', 'jolt']:
            results[model] = resource_manager.validate_model_metadata(openml_task_id, model_lower)
        else:
            results[model] = {
                'valid': True,  # Non-metadata models are always "valid"
                'missing_files': [],
                'errors': [],
                'warnings': [f"Model {model} does not require metadata validation"],
                'dataset_name': None
            }
    
    return results


def generate_metadata_coverage_report(task_ids: List[int] = None, models: List[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive metadata coverage report.
    
    Args:
        task_ids: List of OpenML task IDs to check (if None, uses all from mappings)
        models: List of models to check (default: ['tabllm', 'jolt'])
        
    Returns:
        Dictionary with coverage statistics and detailed results
    """
    if models is None:
        models = ['tabllm', 'jolt']
    
    resource_manager = get_resource_manager()
    
    # Get all available task IDs if not provided
    if task_ids is None:
        task_ids = set()
        
        for model in models:
            if model in ['tabllm', 'jolt']:
                task_mapping = resource_manager.config_manager.get_openml_task_mapping(model)
                if task_mapping:
                    task_ids.update(task_mapping.values())
        
        task_ids = sorted(list(task_ids))
    
    report = {
        'summary': {
            'total_tasks': len(task_ids),
            'models_checked': models,
            'coverage_by_model': {},
            'tasks_with_full_coverage': 0,
            'tasks_with_partial_coverage': 0,
            'tasks_with_no_coverage': 0
        },
        'detailed_results': {}
    }
    
    # Initialize coverage counters
    for model in models:
        report['summary']['coverage_by_model'][model] = {
            'valid': 0,
            'invalid': 0,
            'coverage_percentage': 0.0
        }
    
    # Check each task
    full_coverage_count = 0
    partial_coverage_count = 0
    no_coverage_count = 0
    
    for task_id in task_ids:
        task_results = validate_metadata_for_models(task_id, models)
        report['detailed_results'][task_id] = task_results
        
        # Count valid models for this task
        valid_models = sum(1 for result in task_results.values() if result['valid'])
        
        if valid_models == len(models):
            full_coverage_count += 1
        elif valid_models > 0:
            partial_coverage_count += 1
        else:
            no_coverage_count += 1
        
        # Update per-model counters
        for model in models:
            if task_results[model]['valid']:
                report['summary']['coverage_by_model'][model]['valid'] += 1
            else:
                report['summary']['coverage_by_model'][model]['invalid'] += 1
    
    # Calculate percentages
    for model in models:
        total = report['summary']['coverage_by_model'][model]['valid'] + report['summary']['coverage_by_model'][model]['invalid']
        if total > 0:
            report['summary']['coverage_by_model'][model]['coverage_percentage'] = (
                report['summary']['coverage_by_model'][model]['valid'] / total * 100
            )
    
    # Update summary counts
    report['summary']['tasks_with_full_coverage'] = full_coverage_count
    report['summary']['tasks_with_partial_coverage'] = partial_coverage_count
    report['summary']['tasks_with_no_coverage'] = no_coverage_count
    
    return report


def print_metadata_coverage_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted metadata coverage report.
    
    Args:
        report: Report dictionary from generate_metadata_coverage_report
    """
    summary = report['summary']
    
    print("="*60)
    print("METADATA COVERAGE REPORT")
    print("="*60)
    print(f"Total tasks checked: {summary['total_tasks']}")
    print(f"Models checked: {', '.join(summary['models_checked'])}")
    print()
    
    print("Coverage by Model:")
    for model, stats in summary['coverage_by_model'].items():
        print(f"  {model.upper()}: {stats['valid']}/{stats['valid'] + stats['invalid']} "
              f"({stats['coverage_percentage']:.1f}%)")
    
    print()
    print("Overall Coverage:")
    print(f"  Tasks with full coverage: {summary['tasks_with_full_coverage']}")
    print(f"  Tasks with partial coverage: {summary['tasks_with_partial_coverage']}")
    print(f"  Tasks with no coverage: {summary['tasks_with_no_coverage']}")
    
    # Show some examples of missing metadata
    print()
    print("Examples of Missing Metadata:")
    count = 0
    for task_id, results in report['detailed_results'].items():
        if count >= 5:  # Limit to 5 examples
            break
        
        for model, result in results.items():
            if not result['valid'] and result['errors']:
                print(f"  Task {task_id} ({model}): {result['errors'][0]}")
                count += 1
                break
    
    print("="*60)