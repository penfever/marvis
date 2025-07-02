#!/usr/bin/env python
"""
OpenML mapping utilities for task_id, dataset_id, and dataset_name relationships.

This module provides functions to retrieve and cache the mappings between OpenML task IDs,
dataset IDs, and dataset names. Used primarily for imputing missing task_id values in
wandb analysis when only dataset names are available.
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple, Any
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache file for OpenML mappings
CACHE_DIR = Path.home() / ".marvis" / "cache"
MAPPING_CACHE_FILE = CACHE_DIR / "openml_mappings.pkl"
CC18_TASKS_CACHE_FILE = CACHE_DIR / "openml_cc18_tasks.json"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_openml_cc18_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get comprehensive mapping of OpenML CC18 tasks with task_id, dataset_id, and dataset_name.
    
    TEMPORARY: Disabled cache and fallback mechanisms to force OpenML API usage.
    The centralized cache invalidation system automatically handles cache invalidation
    when the data directory changes, so this function focuses on cache loading/saving.
    
    Returns:
        Dictionary mapping task_id to {'dataset_id': int, 'dataset_name': str, 'num_classes': int, etc.}
    """
    # DISABLED: Try to load from cache first - forcing fresh OpenML API calls
    # if MAPPING_CACHE_FILE.exists():
    #     try:
    #         with open(MAPPING_CACHE_FILE, 'rb') as f:
    #             cached_mapping = pickle.load(f)
    #             logger.debug(f"Loaded OpenML mapping from cache: {len(cached_mapping)} tasks")
    #             return cached_mapping
    #     except Exception as e:
    #         logger.warning(f"Failed to load cached mapping: {e}")
    
    # If cache doesn't exist or failed to load, create new mapping
    logger.info("Creating OpenML mapping from scratch (cache and fallbacks disabled)")
    mapping = _create_openml_mapping()
    
    # DISABLED: Save to cache - preventing incorrect cached mappings
    # try:
    #     with open(MAPPING_CACHE_FILE, 'wb') as f:
    #         pickle.dump(mapping, f)
    #     logger.info(f"Saved OpenML mapping to cache: {len(mapping)} tasks")
    # except Exception as e:
    #     logger.warning(f"Failed to save mapping to cache: {e}")
    
    return mapping

def _discover_tasks_from_data_directory() -> Dict[int, Dict[str, Any]]:
    """
    Discover OpenML task IDs from JSON files in the data directory.
    
    Returns:
        Dictionary mapping task_id to basic task information
    """
    import json
    from pathlib import Path
    
    discovered_tasks = {}
    
    # Get the project root and data directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up to project root
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return discovered_tasks
    
    # Find all JSON files in data directory and subdirectories
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in data directory")
    
    for json_file in json_files:
        # Try to extract task ID from filename (common pattern is <task_id>.json)
        filename = json_file.stem
        try:
            # Check if filename is a number (task ID)
            if filename.isdigit():
                task_id = int(filename)
                
                # Try to load the JSON file to get metadata
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract relevant information
                    dataset_name = metadata.get('dataset_name', metadata.get('dataset', f'task_{task_id}'))
                    dataset_id = metadata.get('dataset_id')
                    target_attr = metadata.get('target_variable', {}).get('name') if isinstance(metadata.get('target_variable'), dict) else metadata.get('target_attribute')
                    
                    # Determine task type
                    task_type = metadata.get('task_type', 'classification')
                    if 'regression' in str(metadata).lower() or task_id > 361000:  # Regression tasks typically have higher IDs
                        task_type = 'regression'
                    
                    discovered_tasks[task_id] = {
                        'dataset_id': dataset_id,
                        'dataset_name': dataset_name,
                        'task_type': task_type,
                        'target_attribute': target_attr,
                        'source_file': str(json_file.relative_to(project_root))
                    }
                    
                    logger.debug(f"Discovered task {task_id}: {dataset_name} ({task_type})")
                    
                except Exception as e:
                    logger.warning(f"Could not parse JSON file {json_file}: {e}")
                    # Still add minimal entry
                    discovered_tasks[task_id] = {
                        'dataset_id': None,
                        'dataset_name': f'task_{task_id}',
                        'task_type': 'unknown',
                        'target_attribute': None,
                        'source_file': str(json_file.relative_to(project_root))
                    }
                    
        except ValueError:
            # Filename is not a number, skip
            continue
    
    return discovered_tasks

def _create_openml_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Create OpenML mapping by:
    1. DISABLED: Starting with fallback mapping (includes hardcoded tasks)
    2. DISABLED: Discovering task IDs from data directory JSON files
    3. Only querying OpenML API for accurate task resolution
    
    Returns:
        Dictionary mapping task_id to task information
    """
    # DISABLED: Start with fallback mapping which includes our manually added tasks
    # mapping = _get_fallback_mapping()
    # logger.info(f"Starting with {len(mapping)} tasks from fallback mapping")
    
    # DISABLED: Discover tasks from data directory
    # discovered_tasks = _discover_tasks_from_data_directory()
    # logger.info(f"Discovered {len(discovered_tasks)} unique task IDs from data directory")
    
    # Start with empty mapping - only use OpenML API
    mapping = {}
    logger.info("Starting with empty mapping - only using OpenML API for task resolution")
    
    # DISABLED: Add discovered tasks to mapping if not already present
    # for task_id, metadata in discovered_tasks.items():
    #     if task_id not in mapping:
    #         mapping[task_id] = metadata
    #         logger.debug(f"Added discovered task {task_id}: {metadata.get('dataset_name', 'unknown')}")
    #     else:
    #         # Update existing entry with discovered metadata if it has more info
    #         if metadata.get('dataset_id') and not mapping[task_id].get('dataset_id'):
    #             mapping[task_id]['dataset_id'] = metadata['dataset_id']
    #         if metadata.get('source_file'):
    #             mapping[task_id]['source_file'] = metadata['source_file']
    
    # Only use OpenML API data - no fallback enrichment
    try:
        import openml
        logger.info("Using only OpenML API data for task resolution")
        
        # Get CC18 study for classification tasks
        try:
            suite = openml.study.get_suite(99)  # 99 is the ID for CC18
            cc18_task_ids = suite.tasks
            logger.info(f"Found {len(cc18_task_ids)} CC18 tasks from OpenML")
            
            # Add CC18 tasks to mapping
            for task_id in cc18_task_ids:
                try:
                    task = openml.tasks.get_task(task_id)
                    dataset = task.get_dataset()
                    
                    mapping[task_id] = {
                        'dataset_id': task.dataset_id,
                        'dataset_name': dataset.name,
                        'num_classes': len(task.class_labels) if hasattr(task, 'class_labels') else None,
                        'num_features': len(dataset.features) if hasattr(dataset, 'features') and isinstance(dataset.features, dict) else None,
                        'target_attribute': task.target_name if hasattr(task, 'target_name') else None,
                        'source': 'openml_api'
                    }
                    logger.debug(f"Added CC18 task {task_id}: {dataset.name} (dataset_id: {task.dataset_id})")
                except Exception as e:
                    logger.warning(f"Failed to get details for CC18 task {task_id}: {e}")
        except Exception as e:
            logger.warning(f"Could not fetch CC18 tasks from OpenML: {e}")
        
        # Get regression study tasks
        try:
            suite = openml.study.get_suite(455)  # 455 is the ID for regression suite
            regression_task_ids = suite.tasks
            logger.info(f"Found {len(regression_task_ids)} regression tasks from OpenML")
            
            # Add regression tasks to mapping
            for task_id in regression_task_ids:
                try:
                    task = openml.tasks.get_task(task_id)
                    dataset = task.get_dataset()
                    
                    mapping[task_id] = {
                        'dataset_id': task.dataset_id,
                        'dataset_name': dataset.name,
                        'task_type': 'regression',
                        'num_features': len(dataset.features) if hasattr(dataset, 'features') and isinstance(dataset.features, dict) else None,
                        'target_attribute': task.target_name if hasattr(task, 'target_name') else None,
                        'source': 'openml_api'
                    }
                    logger.debug(f"Added regression task {task_id}: {dataset.name} (dataset_id: {task.dataset_id})")
                except Exception as e:
                    logger.warning(f"Failed to get details for regression task {task_id}: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not fetch regression tasks from OpenML: {e}")
            
    except ImportError:
        logger.warning("OpenML not available - cannot create mapping without fallbacks")
        return {}
    
    logger.info(f"Final mapping contains {len(mapping)} tasks")
    return mapping

def _get_hardcoded_cc18_tasks() -> list:
    """Get hardcoded list of CC18 task IDs as fallback."""
    return [
        3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 300, 458, 
        469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1489, 1494, 1497, 1501, 1485, 1486, 1487, 1468, 
        1475, 1462, 4534, 1461, 4538, 1478, 40668, 40966, 40982, 40983, 40975, 40984, 40996, 41027, 23517, 40978, 
        40670, 40701
    ]

def resolve_task_id_from_openml_api(task_id: int) -> Optional[Dict[str, Any]]:
    """
    Dynamically resolve a single task_id using OpenML API.
    
    Args:
        task_id: OpenML task ID to resolve
        
    Returns:
        Dictionary with task information if found, None otherwise
    """
    try:
        import openml
        logger.info(f"Attempting to resolve task {task_id} from OpenML API")
        
        # Get task from OpenML API
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        
        result = {
            'dataset_id': task.dataset_id,
            'dataset_name': dataset.name,
            'task_type': str(getattr(task, 'task_type', 'unknown')),
            'target_attribute': getattr(task, 'target_name', None),
            'num_classes': len(task.class_labels) if hasattr(task, 'class_labels') and task.class_labels else None,
            'num_features': len(dataset.features) if hasattr(dataset, 'features') and isinstance(dataset.features, dict) else None,
            'source': 'openml_api'
        }
        
        logger.info(f"Successfully resolved task {task_id}: {result['dataset_name']} (dataset_id: {result['dataset_id']})")
        return result
        
    except ImportError:
        logger.warning("OpenML package not available for dynamic task resolution")
        return None
    except Exception as e:
        logger.warning(f"Failed to resolve task {task_id} from OpenML API: {e}")
        return None


def _get_fallback_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get fallback mapping with known OpenML CC18 relationships.
    This is a curated list based on common CC18 datasets.
    """
    return {
        # Major CC18 datasets with known mappings
        3: {'dataset_id': 3, 'dataset_name': 'kr-vs-kp', 'num_classes': 2, 'num_features': 36, 'target_attribute': 'class'},
        6: {'dataset_id': 6, 'dataset_name': 'letter', 'num_classes': 26, 'num_features': 16, 'target_attribute': 'class'},
        11: {'dataset_id': 11, 'dataset_name': 'balance-scale', 'num_classes': 3, 'num_features': 4, 'target_attribute': 'class'},
        12: {'dataset_id': 12, 'dataset_name': 'mfeat-morphological', 'num_classes': 10, 'num_features': 6, 'target_attribute': 'class'},
        14: {'dataset_id': 14, 'dataset_name': 'mfeat-karhunen', 'num_classes': 10, 'num_features': 64, 'target_attribute': 'class'},
        15: {'dataset_id': 15, 'dataset_name': 'mfeat-zernike', 'num_classes': 10, 'num_features': 47, 'target_attribute': 'class'},
        16: {'dataset_id': 16, 'dataset_name': 'mfeat-pixel', 'num_classes': 10, 'num_features': 240, 'target_attribute': 'class'},
        18: {'dataset_id': 18, 'dataset_name': 'mfeat-factors', 'num_classes': 10, 'num_features': 216, 'target_attribute': 'class'},
        22: {'dataset_id': 22, 'dataset_name': 'mfeat-fourier', 'num_classes': 10, 'num_features': 76, 'target_attribute': 'class'},
        23: {'dataset_id': 23, 'dataset_name': 'cmc', 'num_classes': 3, 'num_features': 9, 'target_attribute': 'Contraceptive_method_used'},
        28: {'dataset_id': 28, 'dataset_name': 'optdigits', 'num_classes': 10, 'num_features': 64, 'target_attribute': 'class'},
        29: {'dataset_id': 29, 'dataset_name': 'credit-approval', 'num_classes': 2, 'num_features': 15, 'target_attribute': 'class'},
        31: {'dataset_id': 31, 'dataset_name': 'credit-g', 'num_classes': 2, 'num_features': 20, 'target_attribute': 'class'},
        32: {'dataset_id': 32, 'dataset_name': 'pendigits', 'num_classes': 10, 'num_features': 16, 'target_attribute': 'class'},
        37: {'dataset_id': 37, 'dataset_name': 'diabetes', 'num_classes': 2, 'num_features': 8, 'target_attribute': 'class'},
        44: {'dataset_id': 44, 'dataset_name': 'spambase', 'num_classes': 2, 'num_features': 57, 'target_attribute': 'class'},
        46: {'dataset_id': 46, 'dataset_name': 'splice', 'num_classes': 3, 'num_features': 60, 'target_attribute': 'class'},
        50: {'dataset_id': 50, 'dataset_name': 'tic-tac-toe', 'num_classes': 2, 'num_features': 9, 'target_attribute': 'class'},
        54: {'dataset_id': 54, 'dataset_name': 'vehicle', 'num_classes': 4, 'num_features': 18, 'target_attribute': 'class'},
        151: {'dataset_id': 151, 'dataset_name': 'electricity', 'num_classes': 2, 'num_features': 8, 'target_attribute': 'class'},
        182: {'dataset_id': 182, 'dataset_name': 'satimage', 'num_classes': 6, 'num_features': 36, 'target_attribute': 'class'},
        188: {'dataset_id': 188, 'dataset_name': 'eucalyptus', 'num_classes': 5, 'num_features': 19, 'target_attribute': 'Utility'},
        38: {'dataset_id': 38, 'dataset_name': 'sick', 'num_classes': 2, 'num_features': 29, 'target_attribute': 'Class'},
        307: {'dataset_id': 307, 'dataset_name': 'vowel', 'num_classes': 11, 'num_features': 13, 'target_attribute': 'Class'},
        300: {'dataset_id': 300, 'dataset_name': 'isolet', 'num_classes': 26, 'num_features': 617, 'target_attribute': 'class'},
        458: {'dataset_id': 458, 'dataset_name': 'analcatdata_authorship', 'num_classes': 4, 'num_features': 70, 'target_attribute': 'class'},
        469: {'dataset_id': 469, 'dataset_name': 'analcatdata_dmft', 'num_classes': 6, 'num_features': 4, 'target_attribute': 'class'},
        554: {'dataset_id': 554, 'dataset_name': 'mnist_784', 'num_classes': 10, 'num_features': 784, 'target_attribute': 'class'},
        1049: {'dataset_id': 1049, 'dataset_name': 'pc4', 'num_classes': 2, 'num_features': 37, 'target_attribute': 'c'},
        1050: {'dataset_id': 1050, 'dataset_name': 'pc3', 'num_classes': 2, 'num_features': 37, 'target_attribute': 'c'},
        1053: {'dataset_id': 1053, 'dataset_name': 'jm1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'defects'},
        1063: {'dataset_id': 1063, 'dataset_name': 'kc2', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'problems'},
        1067: {'dataset_id': 1067, 'dataset_name': 'kc1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'defects'},
        1068: {'dataset_id': 1068, 'dataset_name': 'pc1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'c'},
        1590: {'dataset_id': 1590, 'dataset_name': 'adult', 'num_classes': 2, 'num_features': 14, 'target_attribute': 'class'},
        4134: {'dataset_id': 4134, 'dataset_name': 'bioresponse', 'num_classes': 2, 'num_features': 1776, 'target_attribute': 'target'},
        1489: {'dataset_id': 1489, 'dataset_name': 'phoneme', 'num_classes': 2, 'num_features': 5, 'target_attribute': 'class'},
        1494: {'dataset_id': 1494, 'dataset_name': 'a9a', 'num_classes': 2, 'num_features': 123, 'target_attribute': 'class'},
        1497: {'dataset_id': 1497, 'dataset_name': 'w8a', 'num_classes': 2, 'num_features': 300, 'target_attribute': 'class'},
        1501: {'dataset_id': 1501, 'dataset_name': 'phishing', 'num_classes': 2, 'num_features': 30, 'target_attribute': 'Result'},
        1485: {'dataset_id': 1485, 'dataset_name': 'madelon', 'num_classes': 2, 'num_features': 500, 'target_attribute': 'class'},
        1486: {'dataset_id': 1486, 'dataset_name': 'nomao', 'num_classes': 2, 'num_features': 118, 'target_attribute': 'Class'},
        1487: {'dataset_id': 1487, 'dataset_name': 'connect-4', 'num_classes': 3, 'num_features': 42, 'target_attribute': 'class'},
        1468: {'dataset_id': 1468, 'dataset_name': 'cnae-9', 'num_classes': 9, 'num_features': 856, 'target_attribute': 'class'},
        1475: {'dataset_id': 1475, 'dataset_name': 'first-order-theorem', 'num_classes': 6, 'num_features': 51, 'target_attribute': 'class'},
        1462: {'dataset_id': 1462, 'dataset_name': 'banknote', 'num_classes': 2, 'num_features': 4, 'target_attribute': 'class'},
        4534: {'dataset_id': 4534, 'dataset_name': 'segment', 'num_classes': 7, 'num_features': 19, 'target_attribute': 'class'},
        1461: {'dataset_id': 1461, 'dataset_name': 'bank-marketing', 'num_classes': 2, 'num_features': 16, 'target_attribute': 'y'},
        4538: {'dataset_id': 4538, 'dataset_name': 'vehicle', 'num_classes': 4, 'num_features': 18, 'target_attribute': 'class'},
        1478: {'dataset_id': 1478, 'dataset_name': 'har', 'num_classes': 6, 'num_features': 561, 'target_attribute': 'class'},
        40668: {'dataset_id': 40668, 'dataset_name': 'blood-transfusion', 'num_classes': 2, 'num_features': 4, 'target_attribute': 'class'},
        40966: {'dataset_id': 40966, 'dataset_name': 'MagicTelescope', 'num_classes': 2, 'num_features': 10, 'target_attribute': 'class'},
        40982: {'dataset_id': 40982, 'dataset_name': 'steel-plates', 'num_classes': 7, 'num_features': 33, 'target_attribute': 'class'},
        40983: {'dataset_id': 40983, 'dataset_name': 'wilt', 'num_classes': 2, 'num_features': 5, 'target_attribute': 'class'},
        40975: {'dataset_id': 40975, 'dataset_name': 'car', 'num_classes': 4, 'num_features': 6, 'target_attribute': 'class'},
        40984: {'dataset_id': 40984, 'dataset_name': 'segment', 'num_classes': 7, 'num_features': 19, 'target_attribute': 'class'},
        40996: {'dataset_id': 40996, 'dataset_name': 'Fashion-MNIST', 'num_classes': 10, 'num_features': 784, 'target_attribute': 'class'},
        41027: {'dataset_id': 41027, 'dataset_name': 'jungle_chess', 'num_classes': 3, 'num_features': 6, 'target_attribute': 'class'},
        23517: {'dataset_id': 23517, 'dataset_name': 'higgs', 'num_classes': 2, 'num_features': 28, 'target_attribute': 'class'},
        40978: {'dataset_id': 40978, 'dataset_name': 'mfeat-zernike', 'num_classes': 10, 'num_features': 47, 'target_attribute': 'class'},
        40670: {'dataset_id': 40670, 'dataset_name': 'dna', 'num_classes': 3, 'num_features': 180, 'target_attribute': 'class'},
        40701: {'dataset_id': 40701, 'dataset_name': 'churn', 'num_classes': 2, 'num_features': 20, 'target_attribute': 'class'},
        
        # Regression tasks from 2025 regression suite (OpenML Suite ID 455)
        # Verified against OpenML API on 2025-06-22
        361085: {'dataset_id': 44145, 'dataset_name': 'sulfur', 'task_type': 'regression', 'target_attribute': 'y1'},
        361086: {'dataset_id': 44146, 'dataset_name': 'medical_charges', 'task_type': 'regression', 'target_attribute': 'AverageTotalPayments'},
        361087: {'dataset_id': 44147, 'dataset_name': 'MiamiHousing2016', 'task_type': 'regression', 'target_attribute': 'SALEPRC'},
        361088: {'dataset_id': 44148, 'dataset_name': 'superconduct', 'task_type': 'regression', 'target_attribute': 'criticaltemp'},
        361099: {'dataset_id': 44063, 'dataset_name': 'Bike_Sharing_Demand', 'task_type': 'regression', 'target_attribute': 'count'},
        361103: {'dataset_id': 44068, 'dataset_name': 'particulate-matter-ukair-2017', 'task_type': 'regression', 'target_attribute': 'PM.sub.10..sub..particulate.matter..Hourly.measured.'},
        361104: {'dataset_id': 44069, 'dataset_name': 'SGEMM_GPU_kernel_performance', 'task_type': 'regression', 'target_attribute': 'Run1'},
        363370: {'dataset_id': 46657, 'dataset_name': 'google_qa_answer_type_reason_explanation', 'task_type': 'regression', 'target_attribute': 'answer_type_reason_explanation'},
        363371: {'dataset_id': 46658, 'dataset_name': 'google_qa_question_type_reason_explanation', 'task_type': 'regression', 'target_attribute': 'question_type_reason_explanation'},
        363372: {'dataset_id': 46663, 'dataset_name': 'bookprice_prediction', 'task_type': 'regression', 'target_attribute': 'Price'},
        363373: {'dataset_id': 46661, 'dataset_name': 'jc_penney_products', 'task_type': 'regression', 'target_attribute': 'sale_price'},
        363374: {'dataset_id': 46659, 'dataset_name': 'women_clothing_review', 'task_type': 'regression', 'target_attribute': 'Rating'},
        363375: {'dataset_id': 46656, 'dataset_name': 'ae_price_prediction', 'task_type': 'regression', 'target_attribute': 'price'},
        363376: {'dataset_id': 46662, 'dataset_name': 'news_popularity2', 'task_type': 'regression', 'target_attribute': 'log_shares'},
        363377: {'dataset_id': 46660, 'dataset_name': 'mercari_price_suggestion100K', 'task_type': 'regression', 'target_attribute': 'log_price'},
        363387: {'dataset_id': 46695, 'dataset_name': 'convai2_inferred', 'task_type': 'regression', 'target_attribute': 'ternary_score'},
        363388: {'dataset_id': 46696, 'dataset_name': 'light_inferred', 'task_type': 'regression', 'target_attribute': 'ternary_score'},
        363389: {'dataset_id': 46697, 'dataset_name': 'opensubtitles_inferred', 'task_type': 'regression', 'target_attribute': 'ternary_score'},
        363391: {'dataset_id': 46703, 'dataset_name': 'jigsaw-unintended-bias-in-toxicity', 'task_type': 'regression', 'target_attribute': 'toxicity'},
        363394: {'dataset_id': 46710, 'dataset_name': 'UCC', 'task_type': 'regression', 'target_attribute': 'hostile_confidence'},
        363396: {'dataset_id': 46621, 'dataset_name': 'Wheat', 'task_type': 'regression', 'target_attribute': 'Stability'},
        363397: {'dataset_id': 46638, 'dataset_name': 'Phenotype_202', 'task_type': 'regression', 'target_attribute': 'Survive_time'},
        363399: {'dataset_id': 46586, 'dataset_name': 'QSAR_Bioconcentration_regression', 'task_type': 'regression', 'target_attribute': 'logBCF'},
        363417: {'dataset_id': 46612, 'dataset_name': 'heart_failure_clinical_records', 'task_type': 'regression', 'target_attribute': 'platelets'},
        363418: {'dataset_id': 46613, 'dataset_name': 'infrared_thermography_temperature', 'task_type': 'regression', 'target_attribute': 'aveOralF'},
        363426: {'dataset_id': 46717, 'dataset_name': 'biosses', 'task_type': 'regression', 'target_attribute': 'label'},
        363431: {'dataset_id': 46723, 'dataset_name': 'Violent_Crime_by_County_1975_to_2016', 'task_type': 'regression', 'target_attribute': 'PROPERTY_CRIME_TOTALS'},
        363432: {'dataset_id': 46725, 'dataset_name': 'Crime_Data_from_2010', 'task_type': 'regression', 'target_attribute': 'Victim_Age'},
        363433: {'dataset_id': 46724, 'dataset_name': 'cybersecurity_attacks', 'task_type': 'regression', 'target_attribute': 'Anomaly_Scores'},
        363434: {'dataset_id': 46726, 'dataset_name': 'climate_change_impact_on_agriculture_2024', 'task_type': 'regression', 'target_attribute': 'Economic_Impact_Million_USD'},
        363435: {'dataset_id': 46727, 'dataset_name': 'all-natural-disasters-19002021-eosdis', 'task_type': 'regression', 'target_attribute': 'Total_Affected'},
        363436: {'dataset_id': 46728, 'dataset_name': 'climate_change_dataset2020-2024', 'task_type': 'regression', 'target_attribute': 'Sea_Surface_Temp_degC'},
        363437: {'dataset_id': 46729, 'dataset_name': 'climate_insights_dataset', 'task_type': 'regression', 'target_attribute': 'Sea_Level_Rise'},
        363438: {'dataset_id': 46730, 'dataset_name': 'reddit_opinion_climate_change', 'task_type': 'regression', 'target_attribute': 'post_thumbs_ups'},
        363439: {'dataset_id': 46731, 'dataset_name': 'temperature_emissions_environmental_trends_2000_2024', 'task_type': 'regression', 'target_attribute': 'Forest_Area_pct'},
        363440: {'dataset_id': 46732, 'dataset_name': 'pakistan_hunger_data', 'task_type': 'regression', 'target_attribute': 'Children_Underweight'},
        363442: {'dataset_id': 46734, 'dataset_name': 'world_food_wealth_bank', 'task_type': 'regression', 'target_attribute': 'Value'},
        363443: {'dataset_id': 46735, 'dataset_name': 'sustainable_development_report_zero_hunger', 'task_type': 'regression', 'target_attribute': 'goal_2_score'},
        363444: {'dataset_id': 46736, 'dataset_name': 'methane_emissions_rice_crop', 'task_type': 'regression', 'target_attribute': 'Value'},
        363447: {'dataset_id': 46747, 'dataset_name': 'IoT_Agriculture_2024', 'task_type': 'regression', 'target_attribute': 'water_level'},
        363448: {'dataset_id': 46748, 'dataset_name': 'coffee_distribution_across_94_counties', 'task_type': 'regression', 'target_attribute': 'Total_Supply'},
        363452: {'dataset_id': 46754, 'dataset_name': 'sleep-deprivation-and-cognitive-performance', 'task_type': 'regression', 'target_attribute': 'Stress_Level'},
        363453: {'dataset_id': 46755, 'dataset_name': 'social-media-impact-on-suicide-rates', 'task_type': 'regression', 'target_attribute': 'suicide_rate_change_since_2010'},
    }

def impute_task_id_from_dataset_name(dataset_name: str) -> Optional[int]:
    """
    Impute task_id from dataset_name using OpenML mapping.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        task_id if found, None otherwise
    """
    if not dataset_name:
        return None
    
    mapping = get_openml_cc18_mapping()
    
    # Direct name match
    for task_id, info in mapping.items():
        if info['dataset_name'] == dataset_name:
            logger.debug(f"Found exact match: {dataset_name} -> task_id {task_id}")
            return task_id
    
    # Fuzzy matching for common variations
    dataset_name_clean = dataset_name.lower().replace('-', '_').replace(' ', '_')
    for task_id, info in mapping.items():
        info_name_clean = info['dataset_name'].lower().replace('-', '_').replace(' ', '_')
        if info_name_clean == dataset_name_clean:
            logger.debug(f"Found fuzzy match: {dataset_name} -> {info['dataset_name']} -> task_id {task_id}")
            return task_id
    
    logger.debug(f"No task_id found for dataset_name: {dataset_name}")
    return None

def impute_task_id_from_dataset_id(dataset_id: int) -> Optional[int]:
    """
    Impute task_id from dataset_id using OpenML mapping.
    
    Args:
        dataset_id: OpenML dataset ID
        
    Returns:
        task_id if found, None otherwise
    """
    if not dataset_id:
        return None
    
    mapping = get_openml_cc18_mapping()
    
    for task_id, info in mapping.items():
        if info['dataset_id'] == dataset_id:
            logger.debug(f"Found match: dataset_id {dataset_id} -> task_id {task_id}")
            return task_id
    
    logger.debug(f"No task_id found for dataset_id: {dataset_id}")
    return None

def get_task_info(task_id: int) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about a task.
    
    Args:
        task_id: OpenML task ID
        
    Returns:
        Dictionary with task information or None if not found
    """
    mapping = get_openml_cc18_mapping()
    return mapping.get(task_id)

def clear_cache():
    """Clear all cached OpenML mapping data."""
    try:
        if MAPPING_CACHE_FILE.exists():
            MAPPING_CACHE_FILE.unlink()
        if CC18_TASKS_CACHE_FILE.exists():
            CC18_TASKS_CACHE_FILE.unlink()
        logger.info("Cleared OpenML mapping cache")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

if __name__ == "__main__":
    # Test the mapping functionality
    logging.basicConfig(level=logging.INFO)
    
    mapping = get_openml_cc18_mapping()
    print(f"Loaded mapping for {len(mapping)} tasks")
    
    # Test imputation
    test_cases = [
        ("kr-vs-kp", 3),
        ("adult", 1590),
        ("electricity", 151),
        ("nonexistent", None)
    ]
    
    for dataset_name, expected_task_id in test_cases:
        result = impute_task_id_from_dataset_name(dataset_name)
        status = "✅" if result == expected_task_id else "❌"
        print(f"{status} {dataset_name} -> {result} (expected {expected_task_id})")