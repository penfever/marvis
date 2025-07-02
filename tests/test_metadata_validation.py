#!/usr/bin/env python3
"""
Test script to verify metadata validation works with the new task ID-based approach.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_metadata_validation():
    """Test metadata validation for JOLT and TabLLM with new task ID approach."""
    
    # Silence verbose logging
    logging.getLogger('marvis.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    from marvis.utils.metadata_validation import validate_metadata_for_models
    
    # Test with tasks that we know have configs
    test_cases = [
        (361085, "sulfur regression task"),
        (363373, "jc_penney_products regression task"),
        (23, "cmc classification task"),
    ]
    
    models_to_test = ['jolt', 'tabllm']
    
    logger.info("Testing metadata validation with new task ID approach...")
    
    for task_id, description in test_cases:
        logger.info(f"\n--- Testing {description} (task {task_id}) ---")
        
        try:
            # Test metadata validation for multiple models
            results = validate_metadata_for_models(task_id, models_to_test, feature_count=None)
            
            for model_name, validation in results.items():
                logger.info(f"{model_name.upper()} validation:")
                if validation['valid']:
                    logger.info(f"  ✅ VALID - Dataset: {validation.get('dataset_name', 'Unknown')}")
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            logger.info(f"  ⚠️  Warning: {warning}")
                else:
                    logger.error(f"  ❌ INVALID")
                    for error in validation['errors']:
                        logger.error(f"    - {error}")
                    for missing_file in validation['missing_files']:
                        logger.error(f"    - Missing: {missing_file}")
                        
        except Exception as e:
            logger.error(f"Error validating task {task_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """Main test function."""
    logger.info("=== Metadata Validation Test ===")
    test_metadata_validation()
    logger.info("\n=== Test complete ===")

if __name__ == "__main__":
    main()