#!/usr/bin/env python3
"""
Test the regression metrics fix for MARVIS T-SNE.

This test verifies that regression tasks no longer show misleading 
"accuracy" metrics and instead properly show None for accuracy
and the correct regression metrics.

Usage:
    python tests/test_regression_metrics_fix.py
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_regression_metrics_fix():
    """Test that regression metrics are correctly handled."""
    logger.info("=== Testing Regression Metrics Fix ===")
    
    try:
        from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
        
        # Create regression test data
        np.random.seed(42)
        y_true = np.array([8.5, 9.2, 7.8, 10.1, 8.9, 9.5, 8.3, 9.8, 8.7, 9.4])
        y_pred = y_true + np.random.normal(0, 0.1, len(y_true))  # Small noise
        
        # Test regression metrics
        logger.info("Testing regression metrics with task_type='regression'...")
        metrics = calculate_llm_metrics(
            y_true, y_pred, unique_classes=None, 
            task_type='regression', logger=logger
        )
        
        # Check that accuracy is None (not R²)
        accuracy = metrics.get('accuracy')
        balanced_accuracy = metrics.get('balanced_accuracy')
        r2_score = metrics.get('r2_score')
        
        logger.info(f"accuracy: {accuracy}")
        logger.info(f"balanced_accuracy: {balanced_accuracy}")
        logger.info(f"r2_score: {r2_score}")
        
        # Verify the fix
        tests_passed = 0
        total_tests = 6
        
        # Test 1: accuracy should be None for regression
        if accuracy is None:
            logger.info("✅ accuracy is correctly None for regression")
            tests_passed += 1
        else:
            logger.error(f"❌ accuracy should be None for regression, got {accuracy}")
        
        # Test 2: balanced_accuracy should be None for regression  
        if balanced_accuracy is None:
            logger.info("✅ balanced_accuracy is correctly None for regression")
            tests_passed += 1
        else:
            logger.error(f"❌ balanced_accuracy should be None for regression, got {balanced_accuracy}")
        
        # Test 3: r2_score should have a numeric value
        if r2_score is not None and isinstance(r2_score, (int, float)):
            logger.info(f"✅ r2_score has correct numeric value: {r2_score:.4f}")
            tests_passed += 1
        else:
            logger.error(f"❌ r2_score should be numeric, got {r2_score}")
        
        # Test 4: regression metrics should exist
        expected_reg_metrics = ['mse', 'rmse', 'mae']
        reg_metrics_found = []
        for metric in expected_reg_metrics:
            if metric in metrics and metrics[metric] is not None:
                reg_metrics_found.append(metric)
        
        if len(reg_metrics_found) == len(expected_reg_metrics):
            logger.info(f"✅ All regression metrics present: {reg_metrics_found}")
            tests_passed += 1
        else:
            missing = set(expected_reg_metrics) - set(reg_metrics_found)
            logger.error(f"❌ Missing regression metrics: {missing}")
        
        # Test 5: classification metrics should be None
        expected_cls_metrics = ['f1_macro', 'precision_macro', 'recall_macro']
        cls_metrics_none = []
        for metric in expected_cls_metrics:
            if metric in metrics and metrics[metric] is None:
                cls_metrics_none.append(metric)
        
        if len(cls_metrics_none) == len(expected_cls_metrics):
            logger.info(f"✅ All classification metrics correctly None: {cls_metrics_none}")
            tests_passed += 1
        else:
            not_none = [m for m in expected_cls_metrics if metrics.get(m) is not None]
            logger.error(f"❌ Classification metrics should be None, but found: {not_none}")
        
        # Test 6: No more misleading accuracy=R² mapping
        if accuracy != r2_score:
            logger.info("✅ No more misleading accuracy=R² mapping")
            tests_passed += 1
        else:
            logger.error(f"❌ accuracy still equals r2_score: {accuracy} == {r2_score}")
        
        logger.info(f"\nRegression metrics test: {tests_passed}/{total_tests} passed")
        return tests_passed == total_tests
        
    except Exception as e:
        logger.error(f"❌ Error during regression metrics test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_classification_metrics_unchanged():
    """Test that classification metrics still work correctly."""
    logger.info("\n=== Testing Classification Metrics (Should be unchanged) ===")
    
    try:
        from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
        
        # Create classification test data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])  # Some mistakes
        
        # Test classification metrics
        logger.info("Testing classification metrics with task_type='classification'...")
        metrics = calculate_llm_metrics(
            y_true, y_pred, unique_classes=[0, 1, 2], 
            task_type='classification', logger=logger
        )
        
        accuracy = metrics.get('accuracy')
        balanced_accuracy = metrics.get('balanced_accuracy')
        r2_score = metrics.get('r2_score')
        
        logger.info(f"accuracy: {accuracy}")
        logger.info(f"balanced_accuracy: {balanced_accuracy}")
        logger.info(f"r2_score: {r2_score}")
        
        # Verify classification still works
        tests_passed = 0
        total_tests = 4
        
        # Test 1: accuracy should have numeric value for classification
        if accuracy is not None and isinstance(accuracy, (int, float)):
            logger.info(f"✅ accuracy has correct numeric value: {accuracy:.4f}")
            tests_passed += 1
        else:
            logger.error(f"❌ accuracy should be numeric for classification, got {accuracy}")
        
        # Test 2: balanced_accuracy should have numeric value for classification
        if balanced_accuracy is not None and isinstance(balanced_accuracy, (int, float)):
            logger.info(f"✅ balanced_accuracy has correct numeric value: {balanced_accuracy:.4f}")
            tests_passed += 1
        else:
            logger.error(f"❌ balanced_accuracy should be numeric for classification, got {balanced_accuracy}")
        
        # Test 3: regression metrics should be None for classification
        reg_metrics = ['mse', 'rmse', 'mae']
        reg_none = [m for m in reg_metrics if metrics.get(m) is None]
        
        if len(reg_none) == len(reg_metrics):
            logger.info(f"✅ Regression metrics correctly None for classification: {reg_none}")
            tests_passed += 1
        else:
            not_none = [m for m in reg_metrics if metrics.get(m) is not None]
            logger.error(f"❌ Regression metrics should be None for classification: {not_none}")
        
        # Test 4: classification metrics should have values
        cls_metrics = ['f1_macro', 'precision_macro', 'recall_macro']
        cls_with_values = [m for m in cls_metrics if metrics.get(m) is not None]
        
        if len(cls_with_values) == len(cls_metrics):
            logger.info(f"✅ Classification metrics have values: {cls_with_values}")
            tests_passed += 1
        else:
            none_values = [m for m in cls_metrics if metrics.get(m) is None]
            logger.error(f"❌ Classification metrics should have values, but None: {none_values}")
        
        logger.info(f"\nClassification metrics test: {tests_passed}/{total_tests} passed")
        return tests_passed == total_tests
        
    except Exception as e:
        logger.error(f"❌ Error during classification metrics test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_json_serialization():
    """Test that None values don't break JSON serialization."""
    logger.info("\n=== Testing JSON Serialization with None Values ===")
    
    try:
        import json
        from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
        
        # Create regression data
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        
        metrics = calculate_llm_metrics(
            y_true, y_pred, unique_classes=None, 
            task_type='regression', logger=logger
        )
        
        # Test JSON serialization
        json_str = json.dumps(metrics)
        logger.info("✅ JSON serialization works with None values")
        
        # Test round-trip
        metrics_restored = json.loads(json_str)
        if metrics_restored['accuracy'] is None:
            logger.info("✅ None values preserved through JSON round-trip")
            return True
        else:
            logger.error("❌ None values not preserved through JSON")
            return False
        
    except Exception as e:
        logger.error(f"❌ JSON serialization failed: {e}")
        return False


def main():
    """Run all regression metrics fix tests."""
    logger.info("=== MARVIS T-SNE Regression Metrics Fix Test Suite ===")
    
    tests = [
        ("Regression Metrics Fix", test_regression_metrics_fix),
        ("Classification Metrics Unchanged", test_classification_metrics_unchanged),
        ("JSON Serialization", test_json_serialization)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    logger.info(f"{'='*60}")
    
    if passed_tests == total_tests:
        logger.info("🎉 All regression metrics fix tests PASSED!")
        logger.info("✅ Regression tasks now properly show:")
        logger.info("   - accuracy: None (no longer misleading R²)")
        logger.info("   - balanced_accuracy: None")
        logger.info("   - r2_score: actual R² value")
        logger.info("   - mse, rmse, mae: proper regression metrics")
        logger.info("✅ Classification tasks unchanged:")
        logger.info("   - accuracy: actual accuracy")
        logger.info("   - balanced_accuracy: actual balanced accuracy")
        logger.info("   - regression metrics: None")
        return True
    else:
        logger.error("💥 Some regression metrics fix tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)