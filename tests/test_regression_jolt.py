#!/usr/bin/env python
"""
Test script to reproduce the regression vs classification JOLT issue.
"""

import numpy as np
import pandas as pd
from argparse import Namespace

# Create a simple regression dataset
X_reg = np.random.rand(10, 3)
y_reg = X_reg.sum(axis=1) + np.random.normal(0, 0.1, 10)  # Continuous target

# Create a simple classification dataset  
X_cls = np.random.rand(10, 3)
y_cls = (X_cls.sum(axis=1) > 1.5).astype(int)  # Binary target

# Test datasets
regression_dataset = {
    'name': 'test_regression',
    'id': 999,
    'X': X_reg,
    'y': y_reg,
    'attribute_names': ['feature_1', 'feature_2', 'feature_3'],
    'target_name': 'target'
}

classification_dataset = {
    'name': 'test_classification', 
    'id': 998,
    'X': X_cls,
    'y': y_cls,
    'attribute_names': ['feature_1', 'feature_2', 'feature_3'],
    'target_name': 'target'
}

args = Namespace(
    device='cpu',
    gpu_index=0,
    jolt_model='microsoft/DialoGPT-medium',
    max_test_samples=5,
    num_few_shot_examples=2,
    seed=42
)

print("Testing JOLT with regression dataset...")
print(f"y_reg unique values: {np.unique(y_reg)[:5]} (continuous)")

print("\nTesting JOLT with classification dataset...")  
print(f"y_cls unique values: {np.unique(y_cls)} (discrete)")

# Test task type detection
from marvis.utils.task_detection import detect_task_type

reg_task_type, _ = detect_task_type(dataset=regression_dataset, y=y_reg, manual_override='regression')
cls_task_type, _ = detect_task_type(dataset=classification_dataset, y=y_cls, manual_override='classification')

print(f"\nDetected task types:")
print(f"Regression dataset: {reg_task_type}")
print(f"Classification dataset: {cls_task_type}")

print(f"\nRegression should use mode='sampling', y_column_types=['numerical']")
print(f"Classification should use mode='logpy_only', y_column_types=['categorical']")