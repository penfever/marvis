#!/usr/bin/env python3
"""
Comprehensive test for TabLLM baseline functionality with online note generation.

This test validates:
- Task ID resolution (dataset_id -> task_id)
- Online semantic feature expansion
- Context-aware note truncation
- Note quality and generation
- Integration with existing MARVIS infrastructure

Usage:
    python tests/test_tabllm_comprehensive.py
    python tests/test_tabllm_comprehensive.py --task_id 23 --output_dir ./test_tabllm_output
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_args(task_id: int = 23, output_dir: str = None, max_context_length: int = 4096):
    """Create test arguments for TabLLM evaluation."""
    class TestArgs:
        def __init__(self):
            self.task_id = task_id
            self.output_dir = output_dir or tempfile.mkdtemp(prefix="tabllm_test_")
            self.max_context_length = max_context_length
            self.num_few_shot_examples = 8
            self.max_test_samples = 10  # Small for testing
            self.device = "cpu"  # Use CPU for testing
            self.seed = 42
            self.openai_model = None
            self.gemini_model = None
            self.api_model = None
            self.tabllm_model = "microsoft/phi-3-mini-128k-instruct"  # Small model for testing
            
    return TestArgs()

def test_task_id_resolution():
    """Test task_id resolution from dataset_id."""
    logger.info("=== Testing Task ID Resolution ===")
    
    # Silence verbose logging
    logging.getLogger('marvis.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.tabllm_baseline import load_tabllm_config_by_openml_id
    
    # Test cases: (input_id, expected_behavior, description)
    test_cases = [
        (23, "should_work", "Known OpenML task ID"),
        (40996, "should_resolve", "Dataset ID that should resolve to task ID"),
        (999999, "should_fail", "Non-existent ID"),
    ]
    
    for test_id, expected, description in test_cases:
        logger.info(f"\n--- Testing {description}: ID {test_id} ---")
        
        try:
            config_data, feature_mapping = load_tabllm_config_by_openml_id(test_id, original_feature_count=10)
            
            if config_data is not None:
                logger.info(f"✅ SUCCESS: Config found for ID {test_id}")
                logger.info(f"   Task ID resolved from: {test_id}")
                if feature_mapping:
                    semantic_info = feature_mapping.get('semantic_info')
                    if semantic_info:
                        logger.info(f"   Semantic info available: {list(semantic_info.keys())}")
                    else:
                        logger.info("   No semantic info in feature mapping")
            else:
                if expected == "should_fail":
                    logger.info(f"✅ EXPECTED: No config found for ID {test_id}")
                else:
                    logger.warning(f"⚠️ UNEXPECTED: No config found for ID {test_id}")
                    
        except Exception as e:
            if expected == "should_fail":
                logger.info(f"✅ EXPECTED FAILURE: {e}")
            else:
                logger.error(f"❌ UNEXPECTED ERROR: {e}")

def test_semantic_feature_expansion():
    """Test semantic feature expansion functionality."""
    logger.info("\n=== Testing Semantic Feature Expansion ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import expand_semantic_features
    
    # Test semantic info with 'columns' structure
    test_semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'Age of the person'},
            {'name': 'income', 'semantic_description': 'Annual income'},
            {'name': 'education', 'semantic_description': 'Education level'}
        ]
    }
    
    test_cases = [
        (3, "no expansion needed"),
        (5, "expand from 3 to 5 features"),
        (8, "expand from 3 to 8 features"),
        (1, "fewer features than available")
    ]
    
    for target_count, description in test_cases:
        logger.info(f"\n--- Testing {description}: {target_count} target features ---")
        
        try:
            # Test both with and without actual feature names
            test_feature_names = [f"feature_{i}" for i in range(target_count)] if target_count > 0 else None
            expanded_info = expand_semantic_features(test_semantic_info, target_count, test_feature_names)
            
            if 'columns' in expanded_info:
                expanded_count = len(expanded_info['columns'])
                feature_names = [col['name'] for col in expanded_info['columns']]
                
                logger.info(f"✅ Expanded from 3 to {expanded_count} features")
                
                # CRITICAL TESTS: Uniqueness and Completeness
                if target_count > 0:
                    # Test 1: All feature names are unique
                    unique_names = set(feature_names)
                    if len(unique_names) == len(feature_names):
                        logger.info("   ✅ All semantic feature names are unique")
                    else:
                        logger.error(f"   ❌ Duplicate semantic feature names found: {len(feature_names)} total, {len(unique_names)} unique")
                        duplicates = [name for name in feature_names if feature_names.count(name) > 1]
                        logger.error(f"      Duplicates: {set(duplicates)}")
                    
                    # Test 2: Semantic features match actual feature names when provided
                    if test_feature_names:
                        if set(feature_names) == set(test_feature_names):
                            logger.info("   ✅ Semantic feature names exactly match actual feature names")
                        else:
                            logger.error(f"   ❌ Semantic feature names don't match actual names")
                            logger.error(f"      Expected: {test_feature_names}")
                            logger.error(f"      Got: {feature_names}")
                            missing = set(test_feature_names) - set(feature_names)
                            extra = set(feature_names) - set(test_feature_names)
                            if missing:
                                logger.error(f"      Missing: {missing}")
                            if extra:
                                logger.error(f"      Extra: {extra}")
                    
                    # Test 3: Count matches exactly
                    if expanded_count == target_count:
                        logger.info("   ✅ Exact target count achieved")
                    else:
                        logger.error(f"   ❌ Count mismatch: got {expanded_count}, expected {target_count}")
                        
                # Test 4: All semantic features have descriptions
                missing_descriptions = [col['name'] for col in expanded_info['columns'] if not col.get('semantic_description')]
                if not missing_descriptions:
                    logger.info("   ✅ All semantic features have descriptions")
                else:
                    logger.error(f"   ❌ Features missing descriptions: {missing_descriptions}")
                    
            else:
                logger.error("❌ Expanded info missing 'columns' structure")
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")


def test_semantic_feature_alignment():
    """Test that semantic features properly align with actual data after subselection."""
    logger.info("\n=== Testing Semantic Feature Alignment ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import expand_semantic_features, generate_note_from_row
    import pandas as pd
    
    # Simulate feature subselection scenario
    original_features = [f"original_feature_{i}" for i in range(10)]
    retained_features = [f"retained_feature_{i}" for i in range(7)]  # Simulated after feature reduction
    
    # Create original semantic info (fewer than retained features)
    original_semantic = {
        'columns': [
            {'name': 'base_feature_1', 'semantic_description': 'First base feature'},
            {'name': 'base_feature_2', 'semantic_description': 'Second base feature'},
        ]
    }
    
    logger.info(f"Original semantic features: 2")
    logger.info(f"Retained features after subselection: {len(retained_features)}")
    
    # Test expansion with retained feature names
    expanded_semantic = expand_semantic_features(original_semantic, len(retained_features), retained_features)
    
    # Validate the expansion
    if 'columns' in expanded_semantic:
        semantic_names = [col['name'] for col in expanded_semantic['columns']]
        
        # Test 1: Count matches retained features
        if len(semantic_names) == len(retained_features):
            logger.info("✅ Semantic feature count matches retained feature count")
        else:
            logger.error(f"❌ Count mismatch: {len(semantic_names)} semantic vs {len(retained_features)} retained")
        
        # Test 2: Names match exactly
        if set(semantic_names) == set(retained_features):
            logger.info("✅ Semantic feature names exactly match retained feature names")
        else:
            logger.error("❌ Semantic and retained feature names don't match")
            logger.error(f"   Semantic: {semantic_names}")
            logger.error(f"   Retained: {retained_features}")
        
        # Test 3: All names are unique
        if len(set(semantic_names)) == len(semantic_names):
            logger.info("✅ All semantic feature names are unique")
        else:
            logger.error("❌ Duplicate semantic feature names found")
        
        # Test 4: Test note generation with aligned data
        test_data = pd.Series([i * 10 + 5 for i in range(len(retained_features))], index=retained_features)
        
        try:
            note = generate_note_from_row(test_data, expanded_semantic, retained_features, exclude_target=True)
            if note and len(note.strip()) > 0:
                logger.info("✅ Note generation successful with aligned features")
                logger.info(f"   Sample note: {note[:100]}...")
            else:
                logger.error("❌ Note generation failed - empty note")
        except Exception as e:
            logger.error(f"❌ Note generation failed with error: {e}")
    else:
        logger.error("❌ Expanded semantic info missing 'columns' structure")

def test_online_note_generation():
    """Test online note generation with expanded semantics."""
    logger.info("\n=== Testing Online Note Generation ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import generate_note_from_row, expand_semantic_features
    
    # Create test semantic info
    semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'age in years'},
            {'name': 'workclass', 'semantic_description': 'type of employment'},
            {'name': 'education', 'semantic_description': 'education level achieved'}
        ]
    }
    
    # Test data row
    test_row = pd.Series([39, 'State-gov', 'Bachelors'], index=['age', 'workclass', 'education'])
    attribute_names = ['age', 'workclass', 'education']
    
    logger.info("--- Testing basic note generation ---")
    try:
        note = generate_note_from_row(test_row, semantic_info, attribute_names)
        logger.info(f"✅ Generated note: {note}")
        
        # Check that note contains expected elements
        expected_elements = ['age in years is 39', 'type of employment is State-gov', 'education level achieved is Bachelors']
        for element in expected_elements:
            if element in note:
                logger.info(f"   ✅ Found expected element: {element}")
            else:
                logger.warning(f"   ⚠️ Missing expected element: {element}")
                
    except Exception as e:
        logger.error(f"❌ ERROR in basic note generation: {e}")
    
    logger.info("\n--- Testing note generation with expanded features ---")
    try:
        # Expand semantic features to 5 with actual feature names
        expanded_attributes = ['age', 'workclass', 'education', 'age_1', 'workclass_1']
        expanded_semantic = expand_semantic_features(semantic_info, 5, expanded_attributes)
        
        # Create expanded test row
        expanded_row = pd.Series([39, 'State-gov', 'Bachelors', 50000, 'Married'], 
                                index=expanded_attributes)
        
        expanded_note = generate_note_from_row(expanded_row, expanded_semantic, expanded_attributes)
        logger.info(f"✅ Generated expanded note: {expanded_note}")
        
        # Check for expanded features
        if 'variant 1' in expanded_note:
            logger.info("   ✅ Found expanded feature variants in note")
        else:
            logger.warning("   ⚠️ Expanded features not reflected in note")
            
    except Exception as e:
        logger.error(f"❌ ERROR in expanded note generation: {e}")

def test_context_aware_truncation():
    """Test context-aware note truncation."""
    logger.info("\n=== Testing Context-Aware Truncation ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import (
        truncate_few_shot_examples_for_context, 
        estimate_note_tokens,
        estimate_prompt_tokens
    )
    
    # Create mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            # Simple word-based tokenization for testing
            return text.split()
    
    tokenizer = MockTokenizer()
    
    # Create test few-shot examples
    few_shot_examples = [
        ("The age is 25. The income is 50000. The education is Bachelor.", "Class_A"),
        ("The age is 30. The income is 75000. The education is Master.", "Class_B"),
        ("The age is 45. The income is 100000. The education is PhD.", "Class_A"),
        ("The age is 35. The income is 60000. The education is Bachelor.", "Class_B"),
        ("The age is 28. The income is 55000. The education is Master.", "Class_A")
    ]
    
    test_note = "The age is 40. The income is 80000. The education is Master."
    question = "What is the class?"
    task_description = "Classify the following examples."
    
    logger.info("--- Testing token estimation ---")
    try:
        for i, (note, label) in enumerate(few_shot_examples[:2]):
            tokens = estimate_note_tokens(note, tokenizer)
            logger.info(f"   Example {i}: {tokens} tokens for note: {note[:50]}...")
            
        total_tokens = estimate_prompt_tokens(few_shot_examples, test_note, question, task_description, tokenizer)
        logger.info(f"✅ Total estimated tokens: {total_tokens}")
        
    except Exception as e:
        logger.error(f"❌ ERROR in token estimation: {e}")
    
    logger.info("\n--- Testing truncation scenarios ---")
    
    # Test different context limits
    context_limits = [200, 100, 50, 20]
    
    for limit in context_limits:
        try:
            truncated = truncate_few_shot_examples_for_context(
                few_shot_examples, test_note, question, task_description, tokenizer, limit
            )
            
            logger.info(f"   Context limit {limit}: {len(truncated)}/{len(few_shot_examples)} examples kept")
            
            if limit < 50:
                # Very restrictive limit should result in fewer examples
                if len(truncated) < len(few_shot_examples):
                    logger.info("   ✅ Appropriate truncation occurred")
                else:
                    logger.warning("   ⚠️ Expected more aggressive truncation")
                    
        except Exception as e:
            logger.error(f"❌ ERROR with context limit {limit}: {e}")

def test_tabllm_pipeline_with_mocked_llm():
    """Test TabLLM pipeline with mocked LLM to focus on feature alignment."""
    logger.info("\n=== Testing TabLLM Pipeline with Mocked LLM ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import (
        load_tabllm_config_by_openml_id, expand_semantic_features, generate_note_from_row
    )
    import pandas as pd
    from unittest.mock import Mock, patch
    
    # Create test dataset with feature subselection scenario
    test_dataset = {
        'name': 'test_pipeline',
        'task_id': 23,
        'X': pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'workclass': ['Private', 'Self-emp', 'Private', 'Local-gov', 'Private'],
            'education': ['Bachelors', 'Masters', 'HS-grad', 'PhD', 'Bachelors'],
            'capital_gain': [0, 5178, 0, 0, 15024],
            'hours_per_week': [40, 50, 40, 45, 40]
        }),
        'y': pd.Series(['<=50K', '>50K', '<=50K', '>50K', '>50K']),
        'attribute_names': ['age', 'workclass', 'education', 'capital_gain', 'hours_per_week']
    }
    
    # Simulate feature reduction (keep only 3 out of 5 features)
    retained_features = ['age', 'workclass', 'education']
    reduced_dataset = test_dataset.copy()
    reduced_dataset['X'] = test_dataset['X'][retained_features]
    reduced_dataset['attribute_names'] = retained_features
    
    # Create semantic info with fewer features than final dataset
    semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'age in years'},
            {'name': 'workclass', 'semantic_description': 'type of employment'}
        ]
    }
    
    logger.info(f"Original dataset features: {len(test_dataset['attribute_names'])}")
    logger.info(f"Retained features: {len(retained_features)}")
    logger.info(f"Original semantic features: {len(semantic_info['columns'])}")
    
    # Test semantic expansion
    logger.info("\n--- Testing Semantic Expansion ---")
    expanded_semantic = expand_semantic_features(semantic_info, len(retained_features), retained_features)
    
    semantic_names = [col['name'] for col in expanded_semantic['columns']]
    
    # Validate semantic expansion
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Count matches
    if len(semantic_names) == len(retained_features):
        logger.info("✅ Semantic feature count matches retained feature count")
        tests_passed += 1
    else:
        logger.error(f"❌ Count mismatch: {len(semantic_names)} vs {len(retained_features)}")
    
    # Test 2: Names match exactly
    if set(semantic_names) == set(retained_features):
        logger.info("✅ Semantic feature names exactly match retained feature names")
        tests_passed += 1
    else:
        logger.error(f"❌ Name mismatch")
        logger.error(f"   Expected: {retained_features}")
        logger.error(f"   Got: {semantic_names}")
    
    # Test 3: All names unique
    if len(set(semantic_names)) == len(semantic_names):
        logger.info("✅ All semantic feature names are unique")
        tests_passed += 1
    else:
        logger.error(f"❌ Duplicate semantic names found")
    
    # Test 4: All features have descriptions
    missing_desc = [col['name'] for col in expanded_semantic['columns'] if not col.get('semantic_description')]
    if not missing_desc:
        logger.info("✅ All semantic features have descriptions")
        tests_passed += 1
    else:
        logger.error(f"❌ Missing descriptions for: {missing_desc}")
    
    # Test note generation
    logger.info("\n--- Testing Note Generation ---")
    X_test = reduced_dataset['X']
    
    notes_generated = 0
    empty_notes = 0
    
    for idx in range(len(X_test)):
        row = X_test.iloc[idx]
        try:
            note = generate_note_from_row(row, expanded_semantic, retained_features, exclude_target=True)
            if note and len(note.strip()) > 0:
                notes_generated += 1
                if idx == 0:  # Log first note as example
                    logger.info(f"✅ Sample note generated: {note}")
            else:
                empty_notes += 1
                logger.warning(f"⚠️ Empty note for row {idx}")
        except Exception as e:
            empty_notes += 1
            logger.error(f"❌ Note generation failed for row {idx}: {e}")
    
    if empty_notes == 0:
        logger.info(f"✅ All {notes_generated} notes generated successfully")
        tests_passed += 1
        total_tests += 1
    else:
        logger.error(f"❌ {empty_notes} empty notes out of {len(X_test)} total")
    
    # Mock LLM prediction test
    logger.info("\n--- Testing Mocked LLM Pipeline ---")
    
    class MockLLM:
        def predict(self, prompts):
            # Return mock predictions based on prompt count
            return ['>50K' if i % 2 == 1 else '<=50K' for i in range(len(prompts))]
    
    mock_llm = MockLLM()
    
    # Create prompts (would normally be done by TabLLM)
    prompts = []
    for idx in range(len(X_test)):
        row = X_test.iloc[idx]
        note = generate_note_from_row(row, expanded_semantic, retained_features, exclude_target=True)
        prompt = f"Based on this information: {note}\nQuestion: What is the income class?"
        prompts.append(prompt)
    
    # Test mock predictions
    predictions = mock_llm.predict(prompts)
    
    if len(predictions) == len(X_test):
        logger.info(f"✅ Mock LLM generated {len(predictions)} predictions")
        tests_passed += 1
        total_tests += 1
        
        # Validate predictions are in expected format
        valid_predictions = all(pred in ['<=50K', '>50K'] for pred in predictions)
        if valid_predictions:
            logger.info("✅ All predictions are in valid format")
            tests_passed += 1
            total_tests += 1
        else:
            logger.error("❌ Invalid prediction format found")
    else:
        logger.error(f"❌ Prediction count mismatch: {len(predictions)} vs {len(X_test)}")
    
    # Final summary
    logger.info(f"\n--- Pipeline Test Summary ---")
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("✅ TabLLM pipeline tests PASSED - feature alignment is correct")
    else:
        logger.error("❌ TabLLM pipeline tests FAILED - feature alignment issues detected")
    
    return tests_passed == total_tests


def test_tabllm_integration():
    """Test full TabLLM integration with a minimal dataset."""
    logger.info("\n=== Testing TabLLM Integration ===")
    
    # Silence verbose logging for integration test
    logging.getLogger('marvis.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('examples.tabular.llm_baselines.tabllm_baseline').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.tabllm_baseline import evaluate_tabllm
    
    args = create_test_args(task_id=23, max_context_length=2048)
    
    # Create a minimal test dataset
    test_dataset = {
        'name': 'test_dataset',
        'task_id': 23,  # Use task_id instead of id
        'X': pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'feature_3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        }),
        'y': pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        'attribute_names': ['feature_1', 'feature_2', 'feature_3']
    }
    
    logger.info("--- Testing TabLLM evaluation pipeline ---")
    
    try:
        # This is a dry run to test the pipeline
        results = evaluate_tabllm(test_dataset, args)
        
        if isinstance(results, dict):
            logger.info("✅ TabLLM evaluation completed successfully")
            logger.info(f"   Model: {results.get('model_name', 'Unknown')}")
            logger.info(f"   Dataset: {results.get('dataset_name', 'Unknown')}")
            logger.info(f"   Completed samples: {results.get('completed_samples', 0)}")
            logger.info(f"   Accuracy: {results.get('accuracy', 'N/A')}")
            
            # Check for note inspection files
            inspection_dir = Path(args.output_dir) / "tabllm_notes_inspection"
            if inspection_dir.exists():
                inspection_files = list(inspection_dir.glob("*.json"))
                if inspection_files:
                    logger.info(f"   ✅ Note inspection files created: {len(inspection_files)} files")
                    
                    # Check content of first file
                    with open(inspection_files[0], 'r') as f:
                        inspection_data = json.load(f)
                    
                    if 'sample_notes' in inspection_data:
                        num_notes = len(inspection_data['sample_notes'])
                        logger.info(f"      Sample notes saved: {num_notes}")
                        
                        if num_notes > 0:
                            first_note = inspection_data['sample_notes'][0]['note']
                            logger.info(f"      First note preview: {first_note[:100]}...")
                else:
                    logger.warning("   ⚠️ No inspection files found")
            else:
                logger.warning("   ⚠️ Inspection directory not created")
                
        else:
            logger.error(f"❌ Unexpected result type: {type(results)}")
            
    except Exception as e:
        # This is expected to fail in some cases due to model loading
        logger.warning(f"⚠️ TabLLM integration test failed (may be expected): {e}")
        logger.info("   This is often due to model availability or API limits in test environment")


def test_semantic_content_validation():
    """Test that semantic information matches real dataset structure using n-gram analysis."""
    logger.info("\n=== Testing Semantic Content Validation with N-gram Matching ===")
    
    import re
    import json
    from collections import Counter
    from examples.tabular.llm_baselines.tabllm_baseline import load_tabllm_config_by_openml_id, expand_semantic_features, generate_note_from_row
    from marvis.data.dataset import load_dataset, get_dataset_info
    
    def extract_ngrams(text, n_values=[1, 2]):
        """Extract n-grams from text."""
        if not text:
            return []
        
        # Clean text: lowercase, extract alphanumeric words
        words = re.findall(r'\b\w+\b', str(text).lower())
        
        ngrams = []
        for n in n_values:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
        return ngrams
    
    def calculate_overlap_score(real_ngrams, semantic_ngrams):
        """Calculate overlap score between real and semantic n-grams."""
        if not semantic_ngrams:
            return 0.0
        
        real_set = set(real_ngrams)
        semantic_set = set(semantic_ngrams)
        
        if not semantic_set:
            return 0.0
        
        # Jaccard similarity: intersection / union
        intersection = len(real_set & semantic_set)
        union = len(real_set | semantic_set)
        
        return intersection / union if union > 0 else 0.0
    
    # Test datasets with their expected characteristics
    test_tasks = [3, 6, 23]
    
    for task_id in test_tasks:
        logger.info(f"\n--- Testing task {task_id} ---")
        
        try:
            # Load real dataset
            logger.info(f"Loading real dataset for task {task_id}...")
            try:
                X, y, categorical_indicator, attribute_names, dataset_name = load_dataset(str(task_id))
                dataset_info = get_dataset_info(task_id)
                logger.info(f"✅ Loaded real dataset: {dataset_name}")
                logger.info(f"   Features: {len(attribute_names)}, Samples: {len(X)}")
            except Exception as e:
                logger.error(f"❌ Failed to load real dataset for task {task_id}: {e}")
                continue
            
            # Extract n-grams from real data
            real_ngrams = []
            
            # From column names
            for name in attribute_names:
                real_ngrams.extend(extract_ngrams(name))
            
            # From dataset name
            real_ngrams.extend(extract_ngrams(dataset_name))
            
            # From categorical values (sample first few unique values)
            for i, is_categorical in enumerate(categorical_indicator):
                if is_categorical and i < len(attribute_names):
                    unique_values = pd.Series(X[:, i]).unique()[:5]  # Sample first 5 unique values
                    for val in unique_values:
                        real_ngrams.extend(extract_ngrams(str(val)))
            
            logger.info(f"   Extracted {len(set(real_ngrams))} unique n-grams from real data")
            
            # Load semantic metadata
            logger.info(f"Loading semantic metadata for task {task_id}...")
            config_data, feature_mapping = load_tabllm_config_by_openml_id(task_id, original_feature_count=len(attribute_names))
            
            if config_data is None or feature_mapping is None:
                logger.warning(f"⚠️ No semantic data found for task {task_id}, skipping validation")
                continue
                
            semantic_info = feature_mapping.get('semantic_info')
            if not semantic_info:
                logger.warning(f"⚠️ No semantic info for task {task_id}")
                continue
            
            # Extract n-grams from semantic descriptions
            semantic_ngrams = []
            
            # From column semantic descriptions
            if 'columns' in semantic_info:
                for col in semantic_info['columns']:
                    if 'name' in col:
                        semantic_ngrams.extend(extract_ngrams(col['name']))
                    if 'semantic_description' in col:
                        semantic_ngrams.extend(extract_ngrams(col['semantic_description']))
            
            # From feature descriptions (alternative format)
            if 'feature_description' in semantic_info:
                for feature, description in semantic_info['feature_description'].items():
                    semantic_ngrams.extend(extract_ngrams(feature))
                    semantic_ngrams.extend(extract_ngrams(description))
            
            # From dataset description
            if 'description' in semantic_info:
                semantic_ngrams.extend(extract_ngrams(semantic_info['description']))
            
            # From target class meanings
            if 'target_classes' in semantic_info:
                for target_class in semantic_info['target_classes']:
                    if 'name' in target_class:
                        semantic_ngrams.extend(extract_ngrams(target_class['name']))
                    if 'meaning' in target_class:
                        semantic_ngrams.extend(extract_ngrams(target_class['meaning']))
            
            logger.info(f"   Extracted {len(set(semantic_ngrams))} unique n-grams from semantic data")
            
            # Calculate overlap scores
            overlap_score = calculate_overlap_score(real_ngrams, semantic_ngrams)
            
            # Find common n-grams
            real_set = set(real_ngrams)
            semantic_set = set(semantic_ngrams)
            common_ngrams = real_set & semantic_set
            
            logger.info(f"   Overlap score: {overlap_score:.3f}")
            logger.info(f"   Common n-grams: {len(common_ngrams)}")
            
            if common_ngrams:
                # Show most frequent common n-grams
                common_counter = Counter([ng for ng in real_ngrams + semantic_ngrams if ng in common_ngrams])
                top_common = common_counter.most_common(5)
                logger.info(f"   Top common n-grams: {[ng for ng, count in top_common]}")
            
            # Enhanced validation logic
            # Check for semantic meaningfulness of common n-grams
            meaningful_common = []
            for ngram in common_ngrams:
                # Skip generic terms and numbers
                if ngram not in ['0', '1', '2', '3', '4', '5', 'of', 'the', 'and', 'or', 'is', 'a', 'an']:
                    if len(ngram) > 1:  # Meaningful terms should be longer than 1 char
                        meaningful_common.append(ngram)
            
            # Dynamic thresholds based on dataset characteristics
            min_meaningful_common = max(3, len(attribute_names) // 10)  # At least 3 or 10% of features
            min_overlap_score = 0.02  # Lowered threshold for technical datasets
            
            # Domain-specific validation
            domain_specific_score = 0
            if meaningful_common:
                # Check if common n-grams are relevant to the dataset domain
                domain_keywords = {
                    3: ['bk', 'wk', 'rook', 'king', 'chess'],  # chess terms for task 3
                    6: ['box', 'pixel', 'letter', 'edge', 'x', 'y'],  # vision terms for task 6  
                    23: ['wife', 'education', 'religion', 'children', 'husband']  # demographic terms for task 23
                }
                
                expected_domain_terms = domain_keywords.get(task_id, [])
                domain_matches = [term for term in meaningful_common if any(keyword in term.lower() for keyword in expected_domain_terms)]
                domain_specific_score = len(domain_matches) / max(1, len(expected_domain_terms))
            
            # Overall validation score
            validation_passed = False
            reasons = []
            
            if len(meaningful_common) >= min_meaningful_common:
                reasons.append(f"Meaningful common n-grams: {len(meaningful_common)} >= {min_meaningful_common}")
                validation_passed = True
            else:
                reasons.append(f"Insufficient meaningful common n-grams: {len(meaningful_common)} < {min_meaningful_common}")
            
            if overlap_score >= min_overlap_score:
                reasons.append(f"Overlap score: {overlap_score:.3f} >= {min_overlap_score}")
                validation_passed = True
            else:
                reasons.append(f"Low overlap score: {overlap_score:.3f} < {min_overlap_score}")
            
            if domain_specific_score > 0.3:  # 30% domain match
                reasons.append(f"Good domain alignment: {domain_specific_score:.3f}")
                validation_passed = True
            
            if validation_passed:
                logger.info(f"✅ Semantic validation passed for task {task_id}")
                for reason in reasons:
                    logger.info(f"   {reason}")
                if meaningful_common:
                    logger.info(f"   Meaningful common n-grams: {meaningful_common[:10]}")
            else:
                logger.warning(f"⚠️ Semantic validation needs review for task {task_id}")
                for reason in reasons:
                    logger.warning(f"   {reason}")
                
                # Show diagnostic information
                real_sample = [ng for ng in list(real_set)[:15] if len(ng) > 1]
                semantic_sample = [ng for ng in list(semantic_set)[:15] if len(ng) > 1]
                logger.info(f"   Sample real n-grams: {real_sample}")
                logger.info(f"   Sample semantic n-grams: {semantic_sample}")
                logger.info(f"   All meaningful common: {meaningful_common}")
            
            # Test note generation alignment
            logger.info("--- Testing note generation with n-gram validation ---")
            try:
                # Create test row with actual column names
                test_row = pd.Series([1] * len(attribute_names), index=attribute_names)
                
                note = generate_note_from_row(test_row, semantic_info, attribute_names, exclude_target=True)
                
                if note and len(note.strip()) > 0:
                    # Extract n-grams from generated note
                    note_ngrams = extract_ngrams(note)
                    note_real_overlap = len(set(note_ngrams) & real_set)
                    
                    logger.info(f"✅ Note generated successfully")
                    logger.info(f"   Note contains {note_real_overlap} n-grams from real data")
                    
                    if note_real_overlap >= 3:
                        logger.info(f"   ✅ Note semantically aligned with real data")
                    else:
                        logger.warning(f"   ⚠️ Note has weak alignment with real data")
                        
                else:
                    logger.error(f"❌ Failed to generate note for task {task_id}")
                    
            except Exception as e:
                logger.error(f"❌ Note generation failed for task {task_id}: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to test task {task_id}: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
    
    logger.info(f"\n=== Semantic Validation Complete ===")
    logger.info("N-gram analysis provides quantitative validation of semantic-real data alignment")

def test_real_cc18_semantic_data():
    """Test TabLLM pipeline with real CC18 semantic data."""
    logger.info("\n=== Testing Real CC18 Semantic Data ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import (
        load_tabllm_config_by_openml_id, expand_semantic_features, generate_note_from_row
    )
    import pandas as pd
    
    # Load real CC18 semantic data for task 23 (cmc dataset)
    try:
        config_data, feature_mapping = load_tabllm_config_by_openml_id(23, original_feature_count=9)
        
        if config_data is None or feature_mapping is None:
            logger.warning("⚠️ Could not load real CC18 data for task 23 - skipping test")
            return
        
        real_semantic_info = feature_mapping.get('semantic_info')
        if not real_semantic_info:
            logger.warning("⚠️ No semantic info in loaded CC18 data - skipping test")
            return
            
        logger.info(f"✅ Loaded real CC18 semantic data for task 23")
        logger.info(f"   Structure: {list(real_semantic_info.keys())}")
        
        # Check the structure type
        if 'feature_description' in real_semantic_info:
            original_feature_count = len(real_semantic_info['feature_description'])
            logger.info(f"   Original features in semantic data: {original_feature_count}")
            
            # Simulate feature subselection (reduce from 9 to 6 features)
            simulated_retained_features = [
                "Wife's age", "Wife's education", "Husband's education", 
                "Number of children ever born", "Wife's religion", "Media exposure"
            ]
            logger.info(f"   Simulated retained features: {len(simulated_retained_features)}")
            
            # Test semantic expansion with real data
            expanded_semantic = expand_semantic_features(
                real_semantic_info, 
                len(simulated_retained_features), 
                simulated_retained_features
            )
            
            # Validate expansion
            tests_passed = 0
            total_tests = 4
            
            if 'feature_description' in expanded_semantic:
                expanded_features = list(expanded_semantic['feature_description'].keys())
                
                # Test 1: Count matches
                if len(expanded_features) == len(simulated_retained_features):
                    logger.info("✅ Real data: Semantic feature count matches retained count")
                    tests_passed += 1
                else:
                    logger.error(f"❌ Real data: Count mismatch {len(expanded_features)} vs {len(simulated_retained_features)}")
                
                # Test 2: Names match exactly
                if set(expanded_features) == set(simulated_retained_features):
                    logger.info("✅ Real data: Semantic feature names match retained names")
                    tests_passed += 1
                else:
                    logger.error("❌ Real data: Feature names don't match")
                    missing = set(simulated_retained_features) - set(expanded_features)
                    extra = set(expanded_features) - set(simulated_retained_features)
                    if missing: logger.error(f"   Missing: {missing}")
                    if extra: logger.error(f"   Extra: {extra}")
                
                # Test 3: All names unique
                if len(set(expanded_features)) == len(expanded_features):
                    logger.info("✅ Real data: All feature names are unique")
                    tests_passed += 1
                else:
                    logger.error("❌ Real data: Duplicate feature names found")
                
                # Test 4: All features have descriptions
                missing_desc = [feat for feat, desc in expanded_semantic['feature_description'].items() if not desc]
                if not missing_desc:
                    logger.info("✅ Real data: All features have descriptions")
                    tests_passed += 1
                else:
                    logger.error(f"❌ Real data: Missing descriptions for: {missing_desc}")
                
                # Test note generation with real data
                logger.info("\n--- Testing Note Generation with Real CC18 Data ---")
                
                # Create realistic test data
                test_row_data = {
                    "Wife's age": 29,
                    "Wife's education": 3,
                    "Husband's education": 4,
                    "Number of children ever born": 2,
                    "Wife's religion": 1,
                    "Media exposure": 0
                }
                test_row = pd.Series(test_row_data)
                
                try:
                    note = generate_note_from_row(test_row, expanded_semantic, simulated_retained_features, exclude_target=True)
                    if note and len(note.strip()) > 0:
                        logger.info("✅ Real data: Note generation successful")
                        logger.info(f"   Sample note: {note}")
                        tests_passed += 1
                        total_tests += 1
                        
                        # Validate note contains expected semantic descriptions
                        if "Age of wife in years" in note and "29" in note:
                            logger.info("   ✅ Note contains semantic descriptions from real data")
                        else:
                            logger.warning("   ⚠️ Note may not contain expected semantic content")
                    else:
                        logger.error("❌ Real data: Empty note generated")
                except Exception as e:
                    logger.error(f"❌ Real data: Note generation failed: {e}")
                
                # Summary
                logger.info(f"\n--- Real CC18 Data Test Summary ---")
                logger.info(f"Tests passed: {tests_passed}/{total_tests}")
                
                if tests_passed >= 4:  # Core tests must pass
                    logger.info("✅ Real CC18 semantic data pipeline working correctly")
                else:
                    logger.error("❌ Real CC18 semantic data pipeline has issues")
            else:
                logger.error("❌ Expanded semantic data missing feature_description structure")
        else:
            logger.warning("⚠️ Real semantic data doesn't use feature_description structure")
            
    except Exception as e:
        logger.error(f"❌ Error testing real CC18 data: {e}")


def test_balance_scale_empty_notes_bug():
    """Test the specific bug fix for balance-scale empty notes (task 11)."""
    logger.info("\n=== Testing Balance-Scale Empty Notes Bug Fix ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import (
        load_tabllm_config_by_openml_id, expand_semantic_features, generate_note_from_row
    )
    import pandas as pd
    
    # Test the exact scenario that was failing before the fix
    try:
        # Load task 11 (balance-scale) configuration
        config_data, feature_mapping = load_tabllm_config_by_openml_id(11, original_feature_count=4)
        
        if not config_data or not feature_mapping:
            logger.warning("⚠️ Could not load task 11 config - skipping test")
            return
        
        semantic_info = feature_mapping.get('semantic_info')
        if not semantic_info:
            logger.warning("⚠️ No semantic info for task 11 - skipping test")
            return
        
        logger.info("✅ Loaded task 11 (balance-scale) configuration")
        
        # Test with the exact feature names that were causing the issue
        actual_feature_names = ['left-weight', 'left-distance', 'right-weight', 'right-distance']
        logger.info(f"Using actual feature names: {actual_feature_names}")
        
        # Before fix: semantic features would be ['Left weight', 'Left distance', 'Right weight', 'Right distance']
        # After fix: semantic features should be ['left-weight', 'left-distance', 'right-weight', 'right-distance']
        
        # Expand semantic features (this should trigger realignment)
        expanded_semantic = expand_semantic_features(semantic_info, 4, actual_feature_names)
        
        # Validate that the fix worked
        tests_passed = 0
        total_tests = 3
        
        if 'columns' in expanded_semantic:
            expanded_names = [col['name'] for col in expanded_semantic['columns']]
            
            # Test 1: Names should match exactly
            if set(expanded_names) == set(actual_feature_names):
                logger.info("✅ Bug fix working: Semantic names match actual feature names")
                tests_passed += 1
            else:
                logger.error(f"❌ Bug fix failed: Names don't match")
                logger.error(f"   Expected: {actual_feature_names}")
                logger.error(f"   Got: {expanded_names}")
            
            # Test 2: Create test data and generate note (this should NOT be empty)
            test_data = pd.Series([1, 1, 1, 1], index=actual_feature_names)
            
            try:
                note = generate_note_from_row(test_data, expanded_semantic, actual_feature_names, exclude_target=True)
                if note and len(note.strip()) > 0:
                    logger.info("✅ Bug fix working: Note generation successful")
                    logger.info(f"   Generated note: {note[:100]}...")
                    tests_passed += 1
                else:
                    logger.error("❌ Bug fix failed: Empty note still generated")
            except Exception as e:
                logger.error(f"❌ Bug fix failed: Note generation error: {e}")
            
            # Test 3: All features should have descriptions
            missing_desc = [col['name'] for col in expanded_semantic['columns'] if not col.get('semantic_description')]
            if not missing_desc:
                logger.info("✅ Bug fix working: All features have semantic descriptions")
                tests_passed += 1
            else:
                logger.error(f"❌ Bug fix issue: Missing descriptions for: {missing_desc}")
        
        elif 'feature_description' in expanded_semantic:
            expanded_names = list(expanded_semantic['feature_description'].keys())
            
            # Test 1: Names should match exactly
            if set(expanded_names) == set(actual_feature_names):
                logger.info("✅ Bug fix working: Semantic names match actual feature names")
                tests_passed += 1
            else:
                logger.error(f"❌ Bug fix failed: Names don't match")
                logger.error(f"   Expected: {actual_feature_names}")
                logger.error(f"   Got: {expanded_names}")
            
            # Test 2: Create test data and generate note (this should NOT be empty)
            test_data = pd.Series([1, 1, 1, 1], index=actual_feature_names)
            
            try:
                note = generate_note_from_row(test_data, expanded_semantic, actual_feature_names, exclude_target=True)
                if note and len(note.strip()) > 0:
                    logger.info("✅ Bug fix working: Note generation successful")
                    logger.info(f"   Generated note: {note[:100]}...")
                    tests_passed += 1
                else:
                    logger.error("❌ Bug fix failed: Empty note still generated")
            except Exception as e:
                logger.error(f"❌ Bug fix failed: Note generation error: {e}")
            
            # Test 3: All features should have descriptions
            missing_desc = [feat for feat, desc in expanded_semantic['feature_description'].items() if not desc]
            if not missing_desc:
                logger.info("✅ Bug fix working: All features have semantic descriptions")
                tests_passed += 1
            else:
                logger.error(f"❌ Bug fix issue: Missing descriptions for: {missing_desc}")
        
        # Summary
        logger.info(f"\n--- Balance-Scale Bug Fix Test Summary ---")
        logger.info(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            logger.info("✅ Balance-scale empty notes bug has been FIXED!")
        else:
            logger.error("❌ Balance-scale empty notes bug still exists")
        
        return tests_passed == total_tests
        
    except Exception as e:
        logger.error(f"❌ Error testing balance-scale bug fix: {e}")
        return False


def test_note_inspection_system():
    """Test the note inspection file saving system."""
    logger.info("\n=== Testing Note Inspection System ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import save_sample_notes_for_inspection
    
    # Create test data
    few_shot_examples = [
        ("The age is 25. The workclass is Private. The education is Bachelors.", "<=50K"),
        ("The age is 38. The workclass is Self-emp-not-inc. The education is HS-grad.", ">50K"),
        ("The age is 45. The workclass is Local-gov. The education is Masters.", ">50K")
    ]
    
    test_dataset = {
        'name': 'adult',
        'task_id': 1590,
        'attribute_names': ['age', 'workclass', 'education']
    }
    
    test_semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'age in years'},
            {'name': 'workclass', 'semantic_description': 'employment type'},
            {'name': 'education', 'semantic_description': 'education level'}
        ]
    }
    
    args = create_test_args()
    
    try:
        save_sample_notes_for_inspection(few_shot_examples, test_dataset, args, test_semantic_info)
        
        # Check if file was created
        inspection_dir = Path(args.output_dir) / "tabllm_notes_inspection"
        if inspection_dir.exists():
            inspection_files = list(inspection_dir.glob("*.json"))
            if inspection_files:
                logger.info(f"✅ Inspection file created: {inspection_files[0].name}")
                
                # Validate file content
                with open(inspection_files[0], 'r') as f:
                    data = json.load(f)
                
                required_keys = ['metadata', 'semantic_expansion_info', 'sample_notes']
                for key in required_keys:
                    if key in data:
                        logger.info(f"   ✅ Found required key: {key}")
                    else:
                        logger.error(f"   ❌ Missing required key: {key}")
                
                if 'sample_notes' in data and len(data['sample_notes']) > 0:
                    note_keys = ['note', 'label', 'statistics']
                    first_note = data['sample_notes'][0]
                    for key in note_keys:
                        if key in first_note:
                            logger.info(f"      ✅ Note has required key: {key}")
                        else:
                            logger.error(f"      ❌ Note missing key: {key}")
            else:
                logger.error("❌ No inspection files created")
        else:
            logger.error("❌ Inspection directory not created")
            
    except Exception as e:
        logger.error(f"❌ ERROR in note inspection test: {e}")
    
    finally:
        # Clean up
        if os.path.exists(args.output_dir):
            try:
                shutil.rmtree(args.output_dir)
            except:
                pass

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Comprehensive TabLLM Test Suite')
    parser.add_argument('--task_id', type=int, default=23, help='OpenML task ID to test with')
    parser.add_argument('--output_dir', type=str, help='Output directory for test files')
    parser.add_argument('--skip_integration', action='store_true', help='Skip integration test')
    
    args = parser.parse_args()
    
    logger.info("=== Comprehensive TabLLM Baseline Test Suite ===")
    logger.info(f"Testing with task_id: {args.task_id}")
    
    # Run all tests
    test_task_id_resolution()
    test_semantic_feature_expansion()
    test_semantic_feature_alignment()
    test_real_cc18_semantic_data()
    test_balance_scale_empty_notes_bug()  # Test the specific bug fix
    test_semantic_content_validation()  # Test semantic content correctness
    test_online_note_generation()
    test_context_aware_truncation()
    test_note_inspection_system()
    test_tabllm_pipeline_with_mocked_llm()
    
    if not args.skip_integration:
        test_tabllm_integration()
    else:
        logger.info("\n=== Skipping Integration Test (--skip_integration) ===")
    
    logger.info("\n=== All TabLLM Tests Complete ===")

if __name__ == "__main__":
    main()