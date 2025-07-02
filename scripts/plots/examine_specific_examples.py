#!/usr/bin/env python3
"""
Examine specific examples where top methods disagree to understand reasoning differences.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

ABLATIONS_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"
OUTPUT_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"

TOP_CONFIGS = [
    "tsne_perturbation_axes",
    "tsne_semantic_axes", 
    "tsne_knn",
    "basic_tsne"
]

def load_and_compare_specific_examples():
    """Load and compare specific examples where methods disagree."""
    
    # Load results for top 4 configurations
    results_by_config = {}
    
    test_dirs = [d for d in os.listdir(ABLATIONS_DIR) 
                 if d.startswith('test_vlm_outputs_') and 
                 os.path.isdir(os.path.join(ABLATIONS_DIR, d))]
    
    for test_dir in sorted(test_dirs):
        test_id = test_dir.replace('test_vlm_outputs_', '')
        
        for config_name in TOP_CONFIGS:
            config_path = Path(ABLATIONS_DIR) / test_dir / config_name / "detailed_vlm_outputs.json"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                if config_name not in results_by_config:
                    results_by_config[config_name] = {}
                results_by_config[config_name][test_id] = data
    
    # Find interesting disagreement cases
    examples = []
    
    for test_id in results_by_config[TOP_CONFIGS[0]].keys():
        # Check if all configs have this test
        if not all(test_id in results_by_config[config] for config in TOP_CONFIGS):
            continue
        
        # Get prediction details for each config
        test_data = {}
        for config in TOP_CONFIGS:
            if 'prediction_details' in results_by_config[config][test_id]:
                test_data[config] = results_by_config[config][test_id]['prediction_details']
        
        if len(test_data) < 2:
            continue
        
        # Compare predictions for each test point
        for i in range(min(len(details) for details in test_data.values())):
            point_results = {}
            
            for config in TOP_CONFIGS:
                if config in test_data and i < len(test_data[config]):
                    detail = test_data[config][i]
                    point_results[config] = {
                        'prediction': detail.get('parsed_prediction'),
                        'true_label': detail.get('true_label'),
                        'response': detail.get('vlm_response', ''),
                        'correct': detail.get('parsed_prediction') == detail.get('true_label')
                    }
            
            # Find cases where there's disagreement and different correctness
            if len(point_results) >= 2:
                predictions = [r['prediction'] for r in point_results.values()]
                correctness = [r['correct'] for r in point_results.values()]
                
                # Case 1: One method correct, others wrong
                if sum(correctness) == 1:
                    examples.append({
                        'type': 'one_correct',
                        'test_id': test_id,
                        'test_point': i,
                        'results': point_results
                    })
                
                # Case 2: Mixed predictions but some pattern
                elif len(set(predictions)) > 1 and len(set(correctness)) > 1:
                    examples.append({
                        'type': 'mixed_performance',
                        'test_id': test_id,
                        'test_point': i,
                        'results': point_results
                    })
    
    return examples[:20]  # Return top 20 interesting cases

def analyze_examples(examples):
    """Analyze the collected examples."""
    
    print("SPECIFIC EXAMPLE ANALYSIS")
    print("=" * 50)
    
    for i, example in enumerate(examples[:10]):
        print(f"\nExample {i+1}: {example['type']} (Test {example['test_id']}, Point {example['test_point']})")
        print("-" * 40)
        
        true_label = list(example['results'].values())[0]['true_label']
        print(f"True label: {true_label}")
        print()
        
        for config, result in example['results'].items():
            status = "✓ CORRECT" if result['correct'] else "✗ WRONG"
            print(f"{config}: Predicted {result['prediction']} {status}")
            print(f"  Reasoning: {result['response'][:200]}...")
            print()
        
        print("=" * 50)

def main():
    examples = load_and_compare_specific_examples()
    print(f"Found {len(examples)} interesting disagreement cases")
    
    analyze_examples(examples)
    
    # Save examples
    output_file = os.path.join(OUTPUT_DIR, "reasoning_examples.json")
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Examples saved to: {output_file}")

if __name__ == "__main__":
    main()