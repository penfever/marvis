#!/usr/bin/env python3
"""
Analyze reasoning traces and visualizations for top-performing ablation methods.

This script examines the VLM responses, reasoning patterns, and visualizations
to identify what drives performance improvements in the best configurations.

Usage:
    python scripts/analyze_reasoning_traces.py
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
ABLATIONS_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"
OUTPUT_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"

# Top performing configurations to analyze
TOP_CONFIGS = [
    "tsne_perturbation_axes",
    "tsne_semantic_axes", 
    "tsne_knn",
    "tsne_high_dpi",
    "tsne_use_metadata",
    "tsne_semantic",
    "basic_tsne",
    "tsne_3d_perturbation",
    "tsne_semantic_metadata_combined",
    "tsne_importance_axes"
]

def load_detailed_results(test_dir: str, config_name: str) -> Optional[Dict]:
    """Load detailed VLM outputs for a configuration."""
    config_path = Path(test_dir) / config_name / "detailed_vlm_outputs.json"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {config_path}: {e}")
        return None

def extract_reasoning_features(vlm_response: str) -> Dict:
    """Extract features from VLM reasoning text."""
    if not vlm_response:
        return {}
    
    response_lower = vlm_response.lower()
    
    features = {
        # Spatial reasoning indicators
        'mentions_closest': 'closest' in response_lower or 'nearest' in response_lower,
        'mentions_distance': 'distance' in response_lower or 'far' in response_lower,
        'mentions_cluster': 'cluster' in response_lower or 'group' in response_lower,
        'mentions_region': 'region' in response_lower or 'area' in response_lower,
        'mentions_boundary': 'boundary' in response_lower or 'edge' in response_lower,
        'mentions_density': 'dense' in response_lower or 'sparse' in response_lower,
        
        # Color-based reasoning
        'mentions_color': any(color in response_lower for color in ['blue', 'red', 'green', 'orange', 'purple', 'yellow']),
        'color_count': len([color for color in ['blue', 'red', 'green', 'orange', 'purple', 'yellow'] if color in response_lower]),
        
        # Pattern recognition
        'mentions_pattern': 'pattern' in response_lower or 'structure' in response_lower,
        'mentions_separation': 'separate' in response_lower or 'isolated' in response_lower,
        'mentions_overlap': 'overlap' in response_lower or 'mixed' in response_lower,
        
        # Confidence indicators
        'uses_uncertainty': any(phrase in response_lower for phrase in ['appears', 'seems', 'likely', 'probably', 'might']),
        'uses_certainty': any(phrase in response_lower for phrase in ['clearly', 'obviously', 'definitely', 'certainly']),
        
        # Quantitative reasoning
        'mentions_majority': 'majority' in response_lower or 'most' in response_lower,
        'mentions_surrounding': 'surround' in response_lower or 'around' in response_lower,
        
        # Metadata/semantic reasoning
        'mentions_axis': 'axis' in response_lower or 'dimension' in response_lower,
        'mentions_feature': 'feature' in response_lower or 'attribute' in response_lower,
        'mentions_semantic': 'semantic' in response_lower or 'meaning' in response_lower,
        
        # Response length and complexity
        'response_length': len(vlm_response),
        'word_count': len(vlm_response.split()),
        'sentence_count': len([s for s in vlm_response.split('.') if s.strip()]),
    }
    
    return features

def analyze_prediction_consistency(results_by_config: Dict[str, Dict]) -> Dict:
    """Analyze prediction consistency across configurations for the same test points."""
    
    # Group by test_id and test_point_idx
    test_point_results = defaultdict(lambda: defaultdict(dict))
    
    for config_name, test_results in results_by_config.items():
        for test_id, data in test_results.items():
            if 'prediction_details' in data:
                for detail in data['prediction_details']:
                    test_idx = detail['test_point_idx']
                    test_point_results[test_id][test_idx][config_name] = {
                        'prediction': detail.get('parsed_prediction'),
                        'true_label': detail.get('true_label'),
                        'response': detail.get('vlm_response', ''),
                        'correct': detail.get('parsed_prediction') == detail.get('true_label')
                    }
    
    # Analyze consistency patterns
    consistency_analysis = {
        'agreement_rates': {},
        'flip_patterns': defaultdict(int),
        'performance_correlations': {},
        'reasoning_differences': []
    }
    
    total_points = 0
    agreement_counts = defaultdict(int)
    
    for test_id, points in test_point_results.items():
        for test_idx, config_results in points.items():
            if len(config_results) < 2:  # Need at least 2 configs to compare
                continue
                
            total_points += 1
            configs = list(config_results.keys())
            predictions = [config_results[c]['prediction'] for c in configs]
            
            # Count agreements
            unique_predictions = len(set(predictions))
            if unique_predictions == 1:
                agreement_counts['all_agree'] += 1
            elif unique_predictions == len(predictions):
                agreement_counts['all_disagree'] += 1
            else:
                agreement_counts['partial_agreement'] += 1
            
            # Analyze specific flips between top methods
            for i, config1 in enumerate(configs):
                for config2 in configs[i+1:]:
                    pred1 = config_results[config1]['prediction']
                    pred2 = config_results[config2]['prediction']
                    correct1 = config_results[config1]['correct']
                    correct2 = config_results[config2]['correct']
                    
                    if pred1 != pred2:
                        flip_key = f"{config1}_vs_{config2}"
                        consistency_analysis['flip_patterns'][flip_key] += 1
                        
                        # Track improvement patterns
                        if correct1 and not correct2:
                            consistency_analysis['flip_patterns'][f"{config1}_better_than_{config2}"] += 1
                        elif correct2 and not correct1:
                            consistency_analysis['flip_patterns'][f"{config2}_better_than_{config1}"] += 1
    
    # Calculate agreement rates
    if total_points > 0:
        for key, count in agreement_counts.items():
            consistency_analysis['agreement_rates'][key] = count / total_points
    
    return consistency_analysis, test_point_results

def compare_reasoning_patterns(test_point_results: Dict, top_configs: List[str]) -> Dict:
    """Compare reasoning patterns between high and low performing configurations."""
    
    reasoning_comparison = {
        'feature_differences': {},
        'performance_driven_features': {},
        'example_comparisons': []
    }
    
    # Collect reasoning features for correct vs incorrect predictions
    correct_features = defaultdict(list)
    incorrect_features = defaultdict(list)
    
    config_features = defaultdict(lambda: defaultdict(list))
    
    for test_id, points in test_point_results.items():
        for test_idx, config_results in points.items():
            for config_name, result in config_results.items():
                if config_name not in top_configs:
                    continue
                    
                features = extract_reasoning_features(result['response'])
                
                # Store by config
                for feature, value in features.items():
                    config_features[config_name][feature].append(value)
                
                # Store by correctness
                if result['correct']:
                    for feature, value in features.items():
                        correct_features[feature].append(value)
                else:
                    for feature, value in features.items():
                        incorrect_features[feature].append(value)
    
    # Calculate feature differences between correct and incorrect
    for feature in set(correct_features.keys()) | set(incorrect_features.keys()):
        correct_vals = correct_features.get(feature, [])
        incorrect_vals = incorrect_features.get(feature, [])
        
        if correct_vals and incorrect_vals:
            if isinstance(correct_vals[0], bool):
                correct_rate = np.mean(correct_vals)
                incorrect_rate = np.mean(incorrect_vals)
                diff = correct_rate - incorrect_rate
            else:
                correct_mean = np.mean(correct_vals)
                incorrect_mean = np.mean(incorrect_vals)
                diff = correct_mean - incorrect_mean
            
            reasoning_comparison['performance_driven_features'][feature] = {
                'difference': diff,
                'correct_value': correct_rate if isinstance(correct_vals[0], bool) else correct_mean,
                'incorrect_value': incorrect_rate if isinstance(correct_vals[0], bool) else incorrect_mean
            }
    
    # Calculate differences between configurations
    for feature in set().union(*[f.keys() for f in config_features.values()]):
        feature_by_config = {}
        for config in top_configs:
            if config in config_features and feature in config_features[config]:
                vals = config_features[config][feature]
                if vals:
                    if isinstance(vals[0], bool):
                        feature_by_config[config] = np.mean(vals)
                    else:
                        feature_by_config[config] = np.mean(vals)
        
        if len(feature_by_config) > 1:
            reasoning_comparison['feature_differences'][feature] = feature_by_config
    
    return reasoning_comparison

def analyze_visualization_differences(test_dirs: List[str], top_configs: List[str]) -> Dict:
    """Analyze differences in visualizations between top performing methods."""
    
    viz_analysis = {
        'image_properties': {},
        'color_usage': {},
        'layout_differences': {}
    }
    
    for test_dir in test_dirs:
        test_id = test_dir.split('_')[-1]
        
        for config_name in top_configs:
            config_path = Path(ABLATIONS_DIR) / test_dir / config_name
            
            if not config_path.exists():
                continue
            
            # Look for visualization files
            viz_files = list(config_path.glob("visualization_*.png"))
            multi_viz_files = list(config_path.glob("multi_visualization_*.png"))
            
            viz_analysis['image_properties'][f"{test_id}_{config_name}"] = {
                'has_basic_viz': len(viz_files) > 0,
                'has_multi_viz': len(multi_viz_files) > 0,
                'viz_count': len(viz_files),
                'multi_viz_count': len(multi_viz_files)
            }
    
    return viz_analysis

def create_reasoning_analysis_report(consistency_analysis: Dict, reasoning_comparison: Dict, viz_analysis: Dict) -> str:
    """Create a comprehensive analysis report."""
    
    report = []
    report.append("# VLM Reasoning Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Agreement patterns
    report.append("## Prediction Agreement Patterns")
    report.append("-" * 30)
    agreement_rates = consistency_analysis['agreement_rates']
    for pattern, rate in agreement_rates.items():
        report.append(f"**{pattern.replace('_', ' ').title()}**: {rate:.1%}")
    report.append("")
    
    # Top disagreement patterns
    report.append("## Configuration Disagreement Patterns")
    report.append("-" * 30)
    flip_patterns = consistency_analysis['flip_patterns']
    top_flips = sorted(flip_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for pattern, count in top_flips:
        if 'vs' in pattern:
            report.append(f"**{pattern}**: {count} disagreements")
        elif 'better_than' in pattern:
            report.append(f"**{pattern.replace('_', ' ')}**: {count} cases")
    report.append("")
    
    # Performance-driven reasoning features
    report.append("## Performance-Driven Reasoning Features")
    report.append("-" * 30)
    perf_features = reasoning_comparison['performance_driven_features']
    sorted_features = sorted(perf_features.items(), key=lambda x: abs(x[1]['difference']), reverse=True)
    
    for feature, stats in sorted_features[:15]:
        diff = stats['difference']
        correct_val = stats['correct_value']
        incorrect_val = stats['incorrect_value']
        
        direction = "Higher" if diff > 0 else "Lower"
        report.append(f"**{feature.replace('_', ' ').title()}**:")
        report.append(f"  - {direction} in correct predictions ({correct_val:.3f} vs {incorrect_val:.3f})")
        report.append(f"  - Difference: {diff:+.3f}")
        report.append("")
    
    # Configuration feature differences
    report.append("## Configuration-Specific Reasoning Patterns")
    report.append("-" * 30)
    config_features = reasoning_comparison['feature_differences']
    
    # Find features with highest variance across configs
    high_variance_features = []
    for feature, config_vals in config_features.items():
        if len(config_vals) > 2:
            variance = np.var(list(config_vals.values()))
            high_variance_features.append((feature, variance, config_vals))
    
    high_variance_features.sort(key=lambda x: x[1], reverse=True)
    
    for feature, variance, config_vals in high_variance_features[:10]:
        report.append(f"**{feature.replace('_', ' ').title()}** (variance: {variance:.4f}):")
        sorted_configs = sorted(config_vals.items(), key=lambda x: x[1], reverse=True)
        for config, val in sorted_configs[:5]:
            report.append(f"  - {config}: {val:.3f}")
        report.append("")
    
    # Visualization analysis
    report.append("## Visualization Analysis")
    report.append("-" * 30)
    viz_props = viz_analysis['image_properties']
    
    # Count visualization types
    basic_viz_count = sum(1 for props in viz_props.values() if props['has_basic_viz'])
    multi_viz_count = sum(1 for props in viz_props.values() if props['has_multi_viz'])
    
    report.append(f"**Basic visualizations**: {basic_viz_count} configurations")
    report.append(f"**Multi-panel visualizations**: {multi_viz_count} configurations")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function to run the reasoning analysis."""
    print("Loading top-performing configuration results...")
    
    # Get all test directories
    test_dirs = [d for d in os.listdir(ABLATIONS_DIR) 
                 if d.startswith('test_vlm_outputs_') and 
                 os.path.isdir(os.path.join(ABLATIONS_DIR, d))]
    
    # Load results for top configurations
    results_by_config = defaultdict(dict)
    
    for test_dir in sorted(test_dirs):
        test_id = test_dir.replace('test_vlm_outputs_', '')
        print(f"Processing {test_dir}...")
        
        for config_name in TOP_CONFIGS:
            data = load_detailed_results(os.path.join(ABLATIONS_DIR, test_dir), config_name)
            if data:
                results_by_config[config_name][test_id] = data
            else:
                print(f"  Missing data for {config_name}")
    
    print(f"Loaded data for {len(results_by_config)} configurations")
    
    # Analyze prediction consistency
    print("Analyzing prediction consistency...")
    consistency_analysis, test_point_results = analyze_prediction_consistency(results_by_config)
    
    # Compare reasoning patterns
    print("Comparing reasoning patterns...")
    reasoning_comparison = compare_reasoning_patterns(test_point_results, TOP_CONFIGS)
    
    # Analyze visualization differences
    print("Analyzing visualization differences...")
    viz_analysis = analyze_visualization_differences(test_dirs, TOP_CONFIGS)
    
    # Create comprehensive report
    print("Creating analysis report...")
    report = create_reasoning_analysis_report(consistency_analysis, reasoning_comparison, viz_analysis)
    
    # Save analysis results
    analysis_results = {
        'consistency_analysis': consistency_analysis,
        'reasoning_comparison': reasoning_comparison,
        'visualization_analysis': viz_analysis,
        'configurations_analyzed': TOP_CONFIGS,
        'test_directories': test_dirs
    }
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, "reasoning_analysis_detailed.json")
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"Detailed analysis saved to: {results_file}")
    
    # Save report
    report_file = os.path.join(OUTPUT_DIR, "reasoning_analysis_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Analysis report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("REASONING ANALYSIS SUMMARY")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()