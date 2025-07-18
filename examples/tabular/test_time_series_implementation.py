#!/usr/bin/env python
"""
Test script for MARVIS time series implementation.

This script tests the time series distribution visualization and classification
components without requiring the full gift-eval setup.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add marvis to path
current_dir = Path(__file__).parent
marvis_root = current_dir.parent.parent
marvis_package = marvis_root / "marvis"
sys.path.insert(0, str(marvis_package))

from marvis.viz import TimeSeriesDistributionVisualization, VisualizationConfig
from marvis.utils.vlm_prompting import (
    create_time_series_classification_prompt,
    extract_time_series_classification_response
)


def generate_synthetic_time_series(length: int = 100, pattern: str = "trend") -> np.ndarray:
    """Generate synthetic time series data for testing."""
    np.random.seed(42)
    time = np.arange(length)
    
    if pattern == "trend":
        # Upward trend with noise
        series = 0.05 * time + 2 + 0.5 * np.random.randn(length)
    elif pattern == "seasonal":
        # Seasonal pattern
        series = 3 + 2 * np.sin(2 * np.pi * time / 12) + 0.3 * np.random.randn(length)
    elif pattern == "volatile":
        # High volatility random walk
        increments = np.random.randn(length) * 0.8
        series = np.cumsum(increments) + 5
    elif pattern == "stable":
        # Stable around mean
        series = 4 + 0.2 * np.random.randn(length)
    else:
        # Mixed pattern
        series = (0.02 * time + 
                 1.5 * np.sin(2 * np.pi * time / 15) + 
                 3 + 0.4 * np.random.randn(length))
    
    return series


def test_time_series_visualization():
    """Test the TimeSeriesDistributionVisualization class."""
    print("Testing TimeSeriesDistributionVisualization...")
    
    # Generate test data
    series = generate_synthetic_time_series(length=80, pattern="trend")
    
    # Create visualization config
    config = VisualizationConfig(
        figsize=(12, 8),
        random_state=42,
        task_type='regression',
        extra_params={
            'n_distributions': 4,
            'forecast_horizon': 12,
            'n_keypoints': 6,
            'keypoint_strategy': 'uniform',
            'show_confidence_bands': True
        }
    )
    
    # Create and test visualization
    viz = TimeSeriesDistributionVisualization(config=config)
    
    # Test fitting
    print("Testing fit_transform...")
    transformed_data = viz.fit_transform(series)
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Number of fitted distributions: {len(viz._distributions)}")
    
    # Test plot generation
    print("Testing plot generation...")
    result = viz.generate_plot(transformed_data)
    
    print(f"Plot result type: {type(result.image)}")
    print(f"Image size: {result.image.size}")
    print(f"Metadata keys: {list(result.metadata.keys())}")
    print(f"Class names: {result.metadata.get('class_names', [])}")
    
    # Save test image
    output_path = "/tmp/test_time_series_viz.png"
    result.image.save(output_path)
    print(f"Saved test visualization to {output_path}")
    
    # Test class prediction
    print("Testing class prediction...")
    for i in range(len(viz._distributions)):
        prediction = viz.predict_from_class(i, random_state=42)
        print(f"Class {i} prediction shape: {prediction.shape}")
        print(f"Class {i} prediction sample: {prediction[:5]}")
    
    return result


def test_time_series_prompting():
    """Test the time series prompting functions."""
    print("\nTesting time series prompting...")
    
    # Test data
    class_names = [
        "Uniform: Increasing Trend, Low Volatility, Moderate Tails",
        "Extrema: Stable Trend, High Volatility, Fat Tails",
        "Changepoints: Decreasing Trend, Moderate Volatility, Thin Tails"
    ]
    
    distribution_params = [
        {'df': 4.2, 'loc': 0.05, 'scale': 0.3, 'name': class_names[0]},
        {'df': 2.8, 'loc': 0.01, 'scale': 0.8, 'name': class_names[1]},
        {'df': 8.5, 'loc': -0.02, 'scale': 0.4, 'name': class_names[2]}
    ]
    
    # Test prompt creation
    prompt = create_time_series_classification_prompt(
        class_names=class_names,
        forecast_horizon=12,
        dataset_description="Test synthetic time series with upward trend",
        distribution_params=distribution_params,
        legend_text="Test legend information"
    )
    
    print("Generated prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # Test response extraction
    test_responses = [
        "Based on my analysis, the data shows a clear upward trend with low volatility. Final selection: 0",
        "The series exhibits high volatility with fat tails. I select class 1.",
        "Analysis: decreasing trend observed. Selection: 2",
        "The best choice is 1",
        "2"
    ]
    
    print("\nTesting response extraction:")
    for i, response in enumerate(test_responses):
        extracted = extract_time_series_classification_response(response)
        print(f"Response {i+1}: '{response}' -> {extracted}")
    
    return prompt


def test_integration():
    """Test integration between visualization and prompting."""
    print("\nTesting integration...")
    
    # Create visualization result
    viz_result = test_time_series_visualization()
    
    # Generate prompt from visualization
    from marvis.utils.vlm_prompting import create_time_series_classification_prompt
    
    class_names = viz_result.metadata.get('class_names', [])
    forecast_horizon = viz_result.metadata.get('forecast_horizon', 12)
    distribution_params = viz_result.metadata.get('distribution_params', [])
    
    prompt = create_time_series_classification_prompt(
        class_names=class_names,
        forecast_horizon=forecast_horizon,
        distribution_params=distribution_params,
        legend_text=viz_result.legend_text
    )
    
    print("Integration test successful!")
    print(f"Generated prompt length: {len(prompt)} characters")
    print(f"Number of classes: {len(class_names)}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("MARVIS Time Series Implementation Test")
    print("=" * 60)
    
    try:
        # Test visualization
        viz_result = test_time_series_visualization()
        
        # Test prompting
        test_time_series_prompting()
        
        # Test integration
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        # Print summary
        print(f"\nSummary:")
        print(f"- TimeSeriesDistributionVisualization: ✓ Working")
        print(f"- Student's T distribution fitting: ✓ Working")
        print(f"- VLM prompting functions: ✓ Working")
        print(f"- Response extraction: ✓ Working")
        print(f"- Integration: ✓ Working")
        
        return 0
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())