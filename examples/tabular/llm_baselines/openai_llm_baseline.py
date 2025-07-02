"""
OpenAI LLM baseline for tabular classification.
Supports GPT-4.1, GPT-4o, and other OpenAI text models for tabular data tasks.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

# Add project root to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, examples_dir)

from marvis.utils.model_loader import model_loader, GenerationConfig
from marvis.utils.unified_metrics import MetricsLogger

logger = logging.getLogger(__name__)


class OpenAILLMBaseline:
    """
    OpenAI LLM baseline for tabular data classification.
    Converts tabular data to text format and uses OpenAI models for classification.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gpt-4o", feature_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.model_name = model_name
        self.feature_names = feature_names
        self.model_wrapper = None
        self.is_fitted = False
        self.raw_responses = []
        
        # Validate OpenAI model name
        openai_models = [
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-3.5-turbo", "o3", "o4-mini"
        ]
        if not any(model in model_name for model in openai_models):
            logger.warning(f"Model {model_name} may not be an OpenAI model. Supported: {openai_models}")
    
    def load_model(self):
        """Load OpenAI LLM model through the unified model loader."""
        try:
            logger.info(f"Loading OpenAI LLM model: {self.model_name}")
            
            # Use the unified model loader which will auto-detect OpenAI
            self.model_wrapper = model_loader.load_llm(
                model_name=self.model_name,
                backend="openai"
            )
            
            logger.info("OpenAI LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OpenAI LLM model: {e}")
            raise
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'OpenAILLMBaseline':
        """
        Fit the LLM (no training needed for API models).
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Optional feature names
        """
        if self.model_wrapper is None:
            self.load_model()
            
        if feature_names:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
        self.is_fitted = True
        return self
    
    def _format_tabular_data(self, X: np.ndarray, include_labels: bool = False, 
                           y: Optional[np.ndarray] = None) -> List[str]:
        """Convert tabular data to text format for LLM input."""
        formatted_samples = []
        
        for i, sample in enumerate(X):
            # Create a structured text representation
            sample_text = "Sample data:\n"
            for j, (feature_name, value) in enumerate(zip(self.feature_names, sample)):
                sample_text += f"- {feature_name}: {value:.3f}\n"
            
            if include_labels and y is not None:
                sample_text += f"Class: {self.class_names[y[i]]}\n"
            
            formatted_samples.append(sample_text)
        
        return formatted_samples
    
    def _create_classification_prompt(self, sample_text: str) -> str:
        """Create a classification prompt for a single sample."""
        class_list = ", ".join([f'"{name}"' for name in self.class_names])
        
        prompt = f"""You are a machine learning classifier. Given the following tabular data sample, classify it into one of the available classes.

{sample_text}

Available classes: {class_list}

Classify this sample and respond with just the class name in the format: "Class: [class_name]"

Your classification:"""
        
        return prompt
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using OpenAI LLM.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Running OpenAI LLM predictions on {len(X_test)} samples")
        
        predictions = []
        formatted_samples = self._format_tabular_data(X_test)
        
        # Create generation config optimized for classification
        generation_config = GenerationConfig(
            max_new_tokens=10,  # Short response for classification
            temperature=0.1,    # Low temperature for consistent results
            top_p=0.9,
            do_sample=True
        )
        
        for i, sample_text in enumerate(formatted_samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i+1}/{len(formatted_samples)}")
            
            try:
                # Create classification prompt
                prompt = self._create_classification_prompt(sample_text)
                
                # Generate response
                response = self.model_wrapper.generate(prompt, generation_config)
                
                # Store raw response
                response_entry = {
                    "sample_index": i,
                    "prompt": prompt,
                    "raw_response": response,
                    "model_name": self.model_name,
                    "timestamp": time.time()
                }
                self.raw_responses.append(response_entry)
                
                # Parse response to get class
                predicted_class = self._parse_response(response)
                response_entry["parsed_class"] = predicted_class
                
                # Convert to class index
                prediction_idx = self._class_to_index(predicted_class)
                response_entry["predicted_index"] = prediction_idx
                predictions.append(prediction_idx)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                # Default to first class
                predictions.append(0)
        
        return np.array(predictions)
    
    def _parse_response(self, response: str) -> str:
        """Parse LLM response to extract predicted class."""
        response_lower = response.lower().strip()
        
        # Try to find "class:" pattern
        if "class:" in response_lower:
            try:
                class_part = response.split(":", 1)[1].strip()
                # Remove quotes if present
                class_part = class_part.strip('"\'')
                
                # Try to match with available classes
                for class_name in self.class_names:
                    if class_name.lower() == class_part.lower():
                        return class_name
            except:
                pass
        
        # Fallback: look for any class name in the response
        for class_name in self.class_names:
            if class_name.lower() in response_lower:
                return class_name
        
        # Default to first class
        logger.warning(f"Could not parse class from response: '{response}'. Using fallback: {self.class_names[0]}")
        return self.class_names[0]
    
    def _class_to_index(self, class_name: str) -> int:
        """Convert class name to index."""
        try:
            return self.class_names.index(class_name)
        except ValueError:
            logger.warning(f"Unknown class name: {class_name}. Using index 0.")
            return 0
    
    def save_raw_responses(self, output_path: str, benchmark_name: str = "unknown"):
        """Save raw LLM responses to JSON file."""
        if not self.raw_responses:
            logger.warning("No raw responses to save")
            return
        
        output_data = {
            "metadata": {
                "benchmark": benchmark_name,
                "model_name": self.model_name,
                "num_responses": len(self.raw_responses),
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "feature_names": self.feature_names,
                "saved_at": time.time(),
                "api_model": True,
                "model_type": "openai_llm"
            },
            "responses": self.raw_responses
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.raw_responses)} raw OpenAI responses to {output_path}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 save_raw_responses: bool = False, output_dir: Optional[str] = None,
                 benchmark_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate OpenAI LLM on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_raw_responses: Whether to save raw responses
            output_dir: Directory to save raw responses
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        
        # Save raw responses if requested
        if save_raw_responses and output_dir:
            raw_responses_filename = f"{benchmark_name}_{self.model_name.replace('/', '_')}_llm_responses.json"
            raw_responses_path = os.path.join(output_dir, raw_responses_filename)
            self.save_raw_responses(raw_responses_path, benchmark_name)
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(y_test),
            'predictions': predictions.tolist(),
            'true_labels': y_test.tolist(),
            'num_raw_responses': len(self.raw_responses) if save_raw_responses else 0,
            'model_name': self.model_name,
            'api_model': True,
            'model_type': 'openai_llm'
        }
        
        return results


def evaluate_openai_llm(dataset: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Evaluate OpenAI LLM baseline on tabular dataset.
    
    Args:
        dataset: Dataset dictionary with X, y, name, etc.
        args: Arguments object with configuration
        
    Returns:
        Evaluation results dictionary
    """
    start_time = time.time()
    
    try:
        # Extract data
        X, y = dataset["X"], dataset["y"]
        dataset_name = dataset["name"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        
        # Limit test samples if specified
        if hasattr(args, 'max_test_samples') and args.max_test_samples and args.max_test_samples < len(X_test):
            X_test = X_test[:args.max_test_samples]
            y_test = y_test[:args.max_test_samples]
        
        # Get feature names
        feature_names = dataset.get("attribute_names", [f"feature_{i}" for i in range(X.shape[1])])
        
        # Get class names
        unique_classes = np.unique(y)
        class_names = [f"class_{i}" for i in range(len(unique_classes))]
        
        # Get model name
        model_name = getattr(args, 'openai_model', 'gpt-4o')
        
        # Create and fit model
        model = OpenAILLMBaseline(
            num_classes=len(unique_classes),
            class_names=class_names,
            model_name=model_name,
            feature_names=feature_names
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        results = model.evaluate(
            X_test, y_test,
            save_raw_responses=getattr(args, 'save_raw_responses', False),
            output_dir=getattr(args, 'output_dir', None),
            benchmark_name=dataset_name
        )
        
        # Add dataset info
        results.update({
            'dataset_name': dataset_name,
            'dataset_id': dataset.get('id', 'unknown'),
            'training_time': time.time() - start_time,
            'num_features': X.shape[1],
            'num_classes': len(unique_classes),
            'completed_samples': len(X_test),
            'completion_rate': 1.0  # API models typically complete all samples
        })
        
        logger.info(f"OpenAI LLM achieved {results['accuracy']:.4f} accuracy on {dataset_name}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating OpenAI LLM on {dataset.get('name', 'unknown')}: {e}")
        return {
            'model_name': getattr(args, 'openai_model', 'gpt-4o'),
            'dataset_name': dataset.get('name', 'unknown'),
            'dataset_id': dataset.get('id', 'unknown'),
            'error': str(e),
            'timeout': False,
            'completed_samples': 0,
            'completion_rate': 0.0,
            'api_model': True,
            'model_type': 'openai_llm'
        }