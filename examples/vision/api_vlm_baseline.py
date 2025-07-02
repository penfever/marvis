"""
Unified API Vision-Language Model baseline for image classification.
Supports OpenAI (GPT-4.1, GPT-4o) and Gemini (2.5 Pro, 2.5 Flash) models via API.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score
import time
import os
import sys
import json
from PIL import Image
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marvis.utils.vlm_prompting import create_direct_classification_prompt, parse_vlm_response, create_vlm_conversation
from marvis.utils.model_loader import model_loader, GenerationConfig

logger = logging.getLogger(__name__)


class APIVLMBaseline:
    """
    Unified API VLM baseline for image classification.
    Supports OpenAI and Gemini vision models through a unified interface.
    
    Can be used with either DataLoaders or image paths directly.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gpt-4o", device: Optional[str] = None,
                 use_semantic_names: bool = False):
        self.num_classes = num_classes
        self.class_names = class_names or []
        self.model_name = model_name
        self.model_wrapper = None
        self.device = device or "auto"  # Device is less relevant for API models
        self.use_semantic_names = use_semantic_names
        self.is_fitted = False
        self.raw_responses = []  # Store raw VLM responses for analysis
        
        # Validate model name
        supported_models = [
            # OpenAI models
            "gpt-4.1", "gpt-4o",
            # Gemini models  
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-2.0-flash", "gemini-2.0-pro-experimental"
        ]
        
        if not any(model in model_name for model in supported_models):
            logger.warning(f"Model {model_name} may not be supported. Supported models: {supported_models}")
        
    def load_model(self):
        """Load API VLM model through the unified model loader."""
        try:
            logger.info(f"Loading API VLM model: {self.model_name}")
            
            # Use the unified model loader which will auto-detect the backend
            self.model_wrapper = model_loader.load_vlm(
                model_name=self.model_name,
                device=self.device,
                backend="auto"  # Let it auto-detect OpenAI vs Gemini
            )
            
            logger.info("API VLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load API VLM model: {e}")
            # Fallback to mock implementation for testing
            logger.warning("Using mock API VLM implementation for testing")
            self.model_wrapper = "mock"
    
    def fit(self, train_data=None, train_labels=None, class_names: Optional[List[str]] = None) -> 'APIVLMBaseline':
        """
        Fit the VLM (no training needed for API models).
        
        Args:
            train_data: Either DataLoader or List[str] of image paths (unused, for API compatibility)
            train_labels: Training labels (unused, for API compatibility)
            class_names: Optional list of class names
        """
        if self.model_wrapper is None:
            self.load_model()
        if class_names:
            self.class_names = class_names
        self.is_fitted = True
        return self
    
    def predict(self, test_data) -> np.ndarray:
        """
        Predict using API VLM model.
        
        Args:
            test_data: Either DataLoader or List[str] of image paths
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(test_data, DataLoader):
            return self._predict_from_loader(test_data)
        else:
            return self._predict_from_paths(test_data)
    
    def _predict_from_loader(self, test_loader: DataLoader) -> np.ndarray:
        """Predict using API VLM model from DataLoader."""
        predictions = []
        
        for batch_images, _ in test_loader:
            for i in range(len(batch_images)):
                # Convert tensor to PIL Image
                image_tensor = batch_images[i]
                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = image_tensor * std + mean
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
                # Convert to PIL
                import torchvision.transforms as transforms
                image = transforms.ToPILImage()(image_tensor)
                
                # Predict
                if self.model_wrapper == "mock":
                    prediction = np.random.randint(0, self.num_classes)
                else:
                    prediction = self._predict_single_image(image, image_path=f"batch_{i}_sample_{i}", use_semantic_names=self.use_semantic_names)
                predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_from_paths(self, test_paths: List[str]) -> np.ndarray:
        """Predict using API VLM model from image paths."""
        predictions = []
        
        logger.info(f"Running API VLM predictions on {len(test_paths)} images")
        
        for i, image_path in enumerate(test_paths):
            if i % 100 == 0:
                logger.info(f"Processing image {i+1}/{len(test_paths)}")
            
            try:
                if self.model_wrapper == "mock":
                    # Mock prediction for testing
                    prediction = np.random.randint(0, self.num_classes)
                else:
                    image = Image.open(image_path).convert('RGB')
                    prediction = self._predict_single_image(image, image_path=image_path, use_semantic_names=self.use_semantic_names)
                
                predictions.append(prediction)
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing image {image_path}: {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Default to random prediction
                predictions.append(np.random.randint(0, self.num_classes))
        
        return np.array(predictions)
    
    def _predict_single_image(self, image: Image.Image, modality: str = "image", dataset_description: Optional[str] = None, 
                              image_path: Optional[str] = None, use_semantic_names: bool = False) -> int:
        """Predict single image using API VLM with unified prompting strategy."""
        if not self.class_names:
            raise ValueError("Class names must be provided for VLM prediction")
        
        # Create direct image classification prompt using centralized function
        prompt_text = create_direct_classification_prompt(
            class_names=self.class_names,
            dataset_description=dataset_description,
            use_semantic_names=use_semantic_names
        )
        
        # Create conversation using vlm_prompting utilities
        conversation = create_vlm_conversation(image, prompt_text)
        
        # Create generation config optimized for classification
        generation_config = GenerationConfig(
            max_new_tokens=16384,  # Generous limit for thinking and classification
            temperature=0.1,    # Low temperature for consistent results
            top_p=0.9,
            do_sample=True,
            enable_thinking=True,  # Enable thinking for Gemini models
            thinking_summary=False  # Don't need thought summaries for classification
        )
        
        # Generate response using the API wrapper
        try:
            response = self.model_wrapper.generate_from_conversation(conversation, generation_config)
        except Exception as e:
            import traceback
            error_msg = f"API VLM generation error for {image_path}: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            response = ""
        
        # Store raw response for analysis
        response_entry = {
            "image_path": image_path,
            "prompt": prompt_text,
            "raw_response": response,
            "model_name": self.model_name,
            "timestamp": time.time()
        }
        self.raw_responses.append(response_entry)
        
        # Log first 10 responses for debugging
        if len(self.raw_responses) <= 10:
            logger.info(f"API VLM Response #{len(self.raw_responses)} for {image_path}:")
            logger.info(f"  Prompt: {prompt_text[:200]}...")
            logger.info(f"  Raw Response: '{response}'")
            logger.info(f"  Model: {self.model_name}")
            logger.info("  " + "-"*50)
        
        # Parse response using vlm_prompting utilities
        predicted_class = parse_vlm_response(
            response, 
            unique_classes=self.class_names,
            logger_instance=logger,
            use_semantic_names=use_semantic_names
        )
        
        # Add parsed result to response entry
        response_entry["parsed_class"] = predicted_class
        
        # Convert to class index
        for i, class_name in enumerate(self.class_names):
            if predicted_class == class_name:
                response_entry["predicted_index"] = i
                return i
        
        # Default to first class if no match
        response_entry["predicted_index"] = 0
        return 0
    
    def save_raw_responses(self, output_path: str, benchmark_name: str = "unknown"):
        """Save raw VLM responses to JSON file."""
        if not self.raw_responses:
            logger.warning("No raw responses to save")
            return
        
        # Create output data with metadata
        output_data = {
            "metadata": {
                "benchmark": benchmark_name,
                "model_name": self.model_name,
                "num_responses": len(self.raw_responses),
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "saved_at": time.time(),
                "api_model": True
            },
            "responses": self.raw_responses
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.raw_responses)} raw API responses to {output_path}")
    
    def evaluate(self, test_data, test_labels: List[int], save_raw_responses: bool = False, 
                 output_dir: Optional[str] = None, benchmark_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate API VLM on test data.
        
        Args:
            test_data: Either DataLoader or List[str] of image paths
            test_labels: Ground truth labels
            save_raw_responses: Whether to save raw VLM responses
            output_dir: Directory to save raw responses (if save_raw_responses=True)
            benchmark_name: Name of the benchmark for the raw responses file
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        predictions = self.predict(test_data)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        # Save raw responses if requested
        if save_raw_responses and output_dir:
            raw_responses_filename = f"{benchmark_name}_{self.model_name.replace('/', '_')}_raw_responses.json"
            raw_responses_path = os.path.join(output_dir, raw_responses_filename)
            self.save_raw_responses(raw_responses_path, benchmark_name)
        
        return {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions,
            'true_labels': test_labels,
            'num_raw_responses': len(self.raw_responses) if save_raw_responses else 0,
            'model_name': self.model_name,
            'api_model': True
        }


class BiologicalAPIVLMBaseline(APIVLMBaseline):
    """
    Specialized API VLM baseline for biological image classification.
    Uses domain-specific prompts for biological organisms.
    """
    
    def _predict_single_image(self, image: Image.Image, dataset_description: Optional[str] = None,
                              image_path: Optional[str] = None) -> int:
        """Predict single image using API VLM with biological-specific prompt."""
        # Create biological-specific dataset description if not provided
        if dataset_description is None:
            dataset_description = "Biological image classification dataset with various organisms and specimens"
        
        # Call parent method with biological context
        return super()._predict_single_image(
            image, 
            modality="image",
            dataset_description=dataset_description,
            image_path=image_path,
            use_semantic_names=getattr(self, 'use_semantic_names', False)
        )