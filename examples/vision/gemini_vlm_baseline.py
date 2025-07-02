"""
Gemini Vision-Language Model baseline for image classification.
Supports Gemini 2.5 Pro, 2.5 Flash, and 2.0 models with thinking capabilities.
"""

from .api_vlm_baseline import APIVLMBaseline, BiologicalAPIVLMBaseline
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class GeminiVLMBaseline(APIVLMBaseline):
    """
    Gemini Vision baseline for image classification.
    Specialized for Gemini 2.5 and 2.0 models with thinking capabilities.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gemini-2.5-flash", device: Optional[str] = None,
                 use_semantic_names: bool = False, enable_thinking: bool = True):
        
        # Validate Gemini model name
        gemini_models = [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-2.0-flash", "gemini-2.0-pro-experimental"
        ]
        if not any(model in model_name for model in gemini_models):
            logger.warning(f"Model {model_name} may not be a Gemini model. Supported: {gemini_models}")
            
        self.enable_thinking = enable_thinking
        
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            model_name=model_name,
            device=device,
            use_semantic_names=use_semantic_names
        )
    
    def _predict_single_image(self, image, modality="image", dataset_description=None, 
                              image_path=None, use_semantic_names=False) -> int:
        """Override to use Gemini-specific generation config with thinking."""
        if not self.class_names:
            raise ValueError("Class names must be provided for VLM prediction")
        
        from marvis.utils.vlm_prompting import create_direct_classification_prompt, parse_vlm_response, create_vlm_conversation
        from marvis.utils.model_loader import GenerationConfig
        import time
        
        # Create direct image classification prompt using centralized function
        prompt_text = create_direct_classification_prompt(
            class_names=self.class_names,
            dataset_description=dataset_description,
            use_semantic_names=use_semantic_names
        )
        
        # Create conversation using vlm_prompting utilities
        conversation = create_vlm_conversation(image, prompt_text)
        
        # Create Gemini-optimized generation config
        generation_config = GenerationConfig(
            max_new_tokens=16384,  # Generous limit for thinking and classification
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            enable_thinking=self.enable_thinking,  # Use thinking capabilities
            thinking_summary=False,  # Don't need summaries for classification
        )
        
        # Generate response using the API wrapper
        try:
            response = self.model_wrapper.generate_from_conversation(conversation, generation_config)
        except Exception as e:
            import traceback
            error_msg = f"Gemini VLM generation error for {image_path}: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            response = ""
        
        # Store raw response for analysis
        response_entry = {
            "image_path": image_path,
            "prompt": prompt_text,
            "raw_response": response,
            "model_name": self.model_name,
            "thinking_enabled": self.enable_thinking,
            "timestamp": time.time()
        }
        self.raw_responses.append(response_entry)
        
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


class BiologicalGeminiVLMBaseline(BiologicalAPIVLMBaseline):
    """
    Specialized Gemini VLM baseline for biological image classification.
    Uses domain-specific prompts for biological organisms with Gemini thinking models.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gemini-2.5-flash", device: Optional[str] = None,
                 use_semantic_names: bool = False, enable_thinking: bool = True):
        
        # Validate Gemini model name
        gemini_models = [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-2.0-flash", "gemini-2.0-pro-experimental"
        ]
        if not any(model in model_name for model in gemini_models):
            logger.warning(f"Model {model_name} may not be a Gemini model. Supported: {gemini_models}")
            
        self.enable_thinking = enable_thinking
        
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            model_name=model_name,
            device=device,
            use_semantic_names=use_semantic_names
        )
    
    def _predict_single_image(self, image, dataset_description=None, image_path=None) -> int:
        """Predict single image using Gemini VLM with biological-specific prompt and thinking."""
        # Create biological-specific dataset description if not provided
        if dataset_description is None:
            dataset_description = "Biological image classification dataset with various organisms and specimens"
        
        # Use the parent's Gemini-optimized prediction method
        return GeminiVLMBaseline._predict_single_image(
            self,
            image=image, 
            modality="image",
            dataset_description=dataset_description,
            image_path=image_path,
            use_semantic_names=getattr(self, 'use_semantic_names', False)
        )