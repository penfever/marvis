"""
OpenAI Vision-Language Model baseline for image classification.
Supports GPT-4.1 and GPT-4o vision models.
"""

from .api_vlm_baseline import APIVLMBaseline, BiologicalAPIVLMBaseline
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class OpenAIVLMBaseline(APIVLMBaseline):
    """
    OpenAI GPT Vision baseline for image classification.
    Specialized for OpenAI's GPT-4.1 and GPT-4o vision models.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gpt-4o", device: Optional[str] = None,
                 use_semantic_names: bool = False):
        
        # Validate OpenAI model name
        openai_models = ["gpt-4.1", "gpt-4o"]
        if not any(model in model_name for model in openai_models):
            logger.warning(f"Model {model_name} may not be an OpenAI vision model. Supported: {openai_models}")
            
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            model_name=model_name,
            device=device,
            use_semantic_names=use_semantic_names
        )


class BiologicalOpenAIVLMBaseline(BiologicalAPIVLMBaseline):
    """
    Specialized OpenAI VLM baseline for biological image classification.
    Uses domain-specific prompts for biological organisms with OpenAI models.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "gpt-4o", device: Optional[str] = None,
                 use_semantic_names: bool = False):
        
        # Validate OpenAI model name
        openai_models = ["gpt-4.1", "gpt-4o"]
        if not any(model in model_name for model in openai_models):
            logger.warning(f"Model {model_name} may not be an OpenAI vision model. Supported: {openai_models}")
            
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            model_name=model_name,
            device=device,
            use_semantic_names=use_semantic_names
        )