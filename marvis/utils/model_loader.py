#!/usr/bin/env python
"""
Centralized model loading system with support for both standard transformers and VLLM backends.

This module provides a unified interface for loading and using LLMs and VLMs,
automatically choosing the optimal backend (VLLM for speed when available, 
transformers as fallback) based on model type and availability.
"""

import os
import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# Standard imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
    from transformers.models.auto import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available")

# VLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.molmo import MolmoForCausalLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn("vllm not available, falling back to transformers")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop_tokens: Optional[List[str]] = None
    
    # API-specific parameters
    thinking_budget: Optional[int] = None  # For reasoning models like o3/o4-mini
    enable_thinking: bool = True  # For Gemini thinking capabilities
    thinking_summary: bool = False  # Whether to include thought summaries
    audio_output: bool = False  # For models with native audio output
    response_format: Optional[str] = None  # For structured outputs
    
    def to_vllm_sampling_params(self) -> 'SamplingParams':
        """Convert to VLLM SamplingParams."""
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM not available")
        
        return SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop_tokens
        )
    
    def to_transformers_kwargs(self) -> Dict[str, Any]:
        """Convert to transformers generation kwargs."""
        kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'repetition_penalty': self.repetition_penalty
        }
        
        # Only add sampling parameters if do_sample is True
        if self.do_sample:
            kwargs.update({
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k
            })
        
        return kwargs
    
    def to_openai_kwargs(self) -> Dict[str, Any]:
        """Convert to OpenAI API kwargs."""
        kwargs = {
            'max_tokens': self.max_new_tokens,
            'temperature': self.temperature if self.do_sample else 0,
            'top_p': self.top_p if self.do_sample else 1,
            'stop': self.stop_tokens
        }
        
        # Add response format if specified
        if self.response_format:
            kwargs['response_format'] = {"type": self.response_format}
        
        # Filter out None values
        return {k: v for k, v in kwargs.items() if v is not None}
    
    def to_gemini_kwargs(self) -> Dict[str, Any]:
        """Convert to Gemini API kwargs."""
        generation_config = {
            'max_output_tokens': self.max_new_tokens,
            'temperature': self.temperature if self.do_sample else 0,
            'top_p': self.top_p if self.do_sample else 1,
            'top_k': self.top_k if self.do_sample else None,
            'stop_sequences': self.stop_tokens
        }
        
        # Gemini-specific features
        kwargs = {
            'generation_config': {k: v for k, v in generation_config.items() if v is not None}
        }
        
        # Add thinking controls for Gemini 2.5 models
        if hasattr(self, 'enable_thinking') and not self.enable_thinking:
            kwargs['thinking'] = False
            
        if hasattr(self, 'thinking_summary') and self.thinking_summary:
            kwargs['include_thinking_summary'] = True
        
        return kwargs


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Generate text from inputs."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have a direct cache clearing method like CUDA
            # But we can suggest garbage collection
            import gc
            gc.collect()
    
    def get_model(self):
        """Get the underlying model for compatibility with legacy code."""
        return self._model
    
    def get_tokenizer(self):
        """Get the underlying tokenizer for compatibility with legacy code."""
        return getattr(self, '_tokenizer', None)


class VLLMModelWrapper(BaseModelWrapper):
    """VLLM-based model wrapper for fast LLM inference."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.max_model_len = kwargs.get('max_model_len', None)
        
    def load(self) -> None:
        """Load model using VLLM."""
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM not available")
            
        logger.info(f"Loading {self.model_name} with VLLM backend")
        
        # Configure VLLM parameters
        vllm_kwargs = {
            'model': self.model_name,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'trust_remote_code': True,
            'dtype': 'auto'
        }
        
        if self.max_model_len:
            vllm_kwargs['max_model_len'] = self.max_model_len
            
        # Filter out wrapper-specific kwargs that shouldn't be passed to VLLM
        wrapper_specific_kwargs = {
            'backend', 'tensor_parallel_size', 'gpu_memory_utilization', 
            'max_model_len', 'torch_dtype', 'low_cpu_mem_usage', 'device_map', 'use_cache'
        }
        
        # Add any additional kwargs (excluding wrapper-specific ones)
        vllm_kwargs.update({k: v for k, v in self.kwargs.items() 
                           if k not in wrapper_specific_kwargs})
        
        try:
            self._model = LLM(**vllm_kwargs)
            logger.info(f"Successfully loaded {self.model_name} with VLLM")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} with VLLM: {e}")
            raise
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Generate text using VLLM."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        # Convert config to VLLM sampling params
        sampling_params = config.to_vllm_sampling_params()
        
        # Generate
        outputs = self._model.generate(inputs, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            # Get the generated text (excluding the prompt)
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


class VLLMVisionModelWrapper(BaseModelWrapper):
    """VLLM-based wrapper for Vision Language Models (multimodal)."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.max_model_len = kwargs.get('max_model_len', None)
        self._tokenizer = None
        
    def load(self) -> None:
        """Load VLM using VLLM."""
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM not available")
            
        logger.info(f"Loading {self.model_name} with VLLM VLM backend")
        
        # Load tokenizer for chat template support
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {self.model_name}: {e}")
            self._tokenizer = None
        
        # Configure VLLM parameters for multimodal models
        vllm_kwargs = {
            'model': self.model_name,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'trust_remote_code': True,
            'dtype': 'auto',
            'max_num_seqs': 1,  # VLMs often work better with single sequence
        }
        
        if self.max_model_len:
            vllm_kwargs['max_model_len'] = self.max_model_len
            
        # Filter out wrapper-specific kwargs that shouldn't be passed to VLLM
        wrapper_specific_kwargs = {
            'backend', 'tensor_parallel_size', 'gpu_memory_utilization', 
            'max_model_len', 'torch_dtype', 'low_cpu_mem_usage', 'device_map', 'use_cache'
        }
        
        # Add any additional kwargs (excluding wrapper-specific ones)
        vllm_kwargs.update({k: v for k, v in self.kwargs.items() 
                           if k not in wrapper_specific_kwargs})
        
        try:
            self._model = LLM(**vllm_kwargs)
            logger.info(f"Successfully loaded {self.model_name} with VLLM VLM backend")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} with VLLM VLM: {e}")
            raise
    
    def generate_from_conversation(self, conversation: List[Dict], config: GenerationConfig) -> str:
        """Generate text from a conversation format with image+text input."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Extract image and text from conversation
        text_content = ""
        image = None
        
        for message in conversation:
            if isinstance(message.get('content'), list):
                for content_item in message['content']:
                    if content_item.get('type') == 'text':
                        text_content += content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image = content_item.get('image')
        
        # For VLLM multimodal, we need to create a proper prompt with image placeholders
        if image is not None:
            # Try to use tokenizer's chat template if available
            if self._tokenizer and hasattr(self._tokenizer, 'apply_chat_template'):
                try:
                    # Apply chat template
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    logger.debug(f"Applied chat template, prompt: {formatted_prompt[:200]}...")
                except Exception as e:
                    logger.warning(f"Failed to apply chat template: {e}. Using fallback formatting.")
                    # Fallback to manual formatting
                    formatted_prompt = self._format_prompt_with_placeholder(text_content)
            else:
                # Manual formatting with image placeholders
                formatted_prompt = self._format_prompt_with_placeholder(text_content)
            
            # Create the prompt dictionary for VLLM
            prompt = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            }
        else:
            # Text-only prompt
            prompt = text_content
        
        # Convert config to VLLM sampling params
        sampling_params = config.to_vllm_sampling_params()
        
        # Generate
        outputs = self._model.generate(prompt, sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
    
    def _format_prompt_with_placeholder(self, text_content: str) -> str:
        """Format prompt with appropriate image placeholder for the model."""
        # Determine the correct image placeholder based on model type
        image_placeholder = "<image>"  # Default placeholder
        
        # Model-specific placeholders
        if "Qwen" in self.model_name:
            # Qwen models use <|image_pad|> or <|vision_start|><|image_pad|><|vision_end|>
            image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "llava" in self.model_name.lower():
            image_placeholder = "<image>"
        elif "Phi-3.5-vision" in self.model_name:
            image_placeholder = "<|image_1|>"
        elif "pixtral" in self.model_name.lower():
            image_placeholder = "[IMG]"
        elif "molmo" in self.model_name.lower():
            image_placeholder = "<image>"
        elif "Llama-3.2" in self.model_name and "Vision" in self.model_name:
            image_placeholder = "<|image|>"
        
        # Insert image placeholder at the beginning of the prompt
        formatted_prompt = f"{image_placeholder}\n{text_content}"
        
        return formatted_prompt
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Standard generate method for text-only inputs."""
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        # Convert to VLLM format and generate
        sampling_params = config.to_vllm_sampling_params()
        outputs = self._model.generate(inputs, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


class TransformersModelWrapper(BaseModelWrapper):
    """Transformers-based model wrapper."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        self.low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        self.device_map = kwargs.get('device_map', 'auto' if device == 'auto' else None)
        
    def load(self) -> None:
        """Load model using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
            
        logger.info(f"Loading {self.model_name} with transformers backend")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add pad token if missing
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Configure model loading parameters
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': self.low_cpu_mem_usage
        }
        
        # Device and dtype configuration
        # Import device utilities for proper device detection
        from marvis.utils.device_utils import detect_optimal_device
        
        # Resolve device if set to auto
        actual_device = self.device
        if self.device == "auto":
            actual_device = detect_optimal_device(prefer_mps=True)
            logger.info(f"Auto-detected device: {actual_device}")
        
        # Configure based on device
        if actual_device == "cuda" and torch.cuda.is_available():
            model_kwargs.update({
                'torch_dtype': self.torch_dtype,
                'device_map': self.device_map
            })
        elif actual_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS configuration - don't use device_map, load to CPU then move to MPS
            model_kwargs.update({
                'torch_dtype': torch.float32,  # MPS works better with float32
                'device_map': None,  # Don't use device_map with MPS
                'low_cpu_mem_usage': True  # Use CPU-efficient loading
            })
            logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration")
        elif actual_device == "cpu":
            model_kwargs.update({
                'torch_dtype': torch.float32,  # Use float32 for CPU
                'device_map': 'cpu'
            })
        
        # Filter out wrapper-specific kwargs that shouldn't be passed to the model
        wrapper_specific_kwargs = {
            'backend', 'tensor_parallel_size', 'gpu_memory_utilization', 
            'torch_dtype', 'low_cpu_mem_usage', 'device_map'
        }
        
        # Add any additional kwargs (excluding wrapper-specific ones)
        model_kwargs.update({k: v for k, v in self.kwargs.items() 
                           if k not in wrapper_specific_kwargs})
        
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Manually move to MPS if needed (since we loaded to CPU first)
            if actual_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info(f"Moving model to MPS device...")
                self._model = self._model.to(torch.device('mps'))
                
            logger.info(f"Successfully loaded {self.model_name} with transformers")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} with transformers: {e}")
            raise
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Generate text using transformers."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        # Tokenize inputs
        tokenized = self._tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device  
        if hasattr(self._model, 'device'):
            device = self._model.device
        else:
            # Try to infer device from model parameters
            try:
                device = next(self._model.parameters()).device
            except:
                device = torch.device('cpu')
        
        # Move inputs to the appropriate device
        if device.type != 'cpu':
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Generate
        gen_kwargs = config.to_transformers_kwargs()
        gen_kwargs['pad_token_id'] = self._tokenizer.eos_token_id
        
        # Some models (like Qwen2.5) don't support certain generation flags
        # Use a more conservative approach to avoid warnings
        safe_kwargs = {
            'max_new_tokens': gen_kwargs.get('max_new_tokens', 100),
            'do_sample': gen_kwargs.get('do_sample', True),
            'pad_token_id': gen_kwargs['pad_token_id']
        }
        
        # Only add sampling parameters if do_sample is True and model supports them
        if gen_kwargs.get('do_sample', False):
            # Use a conservative temperature to avoid warnings
            safe_kwargs['temperature'] = max(0.1, gen_kwargs.get('temperature', 0.1))
            # Skip top_p and top_k for models that don't support them reliably
            if 'qwen' not in self.model_name.lower():
                safe_kwargs['top_p'] = gen_kwargs.get('top_p', 0.9)
                safe_kwargs['top_k'] = gen_kwargs.get('top_k', 50)
        
        # Add repetition penalty if present
        if 'repetition_penalty' in gen_kwargs:
            safe_kwargs['repetition_penalty'] = gen_kwargs['repetition_penalty']
        
        with torch.no_grad():
            outputs = self._model.generate(
                **tokenized,
                **safe_kwargs
            )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            # Remove input tokens to get only generated text
            input_length = tokenized['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._tokenizer is not None


class VisionLanguageModelWrapper(BaseModelWrapper):
    """Wrapper for Vision Language Models using transformers."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        self.low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        self.device_map = kwargs.get('device_map', 'auto' if device == 'auto' else None)
        self._processor = None
        
    def load(self) -> None:
        """Load VLM using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
            
        logger.info(f"Loading {self.model_name} with transformers VLM backend")
        
        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Configure model loading parameters
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': self.low_cpu_mem_usage
        }
        
        # Device and dtype configuration
        # Import device utilities for proper device detection
        from marvis.utils.device_utils import detect_optimal_device
        
        # Resolve device if set to auto
        actual_device = self.device
        if self.device == "auto":
            actual_device = detect_optimal_device(prefer_mps=True)
            logger.info(f"Auto-detected device: {actual_device}")
        
        # Configure based on device
        if actual_device == "cuda" and torch.cuda.is_available():
            model_kwargs.update({
                'torch_dtype': self.torch_dtype,
                'device_map': self.device_map
            })
        elif actual_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS configuration - don't use device_map, load to CPU then move to MPS
            model_kwargs.update({
                'torch_dtype': torch.float32,
                'device_map': None,  # Don't use device_map with MPS
                'low_cpu_mem_usage': True  # Use CPU-efficient loading
            })
            logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration")
        elif actual_device == "cpu":
            model_kwargs.update({
                'torch_dtype': torch.float32,
                'device_map': 'cpu'
            })
        
        # Filter out wrapper-specific kwargs that shouldn't be passed to the model
        wrapper_specific_kwargs = {
            'backend', 'tensor_parallel_size', 'gpu_memory_utilization', 
            'max_model_len', 'torch_dtype', 'low_cpu_mem_usage', 'device_map'
        }
        
        # Add any additional kwargs (excluding wrapper-specific ones)
        model_kwargs.update({k: v for k, v in self.kwargs.items() 
                           if k not in wrapper_specific_kwargs})
        
        try:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Manually move to MPS if needed (since we loaded to CPU first)
            if actual_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info(f"Moving model to MPS device...")
                self._model = self._model.to(torch.device('mps'))
                
            logger.info(f"Successfully loaded {self.model_name} with transformers VLM")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} with transformers VLM: {e}")
            raise
    
    def generate_from_conversation(self, conversation: List[Dict], config: GenerationConfig) -> str:
        """Generate text from a conversation format (for VLMs with image+text input)."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Process the conversation
        formatted_text = self._processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Extract image from conversation
        image = None
        for message in conversation:
            if isinstance(message.get('content'), list):
                for content_item in message['content']:
                    if content_item.get('type') == 'image':
                        image = content_item.get('image')
                        break
                if image:
                    break
        
        # Process inputs
        inputs = self._processor(text=formatted_text, images=image, return_tensors="pt")
        
        # Move to device
        if hasattr(self._model, 'device'):
            device = self._model.device
        else:
            # Try to infer device from model parameters
            try:
                device = next(self._model.parameters()).device
            except:
                device = torch.device('cpu')
        
        # Move inputs to the appropriate device
        if device.type != 'cpu':
            inputs = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
        
        # Generate
        gen_kwargs = config.to_transformers_kwargs()
        gen_kwargs['pad_token_id'] = self._processor.tokenizer.eos_token_id
        
        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode response
        response = self._processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Standard generate method for text-only inputs."""
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        for text_input in inputs:
            # Create a simple conversation for text-only input
            conversation = [{"role": "user", "content": [{"type": "text", "text": text_input}]}]
            result = self.generate_from_conversation(conversation, config)
            results.append(result)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._processor is not None
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        super().unload()
        if hasattr(self, '_processor') and self._processor is not None:
            del self._processor
            self._processor = None


class OpenAIModelWrapper(BaseModelWrapper):
    """OpenAI API-based model wrapper for LLM tasks."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._client = None
        # Handle max_model_len for API models - store for use in generation config
        self.max_model_len = kwargs.get('max_model_len', None)
        
    def load(self) -> None:
        """Load OpenAI API client."""
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._client = openai.OpenAI(api_key=api_key)
            logger.info(f"Successfully connected to OpenAI API for model: {self.model_name}")
            
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install 'marvis[api]'")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Generate text using OpenAI API."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        # Override max_new_tokens with max_model_len if provided
        if self.max_model_len is not None:
            config = GenerationConfig(
                max_new_tokens=self.max_model_len,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                stop_tokens=config.stop_tokens
            )
        generation_kwargs = config.to_openai_kwargs()
        
        for text_input in inputs:
            # Retry delays: 15s, 15s, 30s, 30s, 60s
            retry_delays = [15, 15, 30, 30, 60]
            max_retries = 5
            
            for attempt in range(max_retries + 1):
                try:
                    # Use chat completions for all models
                    messages = [{"role": "user", "content": text_input}]
                    
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **generation_kwargs
                    )
                    
                    generated_text = response.choices[0].message.content
                    results.append(generated_text)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    is_rate_limit = (
                        hasattr(e, 'status_code') and e.status_code == 429
                    ) or '429' in str(e) or 'rate limit' in str(e).lower()
                    
                    if is_rate_limit and attempt < max_retries:
                        delay = retry_delays[attempt]
                        logger.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        # Either not a rate limit error, or we've exhausted retries
                        logger.error(f"OpenAI API error: {e}")
                        raise RuntimeError(f"OpenAI API call failed: {e}") from e
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self._client is not None
    
    def unload(self) -> None:
        """Cleanup API client."""
        self._client = None


class OpenAIVisionModelWrapper(BaseModelWrapper):
    """OpenAI API-based wrapper for Vision Language Models."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._client = None
        # Handle max_model_len for API models - store for use in generation config
        self.max_model_len = kwargs.get('max_model_len', None)
        
    def load(self) -> None:
        """Load OpenAI API client."""
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._client = openai.OpenAI(api_key=api_key)
            logger.info(f"Successfully connected to OpenAI API for VLM: {self.model_name}")
            
        except ImportError:
            raise ImportError("OpenAI package not available. Install with: pip install 'marvis[api]'")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_from_conversation(self, conversation: List[Dict], config: GenerationConfig) -> str:
        """Generate text from a conversation format with image+text input."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert conversation format to OpenAI format
            messages = []
            for message in conversation:
                if isinstance(message.get('content'), list):
                    # Handle multimodal content
                    openai_content = []
                    for content_item in message['content']:
                        if content_item.get('type') == 'text':
                            openai_content.append({
                                "type": "text",
                                "text": content_item.get('text', '')
                            })
                        elif content_item.get('type') == 'image':
                            # Handle PIL Image objects
                            image = content_item.get('image')
                            if hasattr(image, 'save'):  # PIL Image
                                import base64
                                import io
                                
                                # Convert PIL Image to base64
                                buffer = io.BytesIO()
                                image.save(buffer, format='PNG')
                                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
                                openai_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    }
                                })
                    
                    messages.append({
                        "role": message.get('role', 'user'),
                        "content": openai_content
                    })
                else:
                    # Text-only content
                    messages.append({
                        "role": message.get('role', 'user'),
                        "content": message.get('content', '')
                    })
            
            # Override max_new_tokens with max_model_len if provided
            if self.max_model_len is not None:
                config = GenerationConfig(
                    max_new_tokens=self.max_model_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    stop_tokens=config.stop_tokens
                )
            generation_kwargs = config.to_openai_kwargs()
            
            # Retry logic for rate limits
            retry_delays = [15, 15, 30, 30, 60]
            max_retries = 5
            
            for attempt in range(max_retries + 1):
                try:
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **generation_kwargs
                    )
                    
                    return response.choices[0].message.content
                    
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    is_rate_limit = (
                        hasattr(e, 'status_code') and e.status_code == 429
                    ) or '429' in str(e) or 'rate limit' in str(e).lower()
                    
                    if is_rate_limit and attempt < max_retries:
                        delay = retry_delays[attempt]
                        logger.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        # Either not a rate limit error, or we've exhausted retries
                        logger.error(f"OpenAI Vision API error: {e}")
                        raise RuntimeError(f"OpenAI Vision API call failed: {e}") from e
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            # Re-raise the exception instead of returning empty string
            raise RuntimeError(f"OpenAI Vision API call failed: {e}") from e
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Standard generate method for text-only inputs."""
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        for text_input in inputs:
            # Create a simple conversation for text-only input
            conversation = [{"role": "user", "content": text_input}]
            result = self.generate_from_conversation(conversation, config)
            results.append(result)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self._client is not None
    
    def unload(self) -> None:
        """Cleanup API client."""
        self._client = None


class GeminiModelWrapper(BaseModelWrapper):
    """Gemini API-based model wrapper for LLM tasks."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._client = None
        self._model = None
        # Handle max_model_len for API models - store for use in generation config
        self.max_model_len = kwargs.get('max_model_len', None)
        
    def load(self) -> None:
        """Load Gemini API client."""
        try:
            import google.generativeai as genai
            import os
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Successfully connected to Gemini API for model: {self.model_name}")
            
        except ImportError:
            raise ImportError("Google GenerativeAI package not available. Install with: pip install 'marvis[api]'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Generate text using Gemini API."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        # Override max_new_tokens with max_model_len if provided
        if self.max_model_len is not None:
            config = GenerationConfig(
                max_new_tokens=self.max_model_len,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                stop_tokens=config.stop_tokens
            )
        generation_kwargs = config.to_gemini_kwargs()
        
        for text_input in inputs:
            try:
                response = self._model.generate_content(
                    text_input,
                    **generation_kwargs
                )
                
                generated_text = response.text if response.text else ""
                results.append(generated_text)
                
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Fallback to empty string
                results.append("")
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self._model is not None
    
    def unload(self) -> None:
        """Cleanup API client."""
        self._model = None


class GeminiVisionModelWrapper(BaseModelWrapper):
    """Gemini API-based wrapper for Vision Language Models."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._model = None
        # Handle max_model_len for API models - store for use in generation config
        self.max_model_len = kwargs.get('max_model_len', None)
        
    def load(self) -> None:
        """Load Gemini API client."""
        try:
            import google.generativeai as genai
            import os
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Successfully connected to Gemini API for VLM: {self.model_name}")
            
        except ImportError:
            raise ImportError("Google GenerativeAI package not available. Install with: pip install 'marvis[api]'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate_from_conversation(self, conversation: List[Dict], config: GenerationConfig) -> str:
        """Generate text from a conversation format with image+text input."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Extract content from conversation
            content_parts = []
            for message in conversation:
                if isinstance(message.get('content'), list):
                    for content_item in message['content']:
                        if content_item.get('type') == 'text':
                            content_parts.append(content_item.get('text', ''))
                        elif content_item.get('type') == 'image':
                            # Gemini can handle PIL Images directly
                            image = content_item.get('image')
                            if image is not None:
                                # Debug: log image info
                                if hasattr(image, 'size') and hasattr(image, 'mode'):
                                    logger.debug(f"Adding image to Gemini request: size={image.size}, mode={image.mode}")
                                else:
                                    logger.warning(f"Image object doesn't have expected PIL attributes: {type(image)}")
                                content_parts.append(image)
                            else:
                                logger.error("Image is None in conversation content")
                else:
                    # Text-only content
                    content_parts.append(message.get('content', ''))
            
            # Override max_new_tokens with max_model_len if provided
            if self.max_model_len is not None:
                config = GenerationConfig(
                    max_new_tokens=self.max_model_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    stop_tokens=config.stop_tokens
                )
            generation_kwargs = config.to_gemini_kwargs()
            
            response = self._model.generate_content(
                content_parts,
                **generation_kwargs
            )
            
            # Handle response more robustly
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == 2:  # MAX_TOKENS
                        logger.warning(f"Gemini response truncated due to token limit")
                    elif candidate.finish_reason == 3:  # SAFETY
                        logger.warning(f"Gemini response blocked for safety reasons")
                
                # Try to extract text from parts
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            return ' '.join(text_parts)
            
            # Fallback to response.text if available
            try:
                return response.text if response.text else ""
            except:
                logger.error("Failed to extract text from Gemini response")
                return ""
            
        except Exception as e:
            logger.error(f"Gemini Vision API error: {e}")
            return ""
    
    def generate(self, inputs: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        """Standard generate method for text-only inputs."""
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        for text_input in inputs:
            # Create a simple conversation for text-only input
            conversation = [{"role": "user", "content": text_input}]
            result = self.generate_from_conversation(conversation, config)
            results.append(result)
        
        return results[0] if single_input else results
    
    def is_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self._model is not None
    
    def unload(self) -> None:
        """Cleanup API client."""
        self._model = None


class ModelLoader:
    """Central model loading system."""
    
    def __init__(self):
        self._loaded_models: Dict[str, BaseModelWrapper] = {}
        
    def load_llm(
        self, 
        model_name: str, 
        backend: str = "auto",
        device: str = "auto",
        **kwargs
    ) -> BaseModelWrapper:
        """
        Load a Large Language Model.
        
        Args:
            model_name: HuggingFace model name
            backend: 'vllm', 'transformers', or 'auto'
            device: Device to load model on
            **kwargs: Additional model loading arguments
            
        Returns:
            Loaded model wrapper
        """
        cache_key = f"{model_name}_{backend}_{device}"
        
        # Return cached model if available
        if cache_key in self._loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self._loaded_models[cache_key]
        
        # Determine backend
        if backend == "auto":
            # Check for API models first
            openai_llm_models = [
                "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                "gpt-4o", "gpt-3.5-turbo", "o3", "o4-mini"
            ]
            gemini_models = [
                "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                "gemini-2.0-flash", "gemini-2.0-pro-experimental"
            ]
            
            if any(model in model_name for model in openai_llm_models):
                backend = "openai"
                logger.info(f"Auto-selected OpenAI backend for LLM: {model_name}")
            elif any(model in model_name for model in gemini_models):
                backend = "gemini"
                logger.info(f"Auto-selected Gemini backend for LLM: {model_name}")
            else:
                # Prefer VLLM for local LLMs if available
                backend = "vllm" if VLLM_AVAILABLE else "transformers"
                logger.info(f"Auto-selected {backend} backend for local LLM: {model_name}")
        
        # Create wrapper
        if backend == "openai":
            wrapper = OpenAIModelWrapper(model_name, device, **kwargs)
        elif backend == "gemini":
            wrapper = GeminiModelWrapper(model_name, device, **kwargs)
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                logger.warning("VLLM not available, falling back to transformers")
                backend = "transformers"
            else:
                wrapper = VLLMModelWrapper(model_name, device, **kwargs)
        
        if backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available")
            wrapper = TransformersModelWrapper(model_name, device, **kwargs)
        elif backend not in ["openai", "gemini", "vllm"]:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Load the model
        wrapper.load()
        
        # Cache and return
        self._loaded_models[cache_key] = wrapper
        logger.info(f"Loaded and cached model: {cache_key}")
        return wrapper
    
    def load_vlm(
        self, 
        model_name: str, 
        device: str = "auto",
        backend: str = "auto",
        **kwargs
    ) -> Union[VisionLanguageModelWrapper, VLLMVisionModelWrapper, OpenAIVisionModelWrapper, GeminiVisionModelWrapper]:
        """
        Load a Vision Language Model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            backend: Backend to use ("auto", "vllm", "transformers")
            **kwargs: Additional model loading arguments
            
        Returns:
            Loaded VLM wrapper
        """
        # Determine backend
        if backend == "auto":
            # Check for API VLM models first
            openai_vlm_models = ["gpt-4.1", "gpt-4o"]  # Both support vision
            gemini_vlm_models = [
                "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                "gemini-2.0-flash", "gemini-2.0-pro-experimental"
            ]
            
            if any(model in model_name for model in openai_vlm_models):
                backend = "openai"
                logger.info(f"Auto-selected OpenAI backend for VLM: {model_name}")
            elif any(model in model_name for model in gemini_vlm_models):
                backend = "gemini"
                logger.info(f"Auto-selected Gemini backend for VLM: {model_name}")
            else:
                # Check if model supports VLLM multimodal
                vlm_supported_models = [
                    "Qwen/Qwen2.5-VL",
                    "Qwen/Qwen2-VL",
                    "llava-hf/llava",
                    "TIGER-Lab/Mantis",
                    "microsoft/Phi-3.5-vision",
                    "mistral-community/pixtral",
                    "allenai/Molmo",
                    "meta-llama/Llama-3.2-11B-Vision"
                ]
                
                # Check if model name matches any supported pattern
                supports_vllm = any(pattern in model_name for pattern in vlm_supported_models)
                
                if supports_vllm and VLLM_AVAILABLE:
                    backend = "vllm"
                    logger.info(f"Auto-selected VLLM backend for VLM: {model_name}")
                else:
                    backend = "transformers"
                    logger.info(f"Auto-selected transformers backend for VLM: {model_name}")
        
        cache_key = f"{model_name}_vlm_{backend}_{device}"
        
        # Return cached model if available
        if cache_key in self._loaded_models:
            logger.info(f"Using cached VLM: {cache_key}")
            return self._loaded_models[cache_key]
        
        # Create appropriate wrapper based on backend
        if backend == "openai":
            wrapper = OpenAIVisionModelWrapper(model_name, device, **kwargs)
        elif backend == "gemini":
            wrapper = GeminiVisionModelWrapper(model_name, device, **kwargs)
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                logger.warning("VLLM not available, falling back to transformers")
                backend = "transformers"
            else:
                wrapper = VLLMVisionModelWrapper(model_name, device, **kwargs)
        
        if backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available")
            wrapper = VisionLanguageModelWrapper(model_name, device, **kwargs)
        elif backend not in ["openai", "gemini", "vllm"]:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Load the model
        wrapper.load()
        
        # Cache and return
        self._loaded_models[cache_key] = wrapper
        logger.info(f"Loaded and cached VLM: {cache_key}")
        return wrapper
    
    def unload_model(self, model_name: str, backend: str = None, device: str = None) -> None:
        """Unload a specific model."""
        # If specific cache key provided
        if backend and device:
            cache_key = f"{model_name}_{backend}_{device}"
            if cache_key in self._loaded_models:
                self._loaded_models[cache_key].unload()
                del self._loaded_models[cache_key]
                logger.info(f"Unloaded model: {cache_key}")
        else:
            # Unload all variants of this model
            keys_to_remove = [k for k in self._loaded_models.keys() if k.startswith(model_name)]
            for key in keys_to_remove:
                self._loaded_models[key].unload()
                del self._loaded_models[key]
                logger.info(f"Unloaded model: {key}")
    
    def unload_all(self) -> None:
        """Unload all models."""
        for wrapper in self._loaded_models.values():
            wrapper.unload()
        self._loaded_models.clear()
        logger.info("Unloaded all models")
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        return list(self._loaded_models.keys())


# Global model loader instance
model_loader = ModelLoader()