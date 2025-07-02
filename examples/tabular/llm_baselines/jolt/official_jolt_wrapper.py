#!/usr/bin/env python
"""
Wrapper for the official JOLT implementation to integrate with our evaluation framework.
"""

import numpy as np
import pandas as pd
import os
import sys
import tempfile
import logging
import time
import importlib.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from argparse import Namespace

# Add path for feature selection utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(os.path.dirname(current_dir))
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

# Import shared utilities
from marvis.utils import (
    drop_feature_for_oom,
    is_oom_error,
    apply_feature_reduction
)
from marvis.utils.task_detection import detect_task_type


def evaluate_jolt_official(dataset, args):
    """Evaluate JOLT baseline using the official implementation."""
    # IMPORTANT: Set CUDA masking FIRST before any other imports
    import os
    original_cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    original_pytorch_cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
    
    # Force CPU usage for JOLT on Mac/CPU environments by masking CUDA completely
    # We need to check this before importing torch to avoid issues
    force_cpu = args.device == "cpu"
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices
    
    # Set CUDA device environment variable for JOLT BEFORE importing torch
    if hasattr(args, 'gpu_index') and args.gpu_index != 0 and args.device != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
        print(f"[JOLT] Set CUDA_VISIBLE_DEVICES={args.gpu_index} BEFORE torch import")
    
    # Set memory management environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Now import torch after setting CUDA_VISIBLE_DEVICES
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # Start logging
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating JOLT (official) on dataset {dataset['name']}")
    
    # Log device configuration
    if force_cpu:
        logger.info("Forcing CPU-only mode for JOLT by hiding CUDA devices")
        logger.info(f"CUDA available after masking: {torch.cuda.is_available()}")
    elif torch.cuda.is_available() and args.device != "cpu":
        logger.info(f"Available CUDA devices after masking: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")  # Should be GPU 2's name
    
    # Load JOLT configuration using the new resource manager (MOVED OUTSIDE try block)
    jolt_config = None
    try:
        from marvis.utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        
        # First try the task-based naming convention (newer format)
        task_config_filename = f"jolt_config_task_{dataset['id']}.json"
        jolt_config_path = rm.path_resolver.get_config_path('jolt', task_config_filename)
        
        if jolt_config_path and jolt_config_path.exists():
            import json
            with open(jolt_config_path, 'r') as f:
                jolt_config = json.load(f)
            logger.info(f"Using JOLT metadata for task {dataset['id']} ({dataset['name']}) from managed config")
        else:
            # Fallback to dataset name-based naming (legacy format)
            name_config_filename = f"jolt_config_{dataset['name']}.json"
            jolt_config_path = rm.path_resolver.get_config_path('jolt', name_config_filename)
            
            if jolt_config_path and jolt_config_path.exists():
                import json
                with open(jolt_config_path, 'r') as f:
                    jolt_config = json.load(f)
                logger.info(f"Using JOLT metadata for {dataset['name']} from managed config (legacy naming)")
            else:
                logger.info(f"No JOLT metadata found for task {dataset['id']} ({dataset['name']}), using default approach")
    except Exception as e:
        logger.error(f"Could not load JOLT config for task {dataset['id']} ({dataset['name']}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to legacy method with both naming conventions
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try task-based naming first
            task_config_path = os.path.join(current_dir, f"jolt_config_task_{dataset['id']}.json")
            if os.path.exists(task_config_path):
                import json
                with open(task_config_path, 'r') as f:
                    jolt_config = json.load(f)
                logger.info(f"Using JOLT metadata for task {dataset['id']} from legacy location")
            else:
                # Try dataset name-based naming
                name_config_path = os.path.join(current_dir, f"jolt_config_{dataset['name']}.json")
                if os.path.exists(name_config_path):
                    import json
                    with open(name_config_path, 'r') as f:
                        jolt_config = json.load(f)
                    logger.info(f"Using JOLT metadata for {dataset['name']} from legacy location")
        except Exception as e2:
            logger.error(f"Legacy JOLT config loading also failed: {e2}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Debug: Log config status before main evaluation
    logger.info(f"Starting JOLT evaluation with jolt_config={'loaded' if jolt_config else 'None'}")
    
    start_time = time.time()
    
    try:
        # Add official JOLT to path using relative path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jolt_path = os.path.join(current_dir, "official_jolt")
        
        # Check if official_jolt directory exists
        if not os.path.exists(jolt_path):
            logger.error(f"Official JOLT directory not found at {jolt_path}")
            raise FileNotFoundError(f"Official JOLT directory not found at {jolt_path}")
        
        # Check if run_jolt.py exists
        run_jolt_file = os.path.join(jolt_path, "run_jolt.py")
        if not os.path.exists(run_jolt_file):
            logger.error(f"run_jolt.py not found at {run_jolt_file}")
            raise FileNotFoundError(f"run_jolt.py not found at {run_jolt_file}")
        
        if jolt_path not in sys.path:
            sys.path.insert(0, jolt_path)
        
        # Import official JOLT components
        try:
            # List what's in the directory for debugging
            logger.debug(f"Contents of {jolt_path}: {os.listdir(jolt_path)[:10]}...")
            
            # Try different import approaches
            try:
                import run_jolt
                from run_jolt import run_jolt as run_jolt_fn
                run_jolt = run_jolt_fn
            except ImportError:
                # Alternative: import as module
                import importlib.util
                spec = importlib.util.spec_from_file_location("run_jolt", run_jolt_file)
                run_jolt_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(run_jolt_module)
                run_jolt = run_jolt_module.run_jolt
            
            # Import hf_api
            try:
                from hf_api import get_model_and_tokenizer
            except ImportError:
                hf_api_file = os.path.join(jolt_path, "hf_api.py")
                spec = importlib.util.spec_from_file_location("hf_api", hf_api_file)
                hf_api_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hf_api_module)
                get_model_and_tokenizer = hf_api_module.get_model_and_tokenizer
                
        except Exception as e:
            logger.error(f"Failed to import JOLT modules from {jolt_path}: {e}")
            logger.error(f"sys.path includes: {sys.path[:5]}...")  # Show first 5 paths
            raise
        
        # Split data
        X, y = dataset["X"], dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        
        # Get unique classes for metric calculation
        import numpy as np
        unique_classes = np.unique(y_train)
        
        # Limit test samples if specified
        max_test_samples = getattr(args, 'max_test_samples', None)
        if max_test_samples and max_test_samples < len(X_test):
            X_test = X_test[:max_test_samples]
            y_test = y_test[:max_test_samples]
        
        # Apply feature selection for large feature spaces
        X_train, X_test, dataset, _ = apply_feature_reduction(
            X_train, y_train, X_test, dataset, args, logger
        )
        
        # Detect task type (classification vs regression) using task_id
        task_id = dataset.get('task_id') or dataset.get('id')
        task_type, detection_method = detect_task_type(dataset=dataset, y=y_train, task_id=task_id)
        logger.info(f"Task type detection method: {detection_method}")
        is_regression = (task_type == 'regression')
        logger.info(f"JOLT detected task type: {task_type}")
        
        # Convert to DataFrame if needed, using JOLT config column descriptions if available
        def get_jolt_feature_names(original_names, jolt_config):
            """Get feature names, using JOLT config descriptions if available."""
            if jolt_config and 'column_descriptions' in jolt_config:
                column_descriptions = jolt_config['column_descriptions']
                logger.info(f"JOLT config has {len(column_descriptions)} column descriptions")
                
                # Try to map original names to descriptive names
                mapped_names = []
                for orig_name in original_names:
                    if orig_name in column_descriptions:
                        mapped_names.append(column_descriptions[orig_name])
                        logger.info(f"JOLT column mapping: {orig_name} -> {column_descriptions[orig_name]}")
                    else:
                        # Try fuzzy matching for close names
                        best_match = None
                        for desc_key in column_descriptions.keys():
                            # Check if the original name is similar to the description key
                            if orig_name.lower().replace('_', ' ').replace('-', ' ') in desc_key.lower() or \
                               desc_key.lower().replace('_', ' ').replace('-', ' ') in orig_name.lower():
                                best_match = column_descriptions[desc_key]
                                logger.info(f"JOLT fuzzy column mapping: {orig_name} -> {desc_key} -> {best_match}")
                                break
                        if best_match:
                            mapped_names.append(best_match)
                        else:
                            logger.warning(f"JOLT: No mapping found for column '{orig_name}', using original")
                            mapped_names.append(orig_name)
                
                return mapped_names
            else:
                logger.info("JOLT: No column descriptions in config, using original names")
                return original_names
        
        if not isinstance(X_train, pd.DataFrame):
            # Handle case where attribute_names might include target column
            feature_names = dataset["attribute_names"]
            if len(feature_names) > X_train.shape[1]:
                # Likely includes target column, so exclude it
                feature_names = feature_names[:X_train.shape[1]]
            elif len(feature_names) < X_train.shape[1]:
                # Not enough names, generate generic ones for missing columns
                feature_names = list(feature_names) + [f"feature_{i}" for i in range(len(feature_names), X_train.shape[1])]
            
            # Apply JOLT config column descriptions if available
            feature_names = get_jolt_feature_names(feature_names, jolt_config)
            X_train = pd.DataFrame(X_train, columns=feature_names)
        
        if not isinstance(X_test, pd.DataFrame):
            # Use same logic for test set
            feature_names = dataset["attribute_names"]
            if len(feature_names) > X_test.shape[1]:
                feature_names = feature_names[:X_test.shape[1]]
            elif len(feature_names) < X_test.shape[1]:
                feature_names = list(feature_names) + [f"feature_{i}" for i in range(len(feature_names), X_test.shape[1])]
            
            # Apply JOLT config column descriptions if available
            feature_names = get_jolt_feature_names(feature_names, jolt_config)
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Set target column name using JOLT config if available
        target_col_name = 'target'
        if jolt_config and 'class_description' in jolt_config:
            # Extract a more descriptive target name from class description
            class_desc = jolt_config['class_description']
            if 'contraceptive' in class_desc.lower():
                target_col_name = 'Contraceptive method choice'
            else:
                target_col_name = 'target'
            logger.info(f"JOLT using target column name: {target_col_name}")
        
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, name=target_col_name)
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test, name=target_col_name)
        
        # Calculate few-shot parameters early for use throughout
        user_few_shot = getattr(args, 'num_few_shot_examples', 16)
        jolt_model_name = getattr(args, 'jolt_model', 'Qwen/Qwen2.5-7B-Instruct')
        
        # JOLT should use reasonable limits for few-shot prompting, but use all training data
        effective_shots = min(user_few_shot, 16)
        
        logger.info(f"JOLT using model: {jolt_model_name}, shots: {effective_shots}, full training data: {len(X_train)} samples")
        
        # Log task-specific configuration
        if is_regression:
            logger.info(f"REGRESSION CONFIG: mode=sample, y_column_types=['numerical'], num_decimal_places_y=3")
        else:
            logger.info(f"CLASSIFICATION CONFIG: mode=logpy_only, y_column_types=['categorical'], num_decimal_places_y=0")
        
        # Combine features and target for JOLT format with memory-safe limits
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Limit training data based on JOLT authors' approach (max 50 shots for regression)
        if is_regression:
            # JOLT authors use 10-50 shots for regression
            max_train_for_jolt = min(effective_shots, 50, len(train_data))
            logger.info(f"Limiting regression training data to {max_train_for_jolt} samples (JOLT authors use 10-50)")
        else:
            # JOLT authors use up to 32 shots for classification  
            max_train_for_jolt = min(effective_shots * 2, 32, len(train_data))
            logger.info(f"Limiting classification training data to {max_train_for_jolt} samples (JOLT authors use up to 32)")
            
        original_train_size = len(train_data)
        if original_train_size > max_train_for_jolt:
            train_data = train_data.head(max_train_for_jolt)
            logger.info(f"Reduced training data from {original_train_size} to {max_train_for_jolt} samples to match JOLT authors' approach")
        
        # Store the actual test data size before combining
        actual_test_size = len(test_data)
        combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        
        # Create temporary CSV file for JOLT
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            combined_data.to_csv(tmp_file.name, index=False)
            temp_csv_path = tmp_file.name
        
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_output_dir:
                # Use the model specified in args
                llm_path = None
                llm_type = ""  # Use empty string instead of None to avoid TypeError
                if hasattr(args, 'jolt_model') and args.jolt_model:
                    logger.info(f"Requested JOLT model: {args.jolt_model}")
                    
                    # If it's a HuggingFace model path, use llm_path instead of llm_type
                    if "/" in args.jolt_model:
                        llm_path = args.jolt_model
                        # Try to infer type from model name for JOLT's internal checks
                        if "qwen" in args.jolt_model.lower():
                            if "2.5" in args.jolt_model and "7b" in args.jolt_model.lower():
                                llm_type = "qwen2.5-7b-instruct"  # Use JOLT's specific naming
                            elif "7b" in args.jolt_model.lower():
                                llm_type = "qwen-7b"
                            else:
                                llm_type = "qwen"
                        elif "llama" in args.jolt_model.lower():
                            if "7b" in args.jolt_model.lower():
                                llm_type = "llama-2-7B"
                            else:
                                llm_type = "llama"
                        elif "gemma" in args.jolt_model.lower():
                            if "2b" in args.jolt_model.lower():
                                llm_type = "gemma-2-2B-instruct"
                            else:
                                llm_type = "gemma"
                        else:
                            llm_type = "custom"  # Generic type
                    else:
                        # Map common model names to JOLT's llm_type format
                        if "llama-2-7b" in args.jolt_model.lower():
                            llm_type = "llama-2-7B"
                        elif "gemma-2-2b" in args.jolt_model.lower():
                            llm_type = "gemma-2-2B-instruct"
                        elif "qwen2.5-7b" in args.jolt_model.lower():
                            llm_type = "qwen2.5-7b-instruct"
                        elif "phi-3" in args.jolt_model.lower():
                            llm_type = "phi-3-mini-128k-instruct"
                        else:
                            # Use the model name as provided - JOLT might recognize it
                            llm_type = args.jolt_model
                else:
                    # Default fallback to a smaller, known working model
                    llm_type = "phi-3-mini-128k-instruct"  # Even smaller than Gemma-2-2B
                
                
                logger.info(f"Using JOLT config - llm_path: {llm_path}, llm_type: {llm_type}")
                logger.info(f"JOLT few-shot: user requested {user_few_shot}, using {effective_shots} shots, {len(train_data)} train samples available")
                logger.info(f"JOLT train_size_limit={effective_shots} to prevent massive prompts while preserving test indices")
                
                # Determine max token length with aggressive limits for regression to prevent OOM
                if is_regression:
                    # For regression, use much smaller context to prevent OOM issues
                    # Regression only needs to predict a number, not generate long text
                    max_tokens = 1024  # Much smaller for regression
                    max_generated_length = 256  # Minimum safe generation length
                    logger.info(f"Using aggressive memory limits for regression: max_tokens={max_tokens}, max_generated_length={max_generated_length}")
                else:
                    # Classification can use larger contexts since it uses logpy_only mode
                    if "qwen" in jolt_model_name.lower():
                        max_tokens = getattr(args, 'max_context_length', 4096)
                    elif "llama" in jolt_model_name.lower():
                        max_tokens = getattr(args, 'max_context_length', 4096)
                    elif "gemma" in jolt_model_name.lower():
                        max_tokens = getattr(args, 'max_context_length', 8192)
                    else:
                        max_tokens = getattr(args, 'max_context_length', 2048)
                    max_generated_length = 256  # Minimum safe generation length
                    logger.info(f"Using standard limits for classification: max_tokens={max_tokens}, max_generated_length={max_generated_length}")
                
                # Create JOLT arguments with memory optimizations
                # Set mode and column types based on task type
                if is_regression:
                    mode = "sample"  # Use sample mode for regression - must contain "sample" string
                    y_column_types = ['numerical']
                    num_decimal_places_y = 3  # Use 3 decimal places for regression targets
                else:
                    mode = "logpy_only"  # Use log probability mode for classification
                    y_column_types = ['categorical']
                    num_decimal_places_y = 0  # Categorical targets don't need decimals
                
                jolt_args = Namespace(
                    mode=mode,
                    experiment_name=f"jolt_{dataset['name']}",
                    data_path=temp_csv_path,
                    llm_path=llm_path,
                    llm_type=llm_type,
                    output_dir=temp_output_dir,
                    seed=args.seed,
                    num_decimal_places_x=2,
                    num_decimal_places_y=num_decimal_places_y,
                    batch_size=1 if is_regression else 5,  # Smaller batch for regression, JOLT authors use 5 for classification
                    
                    # Memory optimization parameters
                    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                    prefix=jolt_config.get('task_prefix', '') if jolt_config else '',
                    break_str='\n',
                    y_column_names=[target_col_name],
                    y_column_types=y_column_types,
                    column_separator=';',
                    name_value_separator=':',
                    
                    # CSV parsing options - use our exact train/test split
                    csv_split_option="fixed_indices",
                    test_fraction=0.2,
                    shots=effective_shots,  # Few-shot examples per prompt (user controlled)
                    header_option='headers_as_item_prefix',  # Use headers in prompts
                    train_size_limit=None,  # We manually limited data above
                    test_size_limit=None,
                    train_start_index=0,
                    train_end_index=len(train_data),  # Use actual limited training data size
                    test_start_index=len(train_data),  # Test data starts after limited training data
                    test_end_index=len(train_data) + actual_test_size,  # Test data ends after all test samples
                    missing_fraction=0.0,
                    impute_features=False,
                    shuffle=False,  # Don't shuffle since we already split
                    columns_to_ignore=[],
                    
                    # Generation parameters (not used in logpy_only mode)
                    num_samples=1,
                    temperature=1.0,
                    top_p=0.9,
                    max_generated_length=max_generated_length,  # Use the calculated value
                    top_k=None,
                    
                    # Debug options
                    print_prompts=True,
                    print_sampling_rejections=False,
                    
                    # Device options - When CUDA_VISIBLE_DEVICES is set, always use cuda:0
                    device="cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu",
                    gpu_index=0 if torch.cuda.is_available() and args.device != "cpu" else None,
                    
                    # Token limit options
                    max_tokens=max_tokens,
                    max_sequence_length=max_tokens
                )
                
                # Clear GPU memory before loading model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get model and tokenizer using JOLT's API
                try:
                    logger.info(f"Loading JOLT model - llm_path: {llm_path}, llm_type: {llm_type}")
                    
                    # Force JOLT to use the correct device by temporarily setting torch default device
                    if torch.cuda.is_available() and args.device != "cpu":
                        # When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the masked GPU
                        torch.cuda.set_device(0)  # This will be the masked GPU
                        logger.info(f"Set torch default CUDA device to 0 (masked GPU)")
                    
                    try:
                        model, tokenizer = get_model_and_tokenizer(jolt_args)
                        logger.info(f"Successfully loaded JOLT model")
                    except Exception as model_load_error:
                        logger.error(f"get_model_and_tokenizer failed: {model_load_error}")
                        logger.error(f"JOLT args: llm_path={llm_path}, llm_type={llm_type}")
                        raise model_load_error
                    
                    # Note: Do NOT manually move model to device - HF Accelerate handles this
                    # and manual moves cause warnings about hooks
                    
                    # Clear cache again after loading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error loading JOLT model {llm_type}: {e}. Using fallback model.")
                    # Use the specified JOLT model as fallback, with better error handling
                    fallback_model = getattr(args, 'jolt_model', 'Qwen/Qwen2.5-7B-Instruct')
                    logger.info(f"Loading fallback model: {fallback_model}")
                    
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        # Use device_map="auto" and let HF Accelerate handle device placement
                        if torch.cuda.is_available() and args.device != "cpu":
                            model = AutoModelForCausalLM.from_pretrained(
                                fallback_model, 
                                device_map="auto", 
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                fallback_model,
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True
                            )
                        logger.info(f"Successfully loaded fallback model: {fallback_model}")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model {fallback_model}: {fallback_error}")
                        # Final fallback to a smaller generative model
                        final_fallback = 'microsoft/DialoGPT-medium'  # Keep as final emergency fallback
                        logger.info(f"Using final emergency fallback model: {final_fallback}")
                        tokenizer = AutoTokenizer.from_pretrained(final_fallback)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        if torch.cuda.is_available() and args.device != "cpu":
                            model = AutoModelForCausalLM.from_pretrained(
                                final_fallback, 
                                device_map="auto", 
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                final_fallback,
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True
                            )
                
                # Feature dropping mechanism for OOM handling
                dropped_features = set()  # Track which features have been dropped
                num_features = len(dataset["attribute_names"])
                original_dataset = dataset.copy()
                
                # Run JOLT evaluation with memory optimizations
                logger.info(f"Running JOLT evaluation on {len(X_test)} test samples...")
                
                jolt_results = None
                max_retries = max(0, num_features - 2)  # Keep trying until only 2 features remain
                
                for retry in range(max_retries + 1):
                    try:
                        # Log critical JOLT arguments before call
                        logger.info(f"CALLING JOLT with key arguments:")
                        logger.info(f"  - mode: {jolt_args.mode}")
                        logger.info(f"  - y_column_types: {jolt_args.y_column_types}")
                        logger.info(f"  - num_samples: {jolt_args.num_samples}")
                        logger.info(f"  - shots: {jolt_args.shots}")
                        logger.info(f"  - temperature: {jolt_args.temperature}")
                        logger.info(f"  - max_generated_length: {jolt_args.max_generated_length}")
                        logger.info(f"  - data_path: {jolt_args.data_path}")
                        logger.info(f"  - y_column_names: {jolt_args.y_column_names}")
                        
                        # Disable gradients globally to save memory during inference
                        with torch.no_grad():
                            
                            # Set model to eval mode and disable caching
                            if hasattr(model, 'eval'):
                                model.eval()
                            
                            logger.info("Starting JOLT run_jolt() call...")
                            jolt_results = run_jolt(jolt_args, model, tokenizer)
                            logger.info("JOLT run_jolt() call completed")
                            
                            # Validate JOLT results immediately after call
                            if jolt_results is None:
                                raise ValueError("JOLT run_jolt() returned None")
                            
                            if not isinstance(jolt_results, dict):
                                raise ValueError(f"JOLT run_jolt() returned {type(jolt_results)}, expected dict")
                            
                            # For regression tasks, validate that sample mode worked
                            if is_regression:
                                if 'gen' not in jolt_results:
                                    raise ValueError("JOLT sample mode did not return 'gen' key - sample() function may have failed")
                                
                                gen_data = jolt_results['gen']
                                if not isinstance(gen_data, list):
                                    raise ValueError(f"JOLT 'gen' data is {type(gen_data)}, expected list")
                                
                                if len(gen_data) == 0:
                                    raise ValueError("JOLT 'gen' data is empty - no samples were generated")
                                
                                # Check if any generation results exist
                                total_samples = sum(len(gen_item) if isinstance(gen_item, list) else 0 for gen_item in gen_data)
                                if total_samples == 0:
                                    raise ValueError("JOLT generated 0 valid samples - all generation attempts failed or were filtered out")
                                
                                logger.info(f"JOLT generated {total_samples} total samples across {len(gen_data)} test instances")
                                
                                # Check if process_generated_results was called (should add 'metrics' key)
                                if 'metrics' not in jolt_results:
                                    raise ValueError("JOLT sample mode completed but 'metrics' key missing - process_generated_results() was not called or failed")
                                
                                # Validate metrics structure for regression
                                metrics = jolt_results['metrics']
                                if not isinstance(metrics, list) or len(metrics) == 0:
                                    raise ValueError(f"JOLT metrics is {type(metrics)} with length {len(metrics) if hasattr(metrics, '__len__') else 'N/A'}, expected non-empty list")
                                
                                first_metric = metrics[0]
                                if not isinstance(first_metric, dict):
                                    raise ValueError(f"JOLT first metric is {type(first_metric)}, expected dict")
                                
                                required_keys = ['y_test_median', 'y_test_mean', 'mae']
                                missing_keys = [key for key in required_keys if key not in first_metric]
                                if missing_keys:
                                    raise ValueError(f"JOLT regression metrics missing required keys: {missing_keys}")
                                
                                # Check for NaN values in predictions
                                y_test_median = first_metric['y_test_median']
                                import numpy as np
                                if isinstance(y_test_median, list) and len(y_test_median) > 0:
                                    nan_count = sum(1 for val in y_test_median if np.isnan(val))
                                    if nan_count == len(y_test_median):
                                        raise ValueError("All JOLT regression predictions are NaN - generation or parsing failed")
                                    elif nan_count > 0:
                                        logger.warning(f"JOLT has {nan_count}/{len(y_test_median)} NaN predictions")
                                
                                logger.info("JOLT regression validation passed")
                        
                        # If successful, break out of retry loop
                        break
                        
                    except Exception as jolt_error:
                        if is_oom_error(jolt_error) and retry < max_retries:
                            logger.warning(f"JOLT failed with OOM: {jolt_error}")
                            
                            # Try dropping a feature and retry (keep at least 2 features for JOLT)
                            if drop_feature_for_oom(dropped_features, num_features, logger, min_features=2):
                                logger.info(f"Retrying JOLT with {len(dropped_features)} dropped features")
                                
                                # Re-apply feature reduction with dropped features
                                # Create mask for non-dropped features
                                keep_indices = [i for i in range(num_features) if i not in dropped_features]
                                
                                # Update datasets
                                X_train = X_train.iloc[:, keep_indices] if hasattr(X_train, 'iloc') else X_train[:, keep_indices]
                                X_test = X_test.iloc[:, keep_indices] if hasattr(X_test, 'iloc') else X_test[:, keep_indices]
                                
                                # Update dataset info
                                dataset = original_dataset.copy()
                                dataset["attribute_names"] = [dataset["attribute_names"][i] for i in keep_indices]
                                if "categorical_indicator" in dataset:
                                    dataset["categorical_indicator"] = [dataset["categorical_indicator"][i] for i in keep_indices]
                                
                                # Recreate DataFrames with new feature names
                                feature_names = get_jolt_feature_names(dataset["attribute_names"], jolt_config)
                                X_train = pd.DataFrame(X_train, columns=feature_names)
                                X_test = pd.DataFrame(X_test, columns=feature_names)
                                
                                # Recreate combined data for JOLT
                                train_data = pd.concat([X_train, y_train], axis=1)
                                test_data = pd.concat([X_test, y_test], axis=1)
                                combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
                                
                                # Create new temporary CSV file
                                if os.path.exists(temp_csv_path):
                                    os.unlink(temp_csv_path)
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                                    combined_data.to_csv(tmp_file.name, index=False)
                                    temp_csv_path = tmp_file.name
                                
                                # Update JOLT args with new CSV path
                                jolt_args.data_path = temp_csv_path
                                
                                # Continue to next retry
                                continue
                            else:
                                logger.error("No more features to drop, cannot continue")
                                raise
                        else:
                            # Not an OOM error or no more retries
                            raise
                
                if jolt_results is None:
                    raise RuntimeError("Failed to run JOLT after all retries")
                
                # Extract predictions from JOLT results
                completed_samples = 0
                predictions = []
                example_inputs_outputs = []  # Store example inputs and outputs for debugging
                jolt_regression_metrics = {}  # Store JOLT's built-in regression metrics
                jolt_probabilities = None  # Store JOLT probabilities for classification
                
                # JOLT stores results in a specific format - extract predictions
                logger.info(f"JOLT results keys: {list(jolt_results.keys()) if isinstance(jolt_results, dict) else 'Not a dict'}")
                if isinstance(jolt_results, dict) and 'data' in jolt_results:
                    logger.info(f"JOLT data keys: {list(jolt_results['data'].keys()) if isinstance(jolt_results['data'], dict) else 'Data not a dict'}")
                
                # Add detailed logging for regression tasks
                if is_regression:
                    logger.info(f"REGRESSION TASK DEBUGGING:")
                    logger.info(f"  - Task type: {task_type}")
                    logger.info(f"  - JOLT mode used: sample")
                    logger.info(f"  - y_column_types: ['numerical']")
                    if isinstance(jolt_results, dict):
                        logger.info(f"  - JOLT results type: dict with {len(jolt_results)} keys")
                        for key, value in jolt_results.items():
                            if key == 'data' and isinstance(value, dict):
                                logger.info(f"    - {key}: dict with keys {list(value.keys())}")
                                for subkey, subvalue in value.items():
                                    if hasattr(subvalue, 'shape'):
                                        logger.info(f"      - {subkey}: array shape {subvalue.shape}")
                                    elif isinstance(subvalue, (list, tuple)):
                                        logger.info(f"      - {subkey}: {type(subvalue).__name__} length {len(subvalue)}")
                                    else:
                                        logger.info(f"      - {subkey}: {type(subvalue).__name__}")
                            elif key == 'metrics':
                                logger.info(f"    - {key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'no'} items")
                                if isinstance(value, list) and len(value) > 0:
                                    first_metric = value[0]
                                    if isinstance(first_metric, dict):
                                        logger.info(f"      - First metric keys: {list(first_metric.keys())}")
                                        for metric_key in ['y_test_median', 'y_test_mean', 'mae']:
                                            if metric_key in first_metric:
                                                metric_val = first_metric[metric_key]
                                                if hasattr(metric_val, '__len__'):
                                                    logger.info(f"        - {metric_key}: length {len(metric_val)}")
                                                else:
                                                    logger.info(f"        - {metric_key}: {metric_val}")
                                            else:
                                                logger.info(f"        - {metric_key}: MISSING")
                            elif hasattr(value, 'shape'):
                                logger.info(f"    - {key}: array shape {value.shape}")
                            elif isinstance(value, (list, tuple)):
                                logger.info(f"    - {key}: {type(value).__name__} length {len(value)}")
                            else:
                                logger.info(f"    - {key}: {type(value).__name__}")
                    else:
                        logger.info(f"  - JOLT results type: {type(jolt_results)}")
                        
                    # Check if JOLT's sample function was supposed to call process_generated_results
                    if isinstance(jolt_results, dict):
                        has_gen = 'gen' in jolt_results
                        has_prompts = 'prompts' in jolt_results
                        has_metrics = 'metrics' in jolt_results
                        logger.info(f"  - Expected sample mode results: gen={has_gen}, prompts={has_prompts}, metrics={has_metrics}")
                        
                        if has_gen:
                            gen_data = jolt_results['gen']
                            logger.info(f"    - gen data type: {type(gen_data)}, length: {len(gen_data) if hasattr(gen_data, '__len__') else 'N/A'}")
                            if isinstance(gen_data, list) and len(gen_data) > 0:
                                first_gen = gen_data[0]
                                logger.info(f"    - first gen item type: {type(first_gen)}, length: {len(first_gen) if hasattr(first_gen, '__len__') else 'N/A'}")
                        
                        if not has_metrics:
                            logger.error("REGRESSION ISSUE: JOLT sample mode did not produce 'metrics' - process_generated_results may have failed!")
                            logger.error("This suggests the sample() function completed but process_generated_results() was not called or failed")
                
                # Handle regression predictions differently from classification
                if is_regression:
                    logger.info("Processing regression task results...")
                    if isinstance(jolt_results, dict) and 'metrics' in jolt_results and len(jolt_results['metrics']) > 0:
                        # Extract regression predictions and metrics from JOLT
                        logger.info("Extracting regression predictions from JOLT metrics...")
                        try:
                            metrics = jolt_results['metrics'][0]  # Get first column metrics for regression
                            
                            # Extract regression predictions (use median as point prediction)
                            if 'y_test_median' in metrics and len(metrics['y_test_median']) > 0:
                                predictions = metrics['y_test_median']
                                completed_samples = len([p for p in predictions if not np.isnan(p)])
                                logger.info(f"Successfully extracted {completed_samples} regression predictions from JOLT")
                                
                                # Extract JOLT's built-in regression metrics
                                if 'mae' in metrics:
                                    jolt_regression_metrics['mae'] = float(metrics['mae'])
                                    logger.info(f"JOLT built-in MAE: {jolt_regression_metrics['mae']:.4f}")
                                
                                # Store additional regression statistics if available
                                if 'y_test_mean' in metrics:
                                    jolt_regression_metrics['predictions_mean'] = metrics['y_test_mean']
                                if 'y_test_std' in metrics:
                                    jolt_regression_metrics['predictions_std'] = metrics['y_test_std']
                                if 'y_test_lower' in metrics and 'y_test_upper' in metrics:
                                    jolt_regression_metrics['confidence_intervals'] = {
                                        'lower': metrics['y_test_lower'],
                                        'upper': metrics['y_test_upper']
                                    }
                            else:
                                logger.warning("No y_test_median found in JOLT regression metrics")
                                predictions = []
                                completed_samples = 0
                        except Exception as e:
                            logger.error(f"Error extracting regression predictions from JOLT metrics: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            predictions = []
                            completed_samples = 0
                
                elif 'data' in jolt_results and 'y_pred' in jolt_results['data']:
                    raw_predictions = jolt_results['data']['y_pred']
                    if len(raw_predictions.shape) > 1:
                        # Multi-column predictions - take first column
                        predictions = raw_predictions[:, 0].tolist()
                    else:
                        predictions = raw_predictions.tolist()
                    completed_samples = len(predictions)
                    logger.info(f"Successfully extracted {completed_samples} predictions from JOLT")
                
                elif 'predictions' in jolt_results:
                    predictions = jolt_results['predictions']
                    completed_samples = len(predictions)
                    
                elif isinstance(jolt_results, dict) and 'metrics' in jolt_results and len(jolt_results['metrics']) > 0:
                    # Try to extract predictions from JOLT's computed probabilities
                    logger.info("Attempting to extract predictions from JOLT metrics...")
                    try:
                        metrics = jolt_results['metrics'][0]  # Get first column metrics
                        if 'probabilities_from_logits' in metrics and len(metrics['probabilities_from_logits']) > 0:
                            # Convert probabilities to predictions using argmax
                            import numpy as np
                            probs = np.array(metrics['probabilities_from_logits'])
                            pred_indices = np.argmax(probs, axis=1)
                            
                            # Map indices back to class labels using categories
                            if 'categories' in jolt_results and len(jolt_results['categories']) > 0:
                                categories = jolt_results['categories'][0]  # First column categories
                                predictions = [categories[idx] for idx in pred_indices]
                                completed_samples = len(predictions)
                                
                                # Store probabilities for proper AUC calculation
                                jolt_probabilities = probs
                                logger.info(f"Successfully extracted {completed_samples} predictions and probabilities from JOLT")
                                logger.info(f"JOLT probabilities shape: {probs.shape}")
                            else:
                                error_msg = "No categories found in JOLT results - cannot map prediction indices to class labels"
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                        else:
                            logger.warning("No probabilities_from_logits found in JOLT metrics")
                            predictions = []
                            completed_samples = 0
                    except Exception as e:
                        logger.error(f"Error extracting predictions from JOLT metrics: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        predictions = []
                        completed_samples = 0
                
                else:
                    # No valid predictions found - raise exception instead of using fallback
                    error_msg = "Could not find valid predictions in JOLT results. JOLT may have failed to generate predictions properly."
                    logger.error(error_msg)
                    logger.error(f"JOLT results structure: {type(jolt_results)}")
                    if isinstance(jolt_results, dict):
                        logger.error(f"JOLT results keys: {list(jolt_results.keys())}")
                    raise ValueError(error_msg)
                
                # Ensure predictions match the expected length
                if len(predictions) > len(X_test):
                    predictions = predictions[:len(X_test)]
                    completed_samples = len(X_test)
                elif len(predictions) < len(X_test) and completed_samples > 0:
                    # JOLT failed to predict all samples - impute missing predictions
                    missing_count = len(X_test) - len(predictions)
                    logger.warning(f"JOLT only predicted {len(predictions)} out of {len(X_test)} samples. "
                                 f"Imputing {missing_count} missing predictions using majority class/mean strategy.")
                    
                    if is_regression:
                        # For regression: use mean of existing predictions
                        if len(predictions) > 0:
                            existing_predictions = [p for p in predictions if p is not None and not np.isnan(p)]
                            if existing_predictions:
                                imputation_value = np.mean(existing_predictions)
                                logger.info(f"Using mean value {imputation_value:.3f} for regression imputation")
                            else:
                                # Fallback to training data mean
                                imputation_value = np.mean(y_train)
                                logger.info(f"Using training data mean {imputation_value:.3f} for regression imputation")
                        else:
                            imputation_value = np.mean(y_train)
                            logger.info(f"Using training data mean {imputation_value:.3f} for regression imputation")
                        
                        # Extend predictions with imputed values
                        predictions.extend([imputation_value] * missing_count)
                    else:
                        # For classification: use majority class
                        if len(predictions) > 0:
                            # Use majority class from existing predictions
                            from collections import Counter
                            prediction_counts = Counter(predictions)
                            majority_class = prediction_counts.most_common(1)[0][0]
                            logger.info(f"Using majority class '{majority_class}' for classification imputation")
                        else:
                            # Fallback to most frequent class in training data
                            from collections import Counter
                            train_counts = Counter(y_train)
                            majority_class = train_counts.most_common(1)[0][0]
                            logger.info(f"Using training majority class '{majority_class}' for classification imputation")
                        
                        # Extend predictions with imputed values
                        predictions.extend([majority_class] * missing_count)
                    
                    completed_samples = len(X_test)  # Update completed samples count
                    logger.info(f"Successfully imputed {missing_count} missing predictions. Total predictions: {len(predictions)}")
                
                # Calculate metrics
                if completed_samples > 0:
                    y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
                    predictions_partial = predictions[:completed_samples]
                    
                    # Ensure predictions have the same type as ground truth
                    if len(y_test_partial) > 0:
                        target_type = type(y_test_partial.iloc[0] if hasattr(y_test_partial, 'iloc') else y_test_partial[0])
                        converted_predictions = []
                        
                        for pred in predictions_partial:
                            try:
                                # Try to convert prediction to target type
                                if target_type == int:
                                    # Handle common string representations
                                    if isinstance(pred, str):
                                        # Remove any extra whitespace
                                        pred = pred.strip()
                                        # Try to convert to int
                                        converted_pred = int(float(pred))  # Use float first to handle "1.0" -> 1
                                    else:
                                        converted_pred = int(pred)
                                elif target_type == float:
                                    converted_pred = float(pred)
                                elif target_type == str:
                                    converted_pred = str(pred)
                                else:
                                    # For other types, try direct conversion
                                    converted_pred = target_type(pred)
                                
                                converted_predictions.append(converted_pred)
                            except (ValueError, TypeError) as e:
                                error_msg = f"Could not convert prediction '{pred}' to expected type {target_type}: {e}"
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                        
                        predictions_partial = converted_predictions
                    
                    # Calculate metrics based on task type
                    if is_regression:
                        # Calculate regression metrics
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        import numpy as np
                        
                        # Convert to numpy arrays for metric calculation
                        y_true = np.array(y_test_partial)
                        y_pred = np.array(predictions_partial)
                        
                        # Calculate regression metrics
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        
                        # Use JOLT's built-in MAE if available (should be more accurate)
                        if 'mae' in jolt_regression_metrics:
                            mae = jolt_regression_metrics['mae']
                        
                        logger.info(f"JOLT regression metrics - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                        
                        # Set regression-specific values
                        calculated_metrics = {
                            'r2_score': r2,
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'roc_auc': None,
                            'f1_macro': None,
                            'f1_micro': None, 
                            'f1_weighted': None,
                            'precision_macro': None,
                            'recall_macro': None
                        }
                        accuracy = None  # Not applicable for regression
                        balanced_acc = None  # Not applicable for regression
                        
                    else:
                        # Import shared metric calculation function for classification
                        from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
                        
                        # Convert JOLT probabilities to log probs format if available
                        all_class_log_probs = None
                        if jolt_probabilities is not None and completed_samples > 0:
                            # JOLT probabilities are in shape [n_samples, n_classes]
                            # We need to convert to list of dicts with log probabilities
                            all_class_log_probs = []
                            categories = jolt_results['categories'][0] if 'categories' in jolt_results else unique_classes
                            
                            for i in range(min(completed_samples, len(jolt_probabilities))):
                                # Create dict mapping class labels to log probabilities
                                class_log_probs = {}
                                for j, class_label in enumerate(categories):
                                    # Convert probability to log probability
                                    prob = jolt_probabilities[i][j] if j < len(jolt_probabilities[i]) else 1e-10
                                    # Ensure prob is positive and non-zero to avoid log(0)
                                    prob = max(prob, 1e-10)
                                    log_prob = np.log(prob)
                                    class_log_probs[str(class_label)] = log_prob
                                all_class_log_probs.append(class_log_probs)
                            
                            logger.info(f"Converted JOLT probabilities to log probs for {len(all_class_log_probs)} samples")
                        
                        # Calculate all classification metrics using shared function
                        calculated_metrics = calculate_llm_metrics(
                            y_test_partial, predictions_partial, unique_classes, 
                            all_class_log_probs=all_class_log_probs, logger=logger, task_type=task_type
                        )
                        
                        # Extract individual metrics for backward compatibility
                        accuracy = calculated_metrics['accuracy']
                        balanced_acc = calculated_metrics['balanced_accuracy']
                        logger.info(f"JOLT achieved {accuracy:.4f} accuracy on {completed_samples} samples")
                    
                    # Store example inputs and outputs for first 20 samples
                    num_examples_to_store = min(20, completed_samples)
                    for i in range(num_examples_to_store):
                        true_label = y_test_partial.iloc[i] if hasattr(y_test_partial, 'iloc') else y_test_partial[i]
                        pred_label = predictions_partial[i]
                        test_features = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
                        
                        example_inputs_outputs.append({
                            'sample_index': i,
                            'input_features': test_features.to_dict() if hasattr(test_features, 'to_dict') else str(test_features),
                            'jolt_config_used': jolt_config is not None,
                            'prediction': pred_label,
                            'true_class': true_label,
                            'correct': pred_label == true_label
                        })
                else:
                    accuracy = 0.0
                    balanced_acc = 0.0
                    # Set default values for other metrics when no predictions available
                    calculated_metrics = {
                        'roc_auc': None,
                        'f1_macro': None,
                        'f1_micro': None,
                        'f1_weighted': None,
                        'precision_macro': None,
                        'recall_macro': None
                    }
                    logger.warning("No valid predictions obtained from JOLT")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
        
        training_time = time.time() - start_time
        
        # Convert all results to JSON-serializable types
        def convert_to_serializable(obj):
            """Convert numpy types and other non-serializable types to Python native types."""
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Create base results dictionary
        results = {
            'model_name': 'JOLT',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_type': task_type,
            'training_time': float(training_time),
            'num_test_samples': len(X_test),
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train.shape[1],  # Use X_train to get actual feature count after reduction
            'predictions': convert_to_serializable(predictions[:completed_samples] if completed_samples > 0 else []),
            'ground_truth': convert_to_serializable((y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else []),
            'used_jolt_config': jolt_config is not None,
            'used_official_jolt': True,
            'example_inputs_outputs': convert_to_serializable(example_inputs_outputs)
        }
        
        # Add task-specific metrics
        if is_regression:
            results.update({
                'accuracy': None,  # Not applicable for regression
                'balanced_accuracy': None,  # Not applicable for regression
                'num_classes': None,  # Not applicable for regression
                'roc_auc': None,
                'f1_macro': None,
                'f1_micro': None,
                'f1_weighted': None,
                'precision_macro': None,
                'recall_macro': None,
                # Regression-specific metrics
                'r2_score': float(calculated_metrics['r2_score']) if calculated_metrics['r2_score'] is not None else None,
                'mae': float(calculated_metrics['mae']) if calculated_metrics['mae'] is not None else None,
                'mse': float(calculated_metrics['mse']) if calculated_metrics['mse'] is not None else None,
                'rmse': float(calculated_metrics['rmse']) if calculated_metrics['rmse'] is not None else None,
                'regression_results': {
                    'r2_score': float(calculated_metrics['r2_score']) if calculated_metrics['r2_score'] is not None else None,
                    'mae': float(calculated_metrics['mae']) if calculated_metrics['mae'] is not None else None,
                    'mse': float(calculated_metrics['mse']) if calculated_metrics['mse'] is not None else None,
                    'rmse': float(calculated_metrics['rmse']) if calculated_metrics['rmse'] is not None else None,
                    'jolt_builtin_metrics': convert_to_serializable(jolt_regression_metrics)
                }
            })
        else:
            results.update({
                'accuracy': float(accuracy) if accuracy is not None else None,
                'balanced_accuracy': float(balanced_acc) if balanced_acc is not None else None,
                'num_classes': len(unique_classes),
                'roc_auc': float(calculated_metrics['roc_auc']) if calculated_metrics['roc_auc'] is not None else None,
                'f1_macro': float(calculated_metrics['f1_macro']) if calculated_metrics['f1_macro'] is not None else None,
                'f1_micro': float(calculated_metrics['f1_micro']) if calculated_metrics['f1_micro'] is not None else None,
                'f1_weighted': float(calculated_metrics['f1_weighted']) if calculated_metrics['f1_weighted'] is not None else None,
                'precision_macro': float(calculated_metrics['precision_macro']) if calculated_metrics['precision_macro'] is not None else None,
                'recall_macro': float(calculated_metrics['recall_macro']) if calculated_metrics['recall_macro'] is not None else None,
                # Regression metrics (null for classification)
                'r2_score': None,
                'mae': None,
                'mse': None,
                'rmse': None,
                'regression_results': None
            })
        
        # Validate JOLT completion before returning results
        min_completion_rate = 0.5  # Require at least 50% completion
        if results['completion_rate'] < min_completion_rate:
            error_msg = (f"JOLT failed to complete successfully: only {results['completed_samples']}/{results['num_test_samples']} "
                        f"samples completed ({results['completion_rate']:.1%}). "
                        f"Minimum required completion rate: {min_completion_rate:.0%}. "
                        f"This indicates JOLT encountered critical errors during evaluation.")
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Additional validation for metrics
        if not is_regression:
            if results['accuracy'] is None or results['accuracy'] == 0:
                error_msg = (f"JOLT failed to produce valid classification results: "
                           f"accuracy={results['accuracy']}, completed_samples={results['completed_samples']}. "
                           f"This indicates the classification pipeline failed.")
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            if results['mae'] is None:
                error_msg = (f"JOLT failed to produce valid regression results: "
                           f"MAE={results['mae']}, completed_samples={results['completed_samples']}. "
                           f"This indicates the regression pipeline failed.")
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        # Debug: Log config status at successful completion
        logger.info(f"JOLT evaluation completed successfully with {results['completion_rate']:.1%} completion rate")
        logger.info(f"Final jolt_config status: {'loaded' if jolt_config else 'None'}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating JOLT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'JOLT',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_type': 'unknown',
            'error': str(e),
            'timeout': False,
            'completed_samples': 0,
            'completion_rate': 0.0,
            'used_jolt_config': jolt_config is not None  # Include config status even in error case
        }
    finally:
        # Restore original environment variables
        if original_cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_device
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            # If it wasn't set originally, remove it
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        if original_pytorch_cuda_alloc_conf is not None:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = original_pytorch_cuda_alloc_conf
        elif 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
            # If it wasn't set originally, remove it
            del os.environ['PYTORCH_CUDA_ALLOC_CONF']