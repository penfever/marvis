#!/usr/bin/env python
"""
Tabula-8B baseline evaluation module.
Contains functions for evaluating the Tabula-8B baseline on tabular datasets.
"""

import os
import numpy as np
import torch
import time
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional, List, Tuple
from marvis.utils.model_loader import model_loader, GenerationConfig

def evaluate_tabula_8b(dataset, args):
    """Evaluate Tabula-8B baseline on a dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating Tabula-8B on dataset {dataset['name']}")
    
    # Import required utilities
    from marvis.utils import (
        drop_feature_for_oom,
        is_oom_error,
        apply_feature_reduction,
        unified_llm_predict
    )
    
    # Save original CUDA device setting
    import os
    original_cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    # Set CUDA device environment variable for Tabula-8B
    if torch.cuda.is_available() and args.device != "cpu" and hasattr(args, 'gpu_index') and args.gpu_index != 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_index} for Tabula-8B")
        
        # Verify the masking worked
        logger.info(f"Available CUDA devices after masking: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")  # Should be specified GPU's name
    
    try:
        # Import Tabula-8B components
        try:
            # First try to import rtfm base module for better error diagnostics
            import rtfm
            logger.debug(f"Successfully imported rtfm from: {rtfm.__file__}")
            
            from rtfm.configs import TrainConfig, TokenizerConfig, SerializerConfig
            from rtfm.inference_utils import InferenceModel
            from rtfm.serialization.serializers import get_serializer
            from rtfm.tokenization.text import prepare_tokenizer
            logger.info("Successfully imported all RTFM dependencies")
        except ImportError as e:
            import sys
            logger.error(f"Tabula-8B dependencies not found: {e}")
            logger.error(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
            logger.error(f"Current working directory: {os.getcwd()}")
            
            # Try to find if rtfm is installed but not importable
            try:
                import pkg_resources
                installed_packages = [d.project_name for d in pkg_resources.working_set]
                rtfm_installed = any('rtfm' in pkg.lower() for pkg in installed_packages)
                logger.error(f"RTFM-related packages found: {[pkg for pkg in installed_packages if 'rtfm' in pkg.lower()]}")
            except:
                logger.error("Could not check installed packages")
            
            logger.error("Please install RTFM package: pip install git+https://github.com/penfever/rtfm.git")
            return {
                'model_name': 'Tabula-8B',
                'dataset_name': dataset['name'],
                'error': f"Missing dependencies: {e}. Install with: pip install git+https://github.com/penfever/rtfm.git"
            }
        
        # Feature selection is handled by apply_feature_reduction() in utils
        
        # Split data
        X, y = dataset["X"], dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        
        # Limit test samples if specified
        if args.max_test_samples and args.max_test_samples < len(X_test):
            X_test = X_test[:args.max_test_samples]
            y_test = y_test[:args.max_test_samples]
        
        # Apply feature selection for large feature spaces
        X_train, X_test, dataset, _ = apply_feature_reduction(
            X_train, y_train, X_test, dataset, args, logger
        )
        
        start_time = time.time()
        
        # Initialize Tabula model configuration
        train_config = TrainConfig(
            model_name=args.tabula_model, 
            context_length=args.max_context_length
        )
        tokenizer_config = TokenizerConfig()
        
        # Load model configuration
        config = AutoConfig.from_pretrained(train_config.model_name)
        config.torch_dtype = 'bfloat16'
        
        # When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the specified GPU
        if torch.cuda.is_available() and args.device != "cpu":
            device = torch.device("cuda:0")  # Always use cuda:0 since CUDA_VISIBLE_DEVICES redirects it
        else:
            device = torch.device("cpu")
        
        # Load model using centralized model loader with VLLM support
        model_kwargs = {
            'config': config,
            'low_cpu_mem_usage': True,
            'use_cache': False  # Disable KV cache to save memory
        }
        
        if torch.cuda.is_available() and args.device != "cpu":
            model_kwargs.update({
                'device_map': "auto",
                'torch_dtype': torch.bfloat16 if config.torch_dtype == 'bfloat16' else torch.float32
            })
        
        # Load using model loader (prefers VLLM for speed)
        model_wrapper = model_loader.load_llm(
            train_config.model_name,
            backend="auto",
            device=args.device,
            **model_kwargs
        )
        
        # For compatibility, get the underlying model
        model = model_wrapper.get_model()
        
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
        # Get serializer - check different ways it might be configured
        try:
            # First try to get it from the train_config
            if hasattr(train_config, 'serializer_cls'):
                serializer = get_serializer(train_config.serializer_cls)
            elif hasattr(train_config, 'serializer'):
                # It might be stored as 'serializer' instead of 'serializer_cls'
                serializer = get_serializer(train_config.serializer)
            else:
                # Create a default SerializerConfig and use it
                from rtfm.configs import SerializerConfig
                serializer_config = SerializerConfig()
                # Try to get the serializer using the config
                if hasattr(serializer_config, 'serializer_cls'):
                    serializer = get_serializer(serializer_config)
                else:
                    # If all else fails, import and instantiate directly
                    from rtfm.serialization.serializers import DefaultSerializer
                    serializer = DefaultSerializer()
        except Exception as e:
            logger.warning(f"Could not initialize serializer: {e}. Using DefaultSerializer directly.")
            from rtfm.serialization.serializers import DefaultSerializer
            serializer = DefaultSerializer()
        
        # Prepare tokenizer
        serializer_tokens = None
        if tokenizer_config.add_serializer_tokens and hasattr(serializer, 'special_tokens'):
            serializer_tokens = serializer.special_tokens
        
        tokenizer, model = prepare_tokenizer(
            model,
            tokenizer=tokenizer,
            pretrained_model_name_or_path=train_config.model_name,
            model_max_length=train_config.context_length,
            use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
            serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
            serializer_tokens=serializer_tokens,
        )
        
        # Create inference model
        inference_model = InferenceModel(model=model, tokenizer=tokenizer, serializer=serializer)
        
        # Create few-shot examples
        n_examples = min(args.num_few_shot_examples, len(X_train))
        
        # Use balanced or random selection based on args
        if getattr(args, 'balanced_few_shot', False):
            # Apply balanced few-shot selection
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            
            # Calculate examples per class (as evenly as possible)
            examples_per_class = n_examples // n_classes
            remainder = n_examples % n_classes
            
            example_indices = []
            for i, class_label in enumerate(unique_classes):
                class_mask = y_train == class_label
                class_indices = np.where(class_mask)[0]
                
                # Add one extra example to first 'remainder' classes
                n_select = examples_per_class + (1 if i < remainder else 0)
                n_select = min(n_select, len(class_indices))
                
                if n_select > 0:
                    selected_class_indices = np.random.RandomState(args.seed).choice(
                        class_indices, n_select, replace=False
                    )
                    example_indices.extend(selected_class_indices)
            
            example_indices = np.array(example_indices)
            logger.info(f"Using balanced few-shot selection: {len(example_indices)} examples ({examples_per_class}+ per class)")
        else:
            example_indices = np.random.choice(len(X_train), n_examples, replace=False)
            logger.info(f"Using random few-shot selection: {n_examples} examples")
        
        # Get target column name
        y_colname = y.name if hasattr(y, 'name') and y.name else 'target'
        
        # Handle both DataFrame and numpy array cases
        if hasattr(X_train, 'iloc'):
            # X_train is a DataFrame
            X_train_examples = X_train.iloc[example_indices].reset_index(drop=True)
        else:
            # X_train is a numpy array
            # Always ensure column count matches by using exactly the right number of names
            num_features = X_train.shape[1]
            original_feature_names = dataset["attribute_names"]
            
            # Create feature names that exactly match the number of columns
            if len(original_feature_names) == num_features:
                # Perfect match
                feature_names = original_feature_names
            elif len(original_feature_names) == num_features + 1:
                # Likely includes target column, exclude it
                feature_names = [name for name in original_feature_names if name != y_colname]
                # If filtering didn't work, use first N names
                if len(feature_names) != num_features:
                    feature_names = original_feature_names[:num_features]
            else:
                # Use first N names or create generic names
                if len(original_feature_names) >= num_features:
                    feature_names = original_feature_names[:num_features]
                else:
                    feature_names = [f"feature_{i}" for i in range(num_features)]
            
            # Final safety check
            assert len(feature_names) == num_features, f"Feature names count {len(feature_names)} != X_train columns {num_features}"
            
            X_train_examples = pd.DataFrame(X_train[example_indices], columns=feature_names)
        
        if hasattr(y_train, 'iloc'):
            # y_train is a Series
            y_train_examples = y_train.iloc[example_indices].reset_index(drop=True)
        else:
            # y_train is a numpy array
            y_train_examples = pd.Series(y_train[example_indices], name=y_colname)
        
        labeled_examples = pd.concat([X_train_examples, y_train_examples], axis=1)
        unique_classes = np.unique(y_train)
        
        # Make predictions with memory management
        predictions = []
        completed_samples = 0
        example_inputs_outputs = []  # Store example inputs and outputs for debugging
        
        # Feature dropping mechanism for OOM handling
        dropped_features = set()  # Track which features have been dropped
        num_features = len(dataset["attribute_names"])
        original_dataset = dataset.copy()
        
        # Set model to eval mode and disable gradients for inference
        if hasattr(model, 'eval'):
            model.eval()
        
        for i in range(len(X_test)):
            # Clear GPU cache periodically to prevent memory buildup
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Get the test example and format it as expected by Tabula-8B
            if hasattr(X_test, 'iloc'):
                # X_test is a DataFrame
                cur_row = X_test.iloc[i].copy()
                # Add a dummy target value (will be predicted)
                cur_row[y_colname] = unique_classes[0]  # Use first class as dummy
                target_example = pd.DataFrame(cur_row).T
            else:
                # X_test is a numpy array
                cur_row_data = X_test[i]
                # Create a Series with feature names
                if len(feature_names) == len(cur_row_data):
                    cur_row = pd.Series(cur_row_data, index=feature_names)
                else:
                    # Fallback: create series with generic names
                    cur_row = pd.Series(cur_row_data, index=[f"feature_{j}" for j in range(len(cur_row_data))])
                
                # Add dummy target value
                cur_row[y_colname] = unique_classes[0]  # Use first class as dummy
                target_example = pd.DataFrame(cur_row).T
            
            # Try prediction with OOM handling
            try:
                # Get model prediction with no gradients to save memory
                with torch.no_grad():
                    output = inference_model.predict(
                        target_example=target_example,
                        target_colname=y_colname,
                        target_choices=[str(cls) for cls in unique_classes],
                        labeled_examples=labeled_examples,
                    )
                
                # Handle different output formats from Tabula-8B
                if isinstance(output, dict) and 'prediction' in output:
                    prediction = output['prediction']
                else:
                    # Sometimes Tabula-8B returns a string directly
                    prediction = output
                
                predictions.append(prediction)
                completed_samples = i + 1
                
            except Exception as pred_error:
                # Check if this is an OOM error
                if is_oom_error(pred_error):
                    logger.warning(f"Tabula-8B prediction failed for sample {i} due to OOM: {pred_error}")
                    
                    # Try dropping a feature and retry this sample
                    if drop_feature_for_oom(dropped_features, num_features, logger):
                        logger.info(f"Retrying sample {i} with {len(dropped_features)} dropped features")
                        
                        # Re-apply feature reduction with dropped features
                        keep_indices = [idx for idx in range(num_features) if idx not in dropped_features]
                        
                        # Update datasets with dropped features
                        X_train = X_train.iloc[:, keep_indices] if hasattr(X_train, 'iloc') else X_train[:, keep_indices]
                        X_test = X_test.iloc[:, keep_indices] if hasattr(X_test, 'iloc') else X_test[:, keep_indices]
                        
                        # Update dataset info
                        dataset = original_dataset.copy()
                        dataset["attribute_names"] = [dataset["attribute_names"][idx] for idx in keep_indices]
                        if "categorical_indicator" in dataset:
                            dataset["categorical_indicator"] = [dataset["categorical_indicator"][idx] for idx in keep_indices]
                        
                        # Update feature names
                        feature_names = dataset["attribute_names"]
                        
                        # Recreate few-shot examples with reduced features
                        if hasattr(X_train, 'iloc'):
                            X_train_examples = X_train.iloc[example_indices].copy()
                        else:
                            X_train_examples = pd.DataFrame(X_train[example_indices], columns=feature_names)
                        
                        labeled_examples = pd.concat([X_train_examples, y_train_examples], axis=1)
                        
                        # Decrement i to retry this sample
                        i -= 1
                        continue
                    else:
                        logger.error("No more features to drop, cannot continue")
                        break
                else:
                    logger.error(f"Tabula-8B prediction failed for sample {i}: {pred_error}")
                    # Use default prediction
                    prediction = unique_classes[0]
                    predictions.append(prediction)
                    completed_samples = i + 1
            
            # Store example inputs and outputs for first 20 samples
            if i < 20:
                true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                example_inputs_outputs.append({
                    'sample_index': i,
                    'input_features': cur_row.drop(y_colname).to_dict() if hasattr(cur_row, 'drop') else str(cur_row),
                    'target_choices': [str(cls) for cls in unique_classes],
                    'prediction': prediction,
                    'prediction_details': output if isinstance(output, dict) else {'prediction': output},
                    'true_class': true_label,
                    'correct': str(prediction) == str(true_label) or prediction == true_label
                })
        
        # Calculate metrics on completed samples
        # Convert predictions to same type as ground truth
        predictions_converted = []
        if completed_samples > 0:
            target_type = type(y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0])
            
            for pred in predictions:
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
                    
                    predictions_converted.append(converted_pred)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert prediction '{pred}' to type {target_type}: {e}")
                    # Keep original prediction as fallback
                    predictions_converted.append(pred)
            
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            # Import shared metric calculation function
            from marvis.utils.llm_evaluation_utils import calculate_llm_metrics
            
            # Calculate all metrics using shared function (no log probabilities available for Tabula-8B)
            calculated_metrics = calculate_llm_metrics(
                y_test_partial, predictions_converted, unique_classes, 
                all_class_log_probs=None, logger=logger
            )
            
            # Extract individual metrics for backward compatibility
            accuracy = calculated_metrics['accuracy']
            balanced_acc = calculated_metrics['balanced_accuracy']
            roc_auc = calculated_metrics['roc_auc']
            f1_macro = calculated_metrics['f1_macro']
            f1_micro = calculated_metrics['f1_micro']
            f1_weighted = calculated_metrics['f1_weighted']
            precision_macro = calculated_metrics['precision_macro']
            recall_macro = calculated_metrics['recall_macro']
        else:
            predictions_converted = predictions
            accuracy = 0.0
            balanced_acc = 0.0
            roc_auc = None
            f1_macro = f1_micro = f1_weighted = None
            precision_macro = recall_macro = None
        
        # Calculate timing - LLMs don't have separate training, so only prediction_time and total_time
        total_time = time.time() - start_time
        prediction_time = total_time  # For LLMs, prediction time includes model loading and inference
        
        results = {
            'model_name': 'Tabula-8B',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id'],  # For consistency with MARVIS extraction logic
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'prediction_time': float(prediction_time),  # Time for inference (includes model loading for LLMs)
            'total_time': float(total_time),  # Same as prediction_time for LLMs (no separate training phase)
            'num_test_samples': len(X_test),
            'num_samples': len(X_train) + len(X_test),  # Total dataset size
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train.shape[1],  # Use X_train to get actual feature count after reduction
            'num_classes': len(unique_classes),
            'predictions': predictions_converted,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics to match evaluate_on_dataset
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'f1_macro': float(f1_macro) if f1_macro is not None else None,
            'f1_micro': float(f1_micro) if f1_micro is not None else None,
            'f1_weighted': float(f1_weighted) if f1_weighted is not None else None,
            'precision_macro': float(precision_macro) if precision_macro is not None else None,
            'recall_macro': float(recall_macro) if recall_macro is not None else None,
            # Tabula-8B-specific metadata
            'example_inputs_outputs': example_inputs_outputs
        }
        
        logger.info(f"Tabula-8B accuracy on {dataset['name']}: {accuracy:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating Tabula-8B: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'Tabula-8B',
            'dataset_name': dataset['name'],
            'error': str(e)
        }
    finally:
        # Restore original CUDA device setting
        if original_cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_device
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            # If it wasn't set originally, remove it
            del os.environ['CUDA_VISIBLE_DEVICES']