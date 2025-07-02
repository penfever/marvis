"""
Baseline models for tabular data evaluation.

This module provides baseline model implementations for tabular data,
including traditional ML models and LLM-based models.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from marvis.data import create_llm_dataset
from marvis.train import evaluate_llm_on_test_set


def load_embeddings_with_limit(cache_file: str, max_test_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from cache file with optional limit on test/val samples."""
    logger = logging.getLogger(__name__)
    cache = np.load(cache_file, allow_pickle=True)
    
    # Load train embeddings fully
    train_embeddings = cache["train_embeddings"]
    y_train_sample = cache["y_train_sample"]
    
    # Get full embeddings to check sizes 
    val_embeddings_full = cache["val_embeddings"]
    test_embeddings_full = cache["test_embeddings"]
    
    if max_test_samples is None:
        # Return full embeddings if no limit
        return train_embeddings, val_embeddings_full, test_embeddings_full, y_train_sample
    
    # Handle embeddings shape - can be either (n_samples, emb_size) or (n_ensemble, n_samples, emb_size)
    if len(val_embeddings_full.shape) == 3:
        # Multi-ensemble embeddings: (n_ensemble, n_samples, emb_size)
        _, val_count, _ = val_embeddings_full.shape
        _, test_count, _ = test_embeddings_full.shape
        
        # Calculate how many samples to load
        val_to_load = min(val_count, max_test_samples)
        test_to_load = min(test_count, max_test_samples)
        
        # Extract only the needed portions
        val_embeddings = val_embeddings_full[:, :val_to_load, :]
        test_embeddings = test_embeddings_full[:, :test_to_load, :]
    else:
        # Single embeddings: (n_samples, emb_size)
        val_count = len(val_embeddings_full)
        test_count = len(test_embeddings_full)
        
        # Calculate how many samples to load
        val_to_load = min(val_count, max_test_samples)
        test_to_load = min(test_count, max_test_samples)
        
        # Extract only the needed portions
        val_embeddings = val_embeddings_full[:val_to_load]
        test_embeddings = test_embeddings_full[:test_to_load]
    
    logger.info(f"Loaded embeddings with limits - Val: {val_to_load}/{val_count}, Test: {test_to_load}/{test_count}")
    
    return train_embeddings, val_embeddings, test_embeddings, y_train_sample


def create_and_evaluate_baseline_model(model_name: str, dataset: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Create and evaluate a baseline model on a dataset.
    
    Args:
        model_name: Name of the model ('catboost', 'tabpfn_v2', 'random_forest', etc.)
        dataset: Dictionary with processed dataset information
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    from marvis.data import preprocess_features
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating and evaluating {model_name} model on dataset {dataset['name']}")
    
    # Extract dataset components
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # For CatBoost, we need to preprocess the data preserving categorical features
    if model_name == 'catboost':
        # Get the categorical indicators
        categorical_indicator = dataset.get('categorical_indicator_raw', [False] * X_train.shape[1])
        
        # For CatBoost, we need to reconstruct the data preserving categorical features
        # We'll apply the preserve_categorical preprocessing to the already-split data
        
        # Apply preprocessing that preserves categorical features
        X_train_processed = preprocess_features(X_train, categorical_indicator, preserve_categorical=True)
        X_test_processed = preprocess_features(X_test, categorical_indicator, preserve_categorical=True)
        
        # Update the variables to use the categorical-preserving data
        X_train = X_train_processed
        X_test = X_test_processed
        
        logger.info(f"Using CatBoost-specific preprocessing: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    # Training data was already limited earlier at the dataset level if max_train_samples was specified
    # No additional limiting needed here - all baseline models use the same pre-limited training data
    logger.info(f"Using training data with {len(X_train)} samples for baseline model training")
    
    # Limit test samples if specified
    if args.max_test_samples and args.max_test_samples < len(X_test):
        X_test = X_test[:args.max_test_samples]
        y_test = y_test[:args.max_test_samples]
        logger.info(f"Limited test set to {args.max_test_samples} samples")
    
    start_time = time.time()
    
    # Initialize and train model based on model_name
    if model_name == 'catboost':
        try:
            if dataset['is_classification']:
                from catboost import CatBoostClassifier as CatBoostModel
            else:
                from catboost import CatBoostRegressor as CatBoostModel
            
            # Get categorical features for CatBoost
            categorical_features = []
            if 'categorical_indicator_raw' in dataset:
                # Find indices of categorical features
                max_features = X_train.shape[1]
                categorical_features = [i for i, is_cat in enumerate(dataset['categorical_indicator_raw']) 
                                      if i < max_features and is_cat]
                task_type = "classification" if dataset['is_classification'] else "regression"
                logger.info(f"CatBoost using {len(categorical_features)} categorical features for {task_type}: {categorical_features}")
                
                # Log data types for debugging
                import pandas as pd
                df_train = pd.DataFrame(X_train)
                for cat_idx in categorical_features[:5]:  # Log first 5 categorical features
                    if cat_idx < df_train.shape[1]:
                        logger.info(f"  Feature {cat_idx}: dtype={df_train.iloc[:, cat_idx].dtype}, "
                                  f"unique values={df_train.iloc[:, cat_idx].nunique()}, "
                                  f"sample values={df_train.iloc[:, cat_idx].unique()[:3].tolist()}")
            
            model = CatBoostModel(
                iterations=args.catboost_iterations,
                depth=args.catboost_depth,
                learning_rate=args.catboost_learning_rate,
                random_seed=args.seed,
                verbose=False
            )
            
            # Train the model
            model.fit(
                X_train, y_train, 
                cat_features=categorical_features,
                verbose=False
            )
            
        except ImportError:
            logger.error("CatBoost not installed. Please install it with 'pip install catboost'.")
            return {
                'model_name': model_name,
                'dataset_name': dataset['name'],
                'error': "CatBoost not installed. Please install it with 'pip install catboost'."
            }
    
    elif model_name == 'tabpfn_v2':
        try:
            if dataset['is_classification']:
                from tabpfn import TabPFNClassifier as TabPFNModel
            else:
                from tabpfn import TabPFNRegressor as TabPFNModel
            
            # Initialize the model
            task_type = "classification" if dataset['is_classification'] else "regression"
            logger.info(f"Using TabPFN v2 for {task_type}")
            # Use n_estimators=8 to align with MARVIS's TabPFN implementation
            device = args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
            n_estimators = 8  # Aligned with MARVIS's TabPFN usage
            logger.info(f"TabPFN v2 initialized with n_estimators={n_estimators}, device={device}")
            
            # Fix for quantiles issue: preprocess targets with too many unique values for regression
            y_train_processed = y_train.copy()
            y_test_processed = y_test.copy()
            target_preprocessor = None
            
            if not dataset['is_classification']:
                # Handle NaN/INF values in targets before processing
                y_train_clean = y_train.copy()
                y_test_clean = y_test.copy()
                
                # Check for NaN/INF values
                train_nan_mask = np.isnan(y_train_clean) | np.isinf(y_train_clean)
                test_nan_mask = np.isnan(y_test_clean) | np.isinf(y_test_clean)
                
                if np.any(train_nan_mask) or np.any(test_nan_mask):
                    logger.warning(f"TabPFN v2: Found {np.sum(train_nan_mask)} NaN/INF values in training targets, "
                                 f"{np.sum(test_nan_mask)} in test targets. Using median imputation.")
                    
                    # Use median imputation for NaN/INF values
                    from sklearn.impute import SimpleImputer
                    target_imputer = SimpleImputer(strategy='median')
                    
                    # Fit on training data (excluding NaN/INF) and transform both
                    y_train_clean = target_imputer.fit_transform(y_train_clean.reshape(-1, 1)).flatten()
                    y_test_clean = target_imputer.transform(y_test_clean.reshape(-1, 1)).flatten()
                
                # For regression, check if we have too many unique values relative to sample size
                unique_targets = np.unique(y_train_clean)
                n_samples = len(y_train_clean)
                n_unique = len(unique_targets)
                
                # If we have more unique values than samples * 0.5, we need to bin the targets
                # This prevents the quantiles error in TabPFN v2's internal preprocessing
                max_quantiles = min(n_samples // 2, 1000)  # Conservative limit for quantiles
                
                if n_unique > max_quantiles:
                    logger.info(f"TabPFN v2: Target has {n_unique} unique values but only {n_samples} samples. "
                               f"Binning to {max_quantiles} quantiles to prevent quantile overflow.")
                    
                    from sklearn.preprocessing import KBinsDiscretizer
                    # Use quantile-uniform strategy to preserve distribution while limiting quantiles
                    target_preprocessor = KBinsDiscretizer(
                        n_bins=max_quantiles, 
                        encode='ordinal', 
                        strategy='quantile',
                        subsample=None  # Use all data for quantile calculation
                    )
                    
                    # Fit on training data and transform both train and test
                    y_train_processed = target_preprocessor.fit_transform(y_train_clean.reshape(-1, 1)).flatten()
                    y_test_processed = target_preprocessor.transform(y_test_clean.reshape(-1, 1)).flatten()
                    
                    logger.info(f"TabPFN v2: Preprocessed target from {n_unique} to {len(np.unique(y_train_processed))} unique values")
                else:
                    logger.info(f"TabPFN v2: Target has {n_unique} unique values with {n_samples} samples - no preprocessing needed")
                    y_train_processed = y_train_clean
                    y_test_processed = y_test_clean
            
            model = TabPFNModel(
                device=device,
                n_estimators=n_estimators,
                ignore_pretraining_limits=True,
            )
            
            # Train (fit) the model with preprocessed targets
            model.fit(X_train, y_train_processed)
            
        except ImportError as e:
            logger.error(f"TabPFN v2 not properly installed: {e}")
            return {
                'model_name': model_name,
                'dataset_name': dataset['name'],
                'error': f"TabPFN v2 not properly installed: {e}"
            }
    
    elif model_name == 'random_forest':
        if dataset['is_classification']:
            from sklearn.ensemble import RandomForestClassifier as RandomForestModel
        else:
            from sklearn.ensemble import RandomForestRegressor as RandomForestModel
        
        task_type = "classification" if dataset['is_classification'] else "regression"
        logger.info(f"Using Random Forest for {task_type}")
        model = RandomForestModel(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=-1  # Use all available cores
        )
        
        # Train the model
        model.fit(X_train, y_train)
    
    elif model_name == 'gradient_boosting':
        if dataset['is_classification']:
            from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingModel
            from sklearn.feature_selection import SelectKBest, f_classif as score_func
        else:
            from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingModel
            from sklearn.feature_selection import SelectKBest, f_regression as score_func
        
        task_type = "classification" if dataset['is_classification'] else "regression"
        logger.info(f"Using Gradient Boosting for {task_type}")
        
        # Limit features to maximum of 500
        max_features = 500
        if X_train.shape[1] > max_features:
            logger.info(f"Limiting gradient boosting features from {X_train.shape[1]} to {max_features}")
            feature_selector = SelectKBest(score_func=score_func, k=max_features)
            X_train_selected = feature_selector.fit_transform(X_train, y_train)
            X_test = feature_selector.transform(X_test)
            logger.info(f"Selected {X_train_selected.shape[1]} features for gradient boosting")
        else:
            X_train_selected = X_train
            logger.info(f"Using all {X_train.shape[1]} features for gradient boosting (under limit)")
        
        model = GradientBoostingModel(
            n_estimators=args.gb_n_estimators,
            learning_rate=args.gb_learning_rate,
            random_state=args.seed
        )
        
        # Train the model
        model.fit(X_train_selected, y_train)
    
    elif model_name == 'logistic_regression':
        from sklearn.preprocessing import StandardScaler
        
        if dataset['is_classification']:
            from sklearn.linear_model import LogisticRegression as LinearModel
            task_type = "classification"
        else:
            from sklearn.linear_model import LinearRegression as LinearModel
            task_type = "regression"
        
        logger.info(f"Using Linear model for {task_type}")
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if dataset['is_classification']:
            model = LinearModel(
                max_iter=args.lr_max_iter,
                C=args.lr_C,
                random_state=args.seed,
                n_jobs=-1  # Use all available cores
            )
        else:
            model = LinearModel(
                n_jobs=-1  # Use all available cores
            )
        
        # Train the model
        model.fit(X_train_scaled, y_train)
    
    else:
        logger.error(f"Unknown model name: {model_name}")
        return {
            'model_name': model_name,
            'dataset_name': dataset['name'],
            'error': f"Unknown model name: {model_name}"
        }
    
    # Measure training time
    training_time = time.time() - start_time
    logger.info(f"Training time for {model_name}: {training_time:.2f} seconds")
    
    # Make predictions
    start_time = time.time()
    
    if model_name == 'logistic_regression':
        # Use the scaled test data
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if dataset['is_classification'] and hasattr(model, 'predict_proba') else None
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if dataset['is_classification'] and hasattr(model, 'predict_proba') else None
    
    # For TabPFN v2 regression with preprocessed targets, inverse transform the predictions
    if model_name == 'tabpfn_v2' and not dataset['is_classification'] and 'target_preprocessor' in locals() and target_preprocessor is not None:
        logger.info(f"TabPFN v2: Inverse transforming predictions from binned space back to original target space")
        
        # Inverse transform predictions back to original scale
        # TabPFN outputs ordinal bin indices, we need to map them back to target values
        
        # Get the bin centers from the preprocessor
        bin_edges = target_preprocessor.bin_edges_[0]  # Get bin edges for the single feature
        
        # Map ordinal predictions to bin centers
        y_pred_original_scale = []
        for pred in y_pred:
            bin_idx = int(np.clip(pred, 0, len(bin_edges) - 2))  # Ensure valid bin index
            # Use bin center as the predicted value
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            y_pred_original_scale.append(bin_center)
        
        y_pred = np.array(y_pred_original_scale)
        logger.info(f"TabPFN v2: Transformed {len(y_pred)} predictions back to original scale")
        
        # Also need to use original scale targets for evaluation
        y_test_for_eval = y_test  # Original targets for evaluation
    else:
        y_test_for_eval = y_test
    
    # Measure prediction time
    prediction_time = time.time() - start_time
    logger.info(f"Prediction time for {model_name}: {prediction_time:.2f} seconds")
    
    # Calculate metrics based on task type
    if dataset['is_classification']:
        # Classification metrics
        accuracy = accuracy_score(y_test_for_eval, y_pred)
        
        # Calculate balanced accuracy
        try:
            balanced_acc = balanced_accuracy_score(y_test_for_eval, y_pred)
            logger.info(f"Accuracy for {model_name} on {dataset['name']}: {accuracy:.4f}, Balanced accuracy: {balanced_acc:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute balanced accuracy: {e}")
            balanced_acc = None
            logger.info(f"Accuracy for {model_name} on {dataset['name']}: {accuracy:.4f}")
        
        # Calculate ROC AUC if probabilities are available
        roc_auc = None
        if y_prob is not None:
            try:
                # Get unique classes
                unique_classes = np.unique(y_test_for_eval)
                
                # For binary classification
                if len(unique_classes) == 2:
                    # Get probabilities for the positive class (usually class 1)
                    pos_class_idx = 1 if 1 in unique_classes else unique_classes[1]
                    binary_truth = np.array([1 if y == pos_class_idx else 0 for y in y_test_for_eval])
                    
                    # Check if class index exists in probability array
                    if y_prob.shape[1] > pos_class_idx:
                        binary_probs = y_prob[:, pos_class_idx]
                        roc_auc = roc_auc_score(binary_truth, binary_probs)
                        logger.info(f"ROC AUC for {model_name} on {dataset['name']}: {roc_auc:.4f}")
                # For multi-class classification
                elif len(unique_classes) > 2:
                    # Use one-vs-rest approach
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(y_test_for_eval, classes=unique_classes)
                    
                    # Make sure we have probabilities for all classes
                    if y_prob.shape[1] >= len(unique_classes):
                        # Get probabilities for the classes that are present
                        probs_array = np.array([y_prob[:, i] for i in unique_classes]).T
                        roc_auc = roc_auc_score(y_bin, probs_array, multi_class='ovr')
                        logger.info(f"ROC AUC (OVR) for {model_name} on {dataset['name']}: {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Generate classification report and confusion matrix
        try:
            report = classification_report(y_test_for_eval, y_pred, output_dict=True)
            cm = confusion_matrix(y_test_for_eval, y_pred)
            logger.info(f"Generated classification report and confusion matrix for {model_name}")
        except Exception as e:
            logger.warning(f"Could not generate detailed metrics: {e}")
            report = None
            cm = None
        
        # Compute frequency distributions for classification
        prediction_distribution = None
        ground_truth_distribution = None
        
        # Log basic distribution info
        unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
        unique_true, true_counts = np.unique(y_test_for_eval, return_counts=True)
        logger.info(f"Prediction distribution: {dict(zip(unique_pred, pred_counts))}")
        logger.info(f"Ground truth distribution: {dict(zip(unique_true, true_counts))}")
        
        # Store classification-specific metrics
        mse = None
        rmse = None
        mae = None
        r2 = None
        
    else:
        # Regression metrics
        mse = mean_squared_error(y_test_for_eval, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_for_eval, y_pred)
        r2 = r2_score(y_test_for_eval, y_pred)
        
        logger.info(f"Regression metrics for {model_name} on {dataset['name']}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # For regression, these classification metrics are not applicable
        accuracy = None
        balanced_acc = None
        roc_auc = None
        report = None
        cm = None
        prediction_distribution = None
        ground_truth_distribution = None
    
    # Calculate total time
    total_time = training_time + prediction_time
    
    # Return results
    results = {
        'model_name': model_name,
        'dataset_name': dataset['name'],
        'dataset_id': dataset['id'],
        'task_type': 'classification' if dataset['is_classification'] else 'regression',
        'training_time': float(training_time),
        'prediction_time': float(prediction_time),
        'total_time': float(total_time),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'num_features': X_train.shape[1],
        'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        'ground_truth': y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    }
    
    # Add task-specific metrics
    if dataset['is_classification']:
        results.update({
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc) if balanced_acc is not None else None,
            'num_classes': len(np.unique(y_train)),
            'prediction_distribution': prediction_distribution,
            'ground_truth_distribution': ground_truth_distribution,
        })
        
        # Add ROC AUC if available
        if roc_auc is not None:
            results['roc_auc'] = float(roc_auc)
        
        if report is not None:
            results['classification_report'] = report
        
        if cm is not None:
            results['confusion_matrix'] = cm.tolist()
    else:
        results.update({
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'target_range': [float(y_test.min()), float(y_test.max())],
            'target_mean': float(y_test.mean()),
            'target_std': float(y_test.std())
        })
    
    return results


def create_and_evaluate_llm_model(model_id: str, dataset: Dict[str, Any], args, cached_model: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Create and evaluate an LLM model from model_id on a dataset.
    
    Args:
        model_id: Hugging Face model ID
        dataset: Dictionary with processed dataset information
        args: Command line arguments
        cached_model: Optional tuple of (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids)
        
    Returns:
        Dictionary with evaluation results
    """
    from marvis.models import prepare_qwen_with_prefix_embedding
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating and evaluating LLM model {model_id} on dataset {dataset['name']}")
    
    start_time = time.time()
    
    # 1. Prepare the LLM model
    if cached_model is not None:
        # Use the cached model
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = cached_model
        logger.info(f"Using cached model {model_id}")
    else:
        # Load the model (backward compatibility)
        try:
            model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
                embedding_size=args.embedding_size,
                model_id=model_id
            )
            
            logger.info(f"Successfully loaded model {model_id}")
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return {
                'model_name': model_id,
                'dataset_name': dataset['name'],
                'error': f"Error loading model: {e}"
            }
    
    # 2. Create dataset for evaluation
    dataset_output_dir = os.path.join(args.output_dir, f"dataset_{dataset['id']}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        dataset["X_train"], dataset["y_train_sample"], 
        dataset["X_val"], dataset["y_val"], 
        dataset["X_test"], dataset["y_test"],
        dataset["train_embeddings"], dataset["val_embeddings"], dataset["test_embeddings"],
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir=dataset_output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )
    
    # 3. Evaluate on test set
    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    results = evaluate_llm_on_test_set(
        model, tokenizer, test_dataset,
        label_encoder, prefix_start_id, prefix_end_id,
        class_token_ids, prefix_data_file, 
        max_test_samples=args.max_test_samples,
        allowed_classes=None  # Auto-detect from test dataset
    )
    
    # Total time
    total_time = time.time() - start_time
    
    # Get predictions and ground truth
    y_test = dataset["y_test"][:len(test_dataset)]
    predictions = results.get('predictions', [])
    
    # Compute frequency distributions
    if len(predictions) > 0:
        prediction_distribution = None
        ground_truth_distribution = None
        
        # Log basic distribution info
        unique_pred, pred_counts = np.unique(predictions, return_counts=True)
        unique_true, true_counts = np.unique(y_test, return_counts=True)
        logger.info(f"Prediction distribution: {dict(zip(unique_pred, pred_counts))}")
        logger.info(f"Ground truth distribution: {dict(zip(unique_true, true_counts))}")
    else:
        prediction_distribution = None
        ground_truth_distribution = None
    
    # Format results
    formatted_results = {
        'model_name': model_id,
        'dataset_name': dataset['name'],
        'dataset_id': dataset['id'],
        'accuracy': float(results['accuracy']),
        'total_time': float(total_time),
        'num_train_samples': len(dataset['X_train']),
        'num_test_samples': len(dataset['X_test']),
        'num_features': dataset['X_train'].shape[1],
        'num_classes': len(np.unique(dataset['y_train']))
    }
    
    # Add frequency distributions
    if prediction_distribution is not None:
        formatted_results['prediction_distribution'] = prediction_distribution
        formatted_results['ground_truth_distribution'] = ground_truth_distribution
        formatted_results['predictions'] = predictions
        formatted_results['ground_truth'] = y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    
    # Add classification report if available
    if 'classification_report' in results:
        formatted_results['classification_report'] = results['classification_report']
    
    # Add confusion matrix if available
    if 'confusion_matrix' in results:
        formatted_results['confusion_matrix'] = results['confusion_matrix'].tolist() if not isinstance(results['confusion_matrix'], list) else results['confusion_matrix']
    
    logger.info(f"Evaluation complete for {model_id}. Accuracy: {formatted_results['accuracy']:.4f}")
    
    return formatted_results