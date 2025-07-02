"""
Training functions for MARVIS models.
"""

import torch
import numpy as np
import os
import logging
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from transformers import AutoModelForCausalLM

from .save_utils import save_checkpoint, save_final_model, should_save_checkpoint

logger = logging.getLogger(__name__)

def train_llm_with_tabpfn_embeddings(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    prefix_start_id: int,
    prefix_end_id: int,
    class_token_ids: List[int],
    prefix_data_file: str,
    output_dir: str = "./tabpfn_llm_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_train_samples: Optional[int] = None,
    lr_initial: float = 5e-5,
    lr_final: float = 1e-5,
    mixup_alpha: float = 0.0,
    min_freq_weight: float = 0.05,
    min_freq_target: float = 0.05,
    save_best_model: bool = True,
    checkpoint_interval: int = 500,
    save_steps: int = 500,
    save_total_limit: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_threshold: float = 0.5,
    permute_examples: bool = False,
    permute_labels: bool = False,
    permute_labels_every_k_steps: Optional[int] = None,
    no_permute_last_k: Optional[int] = None,
    variable_few_shot: bool = False,
    few_shot_min: int = 10,
    few_shot_max: Optional[int] = None,
    wandb_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    resume_from_checkpoint: Optional[str] = None,
    resume_optimizer: bool = False,
    max_steps: Optional[int] = None,
    temperature_initial: float = 1.0,
    temperature_final: float = 0.1,
    temperature_anneal_steps: Optional[int] = None,
    gradient_penalty_weight: float = 0.0,
    gradient_penalty_threshold: float = 10.0,
    unfreeze_last_k_layers: int = 0,
    # Component-specific learning rates
    vq_lr_initial: Optional[float] = None,
    vq_lr_final: Optional[float] = None,
    vq_lr_scheduler_type: Optional[str] = None,
    class_token_lr_initial: Optional[float] = None,
    class_token_lr_final: Optional[float] = None,
    class_token_lr_scheduler_type: Optional[str] = None,
    llm_lr_initial: Optional[float] = None,
    llm_lr_final: Optional[float] = None,
    llm_lr_scheduler_type: Optional[str] = None,
    # Scheduler configs (will be passed to all schedulers)
    lr_scheduler_config: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.Module, Any, List[int]]:
    """
    Train the LLM with TabPFN embeddings as prefix using a custom training loop.
    Uses class tokens for autoregressive prediction and explicitly includes query embeddings.
    Implements a two-phase learning rate schedule, mixup data augmentation, minimum
    frequency regularization, and best model checkpointing.

    Args:
        model: QwenWithPrefixEmbedding model
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
        prefix_data_file: Path to the saved prefix data
        output_dir: Directory to save the model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps for gradient accumulation
        max_train_samples: Maximum number of training samples to use
        lr_initial: Initial learning rate
        lr_final: Final learning rate
        mixup_alpha: Alpha parameter for mixup augmentation
        min_freq_weight: Weight for minimum frequency regularization
        min_freq_target: Target minimum frequency for each class
        save_best_model: Whether to save the best model by training loss
        checkpoint_interval: Save checkpoints every this many steps (deprecated, use save_steps)
        save_steps: Save a checkpoint every this many steps
        save_total_limit: Maximum number of checkpoints to keep (delete oldest)
        early_stopping_patience: Number of steps with no improvement before stopping
        early_stopping_threshold: Minimum loss threshold to start tracking early stopping (only activate
                                 patience counter when batch loss is below this threshold)
        permute_examples: Whether to randomly permute the order of few-shot examples each epoch
                         to discourage memorization
        permute_labels: Whether to randomly permute the class-to-label mapping each epoch
                       to discourage memorization
        permute_labels_every_k_steps: When using max_steps, permute labels every k steps instead of every epoch.
                                     If None, defaults to epoch-based permutation
        no_permute_last_k: If set, disables label permutation for the last k steps when permute_labels is True
        variable_few_shot: Whether to randomly vary the number of few-shot examples during
                          training to improve generalization
        few_shot_min: Minimum number of few-shot examples when using variable_few_shot
        few_shot_max: Maximum number of few-shot examples when using variable_few_shot
                     (defaults to all available if None)
        wandb_callback: Optional callback function for logging metrics to Weights & Biases
        resume_from_checkpoint: Path to a checkpoint directory to resume training from
        resume_optimizer: Whether to resume optimizer state when resuming from checkpoint
        max_steps: Maximum number of training steps. If provided, overrides num_train_epochs
                  and training will stop after this many steps
        temperature_initial: Initial temperature for scaling label logits (default: 1.0)
        temperature_final: Final temperature for scaling label logits (default: 0.1)
        temperature_anneal_steps: Number of steps to anneal temperature over. If None, uses total_steps
        gradient_penalty_weight: Weight for gradient penalty on embeddings (default: 0.0, disabled)
        gradient_penalty_threshold: Gradient norm threshold above which to apply penalty (default: 10.0)
        unfreeze_last_k_layers: Number of last layers to keep unfrozen (default: 0, freeze entire LLM).
                               If k=0, freezes the entire base LLM while keeping embedding_projector trainable.
                               If k>0, unfreezes the last k transformer layers.

    Returns:
        trained_model: Trained model
        tokenizer: Tokenizer for the model
        final_class_token_ids: The class token IDs used in the final epoch (may be permuted)
    """
    logger.info("Setting up custom training loop")
    
    # Load prefix data
    prefix_data = np.load(prefix_data_file)
    original_prefix_embeddings = prefix_data['embeddings']
    original_prefix_class_labels = prefix_data['class_labels']

    # Convert to tensors initially, but we'll permute them later if needed
    prefix_embeddings_tensor = torch.tensor(original_prefix_embeddings, dtype=torch.float32)
    prefix_class_labels_tensor = torch.tensor(original_prefix_class_labels, dtype=torch.long)

    # Set the maximum number of few-shot examples if not provided
    if few_shot_max is None:
        few_shot_max = len(original_prefix_embeddings)

    # Log initial configurations
    logger.info(f"Loaded prefix data from {prefix_data_file}")
    logger.info(f"Embeddings shape: {prefix_embeddings_tensor.shape}, Class labels shape: {prefix_class_labels_tensor.shape}")

    # Log augmentation settings
    if permute_examples:
        logger.info("Example permutation: ENABLED - Example order will be randomly shuffled each epoch")
    else:
        logger.info("Example permutation: DISABLED - Example order will remain fixed")

    if permute_labels:
        logger.info("Label permutation: ENABLED - Class-to-label mapping will be randomly permuted each epoch")
    else:
        logger.info("Label permutation: DISABLED - Class-to-label mapping will remain fixed")

    if variable_few_shot:
        logger.info(f"Variable few-shot: ENABLED - Random number between {few_shot_min} and {few_shot_max} examples")
    else:
        logger.info(f"Variable few-shot: DISABLED - Using fixed {len(original_prefix_embeddings)} examples")
    
    # Determine the number of classes from the dataset
    all_labels = [example["label_id"] for example in train_dataset]
    num_classes = len(set(all_labels))
    logger.info(f"Dataset has {num_classes} classes")
    
    # Initialize tracking for class prediction frequencies - use max possible classes (len of token IDs)
    max_possible_classes = len(class_token_ids)
    class_predictions = {c: 0 for c in range(max_possible_classes)}
    total_predictions = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables for tracking the best model
    best_loss = float('inf')
    best_model_step = 0
    running_loss = 0.0
    running_loss_count = 0
    no_improve_steps = 0
    initial_epoch = 0
    
    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        
        # Load model and tokenizer from checkpoint
        if os.path.exists(resume_from_checkpoint):
            # Get the training state if available
            training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
            
            if os.path.exists(training_state_path):
                logger.info(f"Loading training state from {training_state_path}")
                training_state = torch.load(training_state_path)
                
                # Resume training state
                best_model_step = training_state.get('global_step', 0)
                running_loss = training_state.get('running_loss', 0.0)
                running_loss_count = training_state.get('running_loss_count', 0)
                class_predictions = training_state.get('class_predictions', {c: 0 for c in range(max_possible_classes)})
                total_predictions = training_state.get('total_predictions', 0)
                
                # Calculate initial epoch based on global step
                global_step = training_state.get('global_step', 0)
                steps_per_epoch = len(train_dataset) / per_device_train_batch_size / gradient_accumulation_steps
                initial_epoch = int(global_step / steps_per_epoch)
                
                logger.info(f"Resuming from step {global_step}, estimated epoch {initial_epoch}")
                
                # Get the best loss if available
                best_checkpoint_path = os.path.join(output_dir, "best_model", "training_state.pt")
                if os.path.exists(best_checkpoint_path):
                    best_state = torch.load(best_checkpoint_path)
                    best_loss = best_state.get('running_loss', float('inf')) / max(1, best_state.get('running_loss_count', 1))
                    logger.info(f"Best loss from previous run: {best_loss:.4f}")
                
                # Log the resumed state
                if wandb_callback is not None:
                    wandb_callback({
                        "train/resumed_step": global_step,
                        "train/resumed_epoch": initial_epoch,
                        "train/resumed_loss": running_loss / max(1, running_loss_count),
                        "train/best_loss": best_loss
                    })
        else:
            logger.warning(f"Checkpoint directory {resume_from_checkpoint} not found, starting from scratch")
    
    # If we're truncating, ensure class balance
    if max_train_samples and max_train_samples < len(train_dataset):
        # Get all examples by class
        examples_by_class = {}
        for i, example in enumerate(train_dataset):
            label = example["label_id"]
            if label not in examples_by_class:
                examples_by_class[label] = []
            examples_by_class[label].append(i)
        
        # Calculate how many examples to take from each class
        examples_per_class = max_train_samples // num_classes
        remainder = max_train_samples % num_classes
        
        # Select indices ensuring class balance
        selected_indices = []
        for label, indices in examples_by_class.items():
            # Take examples_per_class + 1 extra if we have remainder
            n_to_take = examples_per_class + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0
            
            # Shuffle indices for this class
            np.random.shuffle(indices)
            
            # Take up to n_to_take or all available
            selected_indices.extend(indices[:min(n_to_take, len(indices))])
        
        # Select the examples
        train_dataset = train_dataset.select(selected_indices)
        
        # Log class distribution in truncated dataset
        trunc_labels = [train_dataset[i]["label_id"] for i in range(len(train_dataset))]
        trunc_unique, trunc_counts = np.unique(trunc_labels, return_counts=True)
        logger.info(f"TRUNCATED training dataset to {len(train_dataset)} examples with class distribution: {dict(zip(trunc_unique, trunc_counts))}")
    
    # Create dataloaders with shuffle=True
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=per_device_train_batch_size,
        shuffle=True,  # Ensure shuffling
        collate_fn=lambda examples: {k: [ex[k] for ex in examples] for k in examples[0]}
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=per_device_train_batch_size,
        collate_fn=lambda examples: {k: [ex[k] for ex in examples] for k in examples[0]}
    )
    
    # Calculate total training steps for learning rate scheduler
    total_steps = max_steps if max_steps is not None else len(train_dataloader) * num_train_epochs // gradient_accumulation_steps
    switch_point = int(0.4 * total_steps)  # Point at which to switch from base_rate_1 to decay
    
    logger.info(f"Training for {total_steps} total steps with learning rate schedule:")
    logger.info(f"  - First {switch_point} steps ({switch_point/total_steps:.1%}): Fixed rate {lr_initial}")
    logger.info(f"  - Remaining steps: Exponential decay from {lr_initial} to {lr_final}")
    logger.info(f"  - Temperature annealing: {temperature_initial} -> {temperature_final} over {temperature_anneal_steps} steps")
    logger.info(f"  - Gradient penalty: weight={gradient_penalty_weight}, threshold={gradient_penalty_threshold}")
    logger.info(f"  - Layer freezing: {'Freeze entire base LLM' if unfreeze_last_k_layers == 0 else f'Unfreeze last {unfreeze_last_k_layers} layers'}")
    logger.info(f"  - Mixup augmentation with alpha={mixup_alpha}")
    logger.info(f"  - Minimum frequency regularization with weight={min_freq_weight}, target={min_freq_target}")
    logger.info(f"  - {'Saving best model by training loss' if save_best_model else 'Saving final model only'}")
    
    # Implement layer freezing
    def freeze_base_model_layers(model, unfreeze_last_k):
        """
        Freeze all but the last k layers of the base model.
        
        When unfreeze_last_k=0 (default), freezes the entire base model including:
        - All transformer layers 
        - Token embeddings (input embeddings)
        - Class token embeddings (<CLASS_0> through <CLASS_9>)
        - Output layer (lm_head)
        
        The embedding_projector remains trainable regardless of this setting.
        
        Args:
            model: QwenWithPrefixEmbedding model
            unfreeze_last_k: Number of last layers to keep unfrozen
        """
        # First, freeze all parameters in the base model
        # This includes token embeddings (input embeddings) and class tokens when unfreeze_last_k is 0
        for param in model.base_model.parameters():
            param.requires_grad = False
        
        # Keep embedding_projector trainable (it's not part of base_model)
        for param in model.embedding_projector.parameters():
            param.requires_grad = True
            
        # When unfreeze_last_k_layers is 0, explicitly ensure token embeddings are frozen
        if unfreeze_last_k == 0:
            # Token embeddings (including class tokens) are already frozen above as part of base_model.parameters()
            # but let's be explicit about this for clarity
            if hasattr(model.base_model, 'get_input_embeddings'):
                input_embeddings = model.base_model.get_input_embeddings()
                for param in input_embeddings.parameters():
                    param.requires_grad = False
                logger.info("Explicitly froze token embeddings including class tokens (unfreeze_last_k_layers=0)")
            else:
                logger.info("Token embeddings including class tokens are frozen (unfreeze_last_k_layers=0)")
            
        if unfreeze_last_k > 0:
            # Identify the transformer layers in the base model
            # For Qwen models, layers are typically in model.layers
            if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'layers'):
                layers = model.base_model.model.layers
                num_layers = len(layers)
                
                # Unfreeze the last k layers
                start_idx = max(0, num_layers - unfreeze_last_k)
                for i in range(start_idx, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                
                logger.info(f"Unfroze last {min(unfreeze_last_k, num_layers)} layers (layers {start_idx}-{num_layers-1})")
                
                # Also unfreeze the output layer (lm_head) if we're unfreezing any layers
                if hasattr(model.base_model, 'lm_head'):
                    for param in model.base_model.lm_head.parameters():
                        param.requires_grad = True
                    logger.info("Also unfroze lm_head (output layer)")
            else:
                logger.warning("Could not identify transformer layers structure. Attempting alternative approach.")
                # Alternative: look for sequential modules containing "layer" in the name
                all_modules = list(model.base_model.named_modules())
                layer_modules = [(name, module) for name, module in all_modules if 'layer' in name.lower() and hasattr(module, 'parameters')]
                
                if layer_modules:
                    # Sort by name to ensure we get the last k
                    layer_modules.sort(key=lambda x: x[0])
                    start_idx = max(0, len(layer_modules) - unfreeze_last_k)
                    
                    for i in range(start_idx, len(layer_modules)):
                        name, module = layer_modules[i]
                        for param in module.parameters():
                            param.requires_grad = True
                        logger.debug(f"Unfroze layer: {name}")
                else:
                    logger.warning("Could not find any layer modules. Base model remains fully frozen.")
        
        # Log the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Layer freezing complete: {trainable_params:,} / {total_params:,} parameters are trainable ({trainable_params/total_params*100:.1f}%)")
    
    # Apply layer freezing
    freeze_base_model_layers(model, unfreeze_last_k_layers)
    
    # Set model to eval mode for frozen layers to save memory
    if unfreeze_last_k_layers == 0:
        # If entire base model is frozen, put it in eval mode
        model.base_model.eval()
        logger.info("Set base model to eval mode (entire model frozen)")
    
    # Set up optimizer with parameter groups for different components
    # Use provided component-specific LRs or fall back to defaults
    vq_lr_init = vq_lr_initial if vq_lr_initial is not None else lr_initial
    vq_lr_fin = vq_lr_final if vq_lr_final is not None else lr_final
    class_token_lr_init = class_token_lr_initial if class_token_lr_initial is not None else lr_initial
    class_token_lr_fin = class_token_lr_final if class_token_lr_final is not None else lr_final
    llm_lr_init = llm_lr_initial if llm_lr_initial is not None else lr_initial
    llm_lr_fin = llm_lr_final if llm_lr_final is not None else lr_final
    
    # Identify parameter groups
    vq_params = []
    class_token_params = []
    llm_params = []
    
    # Check if model has VQ components
    has_vq = hasattr(model, 'vq_layer') or hasattr(model, 'vector_quantizer')
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if has_vq and ('vq_layer' in name or 'vector_quantizer' in name or 'codebook' in name):
            vq_params.append(param)
            logger.debug(f"VQ parameter: {name}")
        elif 'class_token' in name or 'class_embed' in name:
            class_token_params.append(param)
            logger.debug(f"Class token parameter: {name}")
        else:
            llm_params.append(param)
            logger.debug(f"LLM parameter: {name}")
    
    # Create parameter groups
    param_groups = []
    if vq_params:
        param_groups.append({
            'params': vq_params,
            'lr': vq_lr_init,
            'name': 'vq'
        })
        logger.info(f"VQ parameter group: {len(vq_params)} parameters, lr={vq_lr_init}")
    
    if class_token_params:
        param_groups.append({
            'params': class_token_params,
            'lr': class_token_lr_init,
            'name': 'class_tokens'
        })
        logger.info(f"Class token parameter group: {len(class_token_params)} parameters, lr={class_token_lr_init}")
    
    if llm_params:
        param_groups.append({
            'params': llm_params,
            'lr': llm_lr_init,
            'name': 'llm'
        })
        logger.info(f"LLM parameter group: {len(llm_params)} parameters, lr={llm_lr_init}")
    
    # Fall back to simple optimizer if no parameter groups
    if not param_groups:
        logger.warning("No parameter groups created, using default optimizer")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=lr_initial
        )
    else:
        optimizer = torch.optim.AdamW(param_groups)
    
    # Custom learning rate scheduler that handles multiple parameter groups
    class MultiGroupLRScheduler:
        def __init__(self, optimizer, lr_configs, total_steps, current_step=0):
            """
            lr_configs: dict mapping group names to (lr_initial, lr_final, scheduler_type) tuples
            """
            self.optimizer = optimizer
            self.lr_configs = lr_configs
            self.total_steps = total_steps
            self.current_step = current_step
            self.switch_point = int(0.4 * total_steps)  # For legacy scheduler compatibility
            
        def _compute_lr(self, lr_initial, lr_final, scheduler_type=None):
            """Compute learning rate based on scheduler type"""
            if scheduler_type is None or scheduler_type == 'legacy':
                # Legacy two-phase scheduler
                if self.current_step <= self.switch_point:
                    return lr_initial
                else:
                    progress = (self.current_step - self.switch_point) / (self.total_steps - self.switch_point)
                    if lr_initial == lr_final:
                        return lr_initial
                    decay_factor = (lr_final / lr_initial) ** progress
                    return lr_initial * decay_factor
            
            elif scheduler_type == 'constant':
                return lr_initial
                
            elif scheduler_type == 'linear':
                progress = self.current_step / self.total_steps
                return lr_initial + (lr_final - lr_initial) * progress
                
            elif scheduler_type == 'cosine':
                progress = self.current_step / self.total_steps
                return lr_final + (lr_initial - lr_final) * 0.5 * (1 + np.cos(np.pi * progress))
                
            elif scheduler_type == 'exponential':
                if lr_initial == lr_final or lr_initial == 0:
                    return lr_initial
                decay_rate = np.log(lr_final / lr_initial) / self.total_steps
                return lr_initial * np.exp(decay_rate * self.current_step)
                
            else:
                # Default to legacy scheduler
                return self._compute_lr(lr_initial, lr_final, 'legacy')

        def step(self):
            self.current_step += 1
            
            # Update each parameter group
            for param_group in self.optimizer.param_groups:
                group_name = param_group.get('name', 'default')
                
                if group_name in self.lr_configs:
                    lr_initial, lr_final, scheduler_type = self.lr_configs[group_name]
                    lr = self._compute_lr(lr_initial, lr_final, scheduler_type)
                else:
                    # Fall back to first config or default
                    if self.lr_configs:
                        lr_initial, lr_final, scheduler_type = list(self.lr_configs.values())[0]
                        lr = self._compute_lr(lr_initial, lr_final, scheduler_type)
                    else:
                        lr = param_group['lr']  # Keep current LR
                
                param_group['lr'] = lr
            
            return self.get_last_lr()
        
        def get_last_lr(self):
            """Return list of learning rates for all parameter groups"""
            return [group['lr'] for group in self.optimizer.param_groups]
    
    # Temperature annealing scheduler
    class TemperatureScheduler:
        def __init__(self, temp_initial, temp_final, anneal_steps, current_step=0):
            self.temp_initial = temp_initial
            self.temp_final = temp_final
            self.anneal_steps = anneal_steps
            self.current_step = current_step
            
        def get_temperature(self):
            if self.current_step >= self.anneal_steps:
                return self.temp_final
            
            # Linear annealing
            progress = self.current_step / self.anneal_steps
            temp = self.temp_initial + (self.temp_final - self.temp_initial) * progress
            return temp
            
        def step(self):
            self.current_step += 1
            return self.get_temperature()
    
    # If resuming and optimizer state exists, load it
    global_step = 0
    if resume_from_checkpoint and resume_optimizer:
        training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            if 'optimizer' in training_state:
                logger.info("Loading optimizer state from checkpoint")
                optimizer.load_state_dict(training_state['optimizer'])
            
            global_step = training_state.get('global_step', 0)
            scheduler_step = training_state.get('scheduler_step', 0)
            
            logger.info(f"Resuming from global step {global_step}, scheduler step {scheduler_step}")
        else:
            logger.warning("No optimizer state found in checkpoint, starting with fresh optimizer")
    
    # Calculate total training steps for learning rate scheduler
    total_steps = max_steps if max_steps is not None else len(train_dataloader) * num_train_epochs // gradient_accumulation_steps
    switch_point = int(0.4 * total_steps)  # Point at which to switch from base_rate_1 to decay
    
    # Create scheduler with potentially resumed step
    current_step = 0
    temp_current_step = 0
    if resume_from_checkpoint and resume_optimizer:
        training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            current_step = training_state.get('scheduler_step', 0)
            temp_current_step = training_state.get('temp_scheduler_step', current_step)
    
    # Create lr_configs for scheduler
    lr_configs = {}
    
    # Check which parameter groups we have
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', 'default')
        
        if group_name == 'vq':
            scheduler_type = vq_lr_scheduler_type or 'legacy'
            lr_configs['vq'] = (vq_lr_init, vq_lr_fin, scheduler_type)
        elif group_name == 'class_tokens':
            scheduler_type = class_token_lr_scheduler_type or 'legacy'  
            lr_configs['class_tokens'] = (class_token_lr_init, class_token_lr_fin, scheduler_type)
        elif group_name == 'llm':
            scheduler_type = llm_lr_scheduler_type or 'legacy'
            lr_configs['llm'] = (llm_lr_init, llm_lr_fin, scheduler_type)
        else:
            # Default/fallback
            lr_configs[group_name] = (lr_initial, lr_final, 'legacy')
    
    # If no parameter groups were created, use default config
    if not lr_configs:
        lr_configs['default'] = (lr_initial, lr_final, 'legacy')
    
    scheduler = MultiGroupLRScheduler(
        optimizer,
        lr_configs=lr_configs,
        total_steps=total_steps,
        current_step=current_step
    )
    
    # Initialize temperature scheduler
    if temperature_anneal_steps is None:
        temperature_anneal_steps = total_steps
    
    temp_scheduler = TemperatureScheduler(
        temp_initial=temperature_initial,
        temp_final=temperature_final,
        anneal_steps=temperature_anneal_steps,
        current_step=temp_current_step
    )
    
    # For backward compatibility, use save_steps if provided, otherwise use checkpoint_interval
    effective_save_steps = save_steps or checkpoint_interval
    
    # Helper function to save a checkpoint using the new utility
    def save_model_checkpoint(step, is_best=False):
        """Local wrapper around save_checkpoint utility to include training state"""
        # Prepare training state
        training_state = {
            'scheduler_step': scheduler.current_step,
            'temp_scheduler_step': temp_scheduler.current_step,
            'running_loss': running_loss,
            'running_loss_count': running_loss_count,
            'class_predictions': class_predictions,
            'total_predictions': total_predictions
        }
        
        # Call the utility function
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            class_token_ids=class_token_ids,
            is_best=is_best,
            save_optimizer=True
        )
        
        if is_best:
            logger.debug(f"Saved new best model at step {step} with avg loss {best_loss:.4f}")
    
    # Set up training loop
    device = next(model.parameters()).device
    logger.info(f"Training on device: {device}")
    
    # Set training mode, but keep base model in eval if fully frozen
    if unfreeze_last_k_layers == 0:
        model.base_model.eval()  # Keep base model in eval mode
        model.embedding_projector.train()  # Only embedding projector in train mode
        logger.info("Training mode: base_model=eval, embedding_projector=train")
    else:
        model.train()  # Normal training mode for everything
        logger.info("Training mode: full model in train mode")

    # Ensure embedding_projector is on the right device
    if hasattr(model, 'embedding_projector'):
        model.embedding_projector = model.embedding_projector.to(device)
        logger.info(f"Moved embedding_projector to {device}")

    # Original prefix data will stay on CPU to save memory
    # We'll move tensors to device as needed in the epoch loop
    
    # System prompt and instruction text
    system_prompt = "Predict the correct class for the given data."
    instruction = "Look at the data patterns and predict the class."
    
    # Define query separator token ID
    query_separator_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")  # Use EOS as separator if no special token
    if "<QUERY>" in tokenizer.get_vocab():
        query_separator_id = tokenizer.convert_tokens_to_ids("<QUERY>")
    
    # Minimum frequency regularizer
    def min_frequency_regularizer(predictions, max_classes, min_freq=min_freq_target):
        """
        Add regularization to ensure each class is predicted with some minimum frequency.
        Uses global statistics to avoid oscillation.

        Args:
            predictions: Tensor of class predictions (not used in this version)
            max_classes: Maximum number of classes (length of class_token_ids)
            min_freq: Target minimum frequency for each class

        Returns:
            Penalty term to add to the loss
        """
        # Use global class counts instead of batch counts
        if total_predictions == 0:  # Avoid division by zero
            return torch.tensor(0.0, device=device)
        
        # Calculate global frequencies
        global_frequencies = torch.zeros(max_classes, device=device)
        for c in range(max_classes):
            global_frequencies[c] = class_predictions.get(c, 0) / total_predictions

        # Penalize frequencies below minimum with smoothing
        # Use squared penalty for smoother gradients
        penalty = torch.sum(torch.clamp(min_freq - global_frequencies, min=0) ** 2)
        
        # Apply decay factor to reduce oscillations
        decay_factor = min(1.0, total_predictions / (max_classes * 100))
        penalty = penalty * decay_factor

        return penalty
    
    # Mixup function for data augmentation
    def mixup_embeddings(embeddings, labels, alpha=0.2):
        """
        Apply mixup augmentation to embeddings and labels.
        Returns mixed embeddings and interpolated one-hot labels.
        
        Args:
            embeddings: Embeddings to mix
            labels: Labels for the embeddings
            alpha: Mixup hyperparameter
            
        Returns:
            mixed_embeddings: Mixed embeddings
            labels: Original labels
            labels_permuted: Permuted labels
            lam: Mixing coefficient
        """
        batch_size = embeddings.size(0)
        
        # Skip mixup if alpha is zero or the batch only has one example
        if alpha <= 0 or batch_size <= 1:
            return embeddings, labels, labels, 1.0
        
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Ensure lambda is not too close to 0 or 1 (prevents trivial mixups)
        lam = max(min(lam, 0.9), 0.1)
        
        # Shuffle indices
        index = torch.randperm(batch_size).to(embeddings.device)
        
        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        
        # Return mixed embeddings and original labels with mixing coefficient
        return mixed_embeddings, labels, labels[index], lam
    
    # Process batch helper function
    def process_batch(inputs, query_embeddings, label_ids, loss_weight, temperature=1.0):
        """
        Process a batch through the model and compute the weighted loss.
        Also updates the class prediction tracking.
        
        Args:
            inputs: Tokenized inputs
            query_embeddings: Query embeddings tensor
            label_ids: Target label IDs
            loss_weight: Weight to apply to the loss (for mixup)
            temperature: Temperature for scaling class logits
        
        Returns:
            Weighted loss value
        """
        nonlocal class_predictions, total_predictions
        
        # Create labels for causal LM
        labels = inputs["input_ids"].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100  # Mask last token
        
        # Mask positions we don't want to predict
        answer_start = "The class is:"
        answer_start_tokens = tokenizer(answer_start, add_special_tokens=False).input_ids
        
        # Find positions of "The class is:" in each sequence
        for i in range(len(label_ids)):
            input_ids = inputs["input_ids"][i]
            
            # Find start of answer
            for pos in range(len(input_ids) - len(answer_start_tokens)):
                if torch.all(input_ids[pos:pos+len(answer_start_tokens)] == torch.tensor(answer_start_tokens, device=device)):
                    # Set all tokens before answer as -100 (ignored in loss)
                    labels[i, :pos+len(answer_start_tokens)] = -100
                    break
        
        inputs["labels"] = labels
        
        # Get input embeddings and manually replace prefix placeholder tokens
        input_embeds = model.base_model.get_input_embeddings()(inputs["input_ids"])
        
        # Find positions of PREFIX_START and PREFIX_END tokens
        start_positions = (inputs["input_ids"] == prefix_start_id).nonzero(as_tuple=True)
        end_positions = (inputs["input_ids"] == prefix_end_id).nonzero(as_tuple=True)
        
        # Process each sequence in batch
        for i in range(inputs["input_ids"].shape[0]):
            batch_start_positions = torch.where(start_positions[0] == i)[0]
            batch_end_positions = torch.where(end_positions[0] == i)[0]
            
            # Process each PREFIX_START/END pair
            for start_idx_pos, end_idx_pos in zip(batch_start_positions, batch_end_positions):
                start_pos = start_positions[1][start_idx_pos]
                end_pos = end_positions[1][end_idx_pos]
                
                if start_pos >= end_pos - 1:  # Need at least 1 token between markers
                    continue
                
                # Calculate how many tokens we have between markers
                # Convert to int to avoid tensor arithmetic issues
                num_tokens = int((end_pos - start_pos - 1).item()) if isinstance(end_pos, torch.Tensor) else (end_pos - start_pos - 1)

                # Reserve space for the query embedding (10 tokens)
                query_space = min(10, num_tokens // 3)  # Reserve up to 1/3 of available space, max 10 tokens
                example_space = num_tokens - query_space

                # Make sure these are Python integers, not tensors
                query_space = int(query_space) if isinstance(query_space, torch.Tensor) else query_space
                example_space = int(example_space) if isinstance(example_space, torch.Tensor) else example_space
                
                # Get the current example's query embedding
                query_embedding = query_embeddings[i]
                
                # Project query embedding to model hidden size
                # Enable gradient tracking for regularization if needed
                if gradient_penalty_weight > 0:
                    query_embedding.requires_grad_(True)
                projected_query = model.embedding_projector(query_embedding)
                
                # Make sure to use the device-specific tensors for embeddings and class labels
                # These should already be on the right device from our epoch initialization
                embeddings = prefix_embeddings_tensor
                class_labels = prefix_class_labels_tensor

                # Double-check device before using
                if embeddings.device != device:
                    logger.warning(f"Embeddings on {embeddings.device} but model on {device}, moving...")
                    embeddings = embeddings.to(device)
                if class_labels.device != device:
                    logger.warning(f"Class labels on {class_labels.device} but model on {device}, moving...")
                    class_labels = class_labels.to(device)

                # Determine maximum examples we can use based on space
                max_examples = min(example_space // 2, embeddings.shape[0])

                # If variable few-shot is enabled, randomly select the number of examples to use
                if variable_few_shot and max_examples > few_shot_min:
                    # Make sure we're using integer values for randint, not tensors
                    max_ex = min(few_shot_max, max_examples)

                    # When using numpy with PyTorch, ensure all values are on CPU and converted to Python types
                    min_ex = int(few_shot_min)
                    max_ex = int(max_ex) if not isinstance(max_ex, torch.Tensor) else int(max_ex.item())

                    # Randomly choose number of examples between min and max
                    actual_examples = np.random.randint(min_ex, max_ex + 1)

                    # Update progress bar with few-shot info occasionally
                    if batch_idx % 50 == 0:
                        progress_bar.set_postfix(
                            epoch=f"{epoch+1}/{num_train_epochs}",
                            batch=f"{batch_idx}/{len(train_dataloader)}",
                            few_shot=f"{actual_examples}/{max_examples}"
                        )

                    # Limit to this number of examples
                    num_examples = actual_examples
                else:
                    # Use all available examples up to max_examples
                    num_examples = int(max_examples) if isinstance(max_examples, torch.Tensor) else max_examples
                
                # Project example embeddings to model hidden size
                # Make sure we're slicing on the device, not CPU
                examples_to_project = embeddings[:num_examples].to(device)
                # Enable gradient tracking for regularization if needed
                if gradient_penalty_weight > 0:
                    examples_to_project.requires_grad_(True)
                projected_examples = model.embedding_projector(examples_to_project)
                
                # Create a tensor to hold all our embeddings (query + examples)
                all_embeddings = torch.zeros(
                    num_tokens,  # Total space between markers
                    model.config.hidden_size,
                    device=device
                )
                
                # First, add the query embedding with repetition for emphasis
                query_separator = model.base_model.get_input_embeddings()(
                    torch.tensor([query_separator_id], device=device)
                ).squeeze(0)
                
                # Add query separator and repeated query embedding
                all_embeddings[0] = query_separator
                for j in range(1, query_space - 1):
                    all_embeddings[j] = projected_query
                all_embeddings[query_space - 1] = query_separator
                
                # Next, add the interleaved example embeddings and class tokens
                example_offset = query_space
                for j in range(num_examples):
                    # Example embedding
                    all_embeddings[example_offset + j*2] = projected_examples[j]
                    
                    # Class token
                    if j < len(class_labels):
                        # Make sure we get a valid index by converting to integer if needed
                        class_idx = int(class_labels[j].item()) if isinstance(class_labels[j], torch.Tensor) else int(class_labels[j])

                        # Use the current epoch's class token mapping (which may be permuted)
                        class_token_id = class_token_ids[class_idx]

                        # Convert class_token_id to int if it's a tensor
                        if isinstance(class_token_id, torch.Tensor):
                            class_token_id = int(class_token_id.item())

                        class_token_embedding = model.base_model.get_input_embeddings()(
                            torch.tensor([class_token_id], device=device)
                        ).squeeze(0)
                        all_embeddings[example_offset + j*2 + 1] = class_token_embedding
                
                # Replace token embeddings with our custom embeddings
                # +1 to skip the PREFIX_START token
                input_embeds[i, start_pos+1:end_pos, :] = all_embeddings
        
        # Forward pass with the modified embeddings
        outputs = model.base_model(
            input_ids=None,
            attention_mask=inputs["attention_mask"],
            inputs_embeds=input_embeds,
            labels=inputs["labels"]
        )
        
        # Compute gradient penalty if enabled
        gradient_penalty = 0.0
        if gradient_penalty_weight > 0 and model.embedding_projector.weight.grad is not None:
            # Compute gradient norm for embedding projector weights
            grad_norm = torch.norm(model.embedding_projector.weight.grad, p=2)
            
            # Apply penalty only if gradient norm exceeds threshold
            if grad_norm > gradient_penalty_threshold:
                gradient_penalty = gradient_penalty_weight * (grad_norm - gradient_penalty_threshold) ** 2
        
        # Get logits for the final token prediction (class token)
        # This is where the model is deciding which class to predict
        logits = outputs.logits
        
        # Extract the token positions where we expect the class token
        class_positions = []
        for i in range(len(label_ids)):
            # Find the position of the last non-masked token in labels
            class_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(class_pos) > 0:
                class_positions.append((i, class_pos[-1]))
        
        # Extract the predicted class tokens based on highest logit
        predictions = []
        for i, pos in class_positions:
            token_logits = logits[i, pos]

            # Make sure we're using proper indices for the token IDs
            safe_class_token_ids = [int(tid) if isinstance(tid, (int, float)) else int(tid.item()) for tid in class_token_ids]

            # Only consider class token IDs for prediction
            class_logits = torch.stack([token_logits[tid] for tid in safe_class_token_ids])
            
            # Apply temperature scaling to class logits
            if temperature != 1.0:
                class_logits = class_logits / temperature
            
            predicted_idx = torch.argmax(class_logits).item()
            predictions.append(predicted_idx)

            # Update class prediction tracking - ensure we're using Python numbers, not tensors
            weight_value = float(loss_weight) if isinstance(loss_weight, torch.Tensor) else loss_weight
            class_predictions[predicted_idx] += weight_value
            total_predictions += weight_value
        
        # Convert predictions to tensor for regularization
        if predictions:
            predictions_tensor = torch.tensor(predictions, device=device)
            
            # Apply minimum frequency regularization if we have enough accumulated predictions
            if total_predictions > max_possible_classes * 5:  # Wait until we have a reasonable sample
                min_freq_loss = min_frequency_regularizer(predictions_tensor, max_possible_classes)
                
                # Adaptive weight that decreases as we approach target distribution
                # Calculate how far we are from uniform distribution
                global_freqs = torch.zeros(max_possible_classes, device=device)
                for c in range(max_possible_classes):
                    global_freqs[c] = class_predictions.get(c, 0) / total_predictions
                
                # Measure deviation from uniform
                uniform_freq = 1.0 / max_possible_classes
                deviation = torch.sqrt(torch.mean((global_freqs - uniform_freq) ** 2))
                
                # Scale weight based on deviation (less aggressive when closer to target)
                adaptive_weight = min_freq_weight * torch.clamp(deviation * 10, min=0.1, max=1.0)
                
                # Add warmup schedule to prevent early oscillations
                warmup_steps = max_possible_classes * 50  # Warmup over reasonable number of steps
                warmup_factor = min(1.0, global_step / warmup_steps) if global_step < warmup_steps else 1.0
                adaptive_weight = adaptive_weight * warmup_factor
                
                regularized_loss = outputs.loss + adaptive_weight * min_freq_loss + gradient_penalty
            else:
                regularized_loss = outputs.loss + gradient_penalty
        else:
            regularized_loss = outputs.loss + gradient_penalty
        
        # Apply weight to the loss (for mixup)
        weighted_loss = regularized_loss * loss_weight
        
        return weighted_loss
    
    logger.info("Starting training")
    global_step = 0
    
    # Store the number of classes for label permutation
    # Always use 10 classes for consistency, even if dataset has fewer
    num_classes = len(class_token_ids)  # Should always be 10
    
    # Find the actual number of unique classes in the dataset
    unique_train_labels = set()
    for batch in train_dataloader:
        unique_train_labels.update(batch["label_id"])
    actual_num_classes = len(unique_train_labels)
    max_label = max(unique_train_labels) if unique_train_labels else 0
    
    logger.info(f"Model supports {num_classes} classes, dataset has {actual_num_classes} unique classes")
    logger.info(f"Unique labels in dataset: {sorted(unique_train_labels)}")
    
    if max_label >= num_classes:
        raise ValueError(f"Dataset contains label {max_label} but model only supports {num_classes} classes (0-{num_classes-1})")

    # Get device once at the beginning
    device = next(model.parameters()).device
    logger.info(f"Setting up training on device: {device}")

    # Add some debug information about devices
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available, using CPU")

    # Create a top-level progress bar for epochs
    # If num_train_epochs is None but max_steps is provided, set num_train_epochs to a large number
    # Training will be stopped by max_steps check
    if num_train_epochs is None:
        if max_steps is not None:
            # Calculate number of epochs needed to reach max_steps
            steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
            num_train_epochs = (max_steps + steps_per_epoch - 1) // steps_per_epoch  # Ceiling division
            logger.info(f"Setting num_train_epochs to {num_train_epochs} based on max_steps {max_steps}")
        else:
            # If both are None, default to 1 epoch
            num_train_epochs = 1
            logger.warning("Both num_train_epochs and max_steps are None, defaulting to 1 epoch")
    
    epoch_progress = tqdm(range(num_train_epochs), desc="Training Epochs", position=0)
    last_permutation_step = -1  # Track when we last permuted labels, initialize to -1
    
    for epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {epoch+1}/{num_train_epochs}")

        # Create or reset tensors for this epoch
        prefix_embeddings_tensor = torch.tensor(original_prefix_embeddings, dtype=torch.float32, device=device)
        prefix_class_labels_tensor = torch.tensor(original_prefix_class_labels, dtype=torch.long, device=device)
        
        # Store original prefix labels for potential re-permutation during steps
        original_prefix_labels_for_epoch = prefix_class_labels_tensor.clone()

        # Permute the order of examples for each epoch if enabled
        if permute_examples:
            # Generate a random permutation for examples
            perm_indices = torch.randperm(len(original_prefix_embeddings))

            # Apply the permutation to both embeddings and labels
            prefix_embeddings_tensor = prefix_embeddings_tensor[perm_indices]
            prefix_class_labels_tensor = prefix_class_labels_tensor[perm_indices]

            logger.info(f"Permuted order of {len(perm_indices)} few-shot examples for epoch {epoch+1}")

        # Initialize label permutation for this epoch
        label_permutation = None
        # Check if we should skip permutation due to being in the last k steps
        in_last_k_steps = False
        if no_permute_last_k is not None and max_steps is not None:
            in_last_k_steps = global_step >= (max_steps - no_permute_last_k)
            
        if permute_labels and actual_num_classes > 1 and permute_labels_every_k_steps is None and not in_last_k_steps:
            # Create a permutation that only shuffles the actual classes, leaving unused classes in place
            label_permutation = torch.arange(num_classes, device=device)
            
            # Get indices of actual classes and shuffle only those
            actual_indices = list(sorted(unique_train_labels))
            shuffled_indices = actual_indices.copy()
            random.shuffle(shuffled_indices)
            
            # Apply the shuffled mapping
            for orig, new in zip(actual_indices, shuffled_indices):
                label_permutation[orig] = new
            
            # Log the permutation for debugging
            perm_mapping = {i: label_permutation[i].item() for i in actual_indices}
            logger.info(f"Permuted label mapping for epoch {epoch+1}: {perm_mapping} (only showing actual classes)")
            
            # Apply permutation to prefix class labels
            prefix_class_labels_tensor = label_permutation[prefix_class_labels_tensor]
        
        # Training loop with tqdm progress bar (nested under epoch progress bar)
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                          desc=f"Batches", position=1, leave=False)
        for batch_idx, batch in progress_bar:
            # Check if we're in the last k steps
            if no_permute_last_k is not None and max_steps is not None:
                in_last_k_steps = global_step >= (max_steps - no_permute_last_k)
                if in_last_k_steps and label_permutation is not None:
                    # Reset to no permutation
                    label_permutation = None
                    prefix_class_labels_tensor = original_prefix_labels_for_epoch.clone()
                    logger.info(f"Disabled label permutation at step {global_step} (in last {no_permute_last_k} steps)")
            
            # Check if we should permute labels based on steps
            if permute_labels and num_classes > 1 and permute_labels_every_k_steps is not None and not in_last_k_steps:
                if global_step > 0 and global_step % permute_labels_every_k_steps == 0 and global_step != last_permutation_step:
                    # Create a permutation that only shuffles the actual classes
                    label_permutation = torch.arange(num_classes, device=device)
                    
                    # Get indices of actual classes and shuffle only those
                    actual_indices = list(sorted(unique_train_labels))
                    shuffled_indices = actual_indices.copy()
                    random.shuffle(shuffled_indices)
                    
                    # Apply the shuffled mapping
                    for orig, new in zip(actual_indices, shuffled_indices):
                        label_permutation[orig] = new
                    
                    # Log the permutation for debugging
                    perm_mapping = {i: label_permutation[i].item() for i in actual_indices}
                    logger.info(f"Permuted label mapping at step {global_step}: {perm_mapping} (only showing actual classes)")
                    
                    # Apply permutation to prefix class labels
                    prefix_class_labels_tensor = label_permutation[original_prefix_labels_for_epoch]
                    
                    last_permutation_step = global_step
            
            # Get batch data
            label_ids = torch.tensor(batch["label_id"], device=device)
            query_embeddings = torch.tensor(np.array(batch["query_embedding"]), dtype=torch.float32).to(device)
            
            # Apply label permutation if active
            if label_permutation is not None:
                # Permute the labels
                permuted_label_ids = label_permutation[label_ids]
                label_ids = permuted_label_ids
            
            # Apply mixup to query embeddings and labels with probability 0.75
            use_mixup = (mixup_alpha > 0) and (np.random.random() < 0.75)
            
            if use_mixup:
                query_embeddings, orig_label_ids, mixed_label_ids, lam = mixup_embeddings(
                    query_embeddings, label_ids, alpha=mixup_alpha
                )
                # Update progress bar with mixup details
                progress_bar.set_postfix(
                    epoch=f"{epoch+1}/{num_train_epochs}",
                    batch=f"{batch_idx}/{len(train_dataloader)}",
                    mixup=f"λ={lam:.2f}"
                )
                    
                # Prepare prompts for both label sets (we'll need both for mixed loss)
                prompts_orig = []
                prompts_mixed = []
                
                for orig_id, mixed_id in zip(orig_label_ids, mixed_label_ids):
                    # Create prompts with placeholder tokens
                    placeholder_tokens = " ".join(["_"] * 100)

                    # Get orig_id as a regular Python integer
                    orig_idx = orig_id.item() if isinstance(orig_id, torch.Tensor) else int(orig_id)
                    mixed_idx = mixed_id.item() if isinstance(mixed_id, torch.Tensor) else int(mixed_id)

                    # Original class token - use current epoch's mapping which may be permuted
                    orig_class_token = tokenizer.convert_ids_to_tokens(class_token_ids[orig_idx])
                    prompts_orig.append(f"{system_prompt}\n\n<PREFIX_START>{placeholder_tokens}<PREFIX_END>\n\n{instruction}\n\nThe class is: {orig_class_token}")

                    # Mixed class token - use current epoch's mapping which may be permuted
                    mixed_class_token = tokenizer.convert_ids_to_tokens(class_token_ids[mixed_idx])
                    prompts_mixed.append(f"{system_prompt}\n\n<PREFIX_START>{placeholder_tokens}<PREFIX_END>\n\n{instruction}\n\nThe class is: {mixed_class_token}")
                
                # We'll process each set separately and combine the losses
                combined_loss = 0
                
                # Process original labels
                inputs_orig = tokenizer(prompts_orig, padding=True, truncation=True, return_tensors="pt", max_length=2048)
                inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
                current_temp = temp_scheduler.get_temperature()
                batch_loss_orig = process_batch(inputs_orig, query_embeddings, orig_label_ids, lam, temperature=current_temp)
                combined_loss += batch_loss_orig
                
                # Process mixed labels
                inputs_mixed = tokenizer(prompts_mixed, padding=True, truncation=True, return_tensors="pt", max_length=2048)
                inputs_mixed = {k: v.to(device) for k, v in inputs_mixed.items()}
                batch_loss_mixed = process_batch(inputs_mixed, query_embeddings, mixed_label_ids, 1.0 - lam, temperature=current_temp)
                combined_loss += batch_loss_mixed
                
                # Use the combined loss
                loss = combined_loss
                
                # Update running loss with weighted average
                batch_loss_value = batch_loss_orig.item() * lam + batch_loss_mixed.item() * (1.0 - lam)
            else:
                # Standard processing without mixup
                prompts = []
                for label_id in label_ids:
                    placeholder_tokens = " ".join(["_"] * 100)
                    idx = label_id.item() if isinstance(label_id, torch.Tensor) else int(label_id)
                    class_token = tokenizer.convert_ids_to_tokens(class_token_ids[idx])
                    prompts.append(f"{system_prompt}\n\n<PREFIX_START>{placeholder_tokens}<PREFIX_END>\n\n{instruction}\n\nThe class is: {class_token}")
                
                inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                current_temp = temp_scheduler.get_temperature()
                loss = process_batch(inputs, query_embeddings, label_ids, 1.0, temperature=current_temp)
                
                # Update running loss
                batch_loss_value = loss.item()
            
            # Update running loss average
            running_loss += batch_loss_value
            running_loss_count += 1
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate scheduler after gradient accumulation step
                current_lr = scheduler.step()
                current_temp = temp_scheduler.step()
                global_step += 1
                
                # Check if we've reached the maximum number of steps
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps}), stopping training")
                    progress_bar.write(f"Reached max_steps ({max_steps}), stopping training")
                    break
                
                # Calculate current average loss
                if running_loss_count > 0:
                    avg_loss = running_loss / running_loss_count
                    
                    # Only consider early stopping if we're below the threshold
                    below_threshold = batch_loss_value < early_stopping_threshold
                    
                    # Check for sufficient improvement for early stopping counter
                    if avg_loss < best_loss and below_threshold:
                        # Reset early stopping counter on sufficient decrease in loss
                        no_improve_steps = 0
                        
                        # Check if this is good enough to save as best model (using stricter threshold)
                        if save_best_model and (avg_loss < best_loss - 0.03):
                            best_loss = avg_loss
                            best_model_step = global_step
                            save_model_checkpoint(global_step, is_best=True)
                            # Update progress bar with best model information
                            progress_bar.write(f"New best model at step {global_step} with loss {best_loss:.4f}")
                            
                            # Log best model metrics to wandb
                            if wandb_callback is not None:
                                wandb_callback({
                                    "train/best_loss": best_loss,
                                    "train/best_model_step": global_step,
                                })
                        else:
                            # Update best loss for early stopping tracking, but don't save model
                            best_loss = avg_loss
                    else:
                        # Only increment early stopping counter if we're below threshold
                        if below_threshold and early_stopping_patience:
                            no_improve_steps += 1
                            if no_improve_steps >= early_stopping_patience:
                                early_stop_msg = f"Early stopping after {no_improve_steps} steps without improvement"
                                logger.info(early_stop_msg)
                                progress_bar.write(early_stop_msg)
                                break
                
                # Regular checkpointing
                # Check if we should save at this step based on save_steps (effective_save_steps)
                if should_save_checkpoint(global_step, effective_save_steps):
                    save_model_checkpoint(global_step)
                
                # Periodically reset running loss to focus on recent performance
                if global_step % 100 == 0:
                    running_loss = 0.0
                    running_loss_count = 0
            
            # Get learning rates for all groups
            lr_info = {}
            lr_display = []
            for param_group in scheduler.optimizer.param_groups:
                group_name = param_group.get('name', 'default')
                lr_value = param_group['lr']
                lr_info[f"lr_{group_name}"] = lr_value
                lr_display.append(f"{group_name}:{lr_value:.2e}")
            
            # Update progress bar with loss information
            progress_bar.set_postfix(
                epoch=f"{epoch+1}/{num_train_epochs}",
                batch=f"{batch_idx}/{len(train_dataloader)}",
                loss=f"{batch_loss_value:.4f}",
                lr=", ".join(lr_display) if lr_display else f"{scheduler.optimizer.param_groups[0]['lr']:.2e}",
                temp=f"{temp_scheduler.get_temperature():.2f}",
                step=f"{global_step}/{total_steps}"
            )
            
            # Log metrics to wandb if callback is provided
            if wandb_callback is not None and batch_idx % 10 == 0:
                metrics = {
                    "train/loss": batch_loss_value,
                    "train/temperature": temp_scheduler.get_temperature(),
                    "train/epoch": epoch + (batch_idx / len(train_dataloader)),
                    "train/global_step": global_step,
                }
                
                # Add learning rates for each parameter group
                for param_group in scheduler.optimizer.param_groups:
                    group_name = param_group.get('name', 'default')
                    metrics[f"train/lr_{group_name}"] = param_group['lr']
                
                # Add class prediction frequencies if we have enough data
                if total_predictions > 0:
                    for c, count in class_predictions.items():
                        freq = count / total_predictions
                        metrics[f"train/class_{c}_frequency"] = freq
                        
                # Call the wandb callback with metrics
                wandb_callback(metrics)
        
        # Calculate and display epoch average loss
        avg_epoch_loss = running_loss / running_loss_count if running_loss_count > 0 else 0
        epoch_progress.set_postfix(avg_loss=f"{avg_epoch_loss:.4f}")
        
        # Log epoch metrics to wandb
        if wandb_callback is not None:
            epoch_metrics = {
                "train/epoch_avg_loss": avg_epoch_loss,
                "train/epoch": epoch + 1,
            }
            
            # Log class distribution at epoch level
            if total_predictions > 0:
                for c, count in class_predictions.items():
                    freq = count / total_predictions
                    epoch_metrics[f"train/epoch_class_{c}_frequency"] = freq
            
            wandb_callback(epoch_metrics)
        
        # Only consider early stopping if we're below the threshold
        below_threshold = avg_epoch_loss < early_stopping_threshold
        
        # Check for improvement at epoch level
        if avg_epoch_loss < best_loss and below_threshold:
            best_loss = avg_epoch_loss
            no_improve_steps = 0
            
            # Save the model if it's significantly better (using stricter threshold)
            if save_best_model and (avg_epoch_loss < best_loss - 0.03):
                best_model_step = global_step
                save_model_checkpoint(global_step, is_best=True)
                epoch_progress.write(f"New best model at epoch {epoch+1} with loss {best_loss:.4f}")
                
                # Log best model metrics to wandb
                if wandb_callback is not None:
                    wandb_callback({
                        "train/best_loss": best_loss,
                        "train/best_model_step": global_step,
                    })
        # Check for early stopping at epoch level - only if below threshold
        elif early_stopping_patience and below_threshold:
            # Increment counter when no improvement at epoch level
            no_improve_steps += 1
            if no_improve_steps >= early_stopping_patience:
                early_stop_msg = f"Early stopping after {no_improve_steps} epochs without improvement"
                logger.info(early_stop_msg)
                epoch_progress.write(early_stop_msg)
                epoch_progress.set_postfix(early_stopped=True, avg_loss=f"{avg_epoch_loss:.4f}")
                break
            
        # Also check if we've reached max_steps at the epoch level
        if max_steps is not None and global_step >= max_steps:
            max_steps_msg = f"Reached max_steps ({max_steps}), stopping training"
            logger.info(max_steps_msg)
            epoch_progress.write(max_steps_msg)
            epoch_progress.set_postfix(max_steps_reached=True, avg_loss=f"{avg_epoch_loss:.4f}")
            break
    
    # Define final_epoch_token_ids for return if not already defined
    # (will happen if we exit early without saving final model)
    if not 'final_epoch_token_ids' in locals():
        final_epoch_token_ids = class_token_ids.copy()

    # If we've been saving the best model, point out which one was best
    if save_best_model:
        logger.info(f"Best model was at step {best_model_step} with average loss {best_loss:.4f}")
        logger.info(f"Best model saved at: {os.path.join(output_dir, 'best_model')}")

    # Return the appropriate model:
    # 1. Best model if save_best_model=True and the best model exists
    # 2. Otherwise, the current model state
    if save_best_model:
        # We need to create a new instance of our custom model and load the weights
        best_model_path = os.path.join(output_dir, "best_model")
        
        # Check if the best model path exists
        if os.path.exists(best_model_path):
            # Use the model's from_pretrained classmethod if it exists
            if hasattr(model, "from_pretrained"):
                best_model = model.__class__.from_pretrained(
                    best_model_path, 
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            else:
                # Otherwise, load the base model and recreate the wrapper
                from ..models.qwen_prefix import QwenWithPrefixEmbedding
                
                # Load the base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    best_model_path, 
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                
                # Create a new instance of our custom model
                # Make sure we're using original class token IDs, not the permuted ones
                original_tokens = original_class_tokens if permute_labels else class_token_ids

                best_model = QwenWithPrefixEmbedding(
                    base_model=base_model,
                    embedding_size=prefix_embeddings_tensor.shape[1],
                    prefix_start_id=prefix_start_id,
                    prefix_end_id=prefix_end_id,
                    class_token_ids=original_tokens
                )
                
                # Copy the embedding_projector parameters from the current model
                if hasattr(model, 'embedding_projector'):
                    # Make sure the embedding projector has the same parameters
                    best_model.embedding_projector.load_state_dict(model.embedding_projector.state_dict())
                
                # Make sure class token IDs and other attributes are set
                best_model.prefix_start_id = prefix_start_id
                best_model.prefix_end_id = prefix_end_id
                
                # Move to the same device
                best_model = best_model.to(device)
            
            logger.info(f"Loaded best model from {best_model_path}")
            return best_model, tokenizer, final_epoch_token_ids
        else:
            logger.warning(f"Best model path {best_model_path} not found, returning current model")
            return model, tokenizer, final_epoch_token_ids
    else:
        logger.info("Returning current model state")
        return model, tokenizer, final_epoch_token_ids