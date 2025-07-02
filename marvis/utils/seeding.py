"""
Unified random seeding utilities for MARVIS.

This module provides consistent random seeding across all libraries used in MARVIS:
- Python's random module
- NumPy 
- PyTorch (CPU and CUDA)
- Optional deterministic mode for PyTorch
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic mode for PyTorch operations.
                      This may impact performance but ensures complete reproducibility.
    """
    # Python's random module
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch - torch.manual_seed() automatically calls torch.cuda.manual_seed_all()
    # so this seeds both CPU and all CUDA devices in one call
    import torch
    torch.manual_seed(seed)
    
    # Optional deterministic mode for complete reproducibility
    if deterministic:
        # These settings are safe to set regardless of CUDA availability
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set seed {seed} with deterministic mode enabled (may impact performance)")
    else:
        logger.debug(f"Set seed {seed} for all libraries")


def set_seed_with_args(args, deterministic: Optional[bool] = None) -> None:
    """
    Convenience function to set seed from command line arguments.
    
    Args:
        args: Argument namespace with 'seed' attribute
        deterministic: Override deterministic mode. If None, uses getattr(args, 'deterministic', False)
    """
    if not hasattr(args, 'seed'):
        logger.warning("Args object does not have 'seed' attribute, using default seed 42")
        seed = 42
    else:
        seed = args.seed
    
    if deterministic is None:
        deterministic = getattr(args, 'deterministic', False)
    
    set_seed(seed, deterministic=deterministic)


def create_random_state(seed: int):
    """
    Create a random state for use with sklearn and other libraries.
    
    Args:
        seed: Random seed value
        
    Returns:
        Random state that can be passed to sklearn functions
    """
    import numpy as np
    return np.random.RandomState(seed)


# Backwards compatibility - keep the function name used in some existing scripts
def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """
    Backwards compatibility alias for set_seed.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic mode for PyTorch
    """
    set_seed(seed, deterministic=deterministic)