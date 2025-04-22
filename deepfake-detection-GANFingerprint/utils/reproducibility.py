import os
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # For deterministic CUDA operations (PyTorch 1.8+)
    # In utils/reproducibility.py, modify the set_all_seeds function
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)  # Add warn_only=True
        except Exception as e:
            print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    print(f"All seeds set to {seed} for reproducibility")
    
def get_random_state():
    """Get current random state from all sources"""
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    
def set_random_state(random_state):
    """Set random state for all sources"""
    random.setstate(random_state['random'])
    np.random.set_state(random_state['numpy'])
    torch.set_rng_state(random_state['torch'])
    if torch.cuda.is_available() and random_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(random_state['torch_cuda'])