import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    
    # Ensure hash-based functions are seeded
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy random module
    np.random.seed(seed)
    
    # PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior on cuDNN backend (may slow down performance)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Optionally, enforce deterministic algorithms (PyTorch 1.8+)
    # torch.use_deterministic_algorithms(True)

# Example usage:
seed_everything(42)
