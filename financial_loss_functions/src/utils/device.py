import torch
import numpy as np

def get_best_device() -> torch.device:
    if torch.backends.mps.is_available():
        print('Using mps for GPU acceleration.')
        return torch.device('mps')
    
    elif torch.cuda.is_available():
        print('Using cuda for GPU acceleration.')
        return torch.device('cuda')
    
    else:
        print('No GPU acceleration. Using CPU.')
        return torch.device('cpu')

import random

def set_seed(seed=50):
    # Basic Python and Numpy seeds
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Basic PyTorch seed (covers CPU)
    torch.manual_seed(seed)
    
    # NVIDIA CUDA Specifics
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # These two ensure deterministic behavior but may slow down training slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Apple Silicon (MPS) Specifics
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        # Note: MPS is still maturing; some operations might not be 100% deterministic yet
        
    print(f'Seeds set to {seed} across all available backends.')
