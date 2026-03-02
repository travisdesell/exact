import torch
import random
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

def deformtime_device(best_device: torch.device | str) -> torch.device | str:
    """
    Variables
        • best_device
            type: device name or device object
            usage: used to store the preferred runtime device chosen by the project
                   before DeformTime-specific compatibility checks are applied

    This function helps in downgrading DeformTime to CPU when the selected device is
    MPS and its unsupported backward operators would otherwise break training.
    @author: Atharva Vaidya
    """
    # Check whether the selected device is an MPS device object that DeformTime should avoid.
    if isinstance(best_device, torch.device) and best_device.type == 'mps':
        # Return CPU so DeformTime avoids unsupported MPS backward operations.
        print('DeformTime uses CPU because its backward pass requires ops unsupported on MPS.')
        return torch.device('cpu')
    # Check whether the selected device is the string form of MPS from the surrounding runtime.
    if isinstance(best_device, str) and best_device == 'mps':
        # Return CPU in string form so the Trainer receives a safe runtime device.
        print('DeformTime uses CPU because its backward pass requires ops unsupported on MPS.')
        return 'cpu'
    # Keep the originally selected device when no DeformTime-specific MPS workaround is needed.
    return best_device

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
