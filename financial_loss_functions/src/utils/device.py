import torch

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