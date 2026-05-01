import os
import torch
import random
import numpy as np

def get_best_device(gpu_id: int = 0) -> torch.device:
    """
    Get the best GPU/CPU torch device for system being used.

    Args:
        gpu_id (int): By default GPU id is 0. If more than one GPU is available, 
            this can be used to allocate torch device to the specific gpu.
    
    Returns:
        torch.device: Torch device that can used to train pytorch models

    """
    if torch.cuda.is_available():
        print('Using cuda for GPU acceleration.')
        return torch.device(f'cuda:{gpu_id}')
    
    elif torch.backends.mps.is_available():
        print('Using mps for GPU acceleration.')
        return torch.device(f'mps')
    
    else:
        print('No GPU acceleration. Using CPU.')
        return torch.device('cpu')
    
def set_seed(seed=50) -> None:
    """
    Set a fixed seed value for numpy, torch and torch.cuda.

    Args:
        seed (int): seed for random numbers generation. Default = 50.
    """
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

def mpi_setup() -> tuple:
    """
    Setup function for MPI. Loads mpi4py if this function is 
    executed and returns comm and rank specific details.

    Returns:
        tuple[comm, global_rank, size, gpu_id, cpus_per_rank]: 
            A tuple containing MPI.COMM_WORLD, Global Rank, Size, GPU ID, CPU cores per rank.
    """
    # Conditional import of MPI
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()  # Unique ID across all
    size = comm.Get_size()   # Total number of workers
    
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    cpus_per_rank = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        raise RuntimeError('CUDA is required to run MPI version!')
    
    gpu_id = local_rank % num_gpus
    
    return comm, global_rank, size, gpu_id, cpus_per_rank
