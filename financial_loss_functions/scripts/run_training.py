import os
import sys
import signal
from scripts.utils import load_path_config, load_config
from src.training.pipeline import run_training_pipeline

_interrupted = False

def signal_handler(signum, frame):
    """Handle Ctrl+C and other interrupts"""
    global _interrupted
    
    if _interrupted:  # Already handling an interrupt
        print("\nForce quitting immediately...")
        sys.exit(1)
    
    _interrupted = True
    print(f"\n{'='*60}")
    print("INTERRUPTED - Cleaning up before exit...")
    print("Press Ctrl+C again to force quit immediately.")
    print(f"{'='*60}")
    
    # Call your cleanup function
    cleanup_on_interrupt()
    
    # Exit gracefully
    sys.exit(130 if signum == signal.SIGINT else 1)

def cleanup_on_interrupt():
    """Your cleanup logic for interrupts"""
    print('\nCleaning up...')
    
    # 1. Clean up PyTorch memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()
        print('Cleared GPU/MPS memory.')
    except:
        pass
    
    # 2. Kill child processes (if using multiprocessing)
    try:
        import psutil
        current = psutil.Process()
        children = current.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except:
                pass
        print(f'Terminated {len(children)} child processes.')
    except:
        pass
    
    # 3. Close matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        print('Closed matplotlib figures.')
    except:
        pass
    
    print('Cleanup complete.')

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

if __name__ == '__main__':
    # load_dotenv()
    try:
        paths_config = load_path_config(os.path.join('config', 'paths.json'))

        hparams_config = load_config(os.path.join('config', 'hparams.json'))

        run_training_pipeline(paths_config, hparams_config)

    except SystemExit:
        # Already handled by signal handler
        pass
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)