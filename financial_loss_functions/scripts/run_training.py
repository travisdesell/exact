"""
This is a script file to run the Training and Tuning Pipeline.
"""

import os
import sys
import signal
import argparse
from src.utils.io import load_path_config, load_json
from src.training.pipeline import run_tuning_pipeline

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
        parser = argparse.ArgumentParser(description='Training Grid Configuration Script')

        # Grid mode with a default
        parser.add_argument(
            '-gm',
            '--grid_mode', 
            choices=['one_model', 'one'],
            default='one_model', 
            help='Choose the grid mode to run training and tuning'
        )

        # Dependent arguments
        parser.add_argument(
            '-lm',
            '--loss_mode', 
            choices=['all', 'custom'],
            default='custom',
            help="Required if grid_mode is 'all' or 'one_model'"
        )

        parser.add_argument(
            '-m',
            '--model',
            help="Model name required if grid_mode is 'one_model' or 'one'"
        )
        
        parser.add_argument(
            '-l',
            '--loss', 
            help="Loss name required if grid_mode is 'one_loss' or 'one'"
        )

        parser.add_argument('-t', '--tune', action='store_true', help='Hyperparameter tune')
        parser.add_argument('-mpi', '--mpi', action='store_true', help='MPI for HPC')

        args = parser.parse_args()

        # # Rule 1: If grid_mode is 'one_model', model MUST be provided

        if args.grid_mode == 'one_model':
            if not args.model:
                parser.error("--model is required when --grid_mode is 'one_model'")
                
        elif args.grid_mode == 'one':
            if not args.loss or not args.model:
                parser.error("--loss and --model is required when --grid_mode is 'one'")

        paths_config = load_path_config(os.path.join('config', 'paths.json'))
        hparams_config = load_json(os.path.join('config', 'hparams.json'))
        features_config = load_json(os.path.join('config', 'features.json'))

        run_tuning_pipeline(
            paths_config,
            hparams_config, 
            features_config,
            model_name = args.model,
            grid_mode = args.grid_mode, 
            loss_mode = args.loss_mode, 
            loss_name = args.loss,
            tune = args.tune,
            mpi = args.mpi
        )

    except SystemExit:
        # Already handled by signal handler
        pass
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)