import time
from pathlib import Path
from src.data_processing.loading import load_csv_files, load_sp500_rets
from src.utils.io import (
    artifact_paths_setup,
    raise_file_not_found
)

from src.training.loss_functions import LossLibrary
from src.evaluation.metrics import MetricLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary

def _load_processed_data(paths_config: dict) -> tuple:
    
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'processed_test': Path(paths_config['processed_paths']['processed_test']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val']),
        'returns_test': Path(paths_config['processed_paths']['returns_test'])
    }

    # Check if all files exist
    raise_file_not_found(list(processed_files.values()))

    processed_dfs = load_csv_files(processed_files, index_dt=True)
    train_data = processed_dfs['processed_train']
    returns_train = processed_dfs['returns_train']

    val_data = processed_dfs['processed_val']
    returns_val = processed_dfs['returns_val']

    test_data = processed_dfs['processed_test']
    returns_test = processed_dfs['returns_test']

    print('Train shape:', train_data.shape)
    print('Val shape:', val_data.shape)
    print('Test shape:', test_data.shape)

    return train_data, returns_train, val_data, returns_val, test_data, returns_test

def run_evaluation_pipeline(
        paths_config: dict,
        hparams_config: dict,
        features_config: dict, 
        grid_mode: str = 'all',
        model_name: str | None = None,
        loss_name: str | None = None
):
    """
    Run Dynamic Walk-Foward Evaluation on the test set to mimic real-life 
    continous learning for selected models.
    
    Args:
        paths_config (dict): Dictionary containing paths
        hparams_config (dict): Dictionary containing default hyperparameters and tuning ranges
    """
    
    print('\n', '=' * 40, ' Training Grid Pipeline ', '=' * 40)
    start_time = time.time()
    
    artifacts_paths = artifact_paths_setup(paths_config)
    # Registering all NN models to the library
    models_module = paths_config['models_module']
    NNModelLibrary.autodiscover(models_module) # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file

    # -------------------- Loading Processed Data -------------------- #
    process_data_tuple = _load_processed_data(
        paths_config
    )
    train_data = process_data_tuple[0]
    rets_train = process_data_tuple[1]
    val_data = process_data_tuple[2]
    rets_val = process_data_tuple[3]
    test_data = process_data_tuple[4]
    returns_test = process_data_tuple[5]
    
    # Loading S&P 500 for benchmarking
    sp500_rets = load_sp500_rets(paths_config['processed_paths']['benchmark_test'])