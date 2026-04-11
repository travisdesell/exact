import time
from pathlib import Path
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.data_processing.loading import (
    load_csv_files,
    load_sp500_rets,
    ArtifactDataExtractor
)
from src.utils.io import (
    artifact_paths_setup,
    raise_file_not_found
)
from src.utils.formatting import split_combo_names

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
        prev_grid_mode: str, 
        model_losses: list[str],
        mpi: bool = False
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

    # -------------------- Loading Relevant Training Artifacts -------------------- #

    relevant_modl_names = split_combo_names(model_losses, '-')
    
    artifacts_extrator = ArtifactDataExtractor(
        prev_grid_mode,
        artifacts_paths
    )

    if prev_grid_mode == 'one_model':
        model_names = list(set(modl_loss[0] for modl_loss in relevant_modl_names))
        # all_avg_perf = artifacts_extrator.agg_avg_perf('avg_perf', model_names)
        opti_hparams = artifacts_extrator.agg_opti_hparams('optimized', model_names)

    
    elif prev_grid_mode == 'one' and len(relevant_modl_names) == 1:
        opti_hparams = artifacts_extrator.agg_opti_hparams(
            'optimized',
            [f'{relevant_modl_names[0][0]}{MODEL_LOSS_SEP}{relevant_modl_names[0][1]}']
        )

    print(opti_hparams)