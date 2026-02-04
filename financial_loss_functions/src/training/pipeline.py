from torch import optim
from typing import Dict
from pathlib import Path
from src.utils import create_directory
from src.data_processing.dataset import Reshaper
from src.data_processing.dataset import WindowDataset
from src.data_processing.loading import load_csv_files
from src.training.train import (
    Trainer,
    train_val_losses_plot,
    Evaluator,
    CandidatesGrid
)

# Model and Loss Libraries
from src.models.registry import ModelLibrary
from src.training.loss_functions import LossLibrary

def run_training_pipeline(paths_config: Dict, hparams_config: Dict):
    """
    All models training pipeline entry point

    @param paths_config Dict Dictionary containing paths
    @param features_config Dictionary containing hyperparameter information
    """
    print('\n', '=' * 20, ' Training Pipeline ', '=' * 20)
    
    # Create plots directory if it doesnt exist
    plots_dir = (Path(paths_config['artifacts']['plots']))
    create_directory(plots_dir)
    results_dir = Path(paths_config['artifacts']['results'])
    
    # -------------------- Loading Processed Data -------------------- #
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val'])
    }

    processed_dfs = load_csv_files(processed_files)
    train_data = processed_dfs['processed_train']
    returns_train = processed_dfs['returns_train']

    val_data = processed_dfs['processed_val']
    returns_val = processed_dfs['returns_val']

    print('Train shape:', train_data.shape)
    print('Val shape:', val_data.shape)

    # -------------------- Preprocessing (Reshaping) -------------------- #
    reshaper = Reshaper(
        hparams_config['rolling_windows']['in_size'],
        hparams_config['rolling_windows']['out_size'],
        hparams_config['rolling_windows']['stride']
    )
    reshaper.extract_features(train_data)
    
    X_train, y_train, _ = reshaper.reshape(train_data, returns_train)
    print('-'*10, ' train shapes ', '-'*10)
    print('X_train shpe:', X_train.shape)
    print('y_train shape:', y_train.shape)


    X_val, y_val, _ = reshaper.reshape(val_data, returns_val)
    print('-'*10, ' val shapes ', '-'*10)
    print('X_val shape', X_val.shape)
    print('y_val shape:', y_val.shape)

    # -------------------- Training Models -------------------- #
    # Registering all models to the library
    ModelLibrary.autodiscover('src.models') # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file
    
    # Converting to pytorch tensors
    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    # Initializing once to compare all models together
    evaluator = Evaluator(y_val)

    candidates_grid = CandidatesGrid(
        ModelLibrary.items(),
        LossLibrary.items(),
        hparams_config,
        results_dir
    )
    all_alloc_weights = candidates_grid.train_eval_grid(train_ds, val_ds)

    # Calculate returns of all predicted portfolio allocation weights
    for loss_name, models_dict in all_alloc_weights.items():
        for model_name, alloc_weights in models_dict.items():
            evaluator.calc_pf_daily_rets(alloc_weights, f'{model_name}-{loss_name}')
    
    del candidates_grid
    
    # Overall Evaluation/Comparison starts here
    evaluator.calc_eq_wt_daily_rets()
    
    evaluator.plot_windowed_comparison(
        plots_dir /
        (f'Daily Returns' + '.png')
    )

    total_returns = evaluator.calc_total_performance('returns')
    total_sharpes = evaluator.calc_total_performance('sharpe')

    print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)
    print('\n', 'Compounded returns for each window:\n', total_returns)
    print('\n', 'Basic sharpe ratios for each window:\n', total_sharpes)

    #### TODO: 
    # 1. Implement a combination loss
    # 2. Implement addition of tradional models