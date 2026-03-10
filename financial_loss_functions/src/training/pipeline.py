import os
import time
import pandas as pd
from pathlib import Path
from src.utils.io import create_directory, save_to_csv, save_to_json
from src.evaluation.evaluator import Evaluator
from src.data_processing.loading import load_csv_files
from src.utils.device import get_best_device, set_seed, deformtime_device
from src.data_processing.dataset import (
    Reshaper,
    calc_in_out_idx,
    extract_oos_dates,
    extract_sp500_winds
)
from src.visualization.plots import (
    train_val_losses_plot, 
    plot_windowed_comparison,
    plot_models_comparison
)
from src.training.train import (
    CandidatesGrid,
    TradModelsTrainer
)

# Model and Loss Libraries
from src.training.loss_functions import LossLibrary
from src.evaluation.metrics import MetricLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary

# TODO:
# Add Best model ranker
# Unit test NCO

def _common_setup(paths_config):
    # set_seed(seed_value) # Global seed for reproducibility
    # Create plots directory if it doesnt exist
    plots_dir = (Path(paths_config['artifacts']['plots']))
    create_directory(plots_dir)
    results_dir = Path(paths_config['artifacts']['results'])
    
    models_module = paths_config['models_module']

    # Registering all Traditional models to the library
    TradModelLibrary.autodiscover(models_module)
    # Registering all NN models to the library
    NNModelLibrary.autodiscover(models_module) # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file

    best_device = get_best_device()
    
    return plots_dir, results_dir, best_device

def _load_processed_data(paths_config: dict) -> tuple:
    
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

    return train_data, returns_train, val_data, returns_val

def _load_sp500_rets(paths_config: dict):

    # Loading only S&P500 from validation split
    benches = load_csv_files(
        {'benchmark_val': Path(paths_config['processed_paths']['benchmark_val'])}
    )

    return benches['benchmark_val']
    
def _preprocess(
        train_data: pd.DataFrame,
        returns_train: pd.DataFrame,
        val_data: pd.DataFrame,
        returns_val: pd.DataFrame,
        windows_config: dict,
        common_features: list
    ) -> tuple:
    reshaper = Reshaper(
        windows_config['in_size'],
        windows_config['out_size'],
        windows_config['stride'],
        common_features
    )
    reshaper.extract_features(train_data.columns)
    
    X_train, y_train, _ = reshaper.reshape(train_data, returns_train)
    print('-'*10, ' train shapes ', '-'*10)
    print('X_train shpe:', X_train.shape)
    print('y_train shape:', y_train.shape)

    X_val, y_val, _ = reshaper.reshape(val_data, returns_val)
    print('-'*10, ' val shapes ', '-'*10)
    print('X_val shape', X_val.shape)
    print('y_val shape:', y_val.shape)

    # Calculate indexes for input and output windows on split data
    in_wind_indexes, out_wind_indexes = calc_in_out_idx(
        returns_val,
        windows_config['in_size'],
        windows_config['out_size'],
        windows_config['stride']
    ) 

    return X_train, y_train, X_val, y_val, in_wind_indexes, out_wind_indexes

def _print_evaludation_info(in_win_date_cols, out_win_date_cols, **kwargs):
    eval_dates_info = {
        'Input Window Start': [],
        'Input Window End': [],
        'Out Window Start': [],
        'Out Window End': []
    }
    
    for in_date, out_date in zip(in_win_date_cols, out_win_date_cols):
        eval_dates_info['Input Window Start'].append(in_date[0])
        eval_dates_info['Input Window End'].append(in_date[-1])
        eval_dates_info['Out Window Start'].append(out_date[0])
        eval_dates_info['Out Window End'].append(out_date[-1])
    
    print('\nModels evaluated on:')
    print(pd.DataFrame(eval_dates_info))

    print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)

    # Loop over provided dataframes and print
    for metric, df in kwargs.items():
        # Cleaning up the metric name
        title = metric.replace('_', ' ').upper()
        print(f'\n{title.upper()}:\n', df)

def run_training_pipeline(
        paths_config: dict,
        hparams_config: dict,
        features_config: dict, 
        grid_mode: str = 'all', 
        loss_mode: str = 'custom',
        model_name: str | None = None,
        loss_name: str | None = None,
        tune: bool = False
    ):
    """
    All models training pipeline entry point

    @param paths_config Dict Dictionary containing paths
    @param features_config Dictionary containing hyperparameter information
    @param grid_mode str `all`, `one_model` or `one_loss`
    @param loss_mode str `all` or `custom`
    @param model str Name of the model to be run
    @param loss str Name of the loss function to be used
    """
    print('\n', '=' * 40, ' Training Grid Pipeline ', '=' * 40)
    start_time = time.time()
    
    plots_dir, results_dir, best_device = _common_setup(paths_config)
    
    # -------------------- Loading Processed Data -------------------- #
    train_data, returns_train, val_data, returns_val = _load_processed_data(
        paths_config
    )
    
    # -------------------- Preprocessing (Reshaping) -------------------- #
    X_train, y_train, X_val, y_val, in_wind_idxs, out_wind_idxs = _preprocess(
        train_data,
        returns_train,
        val_data,
        returns_val,
        hparams_config['rolling_windows'],
        features_config['common_features']
    )

    # -------------------- Evaluator Setup -------------------- #

    # Initializing once to compare all models together
    evaluator = Evaluator(y_val, MetricLibrary.items())

    # -------------------- Training Tradional Models -------------------- #
    # max_workers = os.cpu_count() - 1
    # trad_grid = TradModelsTrainer(
    #     TradModelLibrary.items(),
    #     hparams_config,
    #     max_workers
    # )
    # trad_alloc_weights = trad_grid.train_all(
    #     in_wind_idxs,
    #     out_wind_idxs,
    #     returns_train,
    #     returns_val
    # )

    # for trad_model_name, alloc_weights in trad_alloc_weights.items():
    #     evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
    
    # del trad_grid

    # -------------------- Training Neural Network Models -------------------- #
    # Building hyperparameter tuning metric
    # System designed to take only linear formulas using + or -
    if tune:
        tune_metric = {
            'sharpe': {
                'func': MetricLibrary.get('sharpe'),
                'sign': '+'
            },
            'max_drawdown': {
                'func': MetricLibrary.get('max_drawdown'),
                'sign': '-'
            }
        }
    else:
        tune_metric = None
    
    candidates_grid = CandidatesGrid(
        model_lib = NNModelLibrary.items(),
        loss_lib = LossLibrary.items(),
        hparams_config = hparams_config,
        torch_device=best_device,
        loss_mode = loss_mode,
        tune = tune,
        tune_metric = tune_metric
    )
    if grid_mode == 'all':
        nn_alloc_weights = candidates_grid.train_eval_grid(
            X_train, X_val, y_val
        )
    elif grid_mode == 'one_model' and model_name is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_model(
            model_name, X_train, y_train, X_val, y_val
        )
    elif grid_mode == 'one_loss' and loss_name is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_loss(
            loss_name, X_train, y_train, X_val, y_val
        )

    elif grid_mode == 'one' and model_name is not None and loss_name is not None:
        nn_alloc_weights = candidates_grid.train_eval_one(
            model_name, loss_name, X_train, y_train, X_val, y_val
        )
    else:
        raise RuntimeError('Incorrect mode arguments while running at entry point.')
    
    if tune:
        optimized_hparams = candidates_grid.get_optimized_hparams()
        save_to_json(
            optimized_hparams,
            Path(paths_config['artifacts']['artifacts_dir']) / 'optimized_hparams.json'
        )

    # Plot training and validation loss curves
    nn_train_loss_curves = candidates_grid.get_train_val_losses()
    for model_loss, model_loss_curves in nn_train_loss_curves.items():
        loss_plot_name = model_loss + ' Loss Curves'
        train_val_losses_plot(
            model_loss_curves['train'],
            model_loss_curves['val'],
            model_loss_curves['eval'],
            loss_plot_name,
            plots_dir / (loss_plot_name + '.png')
        )

    # -------------------- Evaluation on Out-of-Sample data -------------------- #
    # Calculate returns of all predicted portfolio allocation weights
    # Calling on every models output allocation weights to calculate pf returns
    for loss_name, models_dict in nn_alloc_weights.items():
        for nn_model_name, alloc_weights in models_dict.items():
            evaluator.calc_pf_daily_rets(alloc_weights, f'{nn_model_name}-{loss_name}')
    
    del candidates_grid
    
    # Overall Evaluation/Comparison starts here
    evaluator.calc_eq_wt_daily_rets()

    # Extract dates index columns for the rrespective output windows
    in_win_date_cols, out_win_date_cols = extract_oos_dates(
        val_data,
        in_wind_idxs,
        out_wind_idxs
    )
    
    # Loading S&P 500 for benchmarking
    sp500_rets = _load_sp500_rets(paths_config)

    # Extract s&p500 returns column sliced for the respective output windows
    sp500_rets_winds = extract_sp500_winds(
        sp500_rets,
        features_config['sp500_returns'],
        out_wind_idxs
    )

    # Adding s&p500 returns to the evaluator as a benchmark
    evaluator.add_benchmark_rets('S&P500', sp500_rets_winds)
    
    # plot_windowed_comparison(
    #     evaluator.get_all_daily_returns(),
    #     out_win_date_cols,
    #     plots_dir / (f'Daily Returns' + '.png')
    # )

    avg_perf_metrics = evaluator.calc_avg_performance()

    # plot_models_comparison(
    #     total_sharpes,
    #     'Out-of-Sample Sharpe Ratio Comparison',
    #     plots_dir / f'Sharpe Comprison.png'
    # )

    # total_returns = total_returns.describe().T
    save_to_csv(avg_perf_metrics, results_dir / 'avg_performance.csv')
    # total_sharpes = total_sharpes.describe().T
    # total_sharpes.to_csv(results_dir / 'total_sharpes.csv', sep=',')

    _print_evaludation_info(
            in_win_date_cols,
            out_win_date_cols,
            avgerage_performance_metrics=avg_perf_metrics
        )

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')