import os
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from src.utils.device import get_best_device
from src.utils.formatting import serialize_np_dict
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.utils.window import (
    extract_oos_dates,
    get_date_index_col,
    extract_sp500_winds,
    calc_in_out_idx

)
from src.training.train_trad import TradModelsTrainer
from src.data_processing.loading import load_csv_files, load_sp500_rets
from src.visualization.plots import wfv_losses_plot
from src.utils.io import (
    artifact_paths_setup, save_to_csv, save_to_json, raise_file_not_found
)
from src.training.train_nn import CandidatesGrid, MetricModel, WalkForwardValidator
from src.evaluation.evaluator import (
    Evaluator, EqualWeightCalculator
)
from src.data_processing.dataset import WFUtilities
# Model and Loss Libraries
from src.training.loss_functions import LossLibrary
from src.evaluation.metrics import MetricLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary




# def _create_dirs(artifacts_paths: dict[str, Path|str]):
#     """
#     Create directories if they dont exist in artifacts directory.
#     """
#     for dir_path in artifacts_paths.values():
#         create_directory(dir_path)

def _load_processed_data(paths_config: dict) -> tuple:
    
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val'])
    }

    # Check if all files exist
    raise_file_not_found(list(processed_files.values()))

    processed_dfs = load_csv_files(processed_files, index_dt=True)
    train_data = processed_dfs['processed_train']
    returns_train = processed_dfs['returns_train']

    val_data = processed_dfs['processed_val']
    returns_val = processed_dfs['returns_val']

    # print('Train shape:', train_data.shape)
    # print('Val shape:', val_data.shape)

    return train_data, returns_train, val_data, returns_val

def _print_evaludation_info(
        out_win_date_cols, in_win_date_cols: list|None=None, **kwargs
    ):
    if in_win_date_cols:
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
    else:
        eval_dates_info = {
            'Out Window Start': [],
            'Out Window End': []
        }
        for out_date in out_win_date_cols:
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

def mpi_setup() -> tuple:
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

def run_tuning_pipeline(
    paths_config: dict,
    hparams_config: dict,
    features_config: dict, 
    grid_mode: str = 'one_model', 
    loss_mode: str = 'custom',
    model_name: str | None = None,
    loss_name: str | None = None,
    tune: bool = False,
    mpi: bool = False
):
    """
    Tune and train with best hyperparameters and compare against equal weight and S&P 500.

    Args:
        paths_config (dict): Dictionary containing paths
        hparams_config (dict): Dictionary containing default hyperparameters and tuning ranges
        features_config (dict): Dictionary containing hyperparameter information
        grid_mode (str): `all`, `one_model`, `one_loss` of `one`
        loss_mode (str): `all` or `custom`, Default = `custom`
        model (str): Name of the model to be run
        loss (str): Name of the loss function to be used
    """
    print('\n', '=' * 40, ' Training Grid Pipeline ', '=' * 40)
    start_time = time.time()
    
    artifacts_paths = artifact_paths_setup(paths_config)
    
    # Registering all NN models to the library
    models_module = paths_config['models_module']
    NNModelLibrary.autodiscover(models_module) # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file
    
    # -------------------- Loading Processed Data -------------------- #
    train_data, rets_train, val_data, rets_val = _load_processed_data(
        paths_config
    )
    # Loading S&P 500 for benchmarking
    sp500_rets = load_sp500_rets(paths_config['processed_paths']['benchmark_val'])
    
    # -------------------- Prepare Validation Sets -------------------- #
    out_size = hparams_config['rolling_windows']['out_size']
    wf_utils = WFUtilities(out_size)
    num_steps, extra_days = wf_utils.calc_walk_steps(rets_val)

    # # y_val for out of sample evaluation
    y_val, out_wind_idxs =wf_utils.build_eval_windows(rets_val)

    if extra_days != 0:
        raise RuntimeError(
            'Validation data incorrectly adjusted. Number of days must be divisible by out_size.'
        )

    # -------------------- Training Neural Network Models -------------------- #

    # Calculate Equal Weight Portfolio's weights
    eq_wt_calc = EqualWeightCalculator(y_val)
    eq_wt_rets = eq_wt_calc.calc_eq_wt_daily_rets()
    
    # Building hyperparameter tuning metric
    # System designed to take only linear formulas using + or -
    # Must follow the MetricModel structure
    if tune:
        tune_metric = {
            'sharpe': MetricModel(func=MetricLibrary.get('sharpe'), sign='+'),
            # 'cvar': MetricModel(func=MetricLibrary.get('cvar'), sign='+'),
            # # 'max_drawdown': MetricModel(func=MetricLibrary.get('max_drawdown'), sign='+'),
            # 'omega': MetricModel(func=MetricLibrary.get('omega'), sign='+'),
            # 'calmar': MetricModel(func=MetricLibrary.get('calmar'), sign='+')
        }

        tuner_eval_items = {
            'metric': tune_metric,
            'bench_rets': eq_wt_rets,
            'eval_winds': y_val
        }
    else:
        tuner_eval_items = None

    if mpi:

        comm, global_rank, size, gpu_id, _ = mpi_setup()
        torch_device = get_best_device(gpu_id)

        # Create artifact directories if the don't exist
        # Only rank 0 can create if directories don't exist

        candidates_grid = CandidatesGrid(
            model_lib = NNModelLibrary.items(),
            loss_lib = LossLibrary.items(),
            hparams_config = hparams_config,
            num_steps = num_steps,
            common_features=features_config['common_features'],
            torch_device = torch_device,
            loss_mode = loss_mode,
            tune = tune,
            tuner_eval_items = tuner_eval_items,
            mpi = mpi,
            temp_dir = artifacts_paths['temp_dir']
        )
        
        if grid_mode == 'one_model' and model_name is not None:
            nn_alloc_weights = candidates_grid.train_eval_one_model(
                model_name, train_data, rets_train, val_data, rets_val, comm, global_rank, size
            )
            results_suffix = model_name
        
        else:
            raise RuntimeError(
                'Incorrect mode arguments while running at entry point.',
                'If mpi, grid mode must be `one_model` and model name must be provided.'
            )

        # Stop all non zero ranks
        if global_rank != 0:
            print(f'Rank {global_rank}: Work complete. Shutting down.')
            sys.exit(0) # This stops the process for this rank only
        
        if nn_alloc_weights is None:
            print('!!!Rank 0 got empty allocation weights. Needs debug!!!')
        

    else:
        
        # Default cuda or mps device
        torch_device = get_best_device()

        candidates_grid = CandidatesGrid(
            model_lib = NNModelLibrary.items(),
            loss_lib = LossLibrary.items(),
            hparams_config = hparams_config,
            num_steps = num_steps,
            common_features=features_config['common_features'],
            torch_device = torch_device,
            loss_mode = loss_mode,
            tune = tune,
            tuner_eval_items = tuner_eval_items,
            mpi = mpi,
            temp_dir = artifacts_paths['temp_dir']
        )

        if grid_mode == 'one_model' and model_name is not None:
            nn_alloc_weights = candidates_grid.train_eval_one_model(
                model_name, train_data, rets_train, val_data, rets_val, None, None, None
            )
            results_suffix = model_name

        elif grid_mode == 'one' and model_name is not None and loss_name is not None:
            nn_alloc_weights = candidates_grid.train_eval_one(
                model_name, loss_name, train_data, rets_train, val_data, rets_val
            )

            results_suffix = f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
        
        else:
            raise RuntimeError('Incorrect mode arguments while running at entry point.')
    
    if tune:
        opti_file_name = artifacts_paths['hparams_dir'] \
            / f'optimized_{results_suffix}.json'
        optimized_hparams = candidates_grid.get_optimized_hparams()
        save_to_json(
            optimized_hparams,
            opti_file_name
        )

    # Plot training and validation loss curves
    nn_train_loss_curves = candidates_grid.get_train_val_losses()
    for model_loss, model_loss_curves in nn_train_loss_curves.items():
        wfv_plot_name = model_loss + ' WFV Losses'
        wfv_losses_plot(
            model_loss_curves['train'],
            model_loss_curves['eval'],
            wfv_plot_name,
            artifacts_paths['tuned_plots_dir'] / (wfv_plot_name + '.png')
        )

    # -------------------- Evaluator Setup -------------------- #
    
    # Initializing once to compare all models together
    evaluator = Evaluator(y_val, MetricLibrary.items())
    
    # Calculate returns of all predicted portfolio allocation weights
    # Calling on every models output allocation weights to calculate pf returns
    for model_loss, alloc_weights in nn_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, model_loss)

    del candidates_grid

    # # -------------------- Training Tradional Models -------------------- #       
    # print(f'Training All Tradional Models')
    # trad_grid = TradModelsTrainer(
    #     TradModelLibrary.items(),
    #     hparams_config,
    #     num_steps
    # )
    # trad_alloc_weights = trad_grid.train_all(
    #     init_rets_train=rets_train,
    #     init_rets_split=rets_val
    # )

    # for trad_model_name, alloc_weights in trad_alloc_weights.items():
    #     evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
    
    # del trad_grid

    # -------------------- Evaluation on Out-of-Sample data -------------------- #    
    # Extract s&p500 returns column sliced for the respective output windows
    sp500_rets_winds = extract_sp500_winds(
        sp500_rets,
        features_config['sp500_returns'],
        out_wind_idxs
    )

    # Adding s&p500 & equal weight returns to the evaluator as a benchmarks
    evaluator.add_benchmark_rets(EQ_WT_NAME, eq_wt_rets)
    evaluator.add_benchmark_rets(SP500_NAME, sp500_rets_winds)
    
    perf_file_name = artifacts_paths['avg_perf_dir'] \
        / f'avg_perf_{results_suffix}.csv'
    avg_perf_metrics = evaluator.calc_avg_performance()
    save_to_csv(avg_perf_metrics, perf_file_name)

    # Extract dates index columns for the respective output windows
    out_win_date_cols = get_date_index_col(rets_val, out_wind_idxs)

    # Save daily returns
    all_daily_returns = evaluator.get_all_daily_returns()
    all_rets_file_name = artifacts_paths['wfv_perf_dir'] \
        / f'daily_rets_{results_suffix}.json'
    save_to_json(
        serialize_np_dict(all_daily_returns),
        all_rets_file_name
    )

    _print_evaludation_info(
        out_win_date_cols=out_win_date_cols,
        avgerage_performance_metrics=avg_perf_metrics,
    )

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')


# def run_wfv_pipeline(
#     paths_config: dict,
#     hparams_config: dict,
#     features_config: dict,
#     grid_mode: str,
#     model_name: str | None,
#     loss_name: str | None,
#     mpi: bool = False
# ): 
    
#     """
#     Run Dynamic Walk-Foward Validation for selected models that beat the benchmark.
    
#     Args:
#         paths_config (dict): Dictionary containing paths
#         hparams_config (dict): Dictionary containing default hyperparameters and tuning ranges
#     """
#     print('\n', '=' * 40, ' Walk-Forward Validation Grid Pipeline ', '=' * 40)
#     start_time = time.time()

#     artifacts_paths = _common_setup(paths_config)
    
#     # -------------------- Loading Processed Data -------------------- #
#     train_data, returns_train, val_data, returns_val = _load_processed_data(
#         paths_config
#     )

#     # -------------------- Loading Relevant Training Artifacts -------------------- #
#     artifacts_extrator = ArtifactDataExtractor(
#         grid_mode,
#         artifacts_paths
#     )

#     if grid_mode in ['all', 'one_model']:
#         # Get all models from library
#         all_models = []
#         for cat in NNModelLibrary.list_categories():
#             all_models.extend(NNModelLibrary.list_models(cat))
    
#         all_avg_perf = artifacts_extrator.agg_avg_perf('avg_perf', all_models)
#         opti_hparams = artifacts_extrator.agg_opti_hparams('optimized', all_models)

#         all_benches = TradModelLibrary.list_models()
#         all_benches.extend([EQ_WT_NAME, SP500_NAME])

#         # Filter models that beat Equal Weight Portfolio
#         filtered_perf, filtered_models = filter_models(
#             all_avg_perf, EQ_WT_NAME, 'sharpe', all_benches
#         )

#         print(f'\nModels that beat Equal Weight portfolio: {filtered_models}')
#         print('Filtered Avg. Performance Metrics: \n', filtered_perf)
    
#     elif grid_mode == 'one' and model_name is not None and loss_name is not None:
#         opti_hparams = artifacts_extrator.agg_opti_hparams(
#             'optimized', [f'{model_name}-{loss_name}']
#         )
#         filtered_models = None
    

#     # -------------------- Walk-Forward Training & Validation -------------------- #
#     if mpi:
#         comm, global_rank, size, gpu_id, cpus_per_rank = mpi_setup()
#         torch_device = get_best_device(gpu_id)

#         if global_rank == 0:
#             _create_dirs(artifacts_paths)

#         grid_validator = WalkForwardValidator(
#             model_lib = NNModelLibrary.items(),
#             loss_lib = LossLibrary.items(),
#             hparams_config = hparams_config,
#             common_features = features_config['common_features'],
#             torch_device = torch_device,
#             num_steps = num_steps,
#             filtered_models = filtered_models,
#             mpi = mpi,
#             temp_dir = artifacts_paths['temp_dir'],
#             optimized_hparams = opti_hparams
#         )

#         if grid_mode in ['all', 'one_model']:
#             nn_alloc_weights = grid_validator.validate_grid(
#                 init_train, init_rets_train, init_val, init_rets_val, comm, global_rank, size
#             )
#             results_suffix = 'All'
#         else:
#             raise RuntimeError(
#                 'Incorrect mode arguments while running at entry point.',
#                 'If mpi, grid mode must be `all` or `one_model`.'
#             )
        
#         # Stop all non zero ranks
#         if global_rank != 0:
#             print(f'Rank {global_rank}: Work complete. Shutting down.')
#             sys.exit(0) # This stops the process for this rank only
        
#         if nn_alloc_weights is None:
#             print('!!!Rank 0 got empty allocation weights. Needs debug!!!')
    
#     else:
#         _create_dirs(artifacts_paths)

#         # Using default MPS or CUDA
#         torch_device = get_best_device()

#         grid_validator = WalkForwardValidator(
#             model_lib = NNModelLibrary.items(),
#             loss_lib = LossLibrary.items(),
#             hparams_config = hparams_config,
#             common_features = features_config['common_features'],
#             torch_device = torch_device,
#             num_steps = num_steps,
#             filtered_models = filtered_models,
#             mpi = mpi,
#             temp_dir = artifacts_paths['temp_dir'],
#             optimized_hparams = opti_hparams
#         )

#         if grid_mode in ['all', 'one_model']:
#             nn_alloc_weights = grid_validator.validate_grid(
#                 init_train, init_rets_train, init_val, init_rets_val, None, None, None
#             )
            
#             results_suffix = 'All'

#         elif grid_mode == 'one' and model_name is not None and loss_name is not None:
#             nn_alloc_weights = grid_validator.validate_one(
#                 model_name, loss_name,init_train, init_rets_train, init_val, init_rets_val
#             )

#             results_suffix = f'{model_name}-{loss_name}'
        
    
#     nn_train_infer_losses = grid_validator.get_train_infer_losses()
#     for model_loss, model_loss_curves in nn_train_infer_losses.items():
#         wfv_plot_name = model_loss + ' WFV Losses'
#         wfv_losses_plot(
#             model_loss_curves['train'],
#             model_loss_curves['eval'],
#             wfv_plot_name,
#             artifacts_paths['wfv_plots_dir'] / (wfv_plot_name + '.png')
#         )
    
#     print('WFV completed!')
    
#     # -------------------- Evaluator Setup -------------------- #
#     # Initializing Evaluator
#     evaluator = Evaluator(eval_windows, MetricLibrary.items())  
    
#     # -------------------- Training Tradional Models -------------------- #        
#     print(f'Training All Tradional Models workers.')
#     trad_grid = TradModelsTrainer(
#         TradModelLibrary.items(),
#         hparams_config,
#         num_steps
#     )
#     trad_alloc_weights = trad_grid.train_all(
#         init_rets_train=init_rets_train,
#         init_rets_split=init_rets_val
#     )

#     for trad_model_name, alloc_weights in trad_alloc_weights.items():
#         evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
    
#     del trad_grid

#     # -------------------- Out-of-Sample Evaluation -------------------- #

#     for model_loss, alloc_weights in nn_alloc_weights.items():
#         evaluator.calc_pf_daily_rets(alloc_weights, model_loss)

#     # Loading S&P 500 for benchmarking
#     sp500_rets = _load_sp500_rets(paths_config)

#     # Extract s&p500 returns column sliced for the respective output windows
#     sp500_rets_winds = extract_sp500_winds(
#         sp500_rets,
#         features_config['sp500_returns'],
#         out_wind_idxs
#     )  
    
#     # Calculate Equal Weight Portfolio's weeights
#     eq_wt_calc = EqualWeightCalculator(eval_windows)
#     eq_wt_rets = eq_wt_calc.calc_eq_wt_daily_rets()

#     # Adding s&p500 & equal weight returns to the evaluator as a benchmarks
#     evaluator.add_benchmark_rets(EQ_WT_NAME, eq_wt_rets)
#     evaluator.add_benchmark_rets(SP500_NAME, sp500_rets_winds)

        
#     avg_perf_metrics = evaluator.calc_avg_performance()
#     perf_file_name = artifacts_paths['wfv_perf_dir'] \
#         / f'wfv_avg_perf_{results_suffix}.csv'
#     save_to_csv(avg_perf_metrics, perf_file_name)

#     # Extract dates index columns for the respective output windows
#     out_win_date_cols = get_date_index_col(returns_val, out_wind_idxs)


#     all_daily_returns = evaluator.get_all_daily_returns()
#     all_rets_file_name = artifacts_paths['wfv_perf_dir'] \
#         / f'wfv_all_rets_{results_suffix}.json'
#     save_to_json(
#         serialize_np_dict(all_daily_returns),
#         all_rets_file_name
#     )
    
#     plot_windowed_comparison(
#         all_daily_returns,
#         out_win_date_cols,
#         2,
#         artifacts_paths['wfv_plots_dir'] / \
#             (f'WFV Performances_{results_suffix}' + '.png')
#     )
    
#     _print_evaludation_info(
#         out_win_date_cols=out_win_date_cols,
#         avgerage_performance_metrics=avg_perf_metrics
#     )
    
#     time_taken = round((time.time() - start_time) / 60, 3)
#     print(f'Time taken for pipeline = {time_taken} mins')