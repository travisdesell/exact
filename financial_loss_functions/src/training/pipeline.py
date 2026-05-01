"""
Hyperparameter tuning and evaluation on the validation data.
This file contains the entry point for the function to tune hyperparameters and 
evaulate the models on the validation data set.
"""

import sys
import time
from pathlib import Path
from src.utils.device import get_best_device, mpi_setup
from src.utils.formatting import serialize_np_dict, print_evaluation_info
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.utils.window import (
    get_date_index_col,
    extract_sp500_winds

)
from src.data_processing.loading import load_csv_files
from src.visualization.plots import wfv_losses_plot
from src.utils.io import (
    artifact_paths_setup,
    save_to_csv,
    save_to_json
)
from src.training.train_nn import CandidatesGrid, MetricModel
from src.evaluation.evaluator import (
    Evaluator,
    EqualWeightCalculator
)
from src.data_processing.dataset import WFUtilities

# Model and Loss Libraries
from src.training.loss_functions import LossLibrary
from src.evaluation.metrics import MetricLibrary
from src.models.registry import NNModelLibrary


def run_tuning_pipeline(
    paths_config: dict,
    hparams_config: dict,
    features_config: dict, 
    model_name: str,
    grid_mode: str = 'one_model', 
    loss_mode: str = 'custom',
    loss_name: str | None = None,
    tune: bool = False,
    mpi: bool = False
):
    """
    Tune one model with all loss functions, train with best found hyperparameters on the 
    validation data.

    Args:
        paths_config (dict): Dictionary containing paths to all required diretories and files.
        hparams_config (dict): Dictionary containing default hyperparameters and tuning ranges.
        features_config (dict): Dictionary containing hyperparameter information 
            (eg. common features).
        model_name (str): Name of the neural network model acrchitecture to be run.
        grid_mode (str): `one_model` or `one`. 
            - If `one_model`, the pipeline takes one model 
            architecture and pairs it with all available loss functions (based on 'loss_mode'). 
            - If `one`, the pipeline tunes and trains one model-loss combination (model name and 
            loss name must be provided). Default = 'one_model'.
        loss_mode (str): `all` or `custom`. To use all loss functions including objective only 
            functions or only custom loss function. Default = `custom`
        loss_name (str): Name of the loss function to be used. This is only required when grid mode
            is `one`. Default = None.
        tune (bool): Toggle to tune hyperparameters or use default hyperparameters. Default = False.
        mpi (bool): Toggle the use of mpi for distributed training or tuning of model-loss combinations.
    """
    print('\n', '=' * 40, ' Training Grid Pipeline ', '=' * 40)
    start_time = time.time()
    
    artifacts_paths = artifact_paths_setup(paths_config)
    
    # Registering all NN models to the library
    models_module = paths_config['models_module']
    NNModelLibrary.autodiscover(models_module) # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file
    
    # -------------------- Loading Processed Data -------------------- #
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val']),
        'ba_val': Path(paths_config['processed_paths']['ba_val']),
        'benchmark_val': Path(paths_config['processed_paths']['benchmark_val'])
    }

    processed_dfs = load_csv_files(processed_files, index_dt=True)
    train_data = processed_dfs['processed_train']
    rets_train = processed_dfs['returns_train']

    val_data = processed_dfs['processed_val']
    rets_val = processed_dfs['returns_val']

    # Loaded BA Spread data for trading costs
    ba_val = processed_dfs['ba_val']

    # Loaded S&P 500 for benchmarking
    sp500_rets = processed_dfs['benchmark_val']

    print(
        'Train Shape:', train_data.shape,
        'Train Returns Shape:', rets_train.shape
    )
    print(
        'Test Shape:', val_data.shape,
        'Test Returns Shape:', rets_val.shape
    )
    print(
        'Test BA Spread Shape:', ba_val.shape,
        'S&P500 Returns Shape:', sp500_rets.shape
    )
    
    # -------------------- Prepare Validation Sets -------------------- #
    out_size = hparams_config['rolling_windows']['out_size']
    wf_utils = WFUtilities(out_size)
    num_steps, extra_days = wf_utils.calc_walk_steps(rets_val)

    if extra_days != 0:
        raise RuntimeError(
            'Validation data incorrectly adjusted. Number of days must be divisible by out_size.'
        )

    # y_val and the ba_spreads for the same windows for out of sample evaluation
    y_val, out_wind_idxs = wf_utils.build_eval_windows(rets_val)
    y_ba_val = wf_utils.build_ba_for_eval(ba_val, out_wind_idxs)
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
            'eval_winds': y_val,
            'ba_eval_winds': y_ba_val
        }
    else:
        tuner_eval_items = None

    if mpi:

        comm, global_rank, size, gpu_id, _ = mpi_setup()
        torch_device = get_best_device(gpu_id)

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
            print('!!!DEBUG: Rank 0 got empty allocation weights!!!')
        

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
    evaluator = Evaluator(y_val, y_ba_val, MetricLibrary.items())
    
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
    all_rets_file_name = artifacts_paths['wfv_rets_dir'] \
        / f'daily_rets_{results_suffix}.json'
    save_to_json(
        serialize_np_dict(all_daily_returns),
        all_rets_file_name
    )

    print_evaluation_info(
        out_win_date_cols=out_win_date_cols,
        avgerage_performance_metrics=avg_perf_metrics,
    )

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')