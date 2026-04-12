import sys
import time
import pandas as pd
from pathlib import Path
from src.utils.device import get_best_device, mpi_setup
from src.evaluation.test_nn import WFTester
from src.data_processing.dataset import WFUtilities
from src.visualization.plots import wfv_losses_plot
from src.training.train_trad import TradModelsTrainer
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.utils.formatting import serialize_np_dict, print_evaluation_info, reformat_model_perfs
from src.evaluation.evaluator import (
    Evaluator, EqualWeightCalculator
)
from src.utils.window import (
    get_date_index_col,
    extract_sp500_winds
)
from src.data_processing.loading import (
    load_csv_files,
    load_sp500_rets,
    ArtifactDataExtractor
)
from src.utils.io import (
    save_to_csv,
    save_to_json,
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
    TradModelLibrary.autodiscover(models_module)
    # No auto discovery needed for Loss library as all functions are in one file

    # -------------------- Loading Processed Data -------------------- #
    process_data_tuple = _load_processed_data(
        paths_config
    )

    #### Combine train and validation data ####
    train_data = pd.concat([process_data_tuple[0], process_data_tuple[2]], axis=0)
    rets_train = pd.concat([process_data_tuple[1], process_data_tuple[3]], axis=0)
    
    test_data = process_data_tuple[4]
    rets_test = process_data_tuple[5]

    print('Train shape:', train_data.shape)
    print('Test shape:', test_data.shape)
    
    # Loading S&P 500 for benchmarking
    sp500_rets = load_sp500_rets(paths_config['processed_paths']['benchmark_test'])

    # -------------------- Loading Relevant Training Artifacts -------------------- #
    selected_combos = split_combo_names(model_losses, MODEL_LOSS_SEP)
    
    artifacts_extrator = ArtifactDataExtractor(
        prev_grid_mode,
        artifacts_paths
    )

    if prev_grid_mode == 'one_model':
        model_names = list(set(modl_loss[0] for modl_loss in selected_combos))
        # all_avg_perf = artifacts_extrator.agg_avg_perf('avg_perf', model_names)
        opti_hparams = artifacts_extrator.agg_opti_hparams('optimized', model_names)

    
    elif prev_grid_mode == 'one' and len(selected_combos) == 1:
        opti_hparams = artifacts_extrator.agg_opti_hparams(
            'optimized',
            [f'{selected_combos[0][0]}{MODEL_LOSS_SEP}{selected_combos[0][1]}']
        )

    # Remove model+losses that are not relevant
    for key in list(opti_hparams.keys()):
        if key not in model_losses:
            del opti_hparams[key]
    
    # -------------------- Prepare Test Set -------------------- #
    out_size = hparams_config['rolling_windows']['out_size']
    wf_utils = WFUtilities(out_size)
    num_steps, extra_days = wf_utils.calc_walk_steps(rets_test)

    # Use WFUtilities.init_datasets to adjust extra days

    # y_val for out of sample evaluation
    y_val, out_wind_idxs = wf_utils.build_eval_windows(rets_test)

    if extra_days > 0:
        print(f'There are {extra_days} extra days in the test set.')
    
    # -------------------- Walk-Forward Evaluation -------------------- #
    if mpi:

        comm, global_rank, size, gpu_id, _ = mpi_setup()
        torch_device = get_best_device(gpu_id)

        wf_tester = WFTester(
            model_lib = NNModelLibrary.items(),
            loss_lib = LossLibrary.items(),
            hparams_config = hparams_config,
            num_steps = num_steps,
            common_features = features_config['common_features'],
            torch_device = torch_device,
            mpi = mpi,
            temp_dir = artifacts_paths['temp_dir'],
            optimized_hparams = opti_hparams
        )

        if prev_grid_mode == 'one_model':
            nn_alloc_weights = wf_tester.test_all(
                selected_combos, train_data, rets_train, test_data, 
                rets_test, comm, global_rank, size
            )
            results_suffix = 'all'
        else:
            raise RuntimeError(
                'Incorrect mode arguments while running at entry point.',
                'If mpi, previous grid mode must be `one_model`.'
            )
        
        # Stop all non zero ranks
        if global_rank != 0:
            print(f'Rank {global_rank}: Work complete. Shutting down.')
            sys.exit(0) # This stops the process for this rank only
        
        if nn_alloc_weights is None:
            print('!!!DEBUG: Rank 0 got empty allocation weights!!!')
    
    else:
        # Using default MPS or CUDA
        torch_device = get_best_device()

        wf_tester = WFTester(
            model_lib = NNModelLibrary.items(),
            loss_lib = LossLibrary.items(),
            hparams_config = hparams_config,
            num_steps = num_steps,
            common_features = features_config['common_features'],
            torch_device = torch_device,
            mpi = mpi,
            temp_dir = artifacts_paths['temp_dir'],
            optimized_hparams = opti_hparams
        )

        if prev_grid_mode == 'one_model':
            nn_alloc_weights = wf_tester.test_all(
                selected_combos, train_data, rets_train, test_data, rets_test, None, None, None
            )
            results_suffix = 'all'
        
        elif prev_grid_mode == 'one':
            nn_alloc_weights = wf_tester.test_all(
                selected_combos[0][0],
                selected_combos[0][1],
                train_data,
                rets_train,
                test_data,
                rets_test
            )
            results_suffix = f'{selected_combos[0][0]}-{selected_combos[0][1]}'

        else:
            raise RuntimeError('Incorrect mode arguments while running at entry point.')

    # Plot training and validation loss curves
    nn_train_loss_curves = wf_tester.get_train_val_losses()
    for model_loss, model_loss_curves in nn_train_loss_curves.items():
        test_plot_name = model_loss + ' Test WFV Losses'
        wfv_losses_plot(
            model_loss_curves['train'],
            model_loss_curves['eval'],
            test_plot_name,
            artifacts_paths['test_plots_dir'] / (test_plot_name + '.png')
        )
    
    # -------------------- Evaluator Setup -------------------- #
    
    # Initializing once to compare all models together
    evaluator = Evaluator(y_val, MetricLibrary.items())
    
    # Calculate returns of all predicted portfolio allocation weights
    # Calling on every models output allocation weights to calculate pf returns
    for model_loss, alloc_weights in nn_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, model_loss)

    # del wf_testers

    # -------------------- Training Tradional Models -------------------- #       
    print(f'Training All Tradional Models')
    trad_grid = TradModelsTrainer(
        TradModelLibrary.items(),
        hparams_config,
        num_steps
    )
    trad_alloc_weights = trad_grid.train_all(
        init_rets_train=rets_train,
        init_rets_split=rets_test
    )

    for trad_model_name, alloc_weights in trad_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)

    # -------------------- Evaluation on Out-of-Sample data -------------------- # 
    # Extract s&p500 returns column sliced for the respective output windows
    sp500_rets_winds = extract_sp500_winds(
        sp500_rets,
        features_config['sp500_returns'],
        out_wind_idxs
    )
    
    # Calculate Equal Weight Portfolio's weights
    eq_wt_calc = EqualWeightCalculator(y_val)
    eq_wt_rets = eq_wt_calc.calc_eq_wt_daily_rets()

    # Adding s&p500 & equal weight returns to the evaluator as a benchmarks
    evaluator.add_benchmark_rets(EQ_WT_NAME, eq_wt_rets)
    evaluator.add_benchmark_rets(SP500_NAME, sp500_rets_winds)

    # Get all daily returns
    all_daily_returns = evaluator.get_all_daily_returns()

    # Extract dates index columns for the respective output windows
    out_win_date_cols = get_date_index_col(rets_test, out_wind_idxs)

    # Combine all allocation weights
    all_alloc_wts = nn_alloc_weights | trad_alloc_weights
    
    # Serialize dicts and combine everything
    all_daily_returns = reformat_model_perfs(
        serialize_np_dict(all_daily_returns),
        serialize_np_dict(all_alloc_wts),
        out_win_date_cols
    )
    # Save all performance information (daily returns and weights)
    all_rets_file_name = artifacts_paths['test_perf_dir'] \
        / f'test_performance_{results_suffix}.json'
    save_to_json(
        all_daily_returns,
        all_rets_file_name
    )

    # Save average performance
    perf_file_name = artifacts_paths['test_perf_dir'] \
        / f'avg_test_perf_{results_suffix}.csv'
    avg_perf_metrics = evaluator.calc_avg_performance()
    save_to_csv(avg_perf_metrics, perf_file_name)


    print_evaluation_info(
        out_win_date_cols=out_win_date_cols,
        avgerage_performance_metrics=avg_perf_metrics,
    )

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')