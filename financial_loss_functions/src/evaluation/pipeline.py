"""
Test set evaluation pipeline.
This file contains the entry point for the function to evaluate the selected 
models on the test data set.
"""

import sys
import time
import pandas as pd
from pathlib import Path
from src.evaluation.test_nn import WFTester
from src.data_processing.dataset import WFUtilities
from src.training.train_trad import TradModelsTrainer
from src.utils.device import get_best_device, mpi_setup
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.utils.formatting import (
    serialize_np_dict,
    print_evaluation_info,
    split_combo_names,
    reform_returns_w_dates
)
from src.evaluation.evaluator import (
    Evaluator, 
    EqualWeightCalculator
)
from src.utils.window import (
    get_date_index_col,
    extract_sp500_winds
)
from src.data_processing.loading import (
    load_csv_files,
    ArtifactDataExtractor
)
from src.utils.io import (
    save_to_csv,
    save_to_json,
    artifact_paths_setup
)

# Model, Loss and Metrics Libraries
from src.evaluation.metrics import MetricLibrary
from src.training.loss_functions import LossLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary


def run_evaluation_pipeline(
        paths_config: dict,
        hparams_config: dict,
        features_config: dict,
        prev_grid_mode: str, 
        model_losses: list[str],
        mpi: bool = False
):
    """
    Train and evaluate the selected model-loss combinations with best found hyperparameters
    on the test data.

    Args:
        paths_config (dict): Dictionary containing paths to all required diretories and files.
        hparams_config (dict): Dictionary containing default hyperparameters and tuning ranges.
        features_config (dict): Dictionary containing hyperparameter information 
            (eg. common features).
        prev_grid_mode (str): `one_model` or `one`. This is used to collect and aggregate artifacts
            from the previous stage (tuning).
        model_losses (list[str]): List of model-loss combinations that were selected to run on the 
            test data. It must be in format - [<model_name>-<loss_name>,...].
        mpi (bool): Toggle the use of mpi for distributed training and evaluation of model-loss 
            combinations. Default = False
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

    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'processed_test': Path(paths_config['processed_paths']['processed_test']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val']),
        'returns_test': Path(paths_config['processed_paths']['returns_test']),
        'ba_test': Path(paths_config['processed_paths']['ba_test']),
        'benchmark_test': Path(paths_config['processed_paths']['benchmark_test'])
    }

    processed_dfs = load_csv_files(processed_files, index_dt=True)


    #### Combine train and validation data ####
    train_data = pd.concat(
        [processed_dfs['processed_train'],processed_dfs['processed_val']],
        axis=0
    )
    rets_train = pd.concat([
        processed_dfs['returns_train'], processed_dfs['returns_val']],
        axis=0
    )
    
    test_data = processed_dfs['processed_test']
    rets_test = processed_dfs['returns_test']

    # Loaded BA Spread data for trading costs
    ba_test = processed_dfs['ba_test']
    
    # Loaded S&P 500 for benchmarking
    sp500_rets = processed_dfs['benchmark_test']

    print(
        'Train Shape:', train_data.shape, ','
        'Train Returns Shape:', rets_train.shape
    )
    print(
        'Test Shape:', test_data.shape, ','
        'Test Returns Shape:', rets_test.shape
    )
    print(
        'Test BA Spread Shape:', ba_test.shape, ','
        'S&P500 Returns Shape:', sp500_rets.shape
    )
    

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
    print('Rolling Windows Configuration:', hparams_config['rolling_windows'])
    out_size = hparams_config['rolling_windows']['out_size']
    wf_utils = WFUtilities(out_size)
    num_steps, extra_days = wf_utils.calc_walk_steps(rets_test)

    ## Use WFUtilities.init_datasets to adjust extra days if needed ##

    # y_val for out of sample evaluation
    y_test, out_wind_idxs = wf_utils.build_eval_windows(rets_test)
    y_ba_test = wf_utils.build_ba_for_eval(ba_test, out_wind_idxs)

    if extra_days > 0:
        print(f'There are {extra_days} extra days in the test set.')
    
    # -------------------- Neural Networks Walk-Forward Evaluation -------------------- #
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
            results_suffix = f'{selected_combos[0][0]}{MODEL_LOSS_SEP}{selected_combos[0][1]}'

        else:
            raise RuntimeError('Incorrect mode arguments while running at entry point.')

    # Plot training and validation loss curves
    nn_train_loss_curves = wf_tester.get_train_val_losses()
    loss_curves_file = artifacts_paths['test_perf_dir'] \
        / f'test_loss_curves_{results_suffix}.json'
    save_to_json(
        nn_train_loss_curves,
        loss_curves_file
    )
    
    # for model_loss, model_loss_curves in nn_train_loss_curves.items():
    #     test_plot_name = model_loss + ' Test WFV Losses'
    #     wfv_losses_plot(
    #         model_loss_curves['train'],
    #         model_loss_curves['eval'],
    #         test_plot_name,
    #         artifacts_paths['test_plots_dir'] / (test_plot_name + '.png')
    #     )
    
    # -------------------- Neural Networks Returns -------------------- #
    # Extract dates index columns for the respective output windows
    out_win_date_cols = get_date_index_col(rets_test, out_wind_idxs) # For reformatting
    
    nn_daily_returns = {} # store without dates, for evaluation
    nn_daily_rets_w_dates = {} # store with dates
    seed_list = hparams_config.get('seed_list')
    if seed_list:
        for seed in seed_list:
            # Initializing once for ever seed
            evaluator = Evaluator(y_test, y_ba_test)
    
            # Calculate returns of all predicted portfolio allocation weights
            # Calling on every models output allocation weights to calculate pf returns
            for model_loss, alloc_weights in nn_alloc_weights.items():
                evaluator.calc_pf_daily_rets(alloc_weights[seed], model_loss)
            
            seed_returns = evaluator.get_all_daily_returns()
            nn_daily_returns[seed] = seed_returns
            
            seed_rets_reform = reform_returns_w_dates(
                serialize_np_dict(seed_returns),
                out_win_date_cols
            )

            nn_daily_rets_w_dates[seed] = seed_rets_reform
    else:
        # Initializing once for ever seed
        evaluator = Evaluator(y_test, y_ba_test)

        # Calculate returns of all predicted portfolio allocation weights
        # Calling on every models output allocation weights to calculate pf returns
        for model_loss, alloc_weights in nn_alloc_weights.items():
            evaluator.calc_pf_daily_rets(alloc_weights, model_loss)

            nn_daily_returns = evaluator.get_all_daily_returns()

            nn_daily_rets_w_dates = reform_returns_w_dates(
                serialize_np_dict(nn_daily_returns),
                out_win_date_cols
            )

    # del wf_testers

    # -------------------- Tradional Models Walk-Forward Evaluation -------------------- #      
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

    # -------------------- Benchmark Returns -------------------- # 
    evaluator = Evaluator(y_test, y_ba_test)
    for trad_model_name, alloc_weights in trad_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)

    # Calculate Equal Weight Portfolio's weights
    eq_wt_calc = EqualWeightCalculator(y_test)
    eq_wt_rets = eq_wt_calc.calc_eq_wt_daily_rets()

     # Extract s&p500 returns column sliced for the respective output windows
    sp500_rets_winds = extract_sp500_winds(
        sp500_rets,
        features_config['sp500_returns'],
        out_wind_idxs
    )

    # Adding s&p500 & equal weight returns to the evaluator as a benchmarks
    evaluator.add_benchmark_rets(EQ_WT_NAME, eq_wt_rets)
    evaluator.add_benchmark_rets(SP500_NAME, sp500_rets_winds)
    
    bench_daily_returns = evaluator.get_all_daily_returns()
    bench_daily_rets_w_dates = reform_returns_w_dates(
        serialize_np_dict(bench_daily_returns),
        out_win_date_cols
    )
    
    # -------------------- Combining and Savining All Returns -------------------- # 
    all_daily_rets_w_dates = {
        'nn_models': nn_daily_rets_w_dates,
        'benchmarks': bench_daily_rets_w_dates
    }

    all_rets_file_name = artifacts_paths['test_perf_dir'] \
        / f'test_returns_{results_suffix}.json'
    save_to_json(
        all_daily_rets_w_dates,
        all_rets_file_name
    )

    # # -------------------- Overall Evaluation on Out-of-Sample data -------------------- # 

    if seed_list:
        seed_metric_dfs = []
        for seed, seed_returns in nn_daily_returns.items():
            # Initializing once for ever seed
            evaluator = Evaluator(
                eval_returns=None,
                ba_eval=None,
                all_daily_returns=seed_returns,
                metrics_lib=MetricLibrary.items()
            )
            seed_metric_dfs.append(evaluator.calc_avg_performance())
        
        evaluator = Evaluator(
            eval_returns=None,
            ba_eval=None,
            all_daily_returns=bench_daily_returns,
            metrics_lib=MetricLibrary.items()
        )
        seed_metric_dfs.append(evaluator.calc_avg_performance())

        avg_perf_metrics = pd.concat(seed_metric_dfs).groupby(level=0).mean()
        perf_file_name = artifacts_paths['test_perf_dir'] \
            / f'avg_test_perf_{results_suffix}_{len(seed_list)}_seeds.csv'
    
    else:
        evaluator = Evaluator(
            eval_returns=None,
            ba_eval=None,
            all_daily_returns = nn_daily_returns | bench_daily_returns,
            metrics_lib = MetricLibrary.items()
        )
        avg_perf_metrics = evaluator.calc_avg_performance()

        perf_file_name = artifacts_paths['test_perf_dir'] \
            / f'avg_test_perf_{results_suffix}_no_seeds.csv'
    
    
    save_to_csv(avg_perf_metrics, perf_file_name)

    print_evaluation_info(
        out_win_date_cols=out_win_date_cols,
        avgerage_performance_metrics=avg_perf_metrics,
    )

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')