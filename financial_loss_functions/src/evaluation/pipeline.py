import time
import pandas as pd
from pathlib import Path
from src.utils.device import get_best_device
from src.evaluation.evaluate_nn import WFTester
from src.data_processing.dataset import WFUtilities
from src.visualization.plots import wfv_losses_plot
from src.utils.constants import EQ_WT_NAME, SP500_NAME, MODEL_LOSS_SEP
from src.evaluation.evaluator import (
    Evaluator, EqualWeightCalculator
)
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
    selected_combos = split_combo_names(model_losses, '-')
    
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
        print('MPI VERSION NOT IMPLEMENTED YET! EXITING...')
        exit(0)
    
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
            results_suffix = 'All'
        
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

    del wf_tester

    # -------------------- Training Tradional Models -------------------- #       
    print(f'Training All Tradional Models')
    trad_grid = TradModelsTrainer(
        TradModelLibrary.items(),
        hparams_config,
        num_steps
    )
    trad_alloc_weights = trad_grid.train_all(
        init_rets_train=rets_train,
        init_rets_split=rets_val
    )

    for trad_model_name, alloc_weights in trad_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
    
    del trad_grid

    # -------------------- Evaluation on Out-of-Sample data -------------------- # 
    #### TODO: Continue with evaulation



    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')