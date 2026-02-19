import time
from torch import optim
from pathlib import Path
from src.evaluation.evaluator import Evaluator
from src.data_processing.loading import load_csv_files
from src.utils.io import create_directory
from src.utils.device import get_best_device
from src.data_processing.dataset import (
    Reshaper,
    calc_in_out_idx,
    WindowDataset,
    get_date_index_col
)
from src.visualization.plots import (
    train_val_losses_plot, 
    plot_windowed_comparison
)
from src.training.train import (
    CandidatesGrid,
    TradModelsTrainer,
    Trainer
)

# Model and Loss Libraries
from src.training.loss_functions import LossLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary

# TODO:
# 1. Add other NN models
# 2. Add Best model ranker
# 3. Unit test NCO

def _common_setup(paths_config):
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
    
def _preprocess(
        train_data, returns_train, val_data, returns_val, hparams_config: dict
    ) -> tuple:
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

    # Calculate indexes for input and output windows on split data
    in_wind_indexes, out_wind_indexes = calc_in_out_idx(
        returns_val,
        hparams_config['rolling_windows']['in_size'],
        hparams_config['rolling_windows']['out_size'],
        hparams_config['rolling_windows']['stride']
    ) 

    return X_train, y_train, X_val, y_val, in_wind_indexes, out_wind_indexes

def run_training_pipeline(
        paths_config: dict,
        hparams_config: dict, 
        grid_mode: str = 'all', 
        loss_mode: str = 'all',
        model_name: str | None = None,
        loss_name: str | None = None
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
    train_data, returns_train, val_data, returns_val = _load_processed_data(paths_config)
    
    # -------------------- Preprocessing (Reshaping) -------------------- #
    X_train, y_train, X_val, y_val, in_wind_idxs, out_wind_idxs = _preprocess(
        train_data,
        returns_train,
        val_data,
        returns_val,
        hparams_config
    )

    # -------------------- Evaluator Setup -------------------- #

    # Initializing once to compare all models together
    evaluator = Evaluator(y_val)

    # -------------------- Training Tradional Models -------------------- #
    trad_grid = TradModelsTrainer(TradModelLibrary.items(), hparams_config)
    trad_alloc_weights = trad_grid.train_all(
        in_wind_idxs,
        out_wind_idxs,
        returns_train,
        returns_val
    )

    for trad_model_name, alloc_weights in trad_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
    
    del trad_grid

    # -------------------- Training Neural Network Models -------------------- #
    
    # Converting to pytorch tensors
    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    candidates_grid = CandidatesGrid(
        model_lib = NNModelLibrary.items(),
        loss_lib = LossLibrary.items(),
        hparams_config = hparams_config,
        torch_device=best_device,
        loss_mode = loss_mode
    )
    if grid_mode == 'all':
        nn_alloc_weights = candidates_grid.train_eval_grid(train_ds, val_ds)
    elif grid_mode == 'one_model' and model_name is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_model(
            model_name, train_ds, val_ds
        )
    elif grid_mode == 'one_loss' and loss_name is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_loss(loss_name, train_ds, val_ds)
    else:
        raise RuntimeError('Incorrect mode arguments while running at entry point.')

    # Plot training and validation loss curves
    nn_train_loss_curves = candidates_grid.get_train_val_losses()
    for model_loss, model_loss_curves in nn_train_loss_curves.items():
        loss_plot_name = model_loss + ' Loss Curves'
        train_val_losses_plot(
            model_loss_curves['train'],
            model_loss_curves['val'],
            loss_plot_name,
            plots_dir / (loss_plot_name + '.png')
        )

    # Calculate returns of all predicted portfolio allocation weights
    # Calling on every models output allocation weights to calculate pf returns
    for loss_name, models_dict in nn_alloc_weights.items():
        for nn_model_name, alloc_weights in models_dict.items():
            evaluator.calc_pf_daily_rets(alloc_weights, f'{nn_model_name}-{loss_name}')
    
    del candidates_grid
    
    # Overall Evaluation/Comparison starts here
    evaluator.calc_eq_wt_daily_rets()
    
    plot_windowed_comparison(
        evaluator.get_all_daily_returns(),
        get_date_index_col(val_data, out_wind_idxs),
        plots_dir / (f'Daily Returns' + '.png')
    )

    total_returns = evaluator.calc_total_performance('returns')
    total_returns.to_csv(results_dir / 'total_returns.csv', sep=',')
    total_sharpes = evaluator.calc_total_performance('sharpe')
    total_sharpes.to_csv(results_dir / 'total_sharpes.csv', sep=',')

    print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)
    print('\n', 'Compounded returns for each window:\n', total_returns)
    print('\n', 'Basic sharpe ratios for each window:\n', total_sharpes)

    time_taken = round((time.time() - start_time) / 60, 3)
    print(f'Time taken for pipeline = {time_taken} mins')

def run_training_one_model(
        paths_config: dict,
        hparams_config: dict,
        model_cat: str, 
        model_name: str,
        loss_name: str,
        loss_cat: str
    ):
    """
    Entry point to train one model with one loss function. 
    Both have to be specified in arguments.
    
    @param paths_config Dict Dictionary containing paths
    @param features_config Dictionary containing hyperparameter information
    @param model str Name of the model to be run
    @param loss str Name of the loss function to be used
    """
    print('\n', '=' * 40, ' Training One Model with One Loss ', '=' * 40)
    start_time = time.time()

    if loss_cat not in ['objectives', 'custom']:
        raise ValueError('Loss category must be `objectives` or `custom`.')
    
    plots_dir, results_dir, best_device = _common_setup(paths_config)
    
    # -------------------- Model and loss search -------------------- #    
    model_cls = NNModelLibrary.get(model_cat, model_name)
    loss_func = LossLibrary.get(loss_cat, loss_name)

    if model_cls and loss_func:
        # -------------------- Loading Processed Data -------------------- #
        train_data, returns_train, val_data, returns_val = _load_processed_data(paths_config)
        
        # -------------------- Preprocessing (Reshaping) -------------------- #
        X_train, y_train, X_val, y_val, in_wind_idxs, out_wind_idxs = _preprocess(
            train_data,
            returns_train,
            val_data,
            returns_val,
            hparams_config
        )

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)

        # -------------------- Evaluator Setup -------------------- #
        # Initializing once to compare all models together
        evaluator = Evaluator(y_val)

        # -------------------- Training Tradional Models -------------------- #
        trad_grid = TradModelsTrainer(TradModelLibrary.items(), hparams_config)
        trad_alloc_weights = trad_grid.train_all(
            in_wind_idxs,
            out_wind_idxs,
            returns_train,
            returns_val
        )

        for trad_model_name, alloc_weights in trad_alloc_weights.items():
            evaluator.calc_pf_daily_rets(alloc_weights, trad_model_name)
        
        del trad_grid

        # -------------------- Training Neural Network -------------------- #
        print('\n', '-'*10, f' Training {model_name}-{loss_name} ', '-'*10)
        try:
            trainer = Trainer(
                model=model_cls,
                optimizer=optim.AdamW,
                loss=loss_func,
                model_hparams=hparams_config['nn_models'][model_name]['model'],
                optimizer_hparams=hparams_config['nn_models'][model_name]['optimizer'],
                train_hparams=hparams_config['nn_models'][model_name]['train'],
                in_size=X_train.shape[2],
                num_stocks=y_train.shape[2],
                loss_hparams=hparams_config['losses'].get(loss_name),
                device=best_device
            )

            trainer.train(train_ds)
            trainer.evaluate(val_ds)

            loss_plot_name = model_name + f'-{loss_name}' + ' Loss Curves'
            
            # Plot loss curves
            train_val_losses_plot(
                trainer.train_losses,
                trainer.val_losses,
                loss_plot_name,
                plots_dir / (loss_plot_name + '.png')
            )

            alloc_weights = trainer.get_val_alloc_weights()

            # Call on every models output allocation weights to calculate pf returns
            evaluator.calc_pf_daily_rets(alloc_weights, f'{model_name}-{loss_name}')
        except KeyError as ke:
            print('KeyError: Key not found.', ke)
        except Exception as error:
            print(f'DEBUG: Error while training {model_name}. Skipping.', error)
        

        # Overall Evaluation/Comparison
        evaluator.calc_eq_wt_daily_rets()

        plot_windowed_comparison(
            evaluator.get_all_daily_returns(),
            get_date_index_col(val_data, out_wind_idxs),
            plots_dir /
            (f'Daily Returns_{model_name}-{loss_name}' + '.png')
        )

        total_returns = evaluator.calc_total_performance('returns')
        total_returns.to_csv(
            results_dir / f'total_returns_{model_name}-{loss_name}.csv', sep=','
        )
        total_sharpes = evaluator.calc_total_performance('sharpe')
        total_sharpes.to_csv(
            results_dir / f'total_sharpes_{model_name}-{loss_name}.csv', sep=','
        )

        print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)
        print('\n', 'COMPOUNDED RETURNS for each window:\n', total_returns)
        print('\n', 'SHARPE RATIOS for each window:\n', total_sharpes)

        time_taken = round((time.time() - start_time) / 60, 3)
        print(f'Time taken for pipeline = {time_taken} mins')
    
    elif model_cls is None:
        raise ValueError(f'Model {model_name} of {model_cat} not found.')

    else:
        raise ValueError(f'Loss Function {loss_name} not found.')