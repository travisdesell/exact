from pathlib import Path
from src.utils import create_directory
from src.data_processing.dataset import Reshaper
from src.data_processing.dataset import WindowDataset
from src.data_processing.loading import load_csv_files
from src.training.train import (
    Evaluator,
    CandidatesGrid,
    TradModelsTrainer
)

# Model and Loss Libraries
from src.training.loss_functions import LossLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary

def load_processed_data(paths_config: dict) -> tuple:
    
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
    
def preprocess(
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

    return X_train, y_train, X_val, y_val


def run_training_pipeline(
        paths_config: dict,
        hparams_config: dict, 
        grid_mode: str = 'all', 
        loss_mode: str = 'all',
        model: str | None = None,
        loss: str | None = None
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
    print('\n', '=' * 20, ' Training Pipeline ', '=' * 20)
    
    # Create plots directory if it doesnt exist
    plots_dir = (Path(paths_config['artifacts']['plots']))
    create_directory(plots_dir)
    results_dir = Path(paths_config['artifacts']['results'])
    models_module = paths_config['models_module']
    
    # -------------------- Loading Processed Data -------------------- #
    train_data, returns_train, val_data, returns_val = load_processed_data(paths_config)
    
    # -------------------- Preprocessing (Reshaping) -------------------- #
    X_train, y_train, X_val, y_val = preprocess(
        train_data,
        returns_train,
        val_data,
        returns_val,
        hparams_config
    )

    # -------------------- Training Tradional Models -------------------- #
    # Initializing once to compare all models together
    evaluator = Evaluator(y_val)

    # Registering all Traditional models to the library
    TradModelLibrary.autodiscover(models_module)

    trad_grid = TradModelsTrainer(
        TradModelLibrary.items(),
        hparams_config['rolling_windows']['in_size'],
        hparams_config['rolling_windows']['out_size'],
        hparams_config['rolling_windows']['stride']
    )
    trad_alloc_weights = trad_grid.train_all(returns_train, returns_val)

    for model_name, alloc_weights in trad_alloc_weights.items():
        evaluator.calc_pf_daily_rets(alloc_weights, model_name)
    
    del trad_grid

    # -------------------- Training Neural Network Models -------------------- #
    # Registering all NN models to the library
    NNModelLibrary.autodiscover(models_module) # MUST be executed for model registration
    # No auto discovery needed for Loss library as all functions are in one file
    
    # Converting to pytorch tensors
    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    candidates_grid = CandidatesGrid(
        model_lib = NNModelLibrary.items(),
        loss_lib = LossLibrary.items(),
        hparams_config = hparams_config,
        results_dir = results_dir,
        loss_mode = loss_mode
    )
    if grid_mode == 'all':
        nn_alloc_weights = candidates_grid.train_eval_grid(train_ds, val_ds)
    elif grid_mode == 'one_model' and model is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_model(
            model, train_ds, val_ds
        )
    elif grid_mode == 'one_loss' and loss is not None:
        nn_alloc_weights = candidates_grid.train_eval_one_loss(loss, train_ds, val_ds)
    else:
        raise RuntimeError('Incorrect mode arguments while running at entry point.')

    # Calculate returns of all predicted portfolio allocation weights
    for loss_name, models_dict in nn_alloc_weights.items():
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
    total_returns.to_csv(results_dir / 'total_returns.csv', sep=',')
    total_sharpes = evaluator.calc_total_performance('sharpe')
    total_sharpes.to_csv(results_dir / 'total_sharpes.csv', sep=',')

    print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)
    print('\n', 'Compounded returns for each window:\n', total_returns)
    print('\n', 'Basic sharpe ratios for each window:\n', total_sharpes)

    #### TODO: 
    # 1. Implement a combination loss


# #### FOR TESTING & REFACTORING ####
#     from src.training.train import Trainer
#     from src.models.lstm import BaseLSTM
#     from torch import optim
#     from src.training.loss_functions import differentiable_sharpe_loss
#     from src.training.train import train_val_losses_plot
#     #### BaseLSTM ####
#     model1_name = 'BaseLSTM'
#     print('\n', '-'*10, f' Training {model1_name} ', '-'*10)
#     try:
#         trainer = Trainer(
#             model=BaseLSTM,
#             optimizer=optim.AdamW,
#             loss=differentiable_sharpe_loss,
#             model_hparams=hparams_config[model1_name]['model'],
#             optimizer_hparams=hparams_config[model1_name]['optimizer'],
#             train_hparams=hparams_config[model1_name]['train'],
#             in_size=X_train.shape[2],
#             num_stocks=y_train.shape[2]
#         )

#         trainer.train(train_ds)
#         trainer.evaluate(val_ds)

#         # Plot loss curves
#         train_val_losses_plot(
#             trainer.train_losses,
#             trainer.val_losses,
#             model1_name + ' Loss Curves',
#             Path(paths_config['artifacts']['plots']) /
#             (model1_name + ' Loss Curves' + '.png')
#         )

#         alloc_weights = trainer.get_val_alloc_weights()

#         # Call on every models output allocation weights to caluclated weighted returns
#         # Add daily returns for BaseLSTM generated weights
#         # evaluator.calc_pf_daily_rets(alloc_weights, model1_name)
    
#     except Exception as error:
#         print(f'DEBUG: Error while training {model1_name}. Skipping.', error)