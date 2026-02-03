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
    create_directory(Path(paths_config['artifacts']['plots']))
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
    print(all_alloc_weights)
    
    #### TODO: 
    # 1. Implement evaluation for all model predictions
    # 2. Compare all in plot
    # 3. Implement a combination loss
    # See old code below
    

    






    ################################
    
    
    # #### BaseLSTM ####
    # model1_name = 'BaseLSTM'
    # model1 = ModelLibrary.get('lstm', model1_name)
    # loss = LossLibrary.get('objectives', name='differentiable_sharpe_loss')
    # print('\n', '-'*10, f' Training {model1_name} ', '-'*10)
    # try:
    #     trainer = Trainer(
    #         model=model1,
    #         optimizer=optim.AdamW,
    #         loss=loss,
    #         model_hparams=hparams_config[model1_name]['model'],
    #         optimizer_hparams=hparams_config[model1_name]['optimizer'],
    #         train_hparams=hparams_config[model1_name]['train'],
    #         in_size=X_train.shape[2],
    #         num_stocks=y_train.shape[2]
    #     )

    #     trainer.train(train_ds)
    #     trainer.evaluate(val_ds)

    #     # Plot loss curves
    #     train_val_losses_plot(
    #         trainer.train_losses,
    #         trainer.val_losses,
    #         model1_name + ' Loss Curves',
    #         Path(paths_config['artifacts']['plots']) /
    #         (model1_name + ' Loss Curves' + '.png')
    #     )

    #     alloc_weights = trainer.get_val_alloc_weights()

    #     # Call on every models output allocation weights to caluclated weighted returns
    #     # Add daily returns for BaseLSTM generated weights
    #     evaluator.calc_pf_daily_rets(alloc_weights, model1_name)
    
    # except Exception as error:
    #     print(f'DEBUG: Error while training {model1_name}. Skipping.', error)
    

    # #### Attention LSTM ####
    # model2_name = 'AttentionLSTM'
    # model2 = ModelLibrary.get('lstm', model2_name)
    # print('\n', '-'*10, f' Training {model2_name} ', '-'*10)
    # try:
    #     trainer = Trainer(
    #         model=model2,
    #         optimizer=optim.AdamW,
    #         loss=loss,
    #         model_hparams=hparams_config[model2_name]['model'],
    #         optimizer_hparams=hparams_config[model2_name]['optimizer'],
    #         train_hparams=hparams_config[model2_name]['train'],
    #         in_size=X_train.shape[2],
    #         num_stocks=y_train.shape[2]
    #     )

    #     trainer.train(train_ds)
    #     trainer.evaluate(val_ds)

    #     # Plot loss curves
    #     train_val_losses_plot(
    #         trainer.train_losses,
    #         trainer.val_losses,
    #         model2_name + ' Loss Curves',
    #         Path(paths_config['artifacts']['plots']) /
    #         (model2_name + ' Loss Curves' + '.png')
    #     )

    #     alloc_weights = trainer.get_val_alloc_weights()

    #     # Add daily returns for AttentionLSTM generated weights
    #     evaluator.calc_pf_daily_rets(alloc_weights, model2_name)
    # except Exception as error:
    #     print(f'DEBUG: Error while training {model2_name}. Skipping.', error)
    
    # # Evaluation/Comparison starts here
    # evaluator.calc_eq_wt_daily_rets()
    
    # evaluator.plot_windowed_comparison(
    #     Path(paths_config['artifacts']['plots']) /
    #     (f'Daily Returns' + '.png')
    # )

    # total_returns = evaluator.calc_total_performance('returns')
    # total_sharpes = evaluator.calc_total_performance('sharpe')

    # print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)
    # print('\n', 'Compounded returns for each window:\n', total_returns)
    # print('\n', 'Basic sharpe ratios for each window:\n', total_sharpes)