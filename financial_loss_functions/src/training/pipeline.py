from torch import optim
from typing import Dict
from pathlib import Path
from src.data_processing.dataset import Reshaper
from src.data_processing.dataset import WindowDataset
from src.data_processing.loading import load_csv_files
from src.models.lstm import BaseLSTM, SimpleAttentionLSTM
from src.training.train import (
    Trainer,
    train_val_losses_plot,
    Evaluator
)
from src.training.loss_functions import (
    raw_sharpe_loss,
    differentiable_sharpe_loss
)

def run_training_pipeline(paths_config: Dict, hparams_config: Dict):
    print('=' * 20, ' Training Pipeline ', '=' * 20)
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
    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    # BaseLSTM
    model1_name = 'BaseLSTM'
    print('\n')
    print('-'*10, f' Training {model1_name} ', '-'*10)
    trainer = Trainer(
        model=BaseLSTM,
        optimizer=optim.AdamW,
        loss=differentiable_sharpe_loss,
        hparams=hparams_config['BaseLSTM'],
        in_size=X_train.shape[2],
        num_stocks=y_train.shape[2]
    )

    trainer.train(train_ds)
    trainer.evaluate(val_ds)

    train_val_losses_plot(
        trainer.train_losses,
        trainer.val_losses,
        model1_name + ' Loss Curves',
        Path(paths_config['artifacts']['plots']) /
        (model1_name + ' Loss Curves' + '.png')
    )

    alloc_weights = trainer.get_val_alloc_weights()
    
    # We can initialize once
    evaluator = Evaluator(y_val)
    evaluator.calc_eq_wt_daily_rets()
    
    # Call on every models output allocation weights
    evaluator.calc_pf_daily_rets(alloc_weights, model1_name)
    
    # evaluator.plot_windowed_comparison(
    #     Path(paths_config['artifacts']['plots']) /
    #     (f'Daily Returns' + '.png')
    # )

    # Attention LSTM
    model2_name = 'AttentionLSTM'
    print('\n')
    print('-'*10, f' Training {model2_name} ', '-'*10)
    trainer = Trainer(
        model=SimpleAttentionLSTM,
        optimizer=optim.AdamW,
        loss=differentiable_sharpe_loss,
        hparams=hparams_config['AttentionLSTM'],
        in_size=X_train.shape[2],
        num_stocks=y_train.shape[2]
    )

    trainer.train(train_ds)
    trainer.evaluate(val_ds)

    train_val_losses_plot(
        trainer.train_losses,
        trainer.val_losses,
        model2_name + ' Loss Curves',
        Path(paths_config['artifacts']['plots']) /
        (model2_name + ' Loss Curves' + '.png')
    )

    alloc_weights = trainer.get_val_alloc_weights()

    evaluator.calc_pf_daily_rets(alloc_weights, model2_name)
    
    evaluator.plot_windowed_comparison(
        Path(paths_config['artifacts']['plots']) /
        (f'Daily Returns' + '.png')
    )