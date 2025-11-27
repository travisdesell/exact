from torch import optim
from typing import Dict
from pathlib import Path
from src.data_processing.loading import load_csv_files
from src.data_processing.dataset import Reshaper
# from src.training.train import train_lstm_base
from src.models.lstm import FlattenedLSTM
from src.training.loss_functions import sharpe_loss
from src.training.train import Trainer
from src.data_processing.dataset import WindowDataset

def run_training_pipeline(paths_config: Dict, hparams_config: Dict):
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

    # print('Train shape:', train_data.shape)
    # print('Val shape:', val_data.shape)

    # -------------------- Preprocessing (Reshaping) -------------------- #
    reshaper = Reshaper(
        hparams_config['rolling_windows']['in_size'],
        hparams_config['rolling_windows']['out_size'],
        hparams_config['rolling_windows']['stride']
    )
    reshaper.extract_features(train_data)
    
    X_train, y_train, _ = reshaper.reshape(train_data, returns_train)
    # print('-'*10, ' train shapes ', '-'*10)
    # print('X_train shpe:', X_train.shape)
    # print('y_train shape:', y_train.shape)


    X_val, y_val, _ = reshaper.reshape(val_data, returns_val)
    # print('-'*10, ' val shapes ', '-'*10)
    # print('X_val shape', X_val.shape)
    # print('y_val shape:', y_val.shape)

    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    trainer = Trainer(
        model=FlattenedLSTM,
        optimizer=optim.AdamW,
        loss=sharpe_loss,
        hparams=hparams_config['lstm_base'],
        in_size=X_train.shape[2],
        out_size=y_train.shape[2]
    )

    trainer.train(train_ds)
    trainer.eval(val_ds)