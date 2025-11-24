from typing import Dict
from pathlib import Path
from src.data_processing.loading import load_processed_files
from src.data_processing.preprocess import Reshaper

def run_training_pipeline(paths_config: Dict):
    # -------------------- Loading Processed Data -------------------- #
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'processed_test': Path(paths_config['processed_paths']['processed_test']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val']),
        'returns_test': Path(paths_config['processed_paths']['returns_test'])
    }


    processed_dfs = load_processed_files(processed_files)


    reshaper = Reshaper(252, 63, 63)
    train_data = processed_dfs['processed_train']
    returns_train = processed_dfs['returns_train']
    reshaper.reshape(train_data, returns_train)