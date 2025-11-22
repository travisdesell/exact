from typing import Dict
from pathlib import Path
from src.data_processing.loading import load_processed_files

def run_training_pipeline(paths_config: Dict):
    # -------------------- Loading Processed Data -------------------- #
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'processed_test': Path(paths_config['processed_paths']['processed_test'])
    }


    processed_dfs = load_processed_files(processed_files)