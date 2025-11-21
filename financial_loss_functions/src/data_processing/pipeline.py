import sys
from src.utils import data_dir_check
from src.data_processing.preprocess import (
    load_crsp_datasets,
    get_only_returns,
    preprocess_cov,
    clean_inplace,
    Preprocessor
)

def run_processing_pipeline(crsp_data_path: str, processed_data_path: str):
    print('\n','=' * 20, ' Data Processing Pipeline ', '=' * 20)
    
    if not data_dir_check(processed_data_path):
        sys.exit('Data Processing Pipeline Aborted!')
    
    # -------------------- Data Loading -------------------- #
    # Loading CRSP Dataset
    train_data, val_data, test_data = load_crsp_datasets(crsp_data_path)

    # -------------------- Cleaning & Processing -------------------- #
    # Clean dataset inplace
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

    nn_preprocessor = Preprocessor(252*3, 90, 90)
    nn_preprocessor.process_train_data(train_data)