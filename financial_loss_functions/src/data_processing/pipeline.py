import sys
from src.utils import data_dir_check, save_to_csv
from src.data_processing.loading import load_raw_crsp_datasets
from src.data_processing.preprocess import (
    clean_inplace,
    get_only_returns,
    cov_preprocessor,
    Preprocessor
)


def run_processing_pipeline(crsp_data_path: str, processed_data_path: str):
    print('\n','=' * 20, ' Data Processing Pipeline ', '=' * 20)
    
    if not data_dir_check(processed_data_path):
        sys.exit('Data Processing Pipeline Aborted!')
    
    # -------------------- Data Loading -------------------- #
    # Loading CRSP Dataset
    train_data, val_data, test_data = load_raw_crsp_datasets(crsp_data_path)

    # -------------------- Cleaning -------------------- #
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

    
    # -------------------- Preporcessing -------------------- #
    # Common processing (realized returns)
    ret_train, ret_val, ret_test = get_only_returns(train_data, val_data, test_data)
    save_to_csv(ret_train, processed_data_path, 'ret_train.csv')
    save_to_csv(ret_val, processed_data_path, 'ret_val.csv')
    save_to_csv(ret_test, processed_data_path, 'ret_test.csv')
    
    print('Realized returns extracted and saved.')
    
    # Preprocessing for covariance based models
    cov_train, corr_train = cov_preprocessor(ret_train, ret_val)

    # Saves covariance and correlation of the returns
    save_to_csv(cov_train, processed_data_path, 'cov_train.csv')
    save_to_csv(corr_train, processed_data_path, 'corr_train.csv')

    print('Preprocessing for Cov models completed.')

    
    # Preprocessing for NN
    nn_preprocessor = Preprocessor(252*3, 90, 90)
    processed_train = nn_preprocessor.process_train_data(train_data)
    processed_val = nn_preprocessor.process_split_data(val_data)
    processed_test = nn_preprocessor.process_split_data(test_data)

    save_to_csv(processed_train, processed_data_path, 'processed_train.csv')
    save_to_csv(processed_val, processed_data_path, 'processed_val.csv')
    save_to_csv(processed_test, processed_data_path, 'processed_test.csv')

    print('Preprocessing for Neural Networks completed.')

    print('=' * 20, ' All Data Processing Completed! ', '=' * 20)