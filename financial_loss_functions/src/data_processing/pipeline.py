import sys
from src.utils import data_dir_check
from src.data_processing.preprocess import (
    load_raw_crsp_datasets,
    clean_inplace,
    save_to_csv,
    CovPreprocessor,
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

    # -------------------- Processing for Cov Models -------------------- #
    
    cov_preprocessor = CovPreprocessor()
    cov_train, corr_train = cov_preprocessor.process_train_data(
        train_data,
        val_data
    )

    ret_test = cov_preprocessor.process_test_data(test_data)

    # Saves covariance and correlation of the returns
    save_to_csv(cov_train, processed_data_path, 'cov_train.csv')
    save_to_csv(corr_train, processed_data_path, 'corr_train.csv')
    save_to_csv(ret_test, processed_data_path, 'ret_test.csv')

    # -------------------- Processing for NN Models -------------------- #
    # Preprocessing for NN
    nn_preprocessor = Preprocessor(252*3, 90, 90)
    train = nn_preprocessor.process_train_data(train_data)
    val = nn_preprocessor.process_val_data(val_data)
    test = nn_preprocessor.process_test_data(test_data)

    print(train)