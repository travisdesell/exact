"""
Data Processing Pipeline.
This file contains the entry point for the data preprocessing pipeline.
It loads the raw CRSP data, cleans, preprocesses and saves the processed data files 
in 'data/processed/ directory'.
"""

from pathlib import Path
from src.data_processing.dataset import WFUtilities
from src.utils.io import reset_data_stage, save_to_csv
from src.data_processing.loading import load_raw_crsp_datasets
from src.data_processing.preprocess_crsp import clean_inplace, Preprocessor


def run_processing_pipeline(
        paths_config: dict,
        hparams_config: dict,
        features_config: dict
    ):
    """
    Data Processing Pipeline entry point function. It loads, cleans and normalizes the 
    all the data splits.

    Args:
        paths_config (dict): Dictionary containing all paths that are required.
        hparams_config (dict): Dictionary containing rolling window information for adjusting 
            train and validation data sets.
        features_config (dict): Dictionary containing features information.
    """
    print('=' * 20, ' Data Processing Pipeline ', '=' * 20)
    
    # Reset directory
    reset_data_stage(Path(paths_config['data']['processed_dir']))
    
    # -------------------- Data Loading -------------------- #
    # Extracting raw data paths for crsp
    crsp_data_path = Path(paths_config['data']['crsp_dir'])
    train_path = crsp_data_path / paths_config['raw_files']['train']
    val_path = crsp_data_path / paths_config['raw_files']['val']
    test_path = crsp_data_path / paths_config['raw_files']['test']
    
    # Loading CRSP Dataset
    train_data, val_data, test_data = load_raw_crsp_datasets(
        train_path,
        val_path,
        test_path
    )
    
    # -------------------- CRSP Cleaning -------------------- #
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

    # -------------------- Dataset adjustement -------------------- #
    # Extra days from validation is moved to train data to make the validation data be 
    # divided equally among the walk steps.
    data_adjuster = WFUtilities(hparams_config['rolling_windows']['out_size'])
    train_data, val_data = data_adjuster.init_datasets(train_data, val_data)

    # -------------------- Split S&P 500 Returns -------------------- #
    # Extract and save S&P 500 returns for benchmarking.
    sp500_col_name = features_config['sp500_returns']
    
    save_to_csv(
        train_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_train'])
    )

    save_to_csv(
        val_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_val'])
    )

    save_to_csv(
        test_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_test'])
    )

    # -------------------- Preporcessing -------------------- #

    # Preprocessing for NN
    nn_preprocessor = Preprocessor(
        common_features = features_config['common_features']
    )
    processed_train, ret_train = nn_preprocessor.process_train_data(train_data)
    processed_val, ret_val, ba_val = nn_preprocessor.process_split_data(val_data)
    processed_test, ret_test, ba_test = nn_preprocessor.process_split_data(test_data)

    print('Shape of train data:', processed_train.shape)
    print('Shape of validation data:', processed_val.shape)
    print('Shape of test data:', processed_test.shape)

    # -------------------- Saving Processed Files -------------------- #
    # Save returns data
    save_to_csv(
        ret_train,
        Path(paths_config['processed_paths']['returns_train'])
    )
    save_to_csv(
        ret_val,
        Path(paths_config['processed_paths']['returns_val'])
    )
    save_to_csv(
        ret_test,
        Path(paths_config['processed_paths']['returns_test'])
    )
    
    print('Returns data extracted and saved.')

    # Save Bid-Ask Spread data for validation and test sets
    save_to_csv(
        ba_val,
        Path(paths_config['processed_paths']['ba_val'])
    )

    save_to_csv(
        ba_test,
        Path(paths_config['processed_paths']['ba_test'])
    )

    print('BA Spread for validation and test saved.')
    
    # Save all features data
    save_to_csv(
        processed_train,
        Path(paths_config['processed_paths']['processed_train'])
    )
    save_to_csv(
        processed_val,
        Path(paths_config['processed_paths']['processed_val'])
    )
    save_to_csv(
        processed_test,
        Path(paths_config['processed_paths']['processed_test'])
    )

    print('Preprocessing for Neural Networks completed.')

    print('=' * 20, ' All Data Processing Completed! ', '=' * 20)
