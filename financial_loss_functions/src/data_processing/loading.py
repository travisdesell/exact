import os
import pandas as pd
from typing import Tuple

def load_raw_crsp_datasets(dir_path: str)-> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load all CRSP datasets files from a directory which are split into train,
    validation and test.

    Parameters
    ----------
    dir_path : str
        Path to directory where the data is stored.
    
    Returns
    -------
    train_data : pd.DataFrame
        Train data
    val_data : pd.DataFrame
        Validation data
    test_data : pd.DataFrame
        Test data
    """
    train_path = os.path.join(dir_path, 'combined_predictors_train.csv')
    val_path = os.path.join(dir_path, 'combined_predictors_validation.csv')
    test_path = os.path.join(dir_path, 'combined_predictors_test.csv')
    
    # Check if all files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Required file not found: {path}',
                'File names should be: combined_predictors_<split>.csv. <split> = train, val or test'
            )

    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def load_cov_processed(data_dir: str):
    """
    Loads processed data files for cov-based training
    """
    cov_train = pd.read_csv(os.path.join(data_dir, 'cov_train.csv'), index_col=0)
    corr_train = pd.read_csv(os.path.join(data_dir, 'corr_train.csv'), index_col=0)

    ret_test = pd.read_csv(os.path.join(data_dir, 'ret_test.csv'), index_col=0)

    return cov_train, corr_train, ret_test