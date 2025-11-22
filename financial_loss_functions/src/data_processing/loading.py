import os
import pandas as pd
from typing import Tuple

def load_raw_crsp_datasets(
        train_path: str, val_path: str, test_path: str
    )-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CRSP datasets files from a directory which are split into train,
    validation and test.

    Parameters
    ----------
    train_path: str
        Path to raw train data file
    val_path: str
        Path to raw validation data file
    test_path: str
        Path to raw test data file
    
    Returns
    -------
    train_data: pd.DataFrame
        Raw train data
    val_data: pd.DataFrame
        Raw validation data
    test_data: pd.DataFrame
        Raw test data
    """
    # Check if all files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Required file not found: {path}'
            )

    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def load_cov_processed(cov_train_path: str, corr_train_path: str):
    """
    Loads processed data files for cov-based training
    """
    cov_train = pd.read_csv(cov_train_path, index_col=0)
    corr_train = pd.read_csv(corr_train_path, index_col=0)

    return cov_train, corr_train