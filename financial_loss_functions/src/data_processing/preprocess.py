import os
import pandas as pd
from typing import Tuple

def load_crsp_datasets(dir_path: str)-> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load all CRSP datasets files from a directory which are split into train, validation and test.

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
    train_path = os.path.join(dir_path, 'combined_parameters_train.csv')
    val_path = os.path.join(dir_path, 'combined_parameters_validation.csv')
    test_path = os.path.join(dir_path, 'combined_parameters_test.csv')
    
    # Check if all files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required file not found: {path}')

    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def clean_data_returns(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract only return columns from each of the split datasets.

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    val: pd.DataFrame
        Validation data.
    test: pd.DataFrame
        Test data.
    
    Returns
    -------
    train : pd.DataFrame
        Train data with only returns.
    val : pd.DataFrame
        Validation data with only returns.
    test : pd.DataFrame
        Test data with only returns.
    """
    return_cols = []
    for col in train.columns:
        if 'RET' in col:
            return_cols.append(col)

    return train[return_cols], val[return_cols], test[return_cols]
    
def preprocess_cov(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculation of covariance and correlation for unsupervised 
    covariance based models.

    Parameters
    ---------
    data: pd.DataFrame
        Train dataset to be processed
    
    Returns
    -------
    cov: pd.DataFrame
        Covariance matrix
    corr: pd.DataFrame
        Correlation matrix
    """
    cov = data.cov()
    corr = data.corr()

    return cov, corr