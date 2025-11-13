import os
import numpy as np
import pandas as pd
from typing import Tuple

def load_crsp_datasets(dir_path: str)-> Tuple[
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

def get_only_returns(
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

def clean_inplace(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Cleans dataset by removing dupilcate columns and duplicate rows.
    This process is inplace, i.e., Refrence of dataset is used, not copy.
    
    Parameters
    ---------
    train: pd.DataFrame
        train data
    val: pd.DataFrame
        validation data
    test: pd.DataFrame
        test data
    """

    features = train.columns
    if not np.array_equal(train.columns, val.columns) \
        or not np.array_equal(train.columns, test.columns):
        raise ValueError('ERROR: Columns do not match!')
    
    # Remove duplicate s&p500 returns columns
    dup_sp500 = []
    for col in features:
        if 'sprtrn' in col:
            dup_sp500.append(col)
    
    if len(dup_sp500) > 1:
        train.drop(columns=dup_sp500[1:], axis=1, inplace=True)
        val.drop(columns=dup_sp500[1:], axis=1, inplace=True)
        test.drop(columns=dup_sp500[1:], axis=1, inplace=True)

        train.rename(columns={dup_sp500[0]: 'sprtrn'}, inplace=True)
        val.rename(columns={dup_sp500[0]: 'sprtrn'}, inplace=True)
        test.rename(columns={dup_sp500[0]: 'sprtrn'}, inplace=True)

    # Remove duplicate date rows
    train.drop_duplicates(subset=['date'], keep='first', inplace=True)
    val.drop_duplicates(subset=['date'], keep='first', inplace=True)
    test.drop_duplicates(subset=['date'], keep='first', inplace=True)

    # return train, val, test

    