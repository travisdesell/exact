import os
import pandas as pd
from typing import Tuple, List, Dict
from src.utils import check_if_files_exist

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
    check_if_files_exist([train_path, val_path, test_path])

    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def load_processed_files(paths_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Loads processed data files. Provide dictionary of 
    name key and path strings value to be loaded.
    
    Parameters
    ----------
    paths_dict: Dict[str, str]
        Dictionary of name key and path strings value to be loaded.
    
    Returns
    -------
    loaded_dfs: Dict[str, pd.DataFrame]
        Dictionary of name key and loaded dataframe as value
    """
    # Check if all files exist
    check_if_files_exist(list(paths_dict.values()))

    loaded_dfs = {}
    for name, f_path in paths_dict.items():
        temp_df = pd.read_csv(f_path, index_col=0)
        temp_df.index = pd.to_datetime(temp_df.index)
        loaded_dfs[name] = temp_df

    return loaded_dfs