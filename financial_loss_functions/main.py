import os
import pandas as pd
from typing import Tuple
from dotenv import load_dotenv

load_dotenv("../.env")


def load_datasets(dir_path: str)-> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load all datasets from a directory which are split into train, validation and test.

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
    train_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_train.csv')
    val_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_validation.csv')
    test_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_test.csv')
    
    # Load split datasets
    train_data = pd.read_csv(train_path)

    val_data = pd.read_csv(val_path)

    test_data = pd.read_csv(test_path)
    return train_data, val_data, test_data


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR')
    train_data, val_data, test_data = load_datasets(data_dir)

    print('Test Data:')
    print(train_data)

    print('Validation Data:')
    print(val_data)

    print('Test Data:')
    print(test_data)