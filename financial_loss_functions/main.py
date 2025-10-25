import os
import pandas as pd
from typing import Tuple
from dotenv import load_dotenv

load_dotenv("../.env")


def load_datasets(dir_path: str)-> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    train_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_train.csv')
    val_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_validation.csv')
    test_path = os.path.join(dir_path, '2023_sp_500_select_50', 'combined_parameters_test.csv')
    
    # Load split datasets
    train_data = pd.read_csv(train_path)
    print(train_data)

    val_data = pd.read_csv(val_path)

    test_data = pd.read_csv(test_path)
    return train_data, val_path, test_path


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR')
    load_datasets(data_dir)