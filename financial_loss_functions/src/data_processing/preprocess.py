import os
import numpy as np
import pandas as pd
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
from typing import Tuple, List
=======
from typing import Tuple
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/preprocess.py
=======
from typing import Tuple, List
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
from typing import Tuple, List
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1
from sklearn.preprocessing import PowerTransformer


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
    train_path = os.path.join(dir_path, 'combined_predictors_train.csv')
    val_path = os.path.join(dir_path, 'combined_predictors_validation.csv')
    test_path = os.path.join(dir_path, 'combined_predictors_test.csv')
    
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

class Preprocessor:
    def __init__(self, window_in: int, window_out: int, step: int):
        self.window_in = window_in
        self.window_out = window_out
        self.step = step

        self._yeo_john = PowerTransformer(method='yeo-johnson', standardize=False)
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)
=======
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/preprocess.py
=======
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1

    def normalize():
        """
        Scaling
        """
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
=======
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1
        pass 
    
    def _extract_req_cols(self, columns_list: List, suffix: str):
        required_cols = [col for col in columns_list if suffix in col]
        return required_cols
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
=======
        pass

    def _yeo_johnson_transform(self, data: pd.DataFrame, suffix: str, mode: str):
        required_cols = [col for col in data.columns if suffix in col]
        if mode == 'fit':
            data[required_cols] = self._yeo_john.fit_transform(data[required_cols])
        elif mode == 'split':
            data[required_cols] = self._yeo_john.transform(data[required_cols])
        else: 
            raise ValueError('ERROR: Incorrect mode. Must be `fit` or `split`')
        
        return data
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/preprocess.py
=======
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1

    def transform(self, data, mode):
        """
        Transformation of data
        """
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
=======
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1

        # For training split
        if mode == 'fit':
            # Yeo Johnson transformation for VOL_CHANGE
            vol_change_cols = self._extract_req_cols(self.all_col_names, 'VOL_CHANGE')
            data[vol_change_cols] = self._yeo_john.fit_transform(data[vol_change_cols])

            # Box-Cox transoformation for TURNOVER
            turnover_cols = self._extract_req_cols(self.all_col_names, 'TURNOVER')
            data[turnover_cols] = self._box_cox.fit_transform(data[turnover_cols])

        elif mode == 'split':
            # TODO: Transformations on val or test
            pass

<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
=======
        data = self._yeo_johnson_transform(data, 'VOL_CHANGE', mode)
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/preprocess.py
=======
>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1
        return data
        
    def process_train_data(self, train: pd.DataFrame)-> pd.DataFrame:
        """
        Preprocesses given training data
        """
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
<<<<<<< HEAD:financial_loss_functions/src/data_processing/preprocess.py
        self.all_col_names = list(train.columns)

=======
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/preprocess.py
=======
        self.all_col_names = list(train.columns)

>>>>>>> 9e8eb40 (tests updated, box-cox added):financial_loss_functions/preprocess.py
=======
        self.all_col_names = list(train.columns)

>>>>>>> 96d6df7ab41d311095dd73a19e348b3abf0102e1
        train = self.transform(train, 'fit')
        print(train)

    def process_val_data(val: pd.DataFrame) -> pd.DataFrame:
        pass

    def process_test_data(test: pd.DataFrame) -> pd.DataFrame:
        pass