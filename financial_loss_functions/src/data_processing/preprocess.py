import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import PowerTransformer, RobustScaler


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

def clean_inplace(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Cleans dataset by removing dupilcate columns and duplicate rows. It makes date the index.
    This process is inplace, i.e., Refrence of dataset is used, not copy.
    
    Parameters
    ---------
    train: pd.DataFrame
        train data
    val: pd.DataFrame
        validation data
    test: pd.DataFrame
        test data
    
    Returns
    -------
    train: pd.DataFrame
        Cleaned train data
    val: pd.DataFrame
        Cleaned validation data
    test: pd.DataFrame
        Cleaned test data
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
    
    train['date'] = pd.to_datetime(train['date'])
    train.set_index('date', inplace=True)

    val['date'] = pd.to_datetime(val['date'])
    val.set_index('date', inplace=True)

    test['date'] = pd.to_datetime(test['date'])
    test.set_index('date', inplace=True)

    return train, val, test

def save_to_csv(data: pd.DataFrame, output_dir: str, filename: str):
    data.to_csv(os.path.join(output_dir, filename), sep=',')

class CovPreprocessor:
    def _get_only_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only return columns from each of the split datasets.

        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        
        Returns
        -------
        returns : pd.DataFrame
            Train data with only returns.
        """
        return_cols = []
        for col in data.columns:
            if 'RET' in col:
                return_cols.append(col)

        return data[return_cols]
        
    def process_train_data(
            self, train: pd.DataFrame, val: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combines train and Validation data, then calculates
        covariance and correlation matrices

        Parameters
        ----------
        train: pd.DataFrame
            Training split data
        val: pd.DataFrame
            Validation split data
        """
        train = pd.concat([train, val], axis=0)

        train = self._get_only_returns(train)

        cov_train = train.cov()
        corr_train = train.corr()

        return cov_train, corr_train
    
    def process_test_data(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts only returns columns for testing unsupervised models
        """
        test = self._get_only_returns(test)

        return test


class Preprocessor:
    def __init__(self, window_in: int, window_out: int, step: int):
        """
        Initialize Preprocessor which transorforms, normalizes and creates sliding windows.

        Parameters
        ----------
        window_in: int
            size of input window in days
        window_out: int
            size of output window in days
        step: int
            step size in days for rolling windows
        """
        self.window_in = window_in
        self.window_out = window_out
        self.step = step

        self._yeo_john = PowerTransformer(method='yeo-johnson', standardize=False)
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)
        self._robust_scaler = RobustScaler()

    def _extract_req_cols(self, columns_list: List, suffix: str) -> List:
        """
        Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

        Parameters
        ----------
        columns_list: List
            List of all column names.
        suffix: str
            Suffix str to extract its respective columns. e.g., VOL_CHANGE, RETURN
        
        Return
        ------
        required_cols: List
            List of required column names for the given suffix
        """
        required_cols = [col for col in columns_list if suffix in col]
        return required_cols

    def _transform(self, data: pd.DataFrame, mode: str):
        """
        Transformation of data
        """
        vol_change_cols = self._extract_req_cols(self.all_col_names, 'VOL_CHANGE')
        turnover_cols = self._extract_req_cols(self.all_col_names, 'TURNOVER')
        
        # For training split
        if mode == 'fit':
            # Yeo Johnson transformation for VOL_CHANGE
            data[vol_change_cols] = self._yeo_john.fit_transform(data[vol_change_cols])
            # Box-Cox transoformation for TURNOVER
            data[turnover_cols] = self._box_cox.fit_transform(data[turnover_cols])

        # For val or test split
        elif mode == 'split':
            data[vol_change_cols] = self._yeo_john.transform(data[vol_change_cols])
            data[turnover_cols] = self._box_cox.transform(data[turnover_cols])
        else:
            raise ValueError('ERROR: Incorrect mode. Must be `fit` or `split`')

        return data
    
    def _normalize(self, data: pd.DataFrame, mode: str):
        """
        Normalize data set
        """
        # For training split
        if mode == 'fit':
            data[data.columns]= self._robust_scaler.fit_transform(data)
        
        # For val or test split
        elif mode == 'split':
            data[data.columns] = self._robust_scaler.transform(data)
        
        else:
            raise ValueError('ERROR: Incorrect mode. Must be `fit` or `split`')
        
        return data

    def _create_rolling_windows(self):
        """
        Function to create rolling windows based on the input, output and step sizes.
        """
        # TODO: Create rolling windows using initalized sizes
        
        pass

    def process_train_data(self, train: pd.DataFrame)-> pd.DataFrame:
        """
        Preprocesses given training data
        """

        self.all_col_names = list(train.columns)

        train = self._transform(train, 'fit')
        print(train)

        train = self._normalize(train, 'fit')
        print(train)

        # TODO: 2. Call creation of rolling windows

        return train

    def process_val_data(self, val: pd.DataFrame) -> pd.DataFrame:
        pass

    def process_test_data(self, test: pd.DataFrame) -> pd.DataFrame:
        pass