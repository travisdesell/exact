import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import PowerTransformer, RobustScaler


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
        
def cov_preprocessor(
        train: pd.DataFrame, val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines train and Validation returns data, then calculates
    covariance and correlation matrices

    Parameters
    ----------
    train: pd.DataFrame
        Training split data, only returns
    val: pd.DataFrame
        Validation split data, only returns
    """
    train = pd.concat([train, val], axis=0)

    cov_train = train.cov()
    corr_train = train.corr()

    return cov_train, corr_train
    
class Preprocessor:
    def __init__(self):
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
        # self.window_in = window_in
        # self.window_out = window_out
        # self.step = step

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


    def process_train_data(self, train: pd.DataFrame)-> pd.DataFrame:
        """
        Preprocesses given training data

        Parameters
        ----------
        train: pd.DataFrame
            Training data
        
        Return
        ------
        processed_train: pd.DataFrame
            Preprocessed training data
        """

        self.all_col_names = list(train.columns)

        train = self._transform(train, 'fit')

        train = self._normalize(train, 'fit')

        return train

    def process_split_data(self, split_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses given validation or test data based on statistics 
        from the training data.

        Parameters
        ----------
        split_data: pd.DataFrame
            Validation or test data
        
        Return
        ------
        processed_split_data: pd.DataFrame
            Preprocessed validation or test data
        """
        split_data = self._transform(split_data, 'split')

        split_data = self._normalize(split_data, 'split')

        return split_data