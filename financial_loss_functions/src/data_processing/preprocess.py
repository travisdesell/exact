import numpy as np
import pandas as pd
from enum import StrEnum
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
    return_suffix = '_RET'
    for col in train.columns:
        if return_suffix in col:
            return_cols.append(col)
    
    ret_train = train[return_cols]
    ret_val = val[return_cols]
    ret_test = test[return_cols]

    ret_train.columns = [col.replace(return_suffix, '') for col in return_cols]
    ret_val.columns = [col.replace(return_suffix, '') for col in return_cols]
    ret_test.columns = [col.replace(return_suffix, '') for col in return_cols]

    return ret_train, ret_val, ret_test
        
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
    def __init__(
            self, col_sep: str = '_', common_features: List[str] | None = None
        ):
        """
        Initialize Preprocessor which transorforms, normalizes and creates sliding windows.
        
        Parameters
        ----------
        col_sep: str
            Special character that separates the ticker string from the feature string.

        """
        self.common_features = common_features
        self.col_sep = col_sep
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
        vol_change_cols = self._extract_req_cols(self.all_col_names, '_VOL_CHANGE')
        turnover_cols = self._extract_req_cols(self.all_col_names, '_TURNOVER')
        
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

    def _extract_tickers(self):
        tickers = []
        for col in self.all_col_names :
            if col != 'date':
                ticker = col.split(self.col_sep, 1)[0]
                tickers.append(ticker)
        
        if self.common_features:
            tickers = [x for x in sorted(set(tickers)) if x not in self.common_features]
            return tickers
        else:
            return sorted(set(tickers))

    def _broadcast_common(self, data, features: List[str]) -> pd.DataFrame:
        """Broadcast common features to all tickers with names <ticker>_<common_feature>"""

        # Create a dict of new columns: {new_col_name: values} for all ticker-feature pairs
        new_cols = {
            f'{ticker}{self.col_sep}{feat}': data[feat] 
            for feat in features for ticker in self.all_tickers
        }

        # Drop the original common columns after broadcasting
        data = data.drop(columns=features).assign(**new_cols)
        
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
        self.all_tickers = self._extract_tickers()
        
        # III TODO: ##
        # 1. Combine date matched macro data (common features) with CRSP data (training data only)
        # No underscore must be present in macro data column names.

        train = self._transform(train, 'fit')

        train = self._normalize(train, 'fit')

        # IV TODO: ##
        # Combine column names from macro data with list `self.common features` for broadcasting features

        # Broadcast only if common features are present
        if self.common_features:
            train = self._broadcast_common(train, self.common_features)

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

        # TODO: ##
        # 1. Combine date matched macro data (common features) with CRSP data (val/test)

        split_data = self._transform(split_data, 'split')

        split_data = self._normalize(split_data, 'split')

        # Broadcast only if common features are present
        if self.common_features:
            split_data = self._broadcast_common(split_data, self.common_features)

        return split_data

class ReshapeStyle(StrEnum):
    """
    For fixed reshaping styles to avoid reshape errors.
    """
    T_NxF = 'T_NxF' # (Time Steps, Ticker x Features) -> 2D for 3D after Batch
    T_N_F = 'T_N_F' # (Time Steps, Tickers, Features) -> 3D for 4D after Batch
    T_F_N = 'T_F_N' # (Time Steps, Features, Tickers) -> 3D for 4D after Batch  

class Reshaper:
    """
    Reshapes a wide 2D DataFrame with columns like '<ticker>_<feature>',
    into 3D tensors (T, N_stocks, F_features), and builds sliding windows.

    Assumes all columns follow the pattern:
        <ticker>_<feature>
    """

    def __init__(
            self,
            in_size: int,
            out_size: int,
            stride: int,
            col_sep: str = '_',
            layout: ReshapeStyle = ReshapeStyle.T_NxF
        ):
        """
        Initialize Reshaper instance.

        Parameters
        ----------
        in_size: int
            size of input window in terms of time steps
        out_size: int
            size output window in terms of time steps
        stride: int
            step size for the sliding window
        col_sep: str
            Special character that separates the ticker string from the feature string.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        self.col_sep = col_sep
        self.layout = layout
        
        if not isinstance(layout, ReshapeStyle):
            raise TypeError(
                '`layout` must be of type ReshapeStyle to avoid errors.',
                'Use ReshapeStyle from src.data_processing.preprocess.'
            )

        self.tickers = [] # All tickers
        self.features = [] # All features
        self.cols_per_ticker = [] # All features for all tickers
    
    def _split_col(self, col: str) -> Tuple[str, str]:
        """Split column into (ticker, feature) using first underscore only."""
        parts = col.split(self.col_sep, 1)
        if len(parts) != 2:
            raise ValueError(f"Column '{col}' does not match <ticker>_<feature> format")
        return parts[0], parts[1]  # ticker, feature-with-underscores
    
    def extract_features(self, train_df: pd.DataFrame):
        """Extract tickers and features from full DataFrame column names."""
        tickers = []
        features = []

        for col in train_df.columns:
            t, f = self._split_col(col)
            tickers.append(t)
            features.append(f)

        self.tickers = sorted(set(tickers)) # Important to sort

        # Features must be identical for all tickers
        features_by_ticker = {t: set() for t in self.tickers}
        for col in train_df.columns:
            t, f = self._split_col(col)
            features_by_ticker[t].add(f)
        
        # Ensuring all tickers have the same feature list
        all_feature_sets = list(features_by_ticker.values())
        if not all(s == all_feature_sets[0] for s in all_feature_sets):
            raise ValueError('Different tickers have different feature sets!')

        self.features = sorted(list(all_feature_sets[0])) # Important to sort

        # Deterministic order for reshaping
        self.cols_per_ticker = [
            f'{t}{self.col_sep}{f}' for t in self.tickers for f in self.features
        ]
    
    def transform_one_window(self, df_window: pd.DataFrame) -> np.ndarray:
        """
        Convert a single (T_in x flat-columns) window into a tensor:
            'T_N_F' -> (T, N_stocks, F_features)
            'T_F_N' -> (T, F_features, N_stocks)
        """
        T = len(df_window)
        N = len(self.tickers)
        F = len(self.features)

        if self.layout == ReshapeStyle.T_NxF:
            out = np.zeros((T, N*F), dtype=float)
            for j, t in enumerate(self.tickers):
                for k, f in enumerate(self.features):
                    col = f'{t}{self.col_sep}{f}'
                    out[:,j * F + k] = df_window[col].values # flat index = j * F + K
            return out

        elif self.layout == ReshapeStyle.T_N_F:
            out = np.zeros((T, N, F), dtype=float)
            for j, t in enumerate(self.tickers):
                for k, f in enumerate(self.features):
                    col = f'{t}{self.col_sep}{f}'
                    out[:, j, k] = df_window[col].values
            return out

        elif self.layout == ReshapeStyle.T_F_N:
            out = np.zeros((T, F, N), dtype=float)
            for j, t in enumerate(self.tickers):
                for k, f in enumerate(self.features):
                    col = f'{t}{self.col_sep}{f}'
                    out[:, k, j] = df_window[col].values
            return out

        else:
            raise ValueError('layout must be of type `ReshapStyle`')
    
    def _features_check(self):
        if (len(self.tickers) == 0 or 
            len(self.features) == 0 or 
            len(self.cols_per_ticker) == 0):
            raise ValueError('Run `extract_features` before reshaping!')

    def reshape(
            self, features_data: pd.DataFrame, raw_returns: pd.DataFrame
        ) -> np.ndarray:
        
        self._features_check()

        starts = list(
            range(0, len(features_data) - (self.in_size + self.out_size) + 1, self.stride)
        )

        X_list = []
        y_list = []
        good_starts = []

        for s in starts:
            X_df = features_data.iloc[s : s + self.in_size]
            y_df = raw_returns.iloc[
                s + self.in_size : s + self.in_size + self.out_size
            ][self.tickers]

            # skip invalid windows
            if y_df.isna().any().any():
                raise ValueError('Window has missing data. Fix before training.')
        
            X_list.append(self.transform_one_window(X_df))
            y_list.append(y_df.values)
            good_starts.append(s)
        
        X = np.stack(X_list)
        y = np.stack(y_list)
        return X, y, np.array(good_starts)

    def get_tickers(self) -> List:
        return self.tickers
    
    def get_features(self) -> List:
        return self.features