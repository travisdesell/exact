import torch
import numpy as np
import pandas as pd
from enum import StrEnum
from torch.utils.data import Dataset

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
    into 2D or 3D arrays, and builds sliding windows.

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
        Run Reshaper.extract_features on training data after initializing and before reshaping.

        @param in_size int size of input window in terms of time steps
        @param out_size int size output window in terms of time steps
        @param stride int step size for the sliding window
        @param col_sep str Special character that separates the ticker string from the feature string
        @param layout: ReshapeStyle 
            Enum of rehsape style, see `src.data_processing.preprocess.ReshapeStyle`. 
            Deafult = ReshapeStyle.T_NxF
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
    
    def _split_col(self, col: str) -> tuple[str, str]:
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
    
    def _transform_one_window(self, df_window: pd.DataFrame) -> np.ndarray:
        """
        Convert a single (T_in x flat-columns) window into an array of set layout.
        Uses alphabetical ordered for loops to maintain strict ordering to map input
        stocks to ouput nodes neural networks.

        @param df_window pd.DataFrame on window to be reshaped

        @return np.ndarray multi-dimensional reshaped array
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
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshapes 2D DataFrame into set `layout` at initialization.

        @param features_data pd.DataFrame Dataframe containg all processed features
        @param raw_returns pd.DataFrame Dataframe containing only raw returns
        
        @return tuple[np.ndarray, np.ndarray, np.ndarray] 
            Reshaped X and y, Array of good starting points of each window. Helpful for debugging
        """
        
        self._features_check()

        if self.in_size + self.out_size > features_data.shape[0]:
            raise ValueError(
                'Incorrect rolling window sizes. in_size + out_size <= Number os time steps'
            )

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
        
            X_list.append(self._transform_one_window(X_df))
            y_list.append(y_df.values)
            good_starts.append(s)
        
        X = np.stack(X_list)
        y = np.stack(y_list)
        return X, y, np.array(good_starts)

    def get_tickers(self) -> list:
        if len(self.tickers) == 0:
            print('Run `extract_features` on training data first.')
            return None
        else:
            return self.tickers
    
    def get_features(self) -> list:
        if len(self.features) == 0:
            print('Run `extract_features` on training data first.')
            return None
        else:
            return self.features

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Convert given numpy arrays into pytorch windowed dataset.
        
        @param X np.ndarray X input windows
        @param y np.ndarray y output windows
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def get_X_y_shapes(self) -> tuple[torch.Size, torch.Size]:
        """
        returns X and y shapes

        @return tuple[torch.Size, torch.Size] Shape of X and shape of y
        """
        return self.X.shape, self.y.shape

    def __len__(self) -> int:
        """
        Get length of windowed dataset.

        @return int length of windowed dataset (Number of samples)
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Get one window from the dataset.

        @param idx int Index of require window in the windowed dataset
        @return tuple[torch.tensor, torch.tensor] X & y for the given index
        """
        # Return one sample
        return self.X[idx], self.y[idx]

# class DatasetSampler:
#     def __init__(self, in_size: int, out_size: int, stride: int):
#         self.in_size = in_size
#         self.out_size = out_size
#         self.stride = stride
def calc_in_out_idx(
        split_data: pd.DataFrame, in_size: int, out_size: int, stride: int
    ) -> tuple[list[tuple], list[tuple]]:

    #### Implment start index shifting of training data here, if needed ####
    #### Current design intended to use entire train data.

    # Check for missing data
    if split_data.isna().any().any():
        raise ValueError('Split has missing data. Fix before training.')
    
    starts = list(
        range(0, len(split_data) - (in_size + out_size) + 1, stride)
    )

    in_sample_indexes = []
    out_sample_indexes = []
    for strt in starts:
        in_end = strt + in_size
        in_sample_indexes.append((strt, in_end))
        
        # FIX: out_start must be exactly in_end
        out_start = in_end 
        out_end = out_start + out_size
        out_sample_indexes.append((out_start, out_end))

    if len(in_sample_indexes) != len(out_sample_indexes):
        raise RuntimeError('Window count mismatch.')

    return in_sample_indexes, out_sample_indexes

def get_date_index_col(
    split: pd.DataFrame, wind_strt_stops: list[tuple]
) -> list:
    """
    Get the datetime index columns from the provided dataframe using the 
    start and stop indexes.
    """
    date_idx_cols = []
    for idxs in wind_strt_stops:
        date_idx_cols.append(split.index[idxs[0] : idxs[1]])
    
    return date_idx_cols

def build_dataset(
        in_sample_idx: tuple[int, int], # (Start, End)
        out_sample_idx: tuple[int, int],
        returns_train: pd.DataFrame, 
        returns_val: pd.DataFrame,
        returns_test: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
    """
    Dataset builder function for covariance based models (tradional).
    Combines and slices to create in-sample and out-of-sample datasets.
    """
    
    #### If train data must be sliced or shifted, it must be implmented here 
    # after grabbing index from dataset.py
    if returns_test is None:
        returns_is = pd.concat(
            [returns_train, returns_val.iloc[in_sample_idx[0]: in_sample_idx[1]]]
        )
        
        # iloc[200:250] gives rows 200-249 (Exactly 50 rows)
        returns_oos = returns_val.iloc[out_sample_idx[0]: out_sample_idx[1]]
    
    elif returns_test is not None and isinstance(returns_test, pd.DataFrame):
        returns_is = pd.concat(
            [returns_train, returns_val, returns_test.iloc[in_sample_idx[0]: in_sample_idx[1]]]
        )
        returns_oos = returns_test.iloc[out_sample_idx[0]: out_sample_idx[1]]
    else:
        raise ValueError('Incorrect type for test returns.')

    # Sorting in alphabetical order
    return returns_is.sort_index(axis=1), returns_oos.sort_index(axis=1)

def extract_oos_dates(
        split: pd.DataFrame,
        in_wind_idxs: list[tuple],
        out_wind_idxs: list[tuple]
    ):
    in_win_date_cols = get_date_index_col(split, in_wind_idxs)
    out_win_date_cols = get_date_index_col(split, out_wind_idxs)

    return in_win_date_cols, out_win_date_cols