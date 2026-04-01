import torch
import numpy as np
import pandas as pd
from enum import StrEnum
from torch.utils.data import Dataset
from src.utils.formatting import split_col

class ReshapeStyle(StrEnum):
    """
    For fixed reshaping styles to avoid reshape errors.
    """
    T_NxF = 'T_NxF' # (Time Steps, Ticker x Features + Common features) -> 2D for 3D after Batch
    # T_N_F = 'T_N_F' # (Time Steps, Tickers, Features) -> 3D for 4D after Batch
    # T_F_N = 'T_F_N' # (Time Steps, Features, Tickers) -> 3D for 4D after Batch
    T_N_F_plus_C = 'T_N_F_plus_C' # (Time Steps, Tickers, Features + Common Features)
    CNN_2D = 'F_plus_C_T_N' # Channels approach for 2D CNN

class Reshaper:
    """
    Reshapes a wide 2D DataFrame with columns like '<ticker>_<feature>',
    into 2D or 3D arrays, and builds sliding windows.

    Assumes all columns follow the pattern:
        <ticker>_<feature>
    """
    col_sep = '_'
    def __init__(
            self,
            in_size: int,
            out_size: int,
            stride: int,
            common_features: list,
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
        self.common_features = common_features
        self.layout = layout
        
        if not isinstance(layout, ReshapeStyle):
            raise TypeError(
                '`layout` must be of type ReshapeStyle to avoid errors.',
                'Use ReshapeStyle from src.data_processing.preprocess.'
            )
        
        self.num_common_feats = len(common_features)

        self.tickers = [] # All tickers
        self.features = [] # All features

        self.num_tickers = None
        self.num_features = None
    
    def extract_features(self, train_cols: list):
        """Extract tickers and features from full DataFrame column names."""
        tickers = []
        features = []

        for col in train_cols:
            if col != 'date' and col not in self.common_features:
                t, f = split_col(self.col_sep, col)
                tickers.append(t)
                features.append(f)
        
        self.tickers = list(set(tickers)) # Not required to sort here
        # features.extend(common_feats)
        self.features = list(set(features)) # Not required to sort here

        self.num_tickers = len(self.tickers)
        self.num_features = len(self.features)
    
    def transform_one_window(self, wind_data: np.ndarray) -> np.ndarray:        
        # Extract the raw values for ticker features
        # Shape: (T, N * F) + (C)
        
        T = wind_data.shape[0]
        N = self.num_tickers
        F = self.num_features
        C = self.num_common_feats

        ticker_block = wind_data[:, :N*F] 
        # Common Block is the remaining columns at the end
        common_block = wind_data[:, N*F:]

        # Handle Layouts using Vectorized Operations
        # TODO: Other forms of reshaping must be implmented !!!!
        if self.layout == ReshapeStyle.T_NxF:
            # Already in shape (T, N*F)
            return wind_data
        
        elif self.layout == ReshapeStyle.T_N_F_plus_C:
            # BROADCASTING: Each ticker gets the macro features appended to its own features.
            # Reshape tickers to (T, N, F)
            grid = ticker_block.reshape(T, N, F)
            # Reshape common to (T, 1, C) so it can be broadcasted across the N dimension
            common_reshaped = common_block.reshape(T, 1, C)
            common_broadcasted = np.repeat(common_reshaped, N, axis=1)
            # Concatenate on the Feature axis: Result shape (T, N, F + C)
            return np.concatenate([grid, common_broadcasted], axis=2)

        elif self.layout == ReshapeStyle.CNN_2D:
            # CHANNEL APPROACH: (Channels, Time, Tickers)
            # Channel 0..F-1: Ticker Features
            # Channel F..F+C: Macro Features (broadcasted)
            grid = ticker_block.reshape(T, N, F).transpose(2, 0, 1) # (F, T, N)
            
            # Broadcast each macro feature as its own channel
            common_channels = []
            for i in range(C):
                # Take one macro feature, shape (T, 1), broadcast to (T, N), add channel dim
                feat = common_block[:, i:i+1] # (T, 1)
                feat_expanded = np.repeat(feat, N, axis=1)[np.newaxis, :, :] # (1, T, N)
                common_channels.append(feat_expanded)
                
            return np.concatenate([grid] + common_channels, axis=0) # (F+C, T, N)

        else:
            raise ValueError('layout must be of type `ReshapeStyle`')

    def _features_check(self):
        if (len(self.tickers) == 0 or 
            len(self.features) == 0):
            raise ValueError('Run `extract_features` before reshaping!')

    def reshape(
            self, features_data: np.ndarray, raw_returns: np.ndarray
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
            X_df = features_data[s : s + self.in_size]
            y_df = raw_returns[
                s + self.in_size : s + self.in_size + self.out_size
            ]#[self.tickers]

            # skip invalid windows
            if np.isnan(y_df).any():
                raise ValueError('Window has missing data. Fix before training.')
        
            X_list.append(self.transform_one_window(X_df))
            y_list.append(y_df)
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
    
    def update_stride(self, stride: int):
        self.stride = stride

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

class WFAdjustment:
    def __init__(self, out_size: int):
        self.out_size = out_size
        self.num_steps = 0
        self.extra_days = 0
    
    def calc_walk_steps(self, split: pd.DataFrame) -> tuple[int, int]:
        total_oos_days = len(split)
        extra_days = total_oos_days % self.out_size
        num_steps = (total_oos_days - self.extra_days) // self.out_size

        self.extra_days = extra_days
        self.num_steps = num_steps

        return self.num_steps, self.extra_days
    
    def init_datasets(
            self, train: pd.DataFrame, split: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create initial datasets by adjusting for the extra days.
        This method adds the first 'extra' days to the training data.
        We only Adjust teh validation data set, not test set.
        """
        self.calc_walk_steps(split)
        
        init_train = pd.concat(
            [train, split.iloc[:self.extra_days]],
            axis=0
        )

        init_split = split.iloc[self.extra_days:]

        self.init_train = init_train
        self.init_split = init_split

        print(
            f'Evaluation dataset contains {self.num_steps} steps.',
            f'{self.extra_days} days from evaluation dataset moved to training dataset.'
        )

        return self.init_train, self.init_split
    
    def get_num_steps(self) -> int:
        return self.num_steps
    
    def get_extra_days(self) -> int:
        return self.extra_days