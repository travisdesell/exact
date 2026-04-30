import torch
import numpy as np
import pandas as pd
from enum import StrEnum
from torch.utils.data import Dataset
from src.utils.formatting import split_col
from src.utils.window import calc_current_idxs

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
    Reshapes a wide 2D DataFrame with columns like `<ticker>_<feature>`,
    into 2D or 3D arrays, and builds rolling windows.

    Assumes all columns follow the pattern: `<ticker>_<feature>`

    Attributes:
        col_sep (str): Special character that separates the ticker string from the feature string.
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

        Args:
            in_size (int): Size of input window in terms of time steps.
            out_size (int): Size output window in terms of time steps.
            stride (int): step stride size for the rolling windows.
            common_features (list[str]): List of common features that will be placed at the end 
                of the columns or broadcasted if needed.
            layout (ReshapeStyle): Enum for reshaping define the reshaping style, 
                see `src.data_processing.preprocess.ReshapeStyle`. Default = ReshapeStyle.T_NxF.
        
        Raises:
            TypeError: When ReshapeStyle is of the wrong type
            ValueError: When a list of common features in the 2D dataframe is not provided.
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
        
        if not common_features:
            raise ValueError(
                'A list of common features must be provided so that they can get broadcasted if needed.'
            )
        
        self.num_common_feats = len(common_features)

        self.tickers = [] # All tickers
        self.features = [] # All features

        self.total_num_cols = None
        self.num_tickers = None
        self.num_features = None
    
    def extract_features(self, train_cols: list[str]):
        """
        Extract tickers and features from full DataFrame column names.
        
        Args:
            train_cols (list[str]): List of columns in the dataframe.
        """
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

        self.total_num_cols = len(train_cols)
        self.num_tickers = len(self.tickers)
        self.num_features = len(self.features)
    
    def transform_one_window(self, wind_data: np.ndarray) -> np.ndarray:  
        """
        Transform one window of the data array into the desired shape.

        Args:
            wind_data (np.ndarray): Array of one window of data.
        
        Returns:
            np.ndarray: Rehaped array of one window.
        
        Raises:
            ValueError: When layout is not of type `ReshapeStyle`
        """      
        # Extract the raw values for ticker features
        # Shape: (T, N * F) + (C)
        
        T = wind_data.shape[0]
        N = self.num_tickers
        F = self.num_features
        C = self.num_common_feats

        ticker_block = wind_data[:, :N*F] 
        # Common Block is the remaining columns at the end
        common_block = wind_data[:, N*F:]

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
        """
        Utility method to check if features have been extracted from dataframe.
        
        Raises:
            ValueError: If `extract_features` has not run before the execution of this method.
        """
        if (
            not self.num_tickers or 
            not self.num_features or
            not self.total_num_cols
        ):
            raise ValueError('Run `extract_features` before reshaping!')

    def _columns_match_check(self, features_num_cols: int, returns_num_cols: int):
        """
        Utility method to check if the number of columns in features data and returns data match.

        Raises:
            ValueError: When Extracted columns or tickers returns do not match.
        """
        if features_num_cols != self.total_num_cols:
            raise ValueError(
                'Extracted columns do not match provided array shape.'
            )

        if returns_num_cols != self.num_tickers:
            raise ValueError(
                'Number of tickers in extracted columns do not match provided array shape.'
            )

    def reshape(
            self, features_data: np.ndarray, raw_returns: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshapes 2D data into set `layout` at initialization.

        Args:
            features_data (np.ndarray): 2D array containg all processed features.
            raw_returns (np.ndarray): 2D array containing only raw returns.
        
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray] :
                Reshaped X and y, Array of good starting points of each window. Helpful for debugging.
        """
        
        self._features_check()
        self._columns_match_check(features_data.shape[1], raw_returns.shape[1])

        if self.in_size + self.out_size > features_data.shape[0]:
            raise ValueError(
                'Incorrect rolling window sizes. in_size + out_size <= Number of time steps'
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

    def get_tickers(self) -> list | None:
        """
        Get the list of ticker symbols if `extract_features` has been executed.

        Returns:
            tickers (list | None): List of ticker symbols in the dataset.
        """
        if len(self.tickers) == 0:
            print('Run `extract_features` on training data first.')
            return None
        else:
            return self.tickers
    
    def get_features(self) -> list | None:
        """
        Get a list of stock features if `extract_features` method has been executed.

        Returns:
            features (list | None): List stock features in the dataset.
        """
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
        
        Args:
            X (np.ndarray): X input windows.
            y (np.ndarray): y output windows.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def get_X_y_shapes(self) -> tuple[torch.Size, torch.Size]:
        """
        Get the shapes of X and y.

        Returns:
            tuple[torch.Size, torch.Size]: Shape of X and shape of y
        """
        return self.X.shape, self.y.shape

    def __len__(self) -> int:
        """
        Get length of windowed dataset.

        Args:
            int: Length of windowed dataset (Number of samples).
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Get one window from the dataset.

        Args:
            idx (int): Index of required window in the windowed dataset.
        
        Returns:
            tuple[torch.tensor, torch.tensor]: X & y for the given index.
        """
        # Return one sample
        return self.X[idx], self.y[idx]

class WFUtilities:
    def __init__(self, out_size: int):
        self.out_size = out_size
        self.num_steps = None
        self.extra_days = None
    
    def calc_walk_steps(self, split: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate the walk steps in the provided dataframe and extra days remaining 
        after dividing the data by the walk steps.

        Args:
            split (pd.DataFrame): Data split dataframe, where the features data and 
                returns data have the same lengths.
        
        Returns:
            tuple[int, int]: Tuple containing number of steps and number of remaining extra days.
        """
        total_oos_days = len(split)
        extra_days = total_oos_days % self.out_size
        num_steps = total_oos_days // self.out_size

        self.extra_days = extra_days
        self.num_steps = num_steps

        return self.num_steps, self.extra_days
    
    def init_datasets(
            self, train: pd.DataFrame, split: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create initial datasets by adjusting for the extra days.
        This method adds the first 'extra' days to the training data.
        We only Adjust the validation data set, not test set.

        Args:
            train (pd.DataFrame): Dataframe of the train data.
            split (pd.Dataframe): Dataframe of the split data (val or test).
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Adjusted dataframes where extra days 
                from the validation data is moved to the training data.
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

    def build_eval_windows(
            self, split: pd.DataFrame
        ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Build evaluation windows for evaluation of walk forward experiments. 
        These windows are not overlapping.

        Args:
            split (pd.DataFrame): Out-of-Sample split 2D dataframe that is divided to 
                non-overlapping windows.
        
        Returns:
            tuple[np.ndarray, list[tuple[int, int]]]: Windowed evaluation data and list 
                of tuples containing the start and end indexes for each window.
        
        Raises:
            ValueError: If provided dataframe is smaller than num_steps * out_size.
        """  
        if len(split) < self.out_size * self.num_steps:
            raise ValueError('Provided dataframe is smaller than num_steps * out_size.')
        
        eval_windows = []
        out_wind_idxs = []
        for step in range(1, self.num_steps+1):
            current_start, current_end = calc_current_idxs(step, self.out_size)

            walk_rets_eval = split.iloc[current_start : current_end]

            eval_windows.append(walk_rets_eval)

            out_wind_idxs.append((current_start, current_end))
        
        return np.stack(eval_windows), out_wind_idxs
    
    def build_ba_for_eval(
            self, ba_split: pd.DataFrame, out_wind_idxs: list[tuple[int, int]]
        ) -> np.ndarray:
        """
        Build for the Bid-Ask Spread matrix data based on the output window indexes proided.
        This matric contains BA Spread data only for the first day of each evaluation window.

        Args:
            ba_split (pd.DataFrame): Bid-Ask Spread dataframe for the evaluation period.
            out_wind_idxs (list[tuple[int, int]]): List of tuples containing the start 
                and end indexes for each window.
        
        Returns:
            np.ndarray: Array of Bid-Ask Spreads for the first day of every output evaluation window.
        
        Raises:
            ValueError: If provided dataframe is smaller than num_steps * out_size.
        """
        if len(ba_split) < self.out_size * self.num_steps:
            raise ValueError('Provided ba spread dataframe is smaller than num_steps * out_size.')
        
        first_day_bas_winds = []

        for start_idx, _ in out_wind_idxs:
            first_day_bas = ba_split.iloc[start_idx]

            first_day_bas_winds.append(first_day_bas)
        
        return np.stack(first_day_bas_winds)
    
    def get_num_steps(self) -> int:
        """
        Get number of walk steps calculated for the provided dataframes.
        """
        return self.num_steps
    
    def get_extra_days(self) -> int:
        """
        Get the number of extra days remaining after dividing the dataframe by the number of walk steps.
        """
        return self.extra_days