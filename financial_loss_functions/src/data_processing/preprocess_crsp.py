import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import svd
from scipy.linalg import hankel
from sklearn.preprocessing import RobustScaler
from src.utils.formatting import extract_req_cols, split_col


def _handle_missing_data(df: pd.DataFrame, col_suffix: str, limit: int = 1):
    """
    Forward fill data is there are NaNs.

    Args:
        df (pd.DataFrame): Dataframe to be cleaned by forward filling data.
        col_suffix (str): Column suffix to identify columns that need to be 
            handled for missing data.
        limit (int): Number of maximum time steps to forward fill. Default = 1.
    
    Returns:
        df (pd.DataFrame): Dataframe that has its missing values filled using ffill().
    """
    req_cols = extract_req_cols(df.columns, col_suffix)

    df[req_cols] = df[req_cols].ffill(limit=limit)

    return df

def clean_inplace(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans dataset by removing dupilcate columns and duplicate rows. It makes date the index.
    This process is inplace, i.e., Reference of dataset is used, not copy.
    
    Args:
        train (pd.DataFrame): Train dataframe.
        val (pd.DataFrame): Validation dataframe.
        test (pd.DataFrame): test dataframe.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        Cleaned train data, validation data and test data.  
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
    
    # Setting index to datetime
    train['date'] = pd.to_datetime(train['date'])
    train.set_index('date', inplace=True)

    val['date'] = pd.to_datetime(val['date'])
    val.set_index('date', inplace=True)

    test['date'] = pd.to_datetime(test['date'])
    test.set_index('date', inplace=True)

    # Handling missing data #### Add missing data handling for each col here, if needed.
    # CRSP data did not have missing values, but this is just a guard and is used for the sample data
    train = _handle_missing_data(train, '_BA_SPREAD')
    val = _handle_missing_data(val, '_BA_SPREAD')
    test = _handle_missing_data(test, '_BA_SPREAD')
    
    return train, val, test

class SSA:
    """Singular Spectrum Analysis for time series denoising.

    Fits a separate SSA model to each feature (column) of the input data,
    then applies the learned subspace to denoise new data.

    Attributes:
        window_len (int): Window length (L) used to build the trajectory matrix.
        variance_thres (float): Fraction of variance to retain when selecting
            singular components (default 0.90).
        U_dict (dict): Stores the signal subspace (`U_r`) and other metadata
            for each feature column after fitting.
        _input_is_df (bool): Internal flag indicating whether the input was a
            pandas DataFrame (used to restore output type).
        _column_names (list[str] | None): Column names if input was a DataFrame.
    """
    def __init__(self, window_len: int, variance_thres: float = 0.90):
        """Initializes the SSA denoiser.

        Args:
            window_len (int): Window length (L). Must be less than or equal to
                the number of time steps in the training data.
            variance_thres (float): Cumulative variance threshold for
                selecting the number of components to keep. Default = 0.90.
        """
        self.window_len = window_len
        self.variance_thres = variance_thres
        self.U_dict = {}          # stores U_r for each column index
        self._input_is_df = False
        self._column_names = None

    def ssa_fit(self, data: pd.DataFrame | np.ndarray):
        """Fits an SSA model to each feature column of the training data.

        For each column, builds a Hankel trajectory matrix, performs SVD,
        and stores the signal subspace (`U_r`) based on the explained variance
        threshold.

        Args:
            data (pd.DataFrame | np.ndarray): Training data of shape
                (n_samples, n_features). Can be a pandas DataFrame or a
                numpy array.
        """
        if isinstance(data, pd.DataFrame):
            self._input_is_df = True
            self._column_names = data.columns.tolist()
            arr = data.values.astype(np.float64)
        else:
            self._input_is_df = False
            arr = np.asarray(data, dtype=np.float64)
            self._column_names = list(range(arr.shape[1]))

        N, n_features = arr.shape
        L = self.window_len
        K = N - L + 1

        for j in range(n_features):
            col = arr[:, j]
            # Build trajectory matrix
            X = hankel(col[:L], col[L-1:])   # shape (L, K)
            # SVD
            U, s, Vt = svd(X, full_matrices=False)
            # Determine r from explained variance
            var_explained = (s**2) / (s**2).sum()
            cum_var = np.cumsum(var_explained)
            r = np.searchsorted(cum_var, self.variance_thres) + 1
            r = min(r, L)
            U_r = U[:, :r]
            self.U_dict[j] = {
                'U_r': U_r,
                'r': r,
                's': s
            }

    def ssa_transform(
            self, data: pd.DataFrame | np.ndarray
        ) -> pd.DataFrame | np.ndarray:
        """Denoises new data using the previously fitted subspaces.

        For each column, projects the new series onto the signal subspace
        from training and reconstructs the denoised series.

        Args:
            data (pd.DataFrame | np.ndarray): New data of shape
                (n_samples, n_features). Must have the same number of features
                as the training data.

        Returns:
            pd.DataFrame | np.ndarray: Denoised data of the same shape and type
                as the input. If the input was a DataFrame, the returned object
                is a DataFrame with the same index and columns; otherwise, a
                numpy array is returned.

        Raises:
            ValueError: If `ssa_fit` has not been called before this method.
            KeyError: If a feature column index (or name) was not seen during
                fitting (should not happen with consistent data).
        """
        if not self.U_dict:
            raise ValueError('Run `ssa_fit` before calling `ssa_transform`!')

        if isinstance(data, pd.DataFrame):
            arr = data.values.astype(np.float64)
            out_index = data.index
            out_columns = data.columns
        else:
            arr = np.asarray(data, dtype=np.float64)
            out_index = None
            out_columns = None

        N, n_features = arr.shape
        L = self.window_len
        K = N - L + 1
        denoised = np.empty_like(arr)

        for j in range(n_features):
            if j not in self.U_dict:
                raise KeyError(f'Column {j} not seen in training.')
            col = arr[:, j]
            # Build trajectory matrix for new series
            X = hankel(col[:L], col[L-1:])        # (L, K)
            U_r = self.U_dict[j]['U_r']
            coeff = U_r.T @ X                     # (r, K)
            X_rec = U_r @ coeff                   # (L, K)
            # Diagonal averaging (vectorized)
            denoised_col = np.zeros(N)
            count = np.zeros(N)
            for i in range(L):
                denoised_col[i:i+K] += X_rec[i, :]
                count[i:i+K] += 1
            denoised[:, j] = denoised_col / count

        if out_columns is not None:
            return pd.DataFrame(denoised, index=out_index, columns=out_columns)
        else:
            return denoised

class KalmanDenoise:
    """Kalman filter denoising using a local level model.

    Fits a separate local level state-space model to each feature column,
    then applies the fitted filter to denoise new data.

    Attributes:
        method (str): Optimization method used for maximum likelihood estimation.
        maxiter (int): Maximum number of iterations for the optimizer.
        params_dict (dict): Stores the fitted parameters (observation noise and
            level noise) for each feature column (by column index).
        _input_is_df (bool): Internal flag indicating whether the input was a
            pandas DataFrame (used to restore output type).
        _column_names (list[str] | None): Column names if input was a DataFrame.
    """
    def __init__(self, method: str = 'powell', maxiter: int = 100):
        """Initializes the Kalman denoiser.

        Args:
            method (str, optional): Optimization algorithm passed to
                `statsmodels.tsa.UnobservedComponents.fit`. Default = 'powell'.
            maxiter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to 100.
        """
        self.method = method
        self.maxiter = maxiter
        self.params_dict = {}          # key: column index
        self._input_is_df = False
        self._column_names = None

    def kalman_fit(self, data: pd.DataFrame | np.ndarray):
        """Fits a local level Kalman filter to each feature column.

        For each column, estimates the observation noise and level noise
        using maximum likelihood estimation (MLE) via `statsmodels`.

        Args:
            data (pd.DataFrame | np.ndarray): Training data of shape
                (n_samples, n_features). Can be a pandas DataFrame or a
                numpy array.

        Note:
            The fitted parameters are stored internally in `self.params_dict`
            and are later used by `kalman_transform`.
        """
        if isinstance(data, pd.DataFrame):
            self._input_is_df = True
            self._column_names = data.columns.tolist()
            arr = data.values.astype(np.float64)
        else:
            self._input_is_df = False
            arr = np.asarray(data, dtype=np.float64)
            self._column_names = list(range(arr.shape[1]))

        n_features = arr.shape[1]
        for j in range(n_features):
            series = arr[:, j]
            model = sm.tsa.UnobservedComponents(series, 'local level')
            res = model.fit(method=self.method, maxiter=self.maxiter, disp=False)
            self.params_dict[j] = res.params   # store by index

    def kalman_transform(
            self, data: pd.DataFrame | np.ndarray
        ) -> pd.DataFrame | np.ndarray:
        """Applies the fitted Kalman filter to new data.

        Uses the pre-estimated noise parameters from training to filter
        (denoise) each column of the new data.

        Args:
            data (pd.DataFrame | np.ndarray): New data of shape
                (n_samples, n_features). Must have the same number of features
                as the training data.

        Returns:
            pd.DataFrame | np.ndarray: Denoised data of the same shape and type
                as the input. If the input was a DataFrame, the returned object
                is a DataFrame with the same index and columns; otherwise, a
                numpy array is returned.

        Raises:
            ValueError: If `kalman_fit` has not been called before this method.
            KeyError: If a feature column index (or name) was not seen during
                fitting (should not happen with consistent data).
        """
        if not self.params_dict:
            raise ValueError('Run `kalman_fit` on training data before transforming!')

        if isinstance(data, pd.DataFrame):
            arr = data.values.astype(np.float64)
            out_index = data.index
            out_columns = data.columns
        else:
            arr = np.asarray(data, dtype=np.float64)
            out_index = None
            out_columns = None

        n_features = arr.shape[1]
        denoised = np.empty_like(arr)
        for j in range(n_features):
            if j not in self.params_dict:
                raise KeyError(f'Column {j} not seen in training.')
            series = arr[:, j]
            model = sm.tsa.UnobservedComponents(series, 'local level')
            filtered_result = model.filter(self.params_dict[j])
            denoised[:, j] = filtered_result.filtered_state[0]

        if out_columns is not None:
            return pd.DataFrame(denoised, index=out_index, columns=out_columns)
        else:
            return denoised

def calculate_dema(df, span=20):
    """Computes the Double Exponential Moving Average (DEMA) for all columns.

    DEMA = 2 * EMA1 - EMA2, where EMA1 is the standard exponential moving
    average of the original series, and EMA2 is the EMA of EMA1.

    Args:
        df (pd.DataFrame): Input DataFrame where each column is a time series.
        span (int, optional): The span (window size) for the EMA calculation.
            Defaults to 20.

    Returns:
        pd.DataFrame: DataFrame of the same shape as `df` containing the DEMA
            values for each column.
    """
    # 1. Calculate the first EMA
    ema1 = df.ewm(span=span, adjust=False).mean()
    
    # 2. Calculate the EMA of the first EMA
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    
    # 3. Apply the DEMA formula
    dema = (2 * ema1) - ema2
    return dema

class Preprocessor:
    """
    Preprocessor class which splits the Returns, BA Spread for all stocks and 
    then normalizes the given dataset. This class used Robust scaling for normalization 
    (Median/IQR scaling).

    Attributes:
        col_sep (str): Special character that separates the ticker string from the feature string.
        return_suffix (str): Suffix for returns  columns of every stock.
        ba_spread_suffix (str): Suffix for BA spread columns of every stock.
    """
    col_sep = '_' # Special character that separates the ticker string from the feature string.
    return_suffix = '_RET' # Suffix for returns of every stock
    ba_spread_suffix = '_BA_SPREAD' # Suffix for BA spread of every stock
    
    def __init__(
            self,
            common_features: list[str] | None = None,
            broadcast: bool = False
        ):
        """
        Initialize Preprocessor object which splits the Returns, BA Spread for all stocks and 
        then normalizes the given dataset.
        
        Args:
            common_features (list[str]): List of common features in the dataset, eg., sprtrn. 
                Default = None
            broadcast (bool): Toggle to broadcast common features to all stocks. Default = False.
        """
        self.common_features = common_features
        self.broadcast = broadcast

        self._robust_scaler = RobustScaler()

        self.unordered_cols = None
        self.all_tickers = None

        self.return_cols = None
    
    def _normalize(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Normalize data set using robust scaling, i.e., Median/IQR scaling.
        This method treats train and validation or test splits differently.
        Use mode `fit` to fit and transform, use mode `split`
        to transform only and not refit.

        Args:
            data (pd.DataFrame): Dataframe to be normalized using robust scaling.
            mode (str): `fit` or `split`. Determines if dataframe must be used to 
                fit the scaler or the scaler should transform the dataframe.

        Returns:
            data (pd.DataFrame): Normalized dataframe using scaler object.
        
        Raises:
            ValueError: If incorrect mode string is provided. Must be `fit` or `split`.
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

    def _broadcast_common(
            self, data: pd.DataFrame, features: list[str]
        ) -> pd.DataFrame:
        """
        Broadcast common features to all tickers with names <ticker>_<common_feature>.
        
        Args:
            data (pd.DataFrame): Dataset which needs broadcasting of common features.
            features (list[str]): List of features which need to be broadcasted to every stock.

        Returns:
            pd.DataFrame dataframe with broadcasted common features.
        """

        # Build broadcasted columns in a single concatenation to avoid fragmentation
        new_cols = {
            f'{ticker}{self.col_sep}{feat}': data[feat].values
            for feat in features
            for ticker in self.all_tickers
        }
        broadcast_df = pd.DataFrame(new_cols, index=data.index)

        # Drop the original common columns and concatenate once
        remaining = data.drop(columns=features)
        combined = pd.concat([remaining, broadcast_df], axis=1)

        # Copy to defragment the underlying blocks
        return combined.copy()

    def _update_common_features(self, macro_cols: list[str]):
        """
        Merge macro columns with existing common features without duplicates.
        
        Args:
            macro_cols (list[str]): List of column names in macro-economic dataset.
        """
        # TODO: Use set instead of dict (more efficient)
        base_common = self.common_features or []
        combined = list(dict.fromkeys(base_common + macro_cols))
        self.common_features = combined if combined else None

    def _build_feats_order(self) -> tuple[list, list]:
        """
        Extract tickers and features from full DataFrame column names.
        
        Returns:
            tuple[list, list]: Tuple containing list of all features and list of tickers.
        """
        tickers = []
        features = []
        
        for col in self.unordered_cols:
            if col != 'date' and col not in self.common_features:
                t, f = split_col(self.col_sep, col)
                tickers.append(t)
                features.append(f)

        tickers = sorted(set(tickers)) # Important to sort
        features = sorted(set(features)) # Important to sort
        
        # Build list of <ticker>_<feature> in alphabetical order
        all_features = []
        for ticker in tickers:
            for feat in features:
                column_name = f'{ticker}_{feat}'
                if column_name in self.unordered_cols:
                    all_features.append(column_name)
                else:
                    print(f'{column_name}, not found in data, features are not symmetric across tickers')
        
        # Append sorted common features, eg., sprtrn (s&p500)
        all_features.extend(sorted(self.common_features))
        return all_features, tickers

    def _extract_only_returns(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Extract only returns data columns from the given dataframe.

        Args:
            data (pd.DataFrame): Dataframe with returns data, that has to be extracted.
            mode (str): `fit` or `split`. Determines if dataframe must be used to extract 
                return columns names or extract the set return column names from a split dataframe.
        
        Returns:
            returns_data (pd.DataFrame): Dataframe containing only the returns columns.
        
        Raises:
            ValueError: If self.returns_cols is not filled yet, `process_train_data` must be run 
                first to extract return column names from the train data.
        """
        if mode == 'fit':
            self.return_cols = extract_req_cols(data.columns, self.return_suffix)
        else:
            pass
        
        if self.return_cols is not None:
            
            returns_data = data[self.return_cols]

            # Rename columns to remove the '_RET' suffix
            returns_data.columns = [col.replace(self.return_suffix, '') for col in self.return_cols]

            return returns_data.sort_index(axis=1) # Sort to match other files of the dataset

        else:
            raise RuntimeError('Run `process_train_data` first.')

    def _extract_only_ba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only Bid-Ask Spread data columns from the given dataframe.

        Args:
            data (pd.DataFrame): Dataframe with BA Spread data, that has to be extracted.
        
        Returns:
            returns_data (pd.DataFrame): Dataframe containing only BA_Spread columns.
        """
        ba_spread_cols = extract_req_cols(data, self.ba_spread_suffix)

        ba_spreads = data[ba_spread_cols]

        return ba_spreads.sort_index(axis=1) # Sort to match other files of the dataset

    def _build_ba_spread_cols(self):
        """Method to build the ordering of the BA_Spread columns."""
        order_ba_spreads = [f'{ticker}{self.ba_spread_suffix}' for ticker in self.all_tickers]

        return order_ba_spreads

    def process_train_data(
            self, train: pd.DataFrame, macro_data: pd.DataFrame | None = None
        )-> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses given training data. Extracts returns, reorders columns for every 
        dataframe to follow alphabetical ordering, then normalizes the dataframe with all features.

        Args:
            train (pd.DataFrame): Dataframe containing training data.
            macro_data (pd.DataFrame): Macro data aligned to training dates. Default = None.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing Preprocessed training data and 
                returns only data.
        """
        ret_train = self._extract_only_returns(train, 'fit')
        
        macro_cols: list[str] = []
        if macro_data is not None:
            macro_cols = list(macro_data.columns)
            train = pd.concat([train, macro_data], axis=1)

            # Update common features with macro features
            self._update_common_features(macro_cols)

        # Reorder columns in alphabetical order
        self.unordered_cols = list(train.columns)
        self.ordered_cols, self.all_tickers = self._build_feats_order()

        train = train[self.ordered_cols]
        ret_train = ret_train[self.all_tickers]

        # # kalman for denosining
        # self.kalman_filt.kalman_fit(train)
        # train = self.kalman_filt.kalman_transform(train)

        # train = self._transform(train, 'fit')
        train = self._normalize(train, 'fit')

        # Broadcast common features only if needed
        if self.broadcast:
            train = self._broadcast_common(train, self.common_features)

        return train, ret_train

    def process_split_data(
            self, split_data: pd.DataFrame, macro_data: pd.DataFrame | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses given validation or test data based on statistics 
        from the training data. Extracts returns and Bid-Ask spread data, reorders columns for every 
        dataframe to follow alphabetical ordering, then normalizes the dataframe with all features.

        Args:
            split_data (pd.DataFrame): Dataframe containing validation or test data.
            macro_data (pd.DataFrame): Macro data aligned to validation/test dates. Default = None.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing Preprocessed validation or test data, 
                Bid-Ask spread only data, and returns only data.
        
        Raises:
            ValueError: If there are missing columns in the split data, that were present in the training data.
        """
        
        ret_split = self._extract_only_returns(split_data, 'split')
        ba_split = self._extract_only_ba(split_data)

        # macro_cols: list[str] = []
        if macro_data is not None:
            # macro_cols = list(macro_data.columns)
            split_data = pd.concat([split_data, macro_data], axis=1)

            # self._update_common_features(macro_cols)

        # Ensure column alignment with training data before normalization
        missing = set(self.ordered_cols) - set(split_data.columns)
        extra = set(split_data.columns) - set(self.ordered_cols)
        if missing:
            raise ValueError(f'Missing columns in split data: {missing}')
        if extra:
            # Drop any unexpected columns to match training schema
            split_data = split_data[self.ordered_cols]
        else:
            split_data = split_data[self.ordered_cols]

        # Reorder columns to match train data
        split_data = split_data[self.ordered_cols]
        ret_split = ret_split[self.all_tickers]
        ba_split = ba_split[self._build_ba_spread_cols()]
        # split_data = self._transform(split_data, 'split')

        # Kalman filter for denosining
        # split_data = self.kalman_filt.kalman_transform(split_data)

        split_data = self._normalize(split_data, 'split')

        # Broadcast common features only if needed
        if self.broadcast:
            split_data = self._broadcast_common(split_data, self.common_features)

        return split_data, ret_split, ba_split

    def get_common_features(self) -> list[str]:
        """
        Getter method to get the common features list at the current state of the object.

        Returns:
            common_features (list[str]): List of common feature names.
        """
        return self.common_features

def preprocessor2(
        returns_is: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates covariance and correlation matrices for returns data.

    Args: 
        train (pd.DataFrame): Training split data, only returns.
        val (pd.DataFrame): Validation or test split data, only returns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Covariance and correlation matrices.
    """

    returns_is_cov = returns_is.cov()
    returns_is_corr = returns_is.corr()

    return returns_is_cov, returns_is_corr