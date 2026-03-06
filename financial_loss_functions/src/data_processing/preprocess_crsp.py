import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.linalg import hankel
from sklearn.preprocessing import PowerTransformer, RobustScaler
from src.utils.formatting import extract_req_cols, split_col

def _handle_missing_data(df: pd.DataFrame, col_suffix: str, limit: int = 1):
    req_cols = extract_req_cols(df.columns, col_suffix)

    df[req_cols] = df[req_cols].bfill(limit=limit)

    return df

def clean_inplace(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans dataset by removing dupilcate columns and duplicate rows. It makes date the index.
    This process is inplace, i.e., Refrence of dataset is used, not copy.
    
    @param train pd.DataFrame train data
    @param val pd.DataFrame validation data
    @param test pd.DataFrame test data
    
    @return tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] 
            Cleaned train data, validation data and test data
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

    # Handling missing data #### Add missing data handling for each col here, if needed
    train = _handle_missing_data(train, '_BA_SPREAD')
    val = _handle_missing_data(val, '_BA_SPREAD')
    test = _handle_missing_data(test, '_BA_SPREAD')
    
    return train, val, test

def get_only_returns(
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract only return columns from each of the split datasets.

    @param train pd.DataFrame Training data.
    @param val pd.DataFrame Validation data.
    @param test pd.DataFrame Test data.
    
    @return tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] 
            ret_train, ret_val, and ret_test

    """
    # return_cols = []
    return_suffix = '_RET'
    # for col in train.columns:
    #     if return_suffix in col:
    #         return_cols.append(col)
    
    return_cols = extract_req_cols(train.columns, return_suffix)
    
    ret_train = train[return_cols]
    ret_val = val[return_cols]
    ret_test = test[return_cols]

    ret_train.columns = [col.replace(return_suffix, '') for col in return_cols]
    ret_val.columns = [col.replace(return_suffix, '') for col in return_cols]
    ret_test.columns = [col.replace(return_suffix, '') for col in return_cols]

    return (
        ret_train.sort_index(axis=1),
        ret_val.sort_index(axis=1),
        ret_test.sort_index(axis=1)
    )

class SSA:
    def __init__(self, window_len: int, variance_thres: float = 0.90):
        self.window_len = window_len
        self.variance_thres = variance_thres
        self.U_dict = {}
        
        # U_r: eigenvectors of the trajectory matrix (L x r)
        # r: number of components kept
        # s: singular values (for reference)

    def ssa_fit(self, dataframe):
        """
        Fit Singular Spectrum Analysis on a single training series, automatically choosing r.
        Args:
            series: 1D numpy array of length N_train
            L: window length (e.g., 60)
            variance_threshold: fraction of variance to keep (e.g., 0.90)
        
        Returns:
            U_r: eigenvectors of the trajectory matrix (L x r)
            r: number of components kept
            s: singular values (for reference)
        """
        N = len(dataframe)

        for col_name, col in dataframe.items():
            K = N - self.window_len + 1
            # Build trajectory matrix (Hankel matrix)
            X = hankel(col[:self.window_len], col[self.window_len-1:])  # shape (L, K)
        
            # Singular Value Decomposition
            U, s, Vt = svd(X, full_matrices=False)
        
            # Compute explained variance and choose r
            explained_variance = (s**2) / (s**2).sum()
            cumulative_variance = np.cumsum(explained_variance)
            
            r = np.searchsorted(cumulative_variance, self.variance_thres) + 1
            # Ensure at least 1 component
            r = min(r, self.window_len)
            U_r = U[:, :r]

            self.U_dict[col_name] = {
                'U_r': U_r,
                'r': r,
                's': s
            }

    def ssa_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Denoise a new series using the pre-computed subspace U_r.
        Args:
            series: 1D numpy array of length N_val
            U_r: eigenvectors from training (L x r)
            L: same window length used in fit
        Returns:
            denoised_series: 1D array of length N_val
        """
        if not self.U_dict:
            raise ValueError('Run `fit` on the train data before transforming!')
        
        N = len(dataframe)

        denoised_dict = {}
        for col_name, col in dataframe.items():
            K = N - self.window_len + 1
            # Build trajectory matrix for the new series
            X = hankel(col[:self.window_len], col[self.window_len-1:])   # (L, K)
            # Project each column onto the signal subspace
            coeff = self.U_dict[col_name]['U_r'].T @ X                        # (r, K)
            X_rec = self.U_dict[col_name]['U_r'] @ coeff                       # (L, K) reconstructed trajectory
            
            # Vectorized diagonal averaging (your method)
            denoised = np.zeros(N)
            count = np.zeros(N)
            for i in range(self.window_len):
                denoised[i:i+K] += X_rec[i, :]
                count[i:i+K] += 1
            denoised = denoised / count
            denoised_dict[col_name] = denoised
        
        return pd.DataFrame(denoised_dict, index=dataframe.index)

class Preprocessor:
    col_sep = '_'
    
    def __init__(
            self,
            common_features: list[str] | None = None,
            broadcast: bool = False
        ):
        """
        Initialize Preprocessor which transorforms and normalizes the given dataset
        
        @param col_sep str
            Special character that separates the ticker string from the feature string.
        @param common_features list[str] List of common features in the dataset. Default = None
        """
        self.common_features = common_features
        self.broadcast = broadcast
        
        self._yeo_john = PowerTransformer(method='yeo-johnson', standardize=False)
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)

        self.ssa = SSA(window_len=60)

        self._robust_scaler = RobustScaler()

        self.unordered_cols = None
        self.all_tickers = None

    def _transform(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Transforms Volume Change columns using Yeo Johnson Transformation and
        Turnover columns using Box Cox Transformation. Yeo Johnson allows negative values,
        whereas Box Cox doesn't. This method treats train and validation or test 
        splits differently. Use mode `fit` to fit and transform, use mode `split`
        to transform only and not refit.

        @param data pd.DataFrame Dataset to be transformed
        @param mode str `git` or `split`

        @return pd.DataFrame Transformed dataset
        """
        vol_change_cols = extract_req_cols(self.unordered_cols, '_VOL_CHANGE')
        turnover_cols = extract_req_cols(self.unordered_cols, '_TURNOVER')
        
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
    
    def _normalize(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Normalize data set using robust scaling.
        This method treats train and validation or test splits differently.
        Use mode `fit` to fit and transform, use mode `split`
        to transform only and not refit.

        @param data pd.DataFrame Dataframe to be normalized
        @param mode str `fit` or `split`. 
            Determines if dataframe must be used to fit the scaler or the scaler should transform the dataframe.

        @return data pd.DataFrame Normalized using scaler object
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

    # def _extract_tickers(self) -> list[str]:
    #     """
    #     Extract ticker symbols from column names of the dataset.

    #     @return list[str] List of the ticker symbols sorted alphabetically
    #     """
    #     tickers = []
    #     for col in self.unordered_cols :
    #         if col != 'date':
    #             ticker = col.split(self.col_sep, 1)[0]
    #             tickers.append(ticker)
        
    #     if self.common_features:
    #         tickers = [x for x in sorted(set(tickers)) if x not in self.common_features]
    #         return tickers
    #     else:
    #         return sorted(set(tickers))

    def _broadcast_common(
            self, data: pd.DataFrame, features: list[str]
        ) -> pd.DataFrame:
        """
        Broadcast common features to all tickers with names <ticker>_<common_feature>.
        
        @param data pd.DataFrame dataset which needs broadcasting of common features
        @param features list[str] List of features which need to be broadcasted to every stock

        @return pd.DataFrame dataframe with broadcasted common features
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
        
        @param macro_cols list[str] List of column names in macro-economic dataset
        """
        # TODO: Use set instead of dict (more efficient)
        base_common = self.common_features or []
        combined = list(dict.fromkeys(base_common + macro_cols))
        self.common_features = combined if combined else None

    def _build_feats_order(self) -> tuple[list, list]:
        """Extract tickers and features from full DataFrame column names."""
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

    def process_train_data(
            self, train: pd.DataFrame, macro_data: pd.DataFrame | None = None
        )-> pd.DataFrame:
        """
        Preprocesses given training data

        @param train pd.DataFrame Training data
        @param macro_data pd.DataFrame Macro data aligned to training dates. Default = None
        
        @return pd.DataFrame Preprocessed training data
        """

        macro_cols: list[str] = []
        if macro_data:
            macro_cols = list(macro_data.columns)
            train = pd.concat([train, macro_data], axis=1)

            # Update common features with macro features
            self._update_common_features(macro_cols)

        # Reorder columns in alphabetical order
        self.unordered_cols = list(train.columns)
        self.ordered_cols, self.all_tickers = self._build_feats_order()

        train = train[self.ordered_cols]

        # Singular Spectral Analysis for denosining
        self.ssa.ssa_fit(train)
        train = self.ssa.ssa_transform(train)
        
        # train = self._transform(train, 'fit')
        train = self._normalize(train, 'fit')

        # Broadcast common features only if needed
        if self.broadcast:
            train = self._broadcast_common(train, self.common_features)

        return train

    def process_split_data(
            self, split_data: pd.DataFrame, macro_data: pd.DataFrame | None = None
        ) -> pd.DataFrame:
        """
        Preprocesses given validation or test data based on statistics 
        from the training data.

        @param split_data pd.DataFrame Validation or test data
        @param macro_data: pd.DataFrame Macro data aligned to validation/test dates. Default = None
        
        @return pd.DataFrame Preprocessed validation or test data
        """

        # macro_cols: list[str] = []
        if macro_data:
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
        # split_data = self._transform(split_data, 'split')

        # Singular Spectral Analysis for denosining
        split_data = self.ssa.ssa_transform(split_data)

        split_data = self._normalize(split_data, 'split')

        # Broadcast common features only if needed
        if self.broadcast:
            split_data = self._broadcast_common(split_data, self.common_features)

        return split_data

    def get_common_features(self) -> list:
        """
        Getter method to get the common features list at the current state of the object.
        """
        return self.common_features

def preprocessor2(
        returns_is: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates covariance and correlation matrices for returns data

    @param train pd.DataFrame Training split data, only returns
    @param val pd.DataFrame Validation split data, only returns

    @return tuple[pd.DataFrame, pd.DataFrame] covariance and correlation matrices
    """

    returns_is_cov = returns_is.cov()
    returns_is_corr = returns_is.corr()

    return returns_is_cov, returns_is_corr