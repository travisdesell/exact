import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler


class MacroCombiner:
    """
    Utility class to combine macro-economic datasets, upsample them to a daily
    frequency, and align them with CRSP train/val/test splits.
    """

    def __init__(self, resample_freq: str = 'B'):
        self.resample_freq = resample_freq

    def combine_macro_data(self, raw_macro: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate all macro dataframes column-wise after enforcing datetime
        indices and sorting by date.

        @param raw_macro dict[str, pd.DataFrame] 
            Dictionary containing all amcro-econimic dataframes
        
        @return pd.Dataframe Combined dataframe for all macro-economic data
        """
        macro_frames = []
        for df in raw_macro.values():
            temp = df.copy()
            temp.index = pd.to_datetime(temp.index)
            temp.sort_index(inplace=True)
            macro_frames.append(temp)

        macro_df = pd.concat(macro_frames, axis=1)

        # Remove duplicate dates and columns that are entirely NaN
        macro_df = macro_df.loc[~macro_df.index.duplicated(keep='first')]
        macro_df = macro_df.dropna(axis=1, how='all')

        return macro_df

    def to_daily(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample macro data to business-day frequency, forward filling the
        monthly/weekly series to create a daily view.

        @param macro_df pd.DataFrame Dataframe with all macro-economic columns

        @return pd.DataFrame Macro-economic data converted to set resample frequency 
        """
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df.index = pd.to_datetime(macro_df.index)

        macro_df = macro_df.sort_index()
        daily_macro = macro_df.resample(self.resample_freq).ffill()

        # In case the first few rows are missing (no previous value), backfill once
        daily_macro = daily_macro.bfill()

        # Drop any rows where all macro series are NaN (e.g., trailing dates beyond coverage)
        daily_macro = daily_macro.dropna(how='all')

        return daily_macro

    def split_by_crsp_dates(
            self,
            daily_macro: pd.DataFrame,
            train_index: pd.Index,
            val_index: pd.Index,
            test_index: pd.Index
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align the daily macro dataframe to the CRSP train/val/test date indices.

        @param daily_macro pd.DataFrame Macro dataframe with frequency converted
        @param train_index pd.Index Date index from CRSP train data split
        @param val_index pd.Index Date index from CRSP validation data split
        @param train_index pd.DataFrame Date index from CRSP test data split

        @return tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] containing aligned split with CRSP data
        """

        def _align(index: pd.Index) -> pd.DataFrame:
            aligned = daily_macro.reindex(index)
            return aligned.ffill().bfill()

        macro_train = _align(train_index)
        macro_val = _align(val_index)
        macro_test = _align(test_index)

        return macro_train, macro_val, macro_test


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
    
    train['date'] = pd.to_datetime(train['date'])
    train.set_index('date', inplace=True)

    val['date'] = pd.to_datetime(val['date'])
    val.set_index('date', inplace=True)

    test['date'] = pd.to_datetime(test['date'])
    test.set_index('date', inplace=True)

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
    
class Preprocessor:
    def __init__(
            self, col_sep: str = '_', common_features: list[str] | None = None
        ):
        """
        Initialize Preprocessor which transorforms and normalizes the given dataset
        
        @param col_sep str
            Special character that separates the ticker string from the feature string.
        @param common_features list[str] List of common features in the dataset. Default = None
        """
        self.common_features = common_features
        self.col_sep = col_sep
        self._yeo_john = PowerTransformer(method='yeo-johnson', standardize=False)
        self._box_cox = PowerTransformer(method='box-cox', standardize=False)
        self._robust_scaler = RobustScaler()

        self.all_col_names = None
        self.all_tickers = None

    def _extract_req_cols(self, columns_list: list, suffix: str) -> list:
        """
        Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

        @param columns_list list List of all column names.
        @param suffix str 
            Suffix str to extract its respective columns. e.g., VOL_CHANGE, RETURN
        
        @return required_cols List of required column names for the given suffix
        """
        required_cols = [col for col in columns_list if suffix in col]
        return required_cols

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

    def _extract_tickers(self) -> list[str]:
        """
        Extract ticker symbols from column names of the dataset.

        @return list[str] List of the ticker symbols sorted alphabetically
        """
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
        if macro_data is not None:
            macro_cols = list(macro_data.columns)
            train = pd.concat([train, macro_data], axis=1)

        # Update common features before extracting tickers so macro columns are excluded
        self._update_common_features(macro_cols)

        self.all_col_names = list(train.columns)
        self.all_tickers = self._extract_tickers()
        
        # train = self._transform(train, 'fit')

        train = self._normalize(train, 'fit')

        # Broadcasting only if common features are present
        if self.common_features:
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

        macro_cols: list[str] = []
        if macro_data is not None:
            macro_cols = list(macro_data.columns)
            split_data = pd.concat([split_data, macro_data], axis=1)

        if macro_cols:
            self._update_common_features(macro_cols)

        # Ensure column alignment with training data before normalization
        if self.all_col_names is not None:
            missing = set(self.all_col_names) - set(split_data.columns)
            extra = set(split_data.columns) - set(self.all_col_names)
            if missing:
                raise ValueError(f'Missing columns in split data: {missing}')
            if extra:
                # Drop any unexpected columns to match training schema
                split_data = split_data[self.all_col_names]
            else:
                split_data = split_data[self.all_col_names]

        # split_data = self._transform(split_data, 'split')

        split_data = self._normalize(split_data, 'split')

        # Broadcast only if common features are present
        if self.common_features:
            split_data = self._broadcast_common(split_data, self.common_features)

        return split_data

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