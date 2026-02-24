import pandas as pd

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