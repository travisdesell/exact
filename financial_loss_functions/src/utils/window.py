import numpy as np
import pandas as pd

def calc_current_idxs(step: int, stride: int) -> tuple[int, int]:
    """
    Calculate current window's start and end indices based on current step and walk stride.
    
    Args:
        step (int): step count. Must be > 1.
        stride (int): Stride size for the walk.
    
    Returns:
        tuple[current_start, current_end] (tuple[int, int]): Tuple containing the start and end 
        indices for the current step.
    """
    if step == 0:
        raise ValueError('Got step 0. Must be > 0')
    
    current_end = step * stride
    current_start = current_end - stride

    return current_start, current_end

def get_date_index_col(
        split: pd.DataFrame, wind_strt_stops: list[tuple[int, int]]
    ) -> list[pd.DatetimeIndex]:
    """
    Get the datetime index columns from the provided dataframe using the 
    start and stop indexes.

    Args:
        split (pd.DataFrame): Evaluation split of the data.
        wind_strt_stops (list[tuple[int, int]]): Output window indices in the format, [(start, end),..].

    Returns:
        date_idx_cols (list[pd.DatetimeIndex]): List containing the date indices for each output window.
    """
    date_idx_cols = []
    for idxs in wind_strt_stops:
        date_idx_cols.append(split.index[idxs[0] : idxs[1]])
    
    return date_idx_cols

def extract_oos_dates(
        split: pd.DataFrame, 
        in_wind_idxs: list[tuple[int, int]], 
        out_wind_idxs: list[tuple[int, int]]
    ) -> tuple[list[pd.DatetimeIndex], list[pd.DatetimeIndex]]:
    """
    Extract Out-of-Sample dates from the evaluation data.

    Args:
        split (pd.DataFrame): Evaluation split of the data.
        in_wind_idxs (list[tuple[int, int]]): Input window indices in the format, [(start, end),..].
        out_wind_idxs (list[tuple[int, int]]): Output window indices in the format, [(start, end),..].
    
    Returns:
        tuple[list[pd.DatetimeIndex], list[pd.DatetimeIndex]]: Tuple containing input and output 
            date indices.
    """
    in_win_date_cols = get_date_index_col(split, in_wind_idxs)
    out_win_date_cols = get_date_index_col(split, out_wind_idxs)

    return in_win_date_cols, out_win_date_cols

def extract_sp500_winds(
        benchmark_split: pd.DataFrame, col_name: str, out_win_idxs: list[tuple]
    ) -> np.ndarray:
    """
    Reshape 2D dataframe into into windows based on the given output window indices.
    """
    sp500_col = benchmark_split[col_name]

    sp500_windows = []
    for idxs in out_win_idxs:
        sp500_windows.append(sp500_col.iloc[idxs[0] : idxs[1]].to_numpy())
    
    return np.stack(sp500_windows)

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