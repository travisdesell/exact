import numpy as np
import pandas as pd

def calc_current_idxs(step: int, stride: int):
    if step == 0:
        raise ValueError('Got step 0. Must be > 0')
    
    current_end = step * stride
    current_start = current_end - stride

    return current_start, current_end

def build_eval_windows(split: np.ndarray, num_steps, out_size) -> tuple[np.ndarray, list[tuple[int, int]]]:  
    eval_windows = []
    out_wind_idxs = []
    for step in range(1, num_steps+1):
        current_start, current_end = calc_current_idxs(step, out_size)

        walk_rets_eval = split[current_start : current_end]

        eval_windows.append(walk_rets_eval)

        out_wind_idxs.append((current_start, current_end))
    
    return np.stack(eval_windows), out_wind_idxs

def get_date_index_col(split: pd.DataFrame, wind_strt_stops: list[tuple]) -> list:
    """
    Get the datetime index columns from the provided dataframe using the 
    start and stop indexes.
    """
    date_idx_cols = []
    for idxs in wind_strt_stops:
        date_idx_cols.append(split.index[idxs[0] : idxs[1]])
    
    return date_idx_cols

def extract_oos_dates(
        split: pd.DataFrame, in_wind_idxs: list[tuple], out_wind_idxs: list[tuple]
    ) -> tuple[list[tuple], list[tuple]]:
    in_win_date_cols = get_date_index_col(split, in_wind_idxs)
    out_win_date_cols = get_date_index_col(split, out_wind_idxs)

    return in_win_date_cols, out_win_date_cols

def extract_sp500_winds(
        benchmark_split: pd.DataFrame, col_name: str, out_win_idxs: list[tuple]
    ) -> np.ndarray:
    sp500_col = benchmark_split[col_name]

    sp500_windows = []
    for idxs in out_win_idxs:
        sp500_windows.append(sp500_col.iloc[idxs[0] : idxs[1]].to_numpy())
    
    return np.stack(sp500_windows)