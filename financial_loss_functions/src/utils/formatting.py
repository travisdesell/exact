import copy
import numpy as np
import pandas as pd
from typing import Any

def extract_req_cols(columns_list: list, suffix: str) -> list[str]:
    """
    Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

    Args:
        columns_list (list[str]): List of all column names.
        suffix (str): Suffix string to extract its respective columns. e.g., VOL_CHANGE, RETURN.
    
    Returns:
        required_cols (list[str]): List of required column names for the given suffix
    """
    # required_cols = [col for col in columns_list if suffix in col]
    required_cols = [col for col in columns_list if col.endswith(suffix)]
    return required_cols

def split_col(col_sep: str, col: str) -> tuple[str, str]:
    """Split column into (ticker, feature) using first underscore only."""
    parts = col.split(col_sep, 1)
    if len(parts) != 2:
        raise ValueError(f"Column '{col}' does not match <ticker>_<feature> format")
    return parts[0], parts[1]  # ticker, feature-with-underscores

def reformat_hparams(
        model_cfg: dict, loss_cfg: dict
    ) -> dict[str, dict[str, Any]]:
    """
    Reformat the hyperparameters dict, so that each model + loss 
    combination gets its own dict.

    Args:
        model_cfg (dict): Model configuration containing the hyperparameters for a single model.
            Dictionary must contain the keys: `model`, `optimizer`, `train`, `scheduler`;
            Optionally `loss`, if applicable.
    
    Returns:
        hparams (dict[str, dict[str, Any]]): Reformatted dictionary containing
            all hyperparameters for a single mode loss combination.
    """
    hparams = {
        'model': copy.deepcopy(model_cfg['model']),
        'optimizer': copy.deepcopy(model_cfg['optimizer']),
        'train': copy.deepcopy(model_cfg['train']),
        'scheduler': copy.deepcopy(model_cfg.get('scheduler', {})),
        'loss': copy.deepcopy(loss_cfg.get('lambdas', {}))
    }

    return hparams

def split_combo_names(  
        model_losses: list[str], sep: str
    ) -> list[tuple[str, str]]:

    """
    Split the model names and loss functions name from a list of model+loss combination names.
    [<model_name>-<loss_name>] -> [(<model_name>, <loss_name>)]

    Args:
        model_losses (list[str]): List contianining model+loss combination names.
        sep (str): Separater string used between <model_name> and <loss_name>.
    
    Returns:
        split_combos (list[tuple[str, str]]): List of tuples containing the 
            model and loss function names in the tuple items.
    """

    split_combos = []
    for i in model_losses:
        parts = i.split(sep, 1)
        if len(parts) != 2:
            raise ValueError(f'Model + Loss combo name string is incorrect: {i}')
        
        split_combos.append((parts[0], parts[1]))

    return split_combos

def serialize_np_dict(obj: dict):
    """
    Recursive function to convert numpy arrays in a dict to python lists.
    
    Args:
        obj (dict): Dictionary object that will be searched recursively 
            for numpy arrays and convert them to lists.
    
    Returns:
        obj (dict): Dictionary object that has no more numpy arrays.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: serialize_np_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_np_dict(i) for i in obj]
    return obj

def deserialize_np_dict(obj):
    """Recursively converts lists in a dictionary (or list) to NumPy arrays where appropriate.

    The function traverses dictionaries and lists. When a list is encountered, it attempts
    to convert it into a NumPy array. If the list contains only numbers and can be stacked
    into a rectangular array, it returns a NumPy array; otherwise, it recurses into
    the list elements. Empty lists become empty arrays. Non-dict/list objects are
    returned unchanged.

    Args:
        obj (any): Input object (usually a dictionary or list) that may contain nested
            lists that were originally NumPy arrays but were serialised (e.g., to JSON).

    Returns:
        any: The same structure with lists replaced by NumPy arrays wherever possible,
        otherwise the original object.
    """
    if isinstance(obj, dict):
        return {k: deserialize_np_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # If the list is empty, leave as list (or could become empty array)
        if not obj:
            return np.array([])
        # Check if the list contains only numbers or lists (nested)
        # Try to convert to array; if it fails (e.g., mixed types), leave as list and recurse.
        try:
            # Attempt to create a NumPy array
            arr = np.array(obj)
            # If the array has object dtype, it means the conversion didn't produce a numeric array
            if arr.dtype == object:
                # Recursively process each element
                return [deserialize_np_dict(item) for item in obj]
            else:
                return arr
        except (ValueError, TypeError):
            # Not convertible to array, so process elements recursively
            return [deserialize_np_dict(item) for item in obj]
    else:
        return obj

def print_evaluation_info(
        out_win_date_cols: list[pd.DatetimeIndex],
        in_win_date_cols: list[pd.DatetimeIndex] | None = None,
        **kwargs
    ) -> None:
    """
    Printing utility function for pipeline results.

    Args:
        out_win_date_cols (list[pd.DatetimeIndex]): List cotaining pd.DateTimeIndex for 
            each output window of the evaulation split.
        in_win_date_cols: (list[pd.DatetimeIndex] | None):  List containing pd.DataTimeIndex 
            for each input window of the evaluation split. Default = None, since walkforward 
            does not have fixed input windows from the evaluation data.
        **kwargs: Any key object pair can be provided for printing to terminal (console).
            Provided key will be made upper case for printing.
        
    """
    if in_win_date_cols:
        eval_dates_info = {
            'Input Window Start': [],
            'Input Window End': [],
            'Out Window Start': [],
            'Out Window End': []
        }
        
        for in_date, out_date in zip(in_win_date_cols, out_win_date_cols):
            eval_dates_info['Input Window Start'].append(in_date[0])
            eval_dates_info['Input Window End'].append(in_date[-1])
            eval_dates_info['Out Window Start'].append(out_date[0])
            eval_dates_info['Out Window End'].append(out_date[-1])
    else:
        eval_dates_info = {
            'Out Window Start': [],
            'Out Window End': []
        }
        for out_date in out_win_date_cols:
            eval_dates_info['Out Window Start'].append(out_date[0])
            eval_dates_info['Out Window End'].append(out_date[-1])
        
    print('\nModels evaluated on:')
    print(pd.DataFrame(eval_dates_info))

    print('\n', '-'*10, ' Portfolio Perfomance Metrics ', '-'*10)

    # Loop over provided dataframes and print
    for metric, df in kwargs.items():
        # Cleaning up the metric name
        title = metric.replace('_', ' ').upper()
        print(f'\n{title.upper()}:\n', df)

def reform_returns_w_dates(
        daily_returns: dict[str, list[list[float]]],
        out_win_date_cols: list[pd.DatetimeIndex]
) -> dict[str, dict[str, list[float]]]:
    """Reformats daily returns by replacing window indices with date ranges.

    For each model and each window, the function creates a key of the form
    "YYYY-MM-DD_YYYY-MM-DD" (start date to end date) and stores the corresponding
    window's list of returns.

    Args:
        daily_returns (dict[str, list[list[float]]]): Dictionary mapping model name
            to a list of return windows (each window is a list of daily returns).
        out_win_date_cols (list[pd.DatetimeIndex]): List of DatetimeIndex objects,
            one per window, providing the actual dates for that window.

    Returns:
        dict[str, dict[str, list[float]]]: A nested dictionary where the outer key
        is the model name, the inner key is a date range string (start_end), and the
        value is the list of daily returns for that window.
    """
    reformatted_w_dates = {}
    for model, all_winds_ls in daily_returns.items():
        reformatted_w_dates.setdefault(model, {})
        for win_rets, win_dates in zip(all_winds_ls, out_win_date_cols):
            start_date = win_dates[0].strftime('%Y-%m-%d')
            end_date = win_dates[-1].strftime('%Y-%m-%d')
            date_range = f'{start_date}_{end_date}'
            
            reformatted_w_dates[model][date_range] = win_rets
    
    return reformatted_w_dates

def reformat_model_perfs(
        all_daily_returns: dict[str, list[list[float]]],
        alloc_weights: dict[str, list[list[float]]],
        out_win_date_cols: list[pd.DatetimeIndex]
    ) -> dict[str, dict[str, list[float]]]:
    """
    Reformat model performances and combine all information into one dictionary.
    Combines all daily returns, all allocation weights and their respective window date ranges.

    Args:
        all_daily_returns (dict[str, list[list[float]]]): Dictionary containing daily returns 
            for all models and all windows. It must be in the format, 
            {<model_name>: <all_windows_list>[<one_window_list>[<w1>], <one_window_list>[<w2>],..]}
        all_alloc_weights (dict[str, list[list[float]]]):  Dictionary containing all portfolio 
            allocation weights for all models and all windows. It must be in the format, 
            {<model_name>: <all_windows_list>[<one_window_weights>[<w1>], <one_window_weights>[<w2>],..]}
        out_win_date_cols (list[pd.DatetimeIndex]): List cotaining pd.DateTimeIndex for 
            each output window of the evaulation split.

    Returns:
        reformatted_w_dates (dict[str, dict[str, list[float]]]): Reformatted dictionary 
            containing all combined data.
    """

    reformatted_w_dates = {}
    for model, all_winds_ls in all_daily_returns.items():
        reformatted_w_dates.setdefault(model, {})
        all_winds_wts = alloc_weights.get(model)
        if all_winds_wts:
            for win_rets, win_wts, win_dates in zip(all_winds_ls, all_winds_wts, out_win_date_cols):
                start_date = win_dates[0].strftime('%Y-%m-%d')
                end_date = win_dates[-1].strftime('%Y-%m-%d')
                date_range = f'{start_date}_{end_date}'

                reformatted_w_dates[model][date_range] = {
                    'returns': win_rets,
                    'weights': win_wts
                }
        else:
            # Equal weight and S&P500 should not have weights
            for win_rets, win_dates in zip(all_winds_ls, out_win_date_cols):
                start_date = win_dates[0].strftime('%Y-%m-%d')
                end_date = win_dates[-1].strftime('%Y-%m-%d')
                date_range = f'{start_date}_{end_date}'
                reformatted_w_dates[model][date_range] = {
                    'returns': win_rets
                }
            
    return reformatted_w_dates
