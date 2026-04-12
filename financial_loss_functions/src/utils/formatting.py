import copy
import numpy as np
import pandas as pd

def extract_req_cols(columns_list: list, suffix: str) -> list:
    """
    Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

    @param columns_list list List of all column names.
    @param suffix str 
        Suffix str to extract its respective columns. e.g., VOL_CHANGE, RETURN
    
    @return required_cols List of required column names for the given suffix
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

def reformat_hparams(model_cfg: dict, loss_cfg: dict) -> dict:
    """
    Reformat the hyperparameters dict, so that each model + loss 
    combination gets its own dict.
    """
    hparams = {
        'model': copy.deepcopy(model_cfg['model']),
        'optimizer': copy.deepcopy(model_cfg['optimizer']),
        'train': copy.deepcopy(model_cfg['train']),
        'scheduler': copy.deepcopy(model_cfg.get('scheduler')),
        'loss': copy.deepcopy(loss_cfg.get('lambdas', {}))
    }

    return hparams

def split_combo_names(
        model_losses: list[str], sep: str
    ) -> list[tuple[str, str]]:

    split_combos = []
    for i in model_losses:
        parts = i.split(sep, 1)
        if len(parts) != 2:
            raise ValueError(f'Model + Loss combo name string is incorrect: {i}')
        
        split_combos.append((parts[0], parts[1]))

    return split_combos

def serialize_np_dict(obj: dict):
    """
    Convert numpy arrays in a dict to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: serialize_np_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_np_dict(i) for i in obj]
    return obj

def print_evaluation_info(
        out_win_date_cols, in_win_date_cols: list|None=None, **kwargs
    ):
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

def reformat_model_perfs(
        all_daily_returns: dict[str, list[list[float]]],
        alloc_weights: dict[str, list[list[float]]],
        out_win_date_cols
    ) -> dict[str, dict[str, list[float]]]:

    reformatted_w_dates = {}
    for model, all_winds_ls in all_daily_returns.items():
        reformatted_w_dates.setdefault(model, {})
        all_winds_wts = alloc_weights.get(model)
        if all_winds_wts:
            for win_rets, win_wts, win_dates in zip(all_winds_ls, all_winds_wts, out_win_date_cols):
                start_date = win_dates[0].strftime('%Y-%m-%d')
                end_date = win_dates[-1].strftime('%Y-%m-%d')
                date_range = f'{start_date}_{end_date}'

                reformatted_w_dates[model] = {
                    date_range: {
                        'returns': win_rets,
                        'weights': win_wts
                    }
                }
        else:
            # Equal weight and S&P500 should not have weights
            for win_rets, win_dates in zip(all_winds_ls, out_win_date_cols):
                start_date = win_dates[0].strftime('%Y-%m-%d')
                end_date = win_dates[-1].strftime('%Y-%m-%d')
                date_range = f'{start_date}_{end_date}'
                reformatted_w_dates[model] = {
                    date_range: {
                        'returns': win_rets
                    }
                }
            
    return reformatted_w_dates
