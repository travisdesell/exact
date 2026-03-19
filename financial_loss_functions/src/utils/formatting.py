import copy

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
        'loss': copy.deepcopy(loss_cfg.get('lambdas'))
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