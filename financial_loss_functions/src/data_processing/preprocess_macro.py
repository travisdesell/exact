import pandas as pd
from pathlib import Path
from typing import Dict

from src.data_processing.preprocess import MacroCombiner
from src.feature_selection.analysis import (
    align_macro_to_business_days,
    compute_ticker_feature_rankings,
    _load_macro_data,
)
from src.feature_selection.rules import (
    get_sector_prior_weights,
    load_sector_mapping,
)


def select_macro_for_pipeline(
    raw_macro: Dict[str, pd.DataFrame],
    returns_train: pd.DataFrame,
    lags: list[int],
    top_k: int,
    low_corr_threshold: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run feature selection on macro data using CRSP train returns and return
    the daily macro DataFrame filtered to the union of selected features
    across all tickers.

    @param raw_macro Dict[str, pd.DataFrame] Raw macro category DataFrames
    @param returns_train pd.DataFrame CRSP train returns (columns: <TICKER>_RET or <TICKER>)
    @param lags list[int] Business-day lags for correlation search
    @param top_k int Number of top macro features to select per ticker
    @param low_corr_threshold float Threshold for flagging low correlations

    @return tuple[pd.DataFrame, list[str]]
        - Aligned daily macro DataFrame filtered to selected columns
        - Sorted list of selected macro column names
    """
    feature_to_group: Dict[str, str] = {}
    for name, df in raw_macro.items():
        for col in df.columns:
            feature_to_group[col] = name

    macro_aligned = align_macro_to_business_days(raw_macro, returns_train.index)

    ret_suffix = '_RET'
    ticker_cols = [c for c in returns_train.columns if c.endswith(ret_suffix)]
    if ticker_cols:
        tickers = [c.replace(ret_suffix, '') for c in ticker_cols]
    else:
        tickers = list(returns_train.columns)

    sector_mapping = load_sector_mapping()

    all_selected_features: set[str] = set()
    for ticker in tickers:
        col_name = f'{ticker}{ret_suffix}' if f'{ticker}{ret_suffix}' in returns_train.columns else ticker
        target = returns_train[col_name]

        if ticker in sector_mapping:
            prior_weights = get_sector_prior_weights(ticker)
        else:
            prior_weights = {}

        ranking = compute_ticker_feature_rankings(
            ticker=ticker,
            target_returns=target,
            macro_aligned=macro_aligned,
            feature_to_group=feature_to_group,
            lags=lags,
            low_corr_threshold=low_corr_threshold,
            group_prior_weights=prior_weights,
        )
        top_features = ranking[ranking['rank'] <= top_k]['macro_feature'].tolist()
        all_selected_features.update(top_features)

    selected_cols = sorted(all_selected_features)
    filtered_macro = macro_aligned[selected_cols]

    return filtered_macro, selected_cols


def prepare_macro_splits(
    raw_macro: Dict[str, pd.DataFrame],
    train_index: pd.Index,
    val_index: pd.Index,
    test_index: pd.Index,
    selected_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine raw macro, upsample to daily, filter to selected features,
    and split aligned to CRSP date indices. Uses only forward-fill for
    alignment to avoid data leakage.

    @param raw_macro Dict[str, pd.DataFrame] Raw macro category DataFrames
    @param train_index pd.Index CRSP train date index
    @param val_index pd.Index CRSP validation date index
    @param test_index pd.Index CRSP test date index
    @param selected_cols list[str] Macro columns selected by feature selection

    @return tuple of (macro_train, macro_val, macro_test) DataFrames
    """
    combiner = MacroCombiner(resample_freq='B')
    combined = combiner.combine_macro_data(raw_macro)
    daily = combiner.to_daily(combined)

    filtered = daily[selected_cols]

    def _align_ffill_only(index: pd.Index) -> pd.DataFrame:
        aligned = filtered.reindex(index)
        return aligned.ffill().bfill()

    macro_train = _align_ffill_only(train_index)
    macro_val = _align_ffill_only(val_index)
    macro_test = _align_ffill_only(test_index)

    return macro_train, macro_val, macro_test
