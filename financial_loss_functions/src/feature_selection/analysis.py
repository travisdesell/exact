from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.feature_selection.rules import (
    build_sector_mapping_df,
    get_sector_prior_weights,
    load_sector_mapping,
)
from src.data_processing.preprocess import MacroCombiner


@dataclass
class FeatureSelectionArtifacts:
    """Container for the key dataframes/files produced by feature selection."""

    sector_map: pd.DataFrame
    macro_aligned: pd.DataFrame
    ticker_rankings: pd.DataFrame
    ticker_selected: pd.DataFrame
    summary_path: Path


def _ensure_datetime_index(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Normalize dataframe index to sorted DatetimeIndex."""

    cleaned = df.copy()
    if date_col in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[date_col], keep='first')
        cleaned[date_col] = pd.to_datetime(cleaned[date_col])
        cleaned = cleaned.set_index(date_col)
    elif not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)

    cleaned = cleaned.sort_index()
    return cleaned


def _infer_returns_train_file(crsp_dir: Path, train_predictors_name: str) -> Path:
    """Resolve matching returns-train file from predictors-train filename."""

    candidate = crsp_dir / train_predictors_name.replace('predictors', 'returns')
    if candidate.exists():
        return candidate

    fallback = crsp_dir / 'combined_returns_train.csv'
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f'Unable to infer returns train file from {train_predictors_name} in {crsp_dir}'
    )


def _load_train_crsp(paths_config: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CRSP train predictors and returns with aligned datetime index."""

    crsp_dir = Path(paths_config['data']['crsp_dir'])
    train_predictors_name = paths_config['raw_files']['train']
    predictors_path = crsp_dir / train_predictors_name
    returns_path = _infer_returns_train_file(crsp_dir, train_predictors_name)

    if not predictors_path.exists():
        raise FileNotFoundError(f'Missing predictors train file: {predictors_path}')
    if not returns_path.exists():
        raise FileNotFoundError(f'Missing returns train file: {returns_path}')

    predictors_train = _ensure_datetime_index(pd.read_csv(predictors_path), 'date')
    returns_train = _ensure_datetime_index(pd.read_csv(returns_path), 'date')
    return predictors_train, returns_train


def _load_macro_data(
    macro_dir: Path,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Load macro CSVs and build a feature->macro-group lookup."""

    raw_macro: Dict[str, pd.DataFrame] = {}
    feature_to_group: Dict[str, str] = {}

    macro_files = sorted(macro_dir.glob('*.csv'))
    if not macro_files:
        raise FileNotFoundError(f'No macro files found in: {macro_dir}')

    for file_path in macro_files:
        macro_df = pd.read_csv(file_path, index_col=0)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df = macro_df.sort_index()
        raw_macro[file_path.stem] = macro_df

        for col in macro_df.columns:
            feature_to_group[col] = file_path.stem

    return raw_macro, feature_to_group


def align_macro_to_business_days(
    raw_macro: Dict[str, pd.DataFrame],
    target_index: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """
    Combine category-wise macro series, upsample to business days, and align to
    CRSP split dates so every stock row has macro context.
    """

    macro_combiner = MacroCombiner(resample_freq='B')
    combined_macro = macro_combiner.combine_macro_data(raw_macro)
    daily_macro = macro_combiner.to_daily(combined_macro)

    aligned = daily_macro.reindex(pd.DatetimeIndex(target_index)).ffill().bfill()
    aligned = aligned.sort_index()
    return aligned


def compute_ticker_feature_rankings(
    ticker: str,
    target_returns: pd.Series,
    macro_aligned: pd.DataFrame,
    feature_to_group: Dict[str, str],
    lags: list[int],
    low_corr_threshold: float,
    group_prior_weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Rank macro features for one ticker using lagged absolute Spearman
    correlation plus sector-prior weighting.
    """

    rows = []

    for macro_feature in sorted(macro_aligned.columns):
        best_lag = None
        best_corr = 0.0

        for lag in lags:
            # Lag macro feature to model delayed macro impact on returns.
            shifted_macro = macro_aligned[macro_feature].shift(lag)
            corr = target_returns.corr(shifted_macro, method='spearman')
            corr_abs = abs(float(corr)) if pd.notna(corr) else 0.0

            if corr_abs > best_corr:
                best_corr = corr_abs
                best_lag = lag

        macro_group = feature_to_group.get(macro_feature, 'unknown')
        prior_weight = group_prior_weights.get(macro_group, 0.0)

        rows.append(
            {
                'ticker': ticker,
                'macro_feature': macro_feature,
                'macro_group': macro_group,
                'best_lag': int(best_lag) if best_lag is not None else int(lags[0]),
                'max_abs_spearman_corr': best_corr,
                'sector_prior_weight': prior_weight,
            }
        )

    ranking = pd.DataFrame(rows)
    max_corr = float(ranking['max_abs_spearman_corr'].max())
    if max_corr > 0:
        # Normalize per ticker so correlation and prior terms are on comparable scales.
        ranking['normalized_max_abs_corr'] = ranking['max_abs_spearman_corr'] / max_corr
    else:
        ranking['normalized_max_abs_corr'] = 0.0

    # Hybrid score: observed dependency (0.7) + sector/common-sense prior (0.3).
    ranking['composite_score'] = (
        0.7 * ranking['normalized_max_abs_corr'] + 0.3 * ranking['sector_prior_weight']
    )
    ranking['low_corr_flag'] = ranking['max_abs_spearman_corr'] < low_corr_threshold

    # Stable sorting keeps output deterministic for ties across reruns.
    ranking = ranking.sort_values(
        by=['composite_score', 'max_abs_spearman_corr', 'macro_feature'],
        ascending=[False, False, True],
        kind='mergesort',
    ).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    return ranking


def _write_summary(
    summary_path: Path,
    rankings_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    lags: list[int],
    low_corr_threshold: float,
    top_k: int,
) -> None:
    """Write compact markdown summary of pipeline coverage and thresholds."""

    low_corr_count = int(rankings_df['low_corr_flag'].sum()) if not rankings_df.empty else 0
    low_corr_ratio = (low_corr_count / len(rankings_df)) if len(rankings_df) > 0 else 0.0
    covered_tickers = selected_df['ticker'].nunique() if not selected_df.empty else 0

    lines = [
        '# Feature Selection Summary',
        '',
        f'- Tickers in sector map: {len(sector_df)}',
        f'- Tickers with selected features: {covered_tickers}',
        f'- Total ranked ticker-macro pairs: {len(rankings_df)}',
        f'- Low-correlation threshold: {low_corr_threshold}',
        f'- Low-correlation pairs: {low_corr_count} ({low_corr_ratio:.2%})',
        f'- Lags evaluated (business days): {lags}',
        f'- Top-K features per ticker: {top_k}',
        '',
        '## Outputs',
        '- ticker_macro_rankings.csv',
        '- ticker_selected_features.csv',
        '- sector_assignment_50.csv',
    ]
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_feature_selection_pipeline(
    paths_config: Dict,
    output_dir: Path,
    lags: list[int],
    low_corr_threshold: float,
    top_k: int = 10,
) -> FeatureSelectionArtifacts:
    """
    End-to-end feature-selection pipeline.

    Method:
    1) load sector map + CRSP train returns,
    2) align macro features to stock dates,
    3) rank each macro feature per ticker with lag search,
    4) apply sector priors and export Top-K selections.
    """

    if not lags:
        raise ValueError('At least one lag is required.')
    if top_k <= 0:
        raise ValueError('top_k must be >= 1.')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sector mapping is a hard requirement because priors are sector-driven.
    sector_mapping_json = load_sector_mapping()
    sector_df = build_sector_mapping_df(sector_mapping_json)
    if sector_df['sector'].isna().any():
        raise ValueError('Missing sector values in sector classification mapping.')

    sector_csv_path = output_dir / 'sector_assignment_50.csv'
    sector_df.to_csv(sector_csv_path, index=False)

    _, returns_train = _load_train_crsp(paths_config)
    macro_dir = Path(paths_config['data']['raw_macro_dir'])
    raw_macro, feature_to_group = _load_macro_data(macro_dir)
    macro_aligned = align_macro_to_business_days(raw_macro, returns_train.index)

    ticker_cols = [col for col in returns_train.columns if col.endswith('_RET')]
    tickers = [col.replace('_RET', '') for col in ticker_cols]

    missing_tickers = sorted(set(tickers) - set(sector_mapping_json.keys()))
    if missing_tickers:
        raise KeyError(f'Missing sector mapping for tickers: {missing_tickers}')

    all_rankings = []
    for ticker in tickers:
        target = returns_train[f'{ticker}_RET']
        # Priors are computed per ticker from primary/secondary sector assignment.
        prior_weights = get_sector_prior_weights(ticker)
        ticker_ranking = compute_ticker_feature_rankings(
            ticker=ticker,
            target_returns=target,
            macro_aligned=macro_aligned,
            feature_to_group=feature_to_group,
            lags=lags,
            low_corr_threshold=low_corr_threshold,
            group_prior_weights=prior_weights,
        )
        all_rankings.append(ticker_ranking)

    rankings_df = pd.concat(all_rankings, axis=0, ignore_index=True)
    rankings_df = rankings_df.sort_values(
        by=['ticker', 'rank', 'macro_feature'],
        ascending=[True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)
    # Per-ticker final shortlist consumed by downstream modeling.
    selected_df = rankings_df[rankings_df['rank'] <= top_k].copy()

    rankings_path = output_dir / 'ticker_macro_rankings.csv'
    selected_path = output_dir / 'ticker_selected_features.csv'
    summary_path = output_dir / 'feature_selection_summary.md'

    rankings_df.to_csv(rankings_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    _write_summary(
        summary_path=summary_path,
        rankings_df=rankings_df,
        selected_df=selected_df,
        sector_df=sector_df,
        lags=lags,
        low_corr_threshold=low_corr_threshold,
        top_k=top_k,
    )

    return FeatureSelectionArtifacts(
        sector_map=sector_df,
        macro_aligned=macro_aligned,
        ticker_rankings=rankings_df,
        ticker_selected=selected_df,
        summary_path=summary_path,
    )
