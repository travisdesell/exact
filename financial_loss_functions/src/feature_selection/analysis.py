from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_collection.const import FRED_SERIES
from src.data_processing.loading import load_macro_data
from src.data_processing.preprocess import MacroCombiner


DEFAULT_NO_LOGIC_MACRO_IDS = {
    'SP500',  # Duplicates market return information already represented in CRSP.
    'M13002US35620M156NNBR',  # Pre-1997 splice series; not meaningful for this sample window.
}


@dataclass
class FeatureSelectionArtifacts:
    crsp_non_lagged: pd.DataFrame
    crsp_lagged: pd.DataFrame
    macro_aligned: pd.DataFrame
    model_data: pd.DataFrame
    correlation_matrix: pd.DataFrame
    positive_correlations: pd.DataFrame
    negative_correlations: pd.DataFrame
    low_correlations: pd.DataFrame
    pca_feature_scores: pd.Series
    pca_explained_variance: pd.DataFrame
    rf_importance: pd.Series
    permutation_importance: pd.Series
    shap_importance: pd.Series | None
    ranking_comparison: pd.DataFrame
    model_metrics: pd.DataFrame
    dropped_macro_features: pd.DataFrame


def _flatten_fred_ids(series_config: Dict[str, Dict[str, str]]) -> set[str]:
    all_ids: set[str] = set()
    for category_dict in series_config.values():
        all_ids.update(category_dict.values())
    return all_ids


def _load_crsp_predictors(paths_config: Dict) -> tuple[pd.DataFrame, Path]:
    crsp_dir = Path(paths_config['data']['crsp_dir'])
    raw_path = crsp_dir / 'combined_predictors_raw.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        source_path = raw_path
    else:
        train_path = crsp_dir / paths_config['raw_files']['train']
        val_path = crsp_dir / paths_config['raw_files']['val']
        test_path = crsp_dir / paths_config['raw_files']['test']
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        df = pd.concat([train, val, test], axis=0, ignore_index=True)
        source_path = crsp_dir

    if 'date' not in df.columns:
        raise ValueError('CRSP predictors data must contain a `date` column.')

    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.sort_values('date').set_index('date')
    return df, source_path


def aggregate_ticker_features(crsp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all ticker-level CRSP features from <ticker>_<feature> into <feature>
    using row-wise means across tickers.
    """
    tickers = sorted({
        column.rsplit('_RET', 1)[0]
        for column in crsp_df.columns
        if column.endswith('_RET')
    })
    if not tickers:
        raise ValueError('No ticker return columns were found (expected *_RET columns).')

    feature_map: Dict[str, List[str]] = {}
    ticker_set = set(tickers)
    for column in crsp_df.columns:
        if '_' not in column:
            continue
        ticker, feature_name = column.split('_', 1)
        if ticker in ticker_set and feature_name:
            feature_map.setdefault(feature_name, []).append(column)

    if not feature_map:
        raise ValueError('No ticker feature columns could be parsed from CRSP predictors.')

    aggregated = {
        feature_name: crsp_df[feature_columns].mean(axis=1, skipna=True)
        for feature_name, feature_columns in sorted(feature_map.items())
    }
    aggregated_df = pd.DataFrame(aggregated, index=crsp_df.index)

    if 'RET' in aggregated_df.columns:
        cumulative_returns = aggregated_df['RET'].cumsum()
        aggregated_df['CUM_RET'] = cumulative_returns
        aggregated_df['LOG_CUM_RET'] = np.log1p(cumulative_returns.clip(lower=-0.999999))

    return aggregated_df


def create_lagged_features(feature_df: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    lagged_frames = []
    for lag in lags:
        lagged = feature_df.shift(lag)
        lagged = lagged.add_suffix(f'_LAG_{lag}')
        lagged_frames.append(lagged)
    return pd.concat(lagged_frames, axis=1)


def align_macro_features(
    macro_dir: Path,
    target_index: pd.DatetimeIndex,
    no_logic_macro_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_macro = load_macro_data(macro_dir)
    combiner = MacroCombiner(resample_freq='B')
    combined_macro = combiner.combine_macro_data(raw_macro)
    daily_macro = combiner.to_daily(combined_macro)

    aligned_macro = daily_macro.reindex(target_index).ffill().bfill()
    aligned_macro = aligned_macro.loc[:, ~aligned_macro.columns.duplicated(keep='first')]

    known_macro_ids = _flatten_fred_ids(FRED_SERIES)
    excluded_features = []
    drop_ids = set(no_logic_macro_ids or set())

    for column in list(aligned_macro.columns):
        if column in drop_ids:
            excluded_features.append((column, 'configured_no_logic'))
            aligned_macro = aligned_macro.drop(columns=[column])
            continue
        if column not in known_macro_ids:
            excluded_features.append((column, 'unknown_macro_series'))
            aligned_macro = aligned_macro.drop(columns=[column])
            continue
        if aligned_macro[column].isna().all():
            excluded_features.append((column, 'all_nan'))
            aligned_macro = aligned_macro.drop(columns=[column])
            continue
        if np.isclose(aligned_macro[column].var(skipna=True), 0.0):
            excluded_features.append((column, 'near_zero_variance'))
            aligned_macro = aligned_macro.drop(columns=[column])

    excluded_df = pd.DataFrame(excluded_features, columns=['feature', 'reason'])
    return aligned_macro, excluded_df


def _build_correlation_tables(
    non_lagged_df: pd.DataFrame,
    lagged_df: pd.DataFrame,
    low_corr_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    joined = pd.concat([non_lagged_df, lagged_df], axis=1)
    correlation_matrix = joined.corr(method='spearman').loc[non_lagged_df.columns, lagged_df.columns]

    long_corr = correlation_matrix.stack().rename('correlation').reset_index()
    long_corr = long_corr.rename(columns={
        'level_0': 'non_lagged_feature',
        'level_1': 'lagged_feature',
    })
    long_corr['abs_correlation'] = long_corr['correlation'].abs()

    positive = long_corr[long_corr['correlation'] > 0.0].sort_values(
        by='correlation',
        ascending=False,
    )
    negative = long_corr[long_corr['correlation'] < 0.0].sort_values(
        by='correlation',
        ascending=True,
    )
    low = long_corr[long_corr['abs_correlation'] <= low_corr_threshold].sort_values(
        by='abs_correlation',
        ascending=True,
    )
    corr_score = correlation_matrix.abs().mean(axis=0).sort_values(ascending=False)
    return correlation_matrix, positive, negative, low, corr_score


def _run_pca_feature_scoring(feature_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)

    pca = PCA()
    pca.fit(scaled_values)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)

    loadings = np.abs(pca.components_[:n_components_95, :])
    variance_weights = pca.explained_variance_ratio_[:n_components_95]
    weighted_loading_scores = loadings.T @ variance_weights

    feature_scores = pd.Series(
        weighted_loading_scores,
        index=feature_df.columns,
        name='pca_score',
    ).sort_values(ascending=False)

    explained_variance = pd.DataFrame({
        'component': np.arange(1, len(pca.explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_explained_variance': cumulative_variance,
    })
    return feature_scores, explained_variance


def _run_random_forest_importance(
    features_df: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.DataFrame, RandomForestRegressor, pd.DataFrame]:
    model_frame = pd.concat([features_df, target.rename('TARGET_RET')], axis=1).dropna()
    if len(model_frame) < 200:
        raise ValueError(
            f'Not enough rows for model-based feature selection after lagging and joins: {len(model_frame)}'
        )

    split_idx = int(len(model_frame) * 0.8)
    train_df = model_frame.iloc[:split_idx]
    test_df = model_frame.iloc[split_idx:]

    x_train = train_df.drop(columns=['TARGET_RET'])
    y_train = train_df['TARGET_RET']
    x_test = test_df.drop(columns=['TARGET_RET'])
    y_test = test_df['TARGET_RET']

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = pd.DataFrame([{
        'metric': 'r2',
        'value': r2_score(y_test, predictions),
    }, {
        'metric': 'mae',
        'value': mean_absolute_error(y_test, predictions),
    }])

    rf_importance = pd.Series(
        model.feature_importances_,
        index=x_train.columns,
        name='rf_importance',
    ).sort_values(ascending=False)

    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    permutation_scores = pd.Series(
        permutation.importances_mean,
        index=x_train.columns,
        name='permutation_importance',
    ).sort_values(ascending=False)

    shap_scores: pd.Series | None = None
    try:
        import shap  # type: ignore

        sample_size = min(len(x_test), 600)
        shap_sample = x_test.sample(n=sample_size, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)
        shap_values_array = np.array(shap_values)
        if shap_values_array.ndim == 3:
            shap_values_array = shap_values_array[0]
        shap_scores = pd.Series(
            np.abs(shap_values_array).mean(axis=0),
            index=x_train.columns,
            name='shap_importance',
        ).sort_values(ascending=False)
    except Exception:
        shap_scores = None

    return rf_importance, permutation_scores, shap_scores, metrics, model, x_train


def _build_ranking_comparison(method_scores: Dict[str, pd.Series]) -> pd.DataFrame:
    all_features = sorted(set().union(*[set(series.index) for series in method_scores.values()]))
    comparison = pd.DataFrame(index=all_features)

    rank_columns = []
    for method_name, scores in method_scores.items():
        score_col = f'{method_name}_SCORE'
        rank_col = f'{method_name}_RANK'
        comparison[score_col] = scores
        comparison[rank_col] = comparison[score_col].rank(method='min', ascending=False)
        rank_columns.append(rank_col)

    comparison['MEAN_RANK'] = comparison[rank_columns].mean(axis=1, skipna=True)
    comparison = comparison.sort_values(by='MEAN_RANK', ascending=True)
    return comparison


def _save_partial_dependence_plot(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    ranked_features: pd.Series,
    output_path: Path,
) -> None:
    top_features = ranked_features.head(6).index.tolist()
    if not top_features:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes_array = np.array(axes).reshape(-1)

    PartialDependenceDisplay.from_estimator(
        model,
        x_train,
        top_features,
        ax=axes_array[:len(top_features)],
    )
    for empty_ax in axes_array[len(top_features):]:
        empty_ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(
    artifacts: FeatureSelectionArtifacts,
    output_dir: Path,
    lags: Sequence[int],
    source_crsp_path: Path,
) -> None:
    summary_lines = [
        '# Feature Selection Summary',
        '',
        '## Inputs',
        f'- CRSP source: `{source_crsp_path}`',
        f'- Lags used: `{list(lags)}`',
        f'- Non-lagged CRSP features: `{artifacts.crsp_non_lagged.shape[1]}`',
        f'- Lagged CRSP features: `{artifacts.crsp_lagged.shape[1]}`',
        f'- Macro features used: `{artifacts.macro_aligned.shape[1]}`',
        f'- Model rows after lag/macro merge and NaN drop: `{artifacts.model_data.shape[0]}`',
        '',
        '## Macro Features Dropped',
        f'- Count dropped: `{len(artifacts.dropped_macro_features)}`',
    ]

    if not artifacts.dropped_macro_features.empty:
        top_drops = artifacts.dropped_macro_features.head(20)
        summary_lines.append('')
        summary_lines.append('| feature | reason |')
        summary_lines.append('|---|---|')
        for _, row in top_drops.iterrows():
            summary_lines.append(f"| `{row['feature']}` | `{row['reason']}` |")

    summary_lines.extend([
        '',
        '## Correlation Signals (Spearman)',
        '- Correlations are between non-lagged engineered CRSP features and lagged engineered CRSP features.',
        f"- Positive correlations found: `{len(artifacts.positive_correlations)}`",
        f"- Negative correlations found: `{len(artifacts.negative_correlations)}`",
        f"- Low correlations found: `{len(artifacts.low_correlations)}`",
        '',
        '### Top Positive Correlations',
        '| non_lagged_feature | lagged_feature | correlation |',
        '|---|---|---:|',
    ])
    for _, row in artifacts.positive_correlations.head(15).iterrows():
        summary_lines.append(
            f"| `{row['non_lagged_feature']}` | `{row['lagged_feature']}` | `{row['correlation']:.4f}` |"
        )

    summary_lines.extend([
        '',
        '### Top Negative Correlations',
        '| non_lagged_feature | lagged_feature | correlation |',
        '|---|---|---:|',
    ])
    for _, row in artifacts.negative_correlations.head(15).iterrows():
        summary_lines.append(
            f"| `{row['non_lagged_feature']}` | `{row['lagged_feature']}` | `{row['correlation']:.4f}` |"
        )

    summary_lines.extend([
        '',
        '### Lowest-Magnitude Correlations',
        '| non_lagged_feature | lagged_feature | abs_correlation |',
        '|---|---|---:|',
    ])
    for _, row in artifacts.low_correlations.head(15).iterrows():
        summary_lines.append(
            f"| `{row['non_lagged_feature']}` | `{row['lagged_feature']}` | `{row['abs_correlation']:.4f}` |"
        )

    summary_lines.extend([
        '',
        '## Model Metrics',
        '| metric | value |',
        '|---|---:|',
    ])
    for _, row in artifacts.model_metrics.iterrows():
        summary_lines.append(f"| `{row['metric']}` | `{row['value']:.6f}` |")

    summary_lines.extend([
        '',
        '## Top Features by Combined Ranking',
        '| feature | mean_rank |',
        '|---|---:|',
    ])
    for feature, row in artifacts.ranking_comparison.head(50).iterrows():
        summary_lines.append(f'| `{feature}` | `{row["MEAN_RANK"]:.2f}` |')

    summary_path = output_dir / 'feature_selection_summary.md'
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')


def run_feature_selection_pipeline(
    paths_config: Dict,
    output_dir: Path,
    lags: Sequence[int] = (10, 30, 50, 60),
    low_corr_threshold: float = 0.1,
    no_logic_macro_ids: Iterable[str] = DEFAULT_NO_LOGIC_MACRO_IDS,
) -> FeatureSelectionArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    crsp_raw_df, source_crsp_path = _load_crsp_predictors(paths_config)
    crsp_non_lagged = aggregate_ticker_features(crsp_raw_df)
    crsp_lagged = create_lagged_features(crsp_non_lagged, lags)

    macro_dir = Path(paths_config['data']['raw_macro_dir'])
    macro_aligned, dropped_macro_features = align_macro_features(
        macro_dir=macro_dir,
        target_index=crsp_non_lagged.index,
        no_logic_macro_ids=no_logic_macro_ids,
    )

    model_features = pd.concat([crsp_lagged, macro_aligned], axis=1)
    model_target = crsp_non_lagged['RET']
    model_data = pd.concat([model_features, model_target.rename('RET')], axis=1).dropna()

    correlation_matrix, positive_corr, negative_corr, low_corr, corr_score = _build_correlation_tables(
        non_lagged_df=crsp_non_lagged,
        lagged_df=crsp_lagged,
        low_corr_threshold=low_corr_threshold,
    )
    pca_scores, pca_variance = _run_pca_feature_scoring(model_data.drop(columns=['RET']))
    rf_scores, perm_scores, shap_scores, model_metrics, model, x_train = _run_random_forest_importance(
        features_df=model_data.drop(columns=['RET']),
        target=model_data['RET'],
    )

    method_scores: Dict[str, pd.Series] = {
        'CORR': corr_score,
        'PCA': pca_scores,
        'RF': rf_scores,
        'PERM': perm_scores,
    }
    if shap_scores is not None:
        method_scores['SHAP'] = shap_scores

    ranking_comparison = _build_ranking_comparison(method_scores)

    artifacts = FeatureSelectionArtifacts(
        crsp_non_lagged=crsp_non_lagged,
        crsp_lagged=crsp_lagged,
        macro_aligned=macro_aligned,
        model_data=model_data,
        correlation_matrix=correlation_matrix,
        positive_correlations=positive_corr,
        negative_correlations=negative_corr,
        low_correlations=low_corr,
        pca_feature_scores=pca_scores,
        pca_explained_variance=pca_variance,
        rf_importance=rf_scores,
        permutation_importance=perm_scores,
        shap_importance=shap_scores,
        ranking_comparison=ranking_comparison,
        model_metrics=model_metrics,
        dropped_macro_features=dropped_macro_features,
    )

    artifacts.crsp_non_lagged.to_csv(output_dir / 'engineered_non_lagged_crsp.csv')
    artifacts.crsp_lagged.to_csv(output_dir / 'engineered_lagged_crsp.csv')
    artifacts.macro_aligned.to_csv(output_dir / 'macro_aligned_filtered.csv')
    artifacts.model_data.to_csv(output_dir / 'modeling_dataset.csv')

    artifacts.correlation_matrix.to_csv(output_dir / 'correlation_matrix_spearman.csv')
    artifacts.positive_correlations.to_csv(output_dir / 'correlations_positive.csv', index=False)
    artifacts.negative_correlations.to_csv(output_dir / 'correlations_negative.csv', index=False)
    artifacts.low_correlations.to_csv(output_dir / 'correlations_low.csv', index=False)

    artifacts.pca_feature_scores.to_frame('pca_score').to_csv(output_dir / 'pca_feature_scores.csv')
    artifacts.pca_explained_variance.to_csv(output_dir / 'pca_explained_variance.csv', index=False)
    artifacts.rf_importance.to_frame('rf_importance').to_csv(output_dir / 'rf_feature_importance.csv')
    artifacts.permutation_importance.to_frame('permutation_importance').to_csv(
        output_dir / 'permutation_importance.csv'
    )
    if artifacts.shap_importance is not None:
        artifacts.shap_importance.to_frame('shap_importance').to_csv(output_dir / 'shap_importance.csv')

    artifacts.ranking_comparison.to_csv(output_dir / 'feature_ranking_comparison.csv')
    artifacts.ranking_comparison.head(50).to_csv(output_dir / 'top_50_features_comparison.csv')
    artifacts.model_metrics.to_csv(output_dir / 'model_metrics.csv', index=False)
    artifacts.dropped_macro_features.to_csv(output_dir / 'dropped_macro_features.csv', index=False)

    _save_partial_dependence_plot(
        model=model,
        x_train=x_train,
        ranked_features=artifacts.rf_importance,
        output_path=output_dir / 'partial_dependence_top_features.png',
    )
    _write_summary(
        artifacts=artifacts,
        output_dir=output_dir,
        lags=lags,
        source_crsp_path=source_crsp_path,
    )

    return artifacts
