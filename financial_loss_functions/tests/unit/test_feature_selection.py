from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_selection.analysis import (
    align_macro_to_business_days,
    compute_ticker_feature_rankings,
)
from src.feature_selection.rules import (
    BASE_GROUP_WEIGHT,
    build_sector_mapping_df,
    get_sector_prior_weights,
    load_sector_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_sector_mapping_complete_for_50_tickers():
    mapping_path = PROJECT_ROOT / 'config' / 'sector_classification.json'
    mapping_json = load_sector_mapping(mapping_path)
    sector_df = build_sector_mapping_df(mapping_json)

    returns_raw = pd.read_csv(
        PROJECT_ROOT
        / 'data'
        / 'raw'
        / '2023_sp_500_select_50'
        / 'combined_returns_raw.csv',
        nrows=0,
    )
    expected_tickers = sorted(
        col.replace('_RET', '') for col in returns_raw.columns if col.endswith('_RET')
    )

    assert len(expected_tickers) == 50
    assert sorted(mapping_json.keys()) == expected_tickers
    assert len(sector_df) == 50
    assert not sector_df['sector'].isna().any()


def test_sector_prior_weights_for_known_tickers():
    hban_weights = get_sector_prior_weights('HBAN')
    assert hban_weights['Rates_FX'] == 1.0
    assert hban_weights['Money_Credit'] == 1.0
    assert hban_weights['Labor_Market'] < 1.0

    akam_weights = get_sector_prior_weights('AKAM')
    assert akam_weights['Labor_Market'] == 1.0
    assert akam_weights['Prices'] == 1.0
    assert akam_weights['Stock_Market'] == 1.0

    nvr_weights = get_sector_prior_weights('NVR')
    assert nvr_weights['Consumption_Orders_Inventories'] == 1.0
    assert nvr_weights['Rates_FX'] > BASE_GROUP_WEIGHT
    assert nvr_weights['Rates_FX'] < 1.0


def test_macro_alignment_business_day_ffill():
    raw_macro = {
        'Rates_FX': pd.DataFrame(
            {'FEDFUNDS': [1.0, 2.0]},
            index=pd.to_datetime(['2020-01-01', '2020-01-03']),
        )
    }
    target_index = pd.DatetimeIndex(
        pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    )

    aligned = align_macro_to_business_days(raw_macro, target_index)

    assert list(aligned.index) == list(target_index)
    assert aligned.loc[pd.Timestamp('2020-01-02'), 'FEDFUNDS'] == 1.0
    assert aligned.loc[pd.Timestamp('2020-01-03'), 'FEDFUNDS'] == 2.0


def test_correlation_ranking_uses_abs_spearman_with_lags():
    idx = pd.date_range('2022-01-01', periods=8, freq='D')
    target = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx)
    macro = pd.DataFrame(
        {
            'macro_neg': [0, -1, -2, -3, -4, -5, -6, -7],
            'macro_pos': [0, 1, 1, 2, 2, 3, 3, 4],
        },
        index=idx,
    )
    feature_to_group = {'macro_neg': 'Rates_FX', 'macro_pos': 'Rates_FX'}
    prior_weights = {'Rates_FX': 0.0}

    ranking = compute_ticker_feature_rankings(
        ticker='HBAN',
        target_returns=target,
        macro_aligned=macro,
        feature_to_group=feature_to_group,
        lags=[1, 2],
        low_corr_threshold=0.1,
        group_prior_weights=prior_weights,
    )
    row = ranking[ranking['macro_feature'] == 'macro_neg'].iloc[0]

    assert row['best_lag'] == 1
    assert row['max_abs_spearman_corr'] == 1.0


def test_composite_scoring_applies_prior_bonus():
    idx = pd.date_range('2022-01-01', periods=8, freq='D')
    target = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx)
    macro = pd.DataFrame(
        {
            'favored_macro': [0, 1, 2, 3, 4, 5, 6, 7],
            'base_macro': [0, 1, 2, 3, 4, 5, 6, 7],
        },
        index=idx,
    )
    feature_to_group = {
        'favored_macro': 'Rates_FX',
        'base_macro': 'Stock_Market',
    }
    prior_weights = {
        'Rates_FX': 1.0,
        'Stock_Market': BASE_GROUP_WEIGHT,
    }

    ranking = compute_ticker_feature_rankings(
        ticker='HBAN',
        target_returns=target,
        macro_aligned=macro,
        feature_to_group=feature_to_group,
        lags=[1],
        low_corr_threshold=0.1,
        group_prior_weights=prior_weights,
    )

    favored = ranking[ranking['macro_feature'] == 'favored_macro'].iloc[0]
    base = ranking[ranking['macro_feature'] == 'base_macro'].iloc[0]

    assert favored['composite_score'] > base['composite_score']
    assert favored['rank'] < base['rank']


def test_top_k_selection_is_deterministic():
    idx = pd.date_range('2022-01-01', periods=12, freq='D')
    target = pd.Series([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8], index=idx)
    macro = pd.DataFrame(
        {
            'macro_a': [8, 5, 3, 5, 6, 2, 9, 5, 1, 4, 1, 3],
            'macro_b': [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5],
            'macro_c': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        },
        index=idx,
    )
    feature_to_group = {
        'macro_a': 'Labor_Market',
        'macro_b': 'Prices',
        'macro_c': 'Rates_FX',
    }
    prior_weights = {
        'Labor_Market': 1.0,
        'Prices': 1.0,
        'Rates_FX': 1.0,
    }

    ranking_1 = compute_ticker_feature_rankings(
        ticker='AKAM',
        target_returns=target,
        macro_aligned=macro,
        feature_to_group=feature_to_group,
        lags=[1, 2, 3],
        low_corr_threshold=0.1,
        group_prior_weights=prior_weights,
    )
    ranking_2 = compute_ticker_feature_rankings(
        ticker='AKAM',
        target_returns=target,
        macro_aligned=macro,
        feature_to_group=feature_to_group,
        lags=[1, 2, 3],
        low_corr_threshold=0.1,
        group_prior_weights=prior_weights,
    )

    assert_frame_equal(ranking_1, ranking_2)
