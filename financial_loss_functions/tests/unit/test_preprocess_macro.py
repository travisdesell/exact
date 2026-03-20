import numpy as np
import pandas as pd
import pytest

from src.data_processing.preprocess_macro import (
    select_macro_for_pipeline,
    prepare_macro_splits,
)


def _make_synthetic_macro(n_days: int = 60) -> dict[str, pd.DataFrame]:
    """Build small synthetic macro category DataFrames for testing."""
    idx = pd.date_range('2020-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(42)

    rates = pd.DataFrame({
        'FEDFUNDS': np.cumsum(rng.normal(0, 0.01, n_days)),
        'TB3MS': np.cumsum(rng.normal(0, 0.02, n_days)),
    }, index=idx)

    prices = pd.DataFrame({
        'CPIAUCSL': np.linspace(250, 260, n_days) + rng.normal(0, 0.1, n_days),
        'PPICRM': np.linspace(180, 190, n_days) + rng.normal(0, 0.5, n_days),
    }, index=idx)

    return {'Rates_FX': rates, 'Prices': prices}


def _make_synthetic_returns(n_days: int = 60) -> pd.DataFrame:
    """Build small synthetic returns DataFrame with _RET columns."""
    idx = pd.date_range('2020-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(99)

    return pd.DataFrame({
        'AAPL_RET': rng.normal(0.001, 0.02, n_days),
        'MSFT_RET': rng.normal(0.0005, 0.015, n_days),
    }, index=idx)


class TestSelectMacroForPipeline:

    def test_returns_filtered_macro_and_column_list(self):
        raw_macro = _make_synthetic_macro()
        returns = _make_synthetic_returns()

        filtered, cols = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=returns,
            lags=[5, 10],
            top_k=2,
            low_corr_threshold=0.1,
        )

        assert isinstance(filtered, pd.DataFrame)
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert len(cols) <= 4
        assert set(filtered.columns) == set(cols)

    def test_top_k_limits_features_per_ticker(self):
        raw_macro = _make_synthetic_macro()
        returns = _make_synthetic_returns()

        _, cols_k1 = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=returns,
            lags=[5],
            top_k=1,
        )
        _, cols_k4 = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=returns,
            lags=[5],
            top_k=4,
        )
        assert len(cols_k1) <= len(cols_k4)

    def test_filtered_macro_aligned_to_returns_index(self):
        raw_macro = _make_synthetic_macro(80)
        returns = _make_synthetic_returns(60)

        filtered, _ = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=returns,
            lags=[5],
            top_k=2,
        )
        assert set(filtered.index) == set(returns.index)

    def test_handles_returns_without_ret_suffix(self):
        raw_macro = _make_synthetic_macro()
        idx = pd.date_range('2020-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({
            'AAPL': rng.normal(0, 0.02, 60),
            'MSFT': rng.normal(0, 0.02, 60),
        }, index=idx)

        filtered, cols = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=returns,
            lags=[5],
            top_k=2,
        )
        assert len(cols) > 0


class TestPrepareMacroSplits:

    def test_splits_have_correct_indices(self):
        raw_macro = _make_synthetic_macro(200)
        train_idx = pd.date_range('2020-01-01', periods=100, freq='B')
        val_idx = pd.date_range('2020-06-01', periods=50, freq='B')
        test_idx = pd.date_range('2020-09-01', periods=50, freq='B')

        m_train, m_val, m_test = prepare_macro_splits(
            raw_macro=raw_macro,
            train_index=train_idx,
            val_index=val_idx,
            test_index=test_idx,
            selected_cols=['FEDFUNDS', 'CPIAUCSL'],
        )

        assert list(m_train.index) == list(train_idx)
        assert list(m_val.index) == list(val_idx)
        assert list(m_test.index) == list(test_idx)

    def test_splits_contain_only_selected_columns(self):
        raw_macro = _make_synthetic_macro(200)
        train_idx = pd.date_range('2020-01-01', periods=100, freq='B')
        val_idx = pd.date_range('2020-06-01', periods=50, freq='B')
        test_idx = pd.date_range('2020-09-01', periods=50, freq='B')

        selected = ['FEDFUNDS']
        m_train, m_val, m_test = prepare_macro_splits(
            raw_macro=raw_macro,
            train_index=train_idx,
            val_index=val_idx,
            test_index=test_idx,
            selected_cols=selected,
        )

        assert list(m_train.columns) == selected
        assert list(m_val.columns) == selected
        assert list(m_test.columns) == selected

    def test_no_nans_after_alignment(self):
        raw_macro = _make_synthetic_macro(200)
        train_idx = pd.date_range('2020-01-01', periods=100, freq='B')
        val_idx = pd.date_range('2020-06-01', periods=50, freq='B')
        test_idx = pd.date_range('2020-09-01', periods=50, freq='B')

        m_train, m_val, m_test = prepare_macro_splits(
            raw_macro=raw_macro,
            train_index=train_idx,
            val_index=val_idx,
            test_index=test_idx,
            selected_cols=['FEDFUNDS', 'TB3MS'],
        )

        assert not m_train.isna().any().any()
        assert not m_val.isna().any().any()
        assert not m_test.isna().any().any()
