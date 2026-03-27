"""
Unit tests for pyfolio data conversion utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.pyfolio_viz import (
    weights_to_pyfolio,
    build_window_dates,
    comparison_summary,
)


class TestWeightsToPyfolio:

    def test_returns_series_has_correct_dates(self):
        W, T_OUT, N = 2, 5, 3
        tickers = ["A", "B", "C"]
        weights = np.full((W, N), 1.0 / N)
        returns = np.random.default_rng(0).normal(0, 0.01, (W, T_OUT, N))

        all_dates = pd.bdate_range("2020-01-01", periods=20)
        window_dates = [all_dates[0:5], all_dates[5:10]]
        bench = pd.Series(0.001, index=all_dates)

        result = weights_to_pyfolio(weights, returns, tickers, window_dates, bench)

        assert isinstance(result["returns"], pd.Series)
        assert len(result["returns"]) == T_OUT * W
        assert isinstance(result["positions"], pd.DataFrame)
        assert "cash" in result["positions"].columns

    def test_positions_columns_match_tickers(self):
        W, T_OUT, N = 1, 3, 2
        tickers = ["X", "Y"]
        weights = np.array([[0.6, 0.4]])
        returns = np.random.default_rng(1).normal(0, 0.01, (W, T_OUT, N))

        dates = pd.bdate_range("2021-06-01", periods=10)
        window_dates = [dates[0:3]]
        bench = pd.Series(0.0, index=dates)

        result = weights_to_pyfolio(weights, returns, tickers, window_dates, bench)

        pos_cols = set(result["positions"].columns)
        assert {"X", "Y", "cash"}.issubset(pos_cols)


class TestBuildWindowDates:

    def test_correct_number_of_windows(self):
        full_idx = pd.bdate_range("2020-01-01", periods=100)
        starts = np.array([0, 10, 20])
        in_size, out_size = 30, 10

        wd = build_window_dates(full_idx, starts, in_size, out_size)

        assert len(wd) == 3
        for dates in wd:
            assert len(dates) == out_size


class TestComparisonSummary:

    def test_summary_columns(self):
        dates = pd.bdate_range("2020-01-01", periods=50)
        rets = pd.Series(np.random.default_rng(5).normal(0.001, 0.02, 50), index=dates)

        strategies = {
            "A": {"returns": rets, "benchmark_rets": rets * 0},
        }

        summary = comparison_summary(strategies)

        assert "total_return" in summary.columns
        assert "sharpe_ratio" in summary.columns
        assert "max_drawdown" in summary.columns
        assert summary.index[0] == "A"
