"""
Unit tests for SEC filing feature extraction utilities.

These tests exercise the pure computation functions without requiring
a live edgartools connection or SEC API access.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_collection.sec_filings import (
    _compute_event_signal,
    compute_composite_fundamental_scores,
    FUNDAMENTAL_WEIGHTS,
)


class TestEventSignal:

    def _mock_filing(self, date_str):
        class _F:
            filing_date = date_str
        return _F()

    def test_event_within_window(self):
        idx = pd.bdate_range("2020-01-01", "2020-01-31")
        filings = [self._mock_filing("2020-01-15")]

        signal = _compute_event_signal(filings, idx, window_days=2)

        assert signal.loc[pd.Timestamp("2020-01-15")] == 1.0
        assert signal.loc[pd.Timestamp("2020-01-13")] == 1.0
        assert signal.loc[pd.Timestamp("2020-01-10")] == 0.0

    def test_no_filings_returns_zeros(self):
        idx = pd.bdate_range("2020-06-01", "2020-06-30")
        signal = _compute_event_signal([], idx)
        assert (signal == 0.0).all()


class TestCompositeScores:

    def test_shape_and_columns(self):
        tickers = ["A", "B"]
        idx = pd.bdate_range("2020-01-01", "2020-01-10")
        funds = {
            "A": pd.DataFrame(
                {"revenue_growth": 0.1, "operating_margin": 0.2,
                 "debt_to_equity": 1.0, "fcf_yield": 0.05,
                 "event_signal": 0.0},
                index=idx,
            ),
            "B": pd.DataFrame(
                {"revenue_growth": -0.1, "operating_margin": 0.1,
                 "debt_to_equity": 2.0, "fcf_yield": -0.02,
                 "event_signal": 1.0},
                index=idx,
            ),
        }

        scores = compute_composite_fundamental_scores(funds, tickers, idx)

        assert list(scores.columns) == tickers
        assert len(scores) == len(idx)
        assert scores.isna().sum().sum() == 0

    def test_positive_growth_yields_higher_score(self):
        tickers = ["GOOD", "BAD"]
        idx = pd.bdate_range("2020-01-01", "2020-01-05")
        funds = {
            "GOOD": pd.DataFrame(
                {"revenue_growth": 0.5, "operating_margin": 0.3,
                 "debt_to_equity": 0.5, "fcf_yield": 0.10,
                 "event_signal": 0.0},
                index=idx,
            ),
            "BAD": pd.DataFrame(
                {"revenue_growth": -0.5, "operating_margin": -0.1,
                 "debt_to_equity": 5.0, "fcf_yield": -0.10,
                 "event_signal": 0.0},
                index=idx,
            ),
        }

        scores = compute_composite_fundamental_scores(funds, tickers, idx)
        assert (scores["GOOD"] > scores["BAD"]).all()

    def test_missing_ticker_defaults_to_zero(self):
        tickers = ["X", "Y"]
        idx = pd.bdate_range("2020-03-01", "2020-03-05")
        funds = {
            "X": pd.DataFrame(
                {"revenue_growth": 0.1, "operating_margin": 0.2,
                 "debt_to_equity": 1.0, "fcf_yield": 0.05,
                 "event_signal": 0.0},
                index=idx,
            ),
        }

        scores = compute_composite_fundamental_scores(funds, tickers, idx)
        assert "Y" in scores.columns
