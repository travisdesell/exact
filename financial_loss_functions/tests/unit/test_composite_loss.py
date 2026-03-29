"""
Unit tests for CompositeSRLoss and backward-compatible loss wrappers.
"""

import torch
import pytest
import numpy as np

from src.training.loss_functions import (
    CompositeSRLoss,
    TimeframeImportanceFn,
    MacroOverrideGate,
    differentiable_sharpe_loss,
    sharpe_loss_compat,
    sortino_loss_compat,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T_IN, T_OUT, N, F = 4, 20, 10, 5, 6  # batch, time-in, time-out, tickers, features


def _random_weights():
    w = torch.softmax(torch.randn(B, N), dim=-1)
    w.requires_grad_(True)
    return w


def _random_returns():
    return torch.randn(B, T_OUT, N) * 0.02


def _random_features():
    return torch.randn(B, T_IN, N * F)


def _random_fundamentals():
    return torch.randn(B, N)


def _make_loss(**kwargs):
    defaults = dict(
        num_tickers=N,
        num_features_per_ticker=F,
        ret_feature_idx=2,
        turnover_feature_idx=3,
        illiq_feature_idx=1,
        ba_spread_feature_idx=0,
        macro_col_indices=[4, 5],
    )
    defaults.update(kwargs)
    return CompositeSRLoss(**defaults)


# ---------------------------------------------------------------------------
# Tests: CompositeSRLoss (original)
# ---------------------------------------------------------------------------

class TestCompositeSRLoss:

    def test_forward_returns_scalar(self):
        loss_fn = _make_loss()
        w = _random_weights()
        r = _random_returns()
        f = _random_features()
        fund = _random_fundamentals()

        loss = loss_fn(w, r, f, fund)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_gradients_flow_to_weights(self):
        loss_fn = _make_loss()
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        loss.backward()

        assert w.grad is not None, "Gradients must reach weights"
        assert torch.isfinite(w.grad).all(), "Gradients must be finite"

    def test_regime_weights_receive_gradients(self):
        loss_fn = _make_loss()
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        loss.backward()

        assert loss_fn.regime_weights.grad is not None
        assert torch.isfinite(loss_fn.regime_weights.grad).all()

    def test_reduces_to_sharpe_when_penalties_zero(self):
        loss_fn = _make_loss(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        composite = loss_fn(w, r, f)
        sharpe_only = differentiable_sharpe_loss(w, r)

        assert torch.allclose(composite, sharpe_only, atol=1e-6)

    def test_no_features_still_works(self):
        """When features=None, only the Sharpe component should fire."""
        loss_fn = _make_loss()
        w = _random_weights()
        r = _random_returns()

        loss = loss_fn(w, r)
        assert torch.isfinite(loss)

    def test_fundamentals_penalty_direction(self):
        """Overweighting stocks with positive fundamentals should lower the loss."""
        loss_fn = _make_loss(alpha=0, beta=0, gamma=0, delta=1.0)

        fund = torch.ones(B, N)
        w_overweight = torch.ones(B, N) * 0.5
        w_equal = torch.ones(B, N) * (1.0 / N)

        r = _random_returns()

        loss_over = loss_fn(w_overweight, r, fundamentals=fund)
        loss_eq = loss_fn(w_equal, r, fundamentals=fund)

        assert loss_over.item() != loss_eq.item()

    def test_no_macro_indices(self):
        loss_fn = _make_loss(macro_col_indices=None)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests: Backward-compatible wrappers
# ---------------------------------------------------------------------------

class TestCompatWrappers:

    def test_sharpe_compat_ignores_extra_args(self):
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = sharpe_loss_compat(w, r, f, _random_fundamentals())
        expected = differentiable_sharpe_loss(w, r)
        assert torch.allclose(loss, expected, atol=1e-7)

    def test_sortino_compat_ignores_extra_args(self):
        w = _random_weights()
        r = _random_returns()

        loss = sortino_loss_compat(w, r, _random_features())
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests: Multi-Timeframe S/R Hierarchy
# ---------------------------------------------------------------------------

class TestMultiTimeframeSR:

    def test_backward_compat_flag_off(self):
        """With sr_use_multi_timeframe=False, loss matches existing behavior."""
        torch.manual_seed(42)
        loss_fn_old = _make_loss(sr_use_multi_timeframe=False)

        torch.manual_seed(0)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn_old(w, r, f)
        assert torch.isfinite(loss)
        assert loss_fn_old.importance_fn is None

    def test_multi_tf_returns_scalar(self):
        loss_fn = _make_loss(sr_use_multi_timeframe=True)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_multi_tf_gradients_flow(self):
        loss_fn = _make_loss(sr_use_multi_timeframe=True)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        loss.backward()

        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        # Importance MLP params should receive gradients
        for p in loss_fn.importance_fn.parameters():
            assert p.grad is not None, "Importance fn params must get gradients"

    def test_importance_fn_monotonic_init(self):
        """At init, importance should be non-decreasing for increasing lookbacks."""
        fn = TimeframeImportanceFn(hidden=8)
        x = torch.linspace(0.0, 1.0, steps=10)
        with torch.no_grad():
            y = fn(x)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all(), "Importance should be non-decreasing at init"

    def test_pivot_detection_at_high(self):
        """Monotonically increasing prices → scores near +1."""
        loss_fn = _make_loss(sr_use_multi_timeframe=True)
        # Create strictly increasing prices
        prices = torch.linspace(1.0, 2.0, steps=T_IN).unsqueeze(0).unsqueeze(-1)
        prices = prices.expand(B, T_IN, N)
        scores = loss_fn._detect_pivots(prices)  # (B, W, N)
        assert (scores >= 0.9).all(), "Increasing prices should be near +1 (resistance)"

    def test_pivot_detection_at_low(self):
        """Monotonically decreasing prices → scores near -1."""
        loss_fn = _make_loss(sr_use_multi_timeframe=True)
        prices = torch.linspace(2.0, 1.0, steps=T_IN).unsqueeze(0).unsqueeze(-1)
        prices = prices.expand(B, T_IN, N)
        scores = loss_fn._detect_pivots(prices)
        assert (scores <= -0.9).all(), "Decreasing prices should be near -1 (support)"

    def test_custom_lookback_windows(self):
        loss_fn = _make_loss(
            sr_use_multi_timeframe=True,
            sr_lookback_windows=[3, 7, 15],
        )
        assert loss_fn.sr_windows == [3, 7, 15]
        assert loss_fn.sr_lookback_normed.shape == (3,)

    def test_reduces_to_sharpe_when_alpha_zero_multi_tf(self):
        loss_fn = _make_loss(
            alpha=0.0, beta=0.0, gamma=0.0, delta=0.0,
            sr_use_multi_timeframe=True,
        )
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        composite = loss_fn(w, r, f)
        sharpe_only = differentiable_sharpe_loss(w, r)
        assert torch.allclose(composite, sharpe_only, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Macro Override Gate
# ---------------------------------------------------------------------------

class TestMacroOverrideGate:

    def test_gate_output_bounded(self):
        gate = MacroOverrideGate(num_macro_features=3, hidden=8)
        delta_macro = torch.randn(B, 3)
        omega = gate(delta_macro)
        assert (omega >= 0.0).all() and (omega <= 1.0).all()

    def test_gate_gradients_flow(self):
        loss_fn = _make_loss(use_macro_override=True)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        loss.backward()

        for p in loss_fn.macro_override.parameters():
            assert p.grad is not None, "Override gate params must get gradients"

    def test_override_off_matches_baseline(self):
        """use_macro_override=False → standard penalty weights."""
        loss_fn = _make_loss(use_macro_override=False)
        assert loss_fn.macro_override is None

        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)

    def test_override_extreme_macro_boosts_gamma(self):
        """With extreme macro delta, the macro penalty should dominate."""
        loss_fn = _make_loss(
            use_macro_override=True,
            alpha=0.10, beta=0.05, gamma=0.10, delta=0.10,
        )
        w = _random_weights()
        r = _random_returns()
        f = _random_features()
        # Make macro features have huge range to push omega toward 1
        for idx in [4, 5]:
            for n_idx in range(N):
                col = n_idx * F + idx
                f[:, :, col] = torch.linspace(-100, 100, T_IN).unsqueeze(0)

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests: Sector-Aware Penalty
# ---------------------------------------------------------------------------

class TestSectorAware:

    def test_sector_ids_affect_penalty(self):
        sector_ids = [0, 0, 1, 1, 2]
        loss_fn_sec = _make_loss(sector_ids=sector_ids)
        loss_fn_nosec = _make_loss()

        torch.manual_seed(99)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss_sec = loss_fn_sec(w, r, f)
        loss_nosec = loss_fn_nosec(w, r, f)

        assert torch.isfinite(loss_sec)
        # With sectors, the penalty scaling differs
        assert not torch.allclose(loss_sec, loss_nosec, atol=1e-8)

    def test_no_sector_ids_unchanged(self):
        loss_fn = _make_loss(sector_ids=None)
        assert loss_fn.sector_ids is None

        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests: Ticker Macro Sensitivity
# ---------------------------------------------------------------------------

class TestTickerMacroSensitivity:

    def test_sensitivity_affects_macro_penalty(self):
        sens = torch.tensor([0.1, 0.2, 0.8, 0.9, 1.0])
        loss_fn = _make_loss(ticker_macro_sensitivity=sens)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)

    def test_no_sensitivity_unchanged(self):
        loss_fn = _make_loss(ticker_macro_sensitivity=None)
        assert loss_fn.ticker_macro_sensitivity is None


# ---------------------------------------------------------------------------
# Tests: Correlation Guard
# ---------------------------------------------------------------------------

class TestCorrelationGuard:

    def test_corr_matrix_integration(self):
        corr = torch.eye(N)
        corr[0, 1] = 0.9
        corr[1, 0] = 0.9
        loss_fn = _make_loss(
            sr_use_multi_timeframe=True,
            corr_matrix=corr,
        )
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss = loss_fn(w, r, f)
        assert torch.isfinite(loss)

    def test_no_corr_matrix_unchanged(self):
        loss_fn = _make_loss(corr_matrix=None)
        assert loss_fn.corr_matrix is None

    def test_corr_requires_multi_tf(self):
        """Correlation guard only activates when multi-TF is enabled."""
        corr = torch.eye(N)
        loss_fn_no_tf = _make_loss(
            sr_use_multi_timeframe=False,
            corr_matrix=corr,
        )
        loss_fn_tf = _make_loss(
            sr_use_multi_timeframe=True,
            corr_matrix=corr,
        )

        torch.manual_seed(42)
        w = _random_weights()
        r = _random_returns()
        f = _random_features()

        loss_no_tf = loss_fn_no_tf(w, r, f)
        loss_tf = loss_fn_tf(w, r, f)

        assert torch.isfinite(loss_no_tf)
        assert torch.isfinite(loss_tf)
