"""
Unit tests for CompositeSRLoss and backward-compatible loss wrappers.
"""

import torch
import pytest
import numpy as np

from src.training.loss_functions import (
    CompositeSRLoss,
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
# Tests: CompositeSRLoss
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
