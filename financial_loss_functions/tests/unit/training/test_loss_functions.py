"""
Unit tests for financial loss functions.

Tests are registry-driven: every loss in LossLibrary gets shape, gradient, and
finite-output checks automatically. Targeted tests cover numerical-stability
fixes (near-`-1` returns through log-based losses, non-positive temp/beta) and
sign/monotonicity on a handful of representative functions.
"""
from __future__ import annotations

import inspect
import math
from typing import Callable

import pytest
import torch

from src.training.loss_functions import LossLibrary

torch.manual_seed(0)


# ───────────────────────── helpers ─────────────────────────

def _sample_batch(
    B: int = 4, T: int = 16, N: int = 5, seed: int = 123
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (logits, weights, all_returns, pf_returns). `logits` is the leaf."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, N, generator=g, requires_grad=True)
    weights = torch.softmax(logits, dim=-1)

    all_returns = 0.01 * torch.randn(B, T, N, generator=g)
    pf_returns = (weights.unsqueeze(1) * all_returns).sum(dim=-1)
    return logits, weights, all_returns, pf_returns


def _call_loss(fn: Callable, weights, all_returns, pf_returns) -> torch.Tensor:
    """Call a registered loss with whichever of the three tensors it accepts."""
    params = inspect.signature(fn).parameters
    kwargs: dict = {}
    if "weights" in params:
        kwargs["weights"] = weights
    # `all_returns` vs `returns` naming differs across functions.
    if "all_returns" in params:
        kwargs["all_returns"] = all_returns
    elif "returns" in params:
        kwargs["returns"] = all_returns
    # Most losses name the portfolio-return tensor `pf_returns`;
    # `log_return_objective` uses `port_returns`.
    if "pf_returns" in params:
        kwargs["pf_returns"] = pf_returns
    elif "port_returns" in params:
        kwargs["port_returns"] = pf_returns
    # Composites that require lambdas.
    for lam in (
        "lambda1", "lambda2", "cvar_lambda", "risk_p_lambda",
        "ent_lambda", "hhi_lambda", "log_ret_lambda",
    ):
        if lam in params and lam not in kwargs:
            kwargs[lam] = 0.1
    return fn(**kwargs)


def _collect_registered_losses() -> list[tuple[str, str, str, Callable]]:
    """Flatten LossLibrary registry into (category, subcategory, name, fn) tuples."""
    out = []
    for cat, subs in LossLibrary.items().items():
        for sub, fns in subs.items():
            for name, fn in fns.items():
                out.append((cat, sub, name, fn))
    return out


ALL_LOSSES = _collect_registered_losses()
ALL_LOSS_IDS = [f"{c}/{s}/{n}" for c, s, n, _ in ALL_LOSSES]


# ───────────────────────── registry-driven tests ─────────────────────────

@pytest.mark.parametrize(("cat", "sub", "name", "fn"), ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_loss_returns_scalar(cat, sub, name, fn):
    logits, weights, all_returns, pf_returns = _sample_batch()
    out = _call_loss(fn, weights, all_returns, pf_returns)
    assert isinstance(out, torch.Tensor), f"{name}: expected Tensor, got {type(out)}"
    assert out.dim() == 0, f"{name}: expected scalar, got shape {tuple(out.shape)}"


@pytest.mark.parametrize(("cat", "sub", "name", "fn"), ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_loss_is_finite(cat, sub, name, fn):
    logits, weights, all_returns, pf_returns = _sample_batch()
    out = _call_loss(fn, weights, all_returns, pf_returns)
    assert torch.isfinite(out).item(), f"{name}: non-finite output {out.item()}"


@pytest.mark.parametrize(("cat", "sub", "name", "fn"), ALL_LOSSES, ids=ALL_LOSS_IDS)
def test_loss_backward_produces_finite_grads(cat, sub, name, fn):
    logits, weights, all_returns, pf_returns = _sample_batch()
    out = _call_loss(fn, weights, all_returns, pf_returns)
    out.backward()
    # `logits` is the leaf upstream of weights/pf_returns.
    assert logits.grad is not None, f"{name}: no grad flowed to logits"
    assert torch.isfinite(logits.grad).all().item(), f"{name}: non-finite logit grads"


# ───────────────────────── numerical-stability regressions ─────────────────────────

# These correspond to the log(1+r) clamp fix (#2). With returns near -1 the old
# code produced log(≈0) and NaN/-inf; the new code must stay finite.

LOG_BASED_LOSSES = [
    ("log_return_objective", {}),
    ("log_sharpe_objective", {}),
    ("log_sortino_objective", {}),
]


@pytest.mark.parametrize(("loss_name", "extra_kwargs"), LOG_BASED_LOSSES)
def test_log_losses_handle_returns_near_negative_one(loss_name, extra_kwargs):
    """
    Regression for the log(1+r) clamp fix (#2). Before the fix, returns at or
    below `-1 - eps` produced log(<=0) → NaN. After the fix the output must be
    finite. (Gradient finiteness through var.sqrt() is not tested here; that's
    a separate Sharpe-variance issue unrelated to this bugfix.)
    """
    fn = LossLibrary.get("objectives", loss_name)
    B, T = 2, 8
    # Catastrophic window with tiny jitter so downstream var() > 0.
    base = torch.full((B, T), -0.999)
    jitter = 1e-5 * torch.randn(B, T)
    pf = (base + jitter).detach().requires_grad_(True)
    params = inspect.signature(fn).parameters
    kw = dict(extra_kwargs)
    kw["port_returns" if "port_returns" in params else "pf_returns"] = pf
    out = fn(**kw)
    assert torch.isfinite(out).item(), f"{loss_name}: not finite on -0.999 returns"


# These correspond to the temp/beta validation (#3). Bad configs must raise.

BAD_TEMP_BETA_CASES = [
    ("objectives",       "smooth_neglog_sharpe_loss",           {"beta": 0.0}),
    ("objectives",       "smooth_neglog_sortino_objective",     {"beta": 0.0}),
    ("objectives",       "smooth_omega_objective",              {"beta": -1.0}),
    ("objectives",       "log_sortino_objective",               {"beta": 0.0}),
    ("objectives",       "smooth_calmar_objective",             {"mdd_temp": 0.0}),
    ("regularizers",     "smooth_mdd_regularizer",              {"temp": 0.0}),
    ("regularizers",     "smooth_cvar_regularizer",             {"temp": 0.0}),
    ("regularizers",     "smooth_rockafellar_cvar_regularizer", {"temp": -0.1}),
]


@pytest.mark.parametrize(("cat", "name", "bad_kwargs"), BAD_TEMP_BETA_CASES)
def test_bad_temp_beta_raise(cat, name, bad_kwargs):
    # Tail-risk regularizers live under a subcategory.
    subcategory = "tail_risk" if name in (
        "smooth_mdd_regularizer",
        "smooth_cvar_regularizer",
        "smooth_rockafellar_cvar_regularizer",
    ) else None
    fn = LossLibrary.get(cat, name, subcategory=subcategory)
    _, _, _, pf_returns = _sample_batch()
    with pytest.raises(ValueError):
        fn(pf_returns=pf_returns, **bad_kwargs)


# ───────────────────────── targeted semantic checks ─────────────────────────

def test_log_return_monotonic_in_mean():
    """Higher mean returns → lower (more negative) loss."""
    fn = LossLibrary.get("objectives", "log_return_objective")
    low = fn(port_returns=torch.full((2, 10), 0.001))
    high = fn(port_returns=torch.full((2, 10), 0.01))
    assert high.item() < low.item(), "log_return_objective not decreasing in mean return"


def test_raw_sharpe_zero_variance_is_finite():
    """Constant non-zero returns produce zero variance; the eps floor keeps things finite."""
    fn = LossLibrary.get("objectives", "raw_sharpe_objective")
    pf_returns = torch.full((2, 8), 0.005)
    out = fn(pf_returns=pf_returns)
    assert torch.isfinite(out).item()


def test_hhi_uniform_weights_is_zero_when_scaled():
    fn = LossLibrary.get("regularizers", "hhi_regularizer", subcategory="structural")
    N = 10
    weights = torch.full((3, N), 1.0 / N)
    out = fn(weights=weights, scale_to_unit=True)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_hhi_concentrated_weights_is_one_when_scaled():
    fn = LossLibrary.get("regularizers", "hhi_regularizer", subcategory="structural")
    weights = torch.zeros(2, 5)
    weights[:, 0] = 1.0  # all in one asset
    out = fn(weights=weights, scale_to_unit=True)
    assert out.item() == pytest.approx(1.0, abs=1e-6)


def test_entropy_scaled_uniform_is_zero():
    fn = LossLibrary.get("regularizers", "entropy_conc_regularizer", subcategory="structural")
    N = 8
    weights = torch.full((4, N), 1.0 / N)
    out = fn(weights=weights, mode="scaled")
    assert out.item() == pytest.approx(0.0, abs=1e-5)


def test_cvar_topk_matches_empirical_mean():
    """CVaR via top-k should average the worst ceil(alpha*T) losses."""
    fn = LossLibrary.get("regularizers", "cvar_topk_regularizer", subcategory="tail_risk")
    pf_returns = torch.tensor([[0.01, -0.02, 0.03, -0.05, 0.00, 0.02, -0.01, 0.04]])
    alpha = 0.25  # worst 2 out of 8
    out = fn(pf_returns=pf_returns, alpha=alpha).item()
    expected = (0.05 + 0.02) / 2  # the two largest losses (= -returns)
    assert out == pytest.approx(expected, abs=1e-6)


def test_custom_loss_13_uses_multiplied_entropy_term():
    """
    Regression for the `+ → *` fix (#1). When ent_lambda = 0, the entropy term
    must drop out entirely. Two runs with different ent_lambda (0 vs non-zero)
    and identical other lambdas must therefore differ by exactly
    `(lam_b - lam_a) * entropy_term` — which only holds with multiplication.
    """
    fn = LossLibrary.get("custom", "custom_loss_13")
    logits, weights, all_returns, pf_returns = _sample_batch()
    base_kwargs = dict(
        weights=weights, all_returns=all_returns, pf_returns=pf_returns,
        cvar_lambda=0.1, risk_p_lambda=0.1,
    )
    a = fn(**base_kwargs, ent_lambda=0.0).item()
    b = fn(**base_kwargs, ent_lambda=0.5).item()

    # With multiplication the contribution is lam * entropy_term. With addition
    # it would have been just `lam` added on. So b - a ≠ 0.5 is the test.
    assert not math.isclose(b - a, 0.5, abs_tol=1e-6), (
        "custom_loss_13 behaves as if entropy term is ADDED with lambda "
        "(the original bug) instead of multiplied."
    )
