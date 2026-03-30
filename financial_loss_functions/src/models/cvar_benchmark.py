"""
Lightweight Mean-CVaR portfolio optimizer for benchmarking.

Uses CVXPY with the Rockafellar-Uryasev linearization, inspired by
the formulation in ``quantitative-portfolio-optimization`` (NVIDIA cufolio)
but without any cuOpt / GPU dependency.

The optimizer solves:

    minimise   lambda * CVaR_alpha  -  E[w^T r]
    s.t.       sum(w) + cash = 1
               w >= w_min,  w <= w_max
               cash >= 0
               ||w||_1 <= L_tar

where CVaR is expressed via auxiliary variables (t, u):

    CVaR = t + (1 / ((1 - alpha) * S)) * sum(u_s)
    u_s >= 0
    u_s >= -(R_s^T w + t)          for each scenario s
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CVaRParams:
    """Tunable parameters for CVaR benchmark optimization."""

    confidence: float = 0.95
    risk_aversion: float = 1.0
    w_min: float = 0.0
    w_max: float = 0.30
    L_tar: float = 1.6


class CVaRBenchmark:
    """
    Mean-CVaR portfolio optimizer using CVXPY (CPU-only).

    @param params CVaRParams  Optimization parameters.
    """

    def __init__(self, params: Optional[CVaRParams] = None):
        self.params = params or CVaRParams()
        self._cp = None

    def _ensure_cvxpy(self):
        if self._cp is None:
            try:
                import cvxpy as cp
            except ImportError as exc:
                raise ImportError(
                    "cvxpy is required for the CVaR benchmark. "
                    "Install with: pip install cvxpy"
                ) from exc
            self._cp = cp
        return self._cp

    def optimize(self, returns: np.ndarray) -> np.ndarray:
        """
        Solve Mean-CVaR optimization given historical return scenarios.

        @param returns np.ndarray (S, N) -- S scenarios, N assets.
            Each row is one historical daily return vector.

        @return np.ndarray (N,) optimal portfolio weights.
            Falls back to equal-weight if the solver fails.
        """
        cp = self._ensure_cvxpy()

        S, N = returns.shape
        alpha = self.params.confidence
        lam = self.params.risk_aversion

        mu = returns.mean(axis=0)

        w = cp.Variable(N, name="w")
        c = cp.Variable(1, name="cash")
        t = cp.Variable(1, name="VaR")
        u = cp.Variable(S, name="u")

        scenario_ptf_returns = returns @ w  # (S,)

        cvar_risk = t + (1.0 / ((1 - alpha) * S)) * cp.sum(u)
        expected_return = mu @ w

        constraints = [
            u >= 0,
            u + t + scenario_ptf_returns >= 0,
            cp.sum(w) + c == 1,
            w >= self.params.w_min,
            w <= self.params.w_max,
            c >= 0,
            cp.norm1(w) <= self.params.L_tar,
        ]

        objective = cp.Minimize(lam * cvar_risk - expected_return)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except Exception as exc:
                logger.warning("CVaR solve failed: %s – returning equal weights", exc)
                return np.full(N, 1.0 / N)

        if w.value is None:
            logger.warning("CVaR solver returned None – returning equal weights")
            return np.full(N, 1.0 / N)

        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, None)
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.full(N, 1.0 / N)

        # Re-clip to w_max after normalization and redistribute excess
        w_max = self.params.w_max
        for _ in range(10):
            excess_mask = weights > w_max
            if not excess_mask.any():
                break
            excess = (weights[excess_mask] - w_max).sum()
            weights[excess_mask] = w_max
            free_mask = ~excess_mask & (weights < w_max)
            if free_mask.any():
                weights[free_mask] += excess / free_mask.sum()
            else:
                break

        return weights

    def rolling_optimize(
        self,
        X_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Run CVaR optimization on each window's historical returns.

        @param X_returns np.ndarray (num_windows, T_in, N)
            Per-window historical return data (extracted from the input
            windows used by the LSTM).

        @return np.ndarray (num_windows, N) benchmark weights.
        """
        num_windows, T_in, N = X_returns.shape
        all_weights = np.zeros((num_windows, N))

        for i in range(num_windows):
            scenarios = X_returns[i]  # (T_in, N)
            all_weights[i] = self.optimize(scenarios)

        return all_weights
