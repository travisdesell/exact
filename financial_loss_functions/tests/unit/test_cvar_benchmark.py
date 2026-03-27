"""
Unit tests for the CVaR benchmark optimizer.
"""

import pytest
import numpy as np


class TestCVaRBenchmark:

    def _make_benchmark(self, **kwargs):
        from src.models.cvar_benchmark import CVaRBenchmark, CVaRParams
        params = CVaRParams(**kwargs)
        return CVaRBenchmark(params=params)

    def test_optimize_returns_valid_weights(self):
        bench = self._make_benchmark()
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(100, 5))

        weights = bench.optimize(returns)

        assert weights.shape == (5,)
        assert np.all(weights >= -1e-6), "Weights should be non-negative"
        assert abs(weights.sum() - 1.0) < 1e-4, "Weights should sum to 1"

    def test_equal_weight_fallback_on_degenerate_input(self):
        bench = self._make_benchmark(w_max=1.0)
        returns = np.zeros((10, 5))

        weights = bench.optimize(returns)

        assert weights.shape == (5,)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_rolling_optimize_shape(self):
        bench = self._make_benchmark()
        rng = np.random.default_rng(99)
        X_returns = rng.normal(0.0, 0.01, size=(3, 50, 4))

        result = bench.rolling_optimize(X_returns)

        assert result.shape == (3, 4)
        for i in range(3):
            assert abs(result[i].sum() - 1.0) < 1e-4

    def test_respects_w_max_constraint(self):
        bench = self._make_benchmark(w_max=0.25)
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.02, size=(200, 8))

        weights = bench.optimize(returns)

        assert np.all(weights <= 0.25 + 1e-4)
