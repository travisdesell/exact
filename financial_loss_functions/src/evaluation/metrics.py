import numpy as np
from typing import Callable

Registry = dict[str, Callable]  # name -> fn

class MetricLibrary:
    """
    Central registry class for evaluation metrics.
    """
    _registry: Registry = {}

    @classmethod
    def register(
        cls,
        name: str | None = None
    ):
        """
        Decorator to register a standalone function into the class registry.

        Example:
            @LossCollectio.register('regularizers', 'diversification', 'herfindahl_index')
            def herfindahl_index(weights):
                return (weights**2).sum(dim=-1).mean()
        
        Args:
            name (str | None): Name of the loss function. Default = None. 
                If None, name of the function will be used as default.
        """
        def decorator(fn: Callable):
            nm = name or fn.__name__
            cls._registry[nm] = fn
            return fn
        return decorator

    # --- query helpers ---
    @classmethod
    def items(cls) -> Registry:
        """
        Get the entire nesed dictionary of the library of metrics functions.

        Returns:
            Registry: Dictionary of all loss terms and functions.
        """
        return cls._registry

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get a particular model that belongs to a subcategory (with or without subcategory).

        Args:
            category (str): Category that the loss function/term belongs to.
            name (str): Name of the required loss function/term.
            subcategory (str | None): Subcategory of the loss function/term.

        Returns:
            Callable: Callable loss function from the library.
        """
        return cls._registry[name]


@MetricLibrary.register()
def compunded_return(returns_arr: np.ndarray) -> np.float64:
    """Calculate cummulative returns for given window"""
    return np.prod(1 + returns_arr) - 1

@MetricLibrary.register()
def sharpe(
        returns_arr: np.ndarray, risk_free_rate: float = 0.0, annualized: bool = False
    ) -> np.float64:
    """
    Calculates Sharpe ratio for given window of returns.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        risk_free_rate (float): Risk free rate for window used for returns. Default = 0.0.
    
    Returns:
        sharpe (float): Sharpe ratio for the given portfolio.
    """
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr)
    sharpe = (mean_ret - risk_free_rate) / std_ret

    if annualized:
        sharpe = sharpe * np.sqrt(252)

    return sharpe

@MetricLibrary.register()
def sortino(returns_arr: np.ndarray, target: float = 0.0, annualized: bool = False) -> np.float64:
    """
    Calculates Sortino ratio (downside risk only).
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        target (float): Minimum acceptable return (MAR), often 0.
            Default = 0.0.
    """
    downside_returns = returns_arr[returns_arr < target]
    if len(downside_returns) == 0:
        return np.inf
    
    expected_return = np.mean(returns_arr)
    downside_std = np.std(downside_returns)

    sortino = (expected_return - target) / downside_std
    if annualized:
        sortino = sortino * np.sqrt(252)
    
    return sortino

@MetricLibrary.register()
def max_drawdown(returns_arr: np.ndarray) -> np.float64:
    """
    Calculates the maximum peak-to-trough decline, i.e., Max Drawdown.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
    
    Returns:
        float: Maximum peak-to-trough decline for the given portfolio
    """
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return np.min(drawdowns)

@MetricLibrary.register()
def cvar(returns_arr: np.ndarray, alpha: float = 0.05) -> np.float64:
    """Calculates the average loss in the worst alpha% of cases."""
    sorted_returns = np.sort(returns_arr)
    n_cutoff = int(alpha * len(sorted_returns))
    return np.mean(sorted_returns[:n_cutoff])

@MetricLibrary.register()
def omega(returns_arr: np.ndarray, threshold: float = 0.0) -> np.float64:
    """Calculates the ratio of weighted gains to weighted losses."""
    gains = np.sum(returns_arr[returns_arr > threshold] - threshold)
    losses = np.sum(threshold - returns_arr[returns_arr < threshold])
    return gains / losses if losses != 0 else np.inf

@MetricLibrary.register()
def calmar(returns_arr: np.ndarray) -> np.float64:
    """Ratio of annualized return to maximum drawdown."""
    mdd = abs(max_drawdown(returns_arr))

    return np.mean(returns_arr) / mdd if mdd != 0 else 0.0

@MetricLibrary.register()
def turnover(weights: np.ndarray) -> np.float64:
    """
    Average one-way L1 turnover across consecutive windows: mean_t( 0.5 * ||w_t - w_{t-1}||_1 ).

    Args:
        weights (np.ndarray): (num_windows, N) per-window allocation weights.

    Returns:
        float: Mean per-step turnover (0 = never trade, 1 = rotate the whole book each step).
            Returns 0.0 if fewer than two windows are provided.
    """
    if weights.ndim != 2:
        raise ValueError(f"turnover expects 2D (num_windows, N), got {weights.ndim}D.")
    if weights.shape[0] < 2:
        return np.float64(0.0)
    deltas = np.abs(np.diff(weights, axis=0)).sum(axis=1) * 0.5  # (num_windows-1,)
    return np.mean(deltas)


# ─────────────────────────── Bootstrap CI helpers ───────────────────────────

def _stationary_bootstrap_indices(
    n: int, block_mean_len: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Politis & Romano (1994) stationary bootstrap indices.

    Draws `n` indices by chaining geometric-length blocks (mean = block_mean_len)
    sampled from the original series. Preserves short-range autocorrelation
    better than i.i.d. resampling while staying stationary.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    if block_mean_len < 1.0:
        block_mean_len = 1.0
    p = 1.0 / block_mean_len  # per-step probability of starting a new block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    restart = rng.random(n) < p
    for t in range(1, n):
        if restart[t]:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def bootstrap_metric_ci(
    daily_returns: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    ci: float = 0.95,
    block_mean_len: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    Block-bootstrap confidence interval for a scalar metric on a daily-return series.

    Args:
        daily_returns (np.ndarray): (T,) series of daily returns.
        metric_fn (Callable): function mapping (T,)-array → scalar (e.g. sharpe).
        n_boot (int): number of bootstrap resamples. Default = 1000.
        ci (float): two-sided coverage in (0, 1). Default = 0.95.
        block_mean_len (float | None): mean block length for the stationary bootstrap.
            If None, defaults to sqrt(T) — a common rule of thumb for daily data.
        seed (int | None): RNG seed for reproducibility.

    Returns:
        dict with keys 'point' (metric on the full series), 'lower', 'upper'
        (empirical percentile CI), and 'p_value_gt_zero' (fraction of bootstrap
        estimates that are ≤ 0; a small value means "metric > 0" is well-supported).
    """
    daily_returns = np.asarray(daily_returns).flatten()
    T = daily_returns.shape[0]
    if T < 2:
        raise ValueError("bootstrap_metric_ci needs at least 2 observations.")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if block_mean_len is None:
        block_mean_len = float(np.sqrt(T))

    rng = np.random.default_rng(seed)
    point = float(metric_fn(daily_returns))

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block_mean_len, rng)
        boots[b] = float(metric_fn(daily_returns[idx]))

    # Replace non-finite bootstrap draws (e.g. sortino's inf when no downside) with NaN
    # so they drop out of the percentile computation cleanly.
    finite = np.isfinite(boots)
    if not finite.any():
        return {"point": point, "lower": np.nan, "upper": np.nan, "p_value_gt_zero": np.nan}
    draws = boots[finite]

    alpha = 1.0 - ci
    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    p_value = float(np.mean(draws <= 0.0))

    return {"point": point, "lower": lower, "upper": upper, "p_value_gt_zero": p_value}


def bootstrap_paired_diff_ci(
    daily_returns_a: np.ndarray,
    daily_returns_b: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    ci: float = 0.95,
    block_mean_len: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    Paired block-bootstrap CI for metric(a) - metric(b), using *the same* resampling
    indices for both series so the draws are matched in time.

    Useful for claims like "strategy A has a significantly higher Sharpe than B".
    """
    a = np.asarray(daily_returns_a).flatten()
    b = np.asarray(daily_returns_b).flatten()
    if a.shape != b.shape:
        raise ValueError(
            f"paired bootstrap requires equal-length series, got {a.shape} vs {b.shape}"
        )
    T = a.shape[0]
    if T < 2:
        raise ValueError("bootstrap_paired_diff_ci needs at least 2 observations.")
    if block_mean_len is None:
        block_mean_len = float(np.sqrt(T))

    rng = np.random.default_rng(seed)
    point = float(metric_fn(a) - metric_fn(b))

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block_mean_len, rng)
        boots[i] = float(metric_fn(a[idx]) - metric_fn(b[idx]))

    finite = np.isfinite(boots)
    if not finite.any():
        return {"point": point, "lower": np.nan, "upper": np.nan, "p_value_a_gt_b": np.nan}
    draws = boots[finite]

    alpha = 1.0 - ci
    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    # one-sided p-value for H1: metric(a) > metric(b)
    p_value = float(np.mean(draws <= 0.0))

    return {"point": point, "lower": lower, "upper": upper, "p_value_a_gt_b": p_value}