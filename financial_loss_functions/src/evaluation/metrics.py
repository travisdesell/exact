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
def compounded_return(returns_arr: np.ndarray) -> np.float64:
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