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
        Decorator to register a standalone function portfolio performance metric 
        function into the class registry.

        Args:
            name (str | None): Name of the portfolio metric function. Default = None. 
                If None, name of the function will be used as default.
        """
        def decorator(fn: Callable):
            nm = name or fn.__name__
            # Prevent duplicate registration
            if nm in cls._registry:
                raise KeyError(f"Metric function '{nm}' already registered.")
            cls._registry[nm] = fn
            return fn
        return decorator

    # --- query helpers ---
    @classmethod
    def items(cls) -> Registry:
        """
        Get the entire registry (dictionary) of portfolio performance metrics.

        Returns:
            Registry: Dictionary of all portfolio performance metrics.
        """
        return cls._registry

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get a function for a portfolio performance metric.

        Args:
            name (str): Name of the required performance metric.

        Returns:
            Callable: Callable performance metric..
        """
        return cls._registry[name]


@MetricLibrary.register()
def compunded_return(returns_arr: np.ndarray) -> np.float64:
    """
    Calculate compunded (cumulative) returns for given window.

    Args:
        returns_arr (np.ndarray): Array of returns for a window/period.
    """
    return np.prod(1 + returns_arr) - 1

@MetricLibrary.register()
def sharpe(
        returns_arr: np.ndarray, 
        risk_free_rate: float = 0.0, 
        annualized: bool = False,
        days_per_year: int = 252
    ) -> np.float64:
    """
    Calculates Sharpe ratio for for a given portfolio's returns.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        risk_free_rate (float): Risk free rate for window used for returns. Default = 0.0.
        annualized (bool): Multiply the sharpe ratio by sqrt(252) to annualize the metric.
            Default = False.
        days_per_year (int): Number of trading days. Only used if annualized=True. Default = 252
    
    Returns:
        sharpe_value (np.float64): Sharpe ratio for the given portfolio.
    """
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr)
    sharpe_value = (mean_ret - risk_free_rate) / std_ret

    if annualized:
        sharpe_value = sharpe_value * np.sqrt(days_per_year)

    return sharpe_value

@MetricLibrary.register()
def sortino(
    returns_arr: np.ndarray, 
    target: float = 0.0, 
    annualized: bool = False,
    days_per_year: int = 252
) -> np.float64:
    """
    Calculates Sortino ratio (downside risk only) for a given portfolio's returns.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        target (float): Minimum acceptable return (MAR), often 0. Default = 0.0.
        annualized (bool): Multiply the sortino ratio by sqrt(252) to annualize the metric.
            Default = False.
        days_per_year (int): Number of trading days. Only used if annualized=True. Default = 252
    
    Returns:
        sortino_value (np.float64): Sharpe ratio for the given portfolio.
    """
    downside_returns = returns_arr[returns_arr < target]
    if len(downside_returns) == 0:
        return np.inf
    
    expected_return = np.mean(returns_arr)
    downside_std = np.std(downside_returns)

    sortino_value = (expected_return - target) / downside_std
    if annualized:
        sortino_value = sortino_value * np.sqrt(days_per_year)
    
    return sortino_value

@MetricLibrary.register()
def max_drawdown(returns_arr: np.ndarray) -> np.float64:
    """
    Calculates the maximum peak-to-trough decline, i.e., Max Drawdown.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
    
    Returns:
       mdd_value (np.float64): Maximum peak-to-trough decline for the given portfolio.
    """
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    mdd_value = np.min(drawdowns)
    return mdd_value

@MetricLibrary.register()
def cvar(returns_arr: np.ndarray, alpha: float = 0.05) -> np.float64:
    """
    Calculates the Conditional Value-at-Risk, i.e., average loss in the worst alpha % of cases.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        alpha (float): Alpha value as a float. Default = 0.05 (5%).
    
    Returns:
        cvar_value (np.float64): CVaR value for the given portfolio. 
            Most likely this value will always be negative. 
    """
    sorted_returns = np.sort(returns_arr)
    n_cutoff = int(alpha * len(sorted_returns))
    cvar_value = np.mean(sorted_returns[:n_cutoff])
    return cvar_value

@MetricLibrary.register()
def omega(returns_arr: np.ndarray, threshold: float = 0.0) -> np.float64:
    """
    Calculates the Omega Ratio, i.e., ratio of weighted gains to weighted losses.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        threshold (float): threshold value to decide the gains and losses.. Default = 0.0.
    
    Returns:
        omega_value (np.float64): Omega ratio for the given portfolio.
    """
    gains = np.sum(returns_arr[returns_arr > threshold] - threshold)
    losses = np.sum(threshold - returns_arr[returns_arr < threshold])
    omega_value = gains / losses if losses != 0 else np.inf
    return omega_value

@MetricLibrary.register()
def calmar(
    returns_arr: np.ndarray,
    annualized: bool = False,
    days_per_year: int = 252
) -> np.float64:
    """
    Calculates the Calmar Ratio, i.e., ratio of mean returns to maximum drawdown.
    
    Args:
        returns_arr (np.array): (n,) Array of daily returns for a particular portfolio.
        annualize (bool): Annualize the compunded return or use mean of returns. Default = False.
        days_per_year (int): Number of trading days. Only used if annualized=True. Default = 252
    
    Returns:
        calmar_value (np.float64): Calmar ratio for thr given portfolio.
    """
    mdd = abs(max_drawdown(returns_arr))
    if mdd != 0:
        if annualized:
            T = len(returns_arr)
            total_return = np.prod(1 + returns_arr) - 1
            annualized_return = (1 + total_return) ** (days_per_year / T) - 1
            calmar_value = annualized_return / mdd
        else:
            calmar_value = np.mean(returns_arr) / mdd
    
    else:
        calmar_value = 0.0
    
    return calmar_value