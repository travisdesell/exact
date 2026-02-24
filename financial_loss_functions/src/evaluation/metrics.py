import numpy as np

def cumulative_return(returns_arr: np.ndarray) -> np.float64:
    """Calculate cummulative returns for given window"""
    return np.prod(1 + returns_arr) - 1

def basic_sharpe(
        returns_arr: np.ndarray, risk_free_rate: float = 0.0, annualized: bool = False
    ) -> np.float64:
    """
    Calculates non-annualized sharpe for given window.
    
    @param returns_arr np.array (n,)
        array of discrete returns for each time step
    @param risk_free_rate float
        Risk free rate for window used for returns. Default = 0.0
    """
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr)
    sharpe = (mean_ret - risk_free_rate) / std_ret

    if annualized:
        sharpe = sharpe * np.sqrt(252)

    return sharpe