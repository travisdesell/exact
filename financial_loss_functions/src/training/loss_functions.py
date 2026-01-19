import math
import torch
from torch import Tensor

# -------------------- Sharpe -------------------- #
def raw_sharpe_loss(
        weights: Tensor, returns: Tensor, eps: float = 1e-8
    ):
    """
    Raw Sharpe ratio using standard deviation directly.

    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    # portfolio returns per step
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    mean_ret = port_ret.mean(dim=1)          # (B,)
    port_std = port_ret.std(dim=1) + eps      # (B,)
    sharpe = mean_ret / port_std              # (B,)
    # maximize Sharpe -> minimize negative Sharpe
    return -sharpe.mean()

def differentiable_sharpe_loss(
        weights: Tensor, returns: Tensor, eps: float = 1e-6
    ):
    """
    Differentiable Sharpe ratio where we calculate square root of variance 
    instead of standard deviation.
    
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    port_ret = (weights.unsqueeze(1) * returns).sum(-1)   # (B, T)
    mean_ret = port_ret.mean(dim=1)
    var  = port_ret.var(dim=1)          # variance, not std
    # Avoiding the std entirely
    return -(mean_ret / (var.sqrt() + eps)).mean()

def rms_sharpe_loss(weights: Tensor, returns: Tensor, eps: float = 1e-6):
    """
    Sharpe ratio where we use RMS instead of standard deviation.
    RMS is the population standard deviation.
    
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    port_ret = (weights.unsqeeze(1) * returns).sum(-1)
    mean_ret = port_ret.mean(dim=1)

    # RMS (population std)
    rms = torch.sqrt(
        torch.mean((port_ret - mean_ret.unsqueeze(1))**2, dim=1) + eps
    )

    sharpe = mean_ret / rms

    return -sharpe.mean()

# -------------------- Sortino -------------------- #
def raw_sortino_loss(
        weights: Tensor, returns: Tensor, target: float = 0.0, eps: float = 1e-8
    ):
    """
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns
    @param target float Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
    @param eps float Epsilon value to avoid divide by zero error

    @return batch average Sortino Ratio. 
        Negative since NN has to maximize Sortino but minimize loss
    """
    # Portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - port, min=0.0)  # (B, T_out), only positive for downside
    downside_std = downside.std(dim=1) + eps  # (B,)
    
    mean_return = port.mean(dim=1)  # (B,)
    sortino = mean_return / downside_std  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

def rms_sortino_loss(
        weights: Tensor, returns: Tensor, target: float = 0.0, eps: float = 1e-8
    ):
    """
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns
    @param target float Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
    @param eps float Epsilon value to avoid divide by zero error

    @return batch average Sortino Ratio. 
        Negative since NN has to maximize Sortino but minimize loss
    """
    # Portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - port, min=0.0)  # (B, T_out), only positive for downside
    
    # RMS downside: sqrt(mean(downside**2) + eps) -> for stable gradients
    downside_rms = torch.sqrt(torch.mean(downside ** 2, dim=1) + eps)  # (B,)
    
    mean_return = port.mean(dim=1)  # (B,)
    sortino = mean_return / downside_rms  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

# -------------------- Max Drawdown -------------------- #
def smooth_mdd_regularizer(
    weights: Tensor,
    returns: Tensor,
    temp: float = 50.0,
    eps: float = 1e-8,
    min_return: float = -0.999,   # to keep log1p safe
    use_percent: bool = True
) -> Tensor:
    """
    Differentiable smooth Max Drawdown regularizer (to MINIMIZE).

    Assumptions:
      - `weights` are already normalized (e.g., via softmax) and shaped (B, N).
      - `returns` are simple returns shaped (B, T, N) (e.g., 0.01 => +1%).
      - No internal normalization of `weights` is performed.

    @param weights Tensor[B, N] normalized allocation weights.
    @param returns Tensor[B, T, N] per-period simple returns.
    @param temp float temperature for the log-sum-exp smoothing. Higher -> closer to true max.
    @param eps float small constant for numerical stability.
    @param min_return float lower clamp for per-period returns (must be > -1).
    @param use_percent bool if True returns fractional drawdown in [0,1); else returns log-units.

    @return Tensor batch-mean smooth max drawdown
    """
    B, T, N = returns.shape
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)

    # clamp port to > -1 for log1p safety
    port_clamped = torch.clamp(port_ret, min=min_return)

    # cumulative log-wealth (log-returns summed)
    log_ret = torch.log1p(port_clamped)    # (B, T)
    cum_log = torch.cumsum(log_ret, dim=1) # (B, T)

    # running peak in log-space
    running_peak = torch.cummax(cum_log, dim=1).values  # (B, T)

    # drawdown in log-units (>=0)
    drawdown_log = running_peak - cum_log  # (B, T)

    # smooth max via log-sum-exp with bias correction (subtract log(T))
    #  s = temp * drawdown_log
    #  smooth_max_log = (logsumexp(s) - log(T)) / temp
    s = temp * drawdown_log
    lse = torch.logsumexp(s, dim=1)                     # (B,)
    smooth_max_log = (lse - math.log(T)) / (temp + eps) # (B,)

    # convert to percent-like drawdown if desired: dd = 1 - exp(-log_dd)
    if use_percent:
        mdd = 1.0 - torch.exp(-smooth_max_log)  # (B,) in [0,1)
    else:
        mdd = smooth_max_log  # (B,)

    return mdd.mean()