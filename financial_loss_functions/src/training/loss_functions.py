import math
import torch
from torch import Tensor
from typing import Tuple
from torch.nn.functional import softmax

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

# -------------------- CVaR -------------------- #
def cvar_topk_regularizer(
    weights: Tensor,
    returns: Tensor,
    alpha: float = 0.05
) -> Tensor:
    """
    Empirical CVaR (expected shortfall) over the worst alpha fraction of losses.
    Uses torch.topk to compute average of top-k losses.

    @param weights Tensor[B, N] (already softmax-normalized)
    @param returns Tensor[B, T, N] per-period simple returns.
    @param alpha tail fraction (0 < alpha <= 1)

    @return Tensor CVaR averaged across batch (minimize)
    """
    B, T, N = returns.shape
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T)
    
    # losses = -returns (higher = worse)
    losses = -port  # (B, T)

    # number of tail points to average (at least 1)
    k = max(1, math.ceil(alpha * T))

    # top-k largest losses per batch
    topk_vals, _ = torch.topk(losses, k, dim=1, largest=True, sorted=False)  # (B, k)
    cvar_per_batch = topk_vals.mean(dim=1)  # (B,)
    return cvar_per_batch.mean()  # scalar

def smooth_cvar_regularizer(
    weights: Tensor,
    returns: Tensor,
    alpha: float = 0.05,
    temp: float = 1e-2,
    eps: float = 1e-8,
    scale_by_std: bool = True
) -> Tensor:
    """
    Smooth differentiable approximation to CVaR using soft-selection (softmax) over losses.

    The idea: create scores = losses / (temp * std) (or / temp), take softmax across time,
    and compute a weighted average. Tuning `temp` controls concentration on the tail.
    The final value is scaled to approximate the expected loss in the worst alpha fraction.

    @param weights Tensor[B, N], already normalized (softmax)
    @param returns Tensor[B, T, N]
    @param alpha float tail fraction, e.g., 0.05
    @param temp float small positive temperature -> smaller temp => more concentrated on worst losses
    @param scale_by_std float whether to standardize losses per batch for numeric stability

    @return Tensor smooth CVaR approx (minimize)
    """
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T)
    losses = -port  # (B, T)

    if scale_by_std:
        std = losses.std(dim=1, keepdim=True) + eps
        scores = losses / (std * (temp + eps))
    else:
        scores = losses / (temp + eps)

    # soft selection: weights over time (sum to 1)
    sel = softmax(scores, dim=1)  # (B, T)

    # approximate CVaR: mean of worst-alpha fraction. If sel concentrates on worst alpha*T
    # then weighted_mean ~ mean(worst). To expose the alpha scaling, we multiply by (1/alpha).
    # This is an approximation - tune temp so selection mass ~ alpha.
    weighted_mean = (sel * losses).sum(dim=1)  # (B,)
    approx_cvar = weighted_mean / max(alpha, eps)  # scale up to match order-of-magnitude w/ CVaR

    return approx_cvar.mean()

# -------------------- Risk Parity -------------------- #
def sample_covariance(returns: Tensor, unbiased: bool = True):
    """
    @param returns Tensor (B, T, N)
    @return sample covariance per batch -> (B, N, N)
    """
    B, T, N = returns.shape
    mean = returns.mean(dim=1, keepdim=True)  # (B, 1, N)
    X = returns - mean  # (B, T, N)
    # cov = X^T X / (T-1) if unbiased else / T
    denom = (T - 1) if unbiased and T > 1 else T
    cov = X.transpose(1, 2).bmm(X) / float(max(denom, 1))
    return cov

def shrinkage_covariance_torch(cov: Tensor, shrink: float = 0.1):
    """
    Linear shrinkage toward scaled identity:
      cov_shrunk = (1 - shrink) * cov + shrink * (trace(cov)/N) * I
    @param Tensor cov (B, N, N)
    """
    B, N, _ = cov.shape
    # trace per batch (B,1)
    trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)  # (B,1)
    scale = trace / float(N)                                       # (B,1)
    I = torch.eye(N, device=cov.device, dtype=cov.dtype).unsqueeze(0)  # (1,N,N)
    # broadcast scale to (B,1,1)
    scale = scale.view(B, 1, 1)
    return (1.0 - shrink) * cov + shrink * scale * I

def risk_parity_regularizer(
    weights: Tensor,
    returns: Tensor,
    shrink: float = 0.1,
    use_shrink: bool = True,
    shrink_clip: Tuple = (0.0, 0.9),
    eps: float = 1e-8,
    scale_invariant: bool = True
) -> Tensor:
    """
    Differentiable Risk-Parity regularizer with optional linear shrinkage.

    @param weights Tensor (B, N)
    @param returns Tensor (B, T, N)
    @param shrink float in [0,1] shrinkage intensity
    @param use_shrink bool whether to apply shrinkage
    @param shrink_clip Tuple allowed range for shrink (safety)
    @param eps float numerical eps
    @param scale_invariant bool if True divide squared-deviation by 
        sigma2^2 to be scale-invariant

    @return Risk contribution Tensor (mean across batch)
    """
    B, T, N = returns.shape

    cov = sample_covariance(returns, unbiased=True)  # (B, N, N)

    # Optional linear shrinkage (applied per-batch)
    if use_shrink:
        # safety: clip shrink to reasonable range
        shrink = float(shrink)
        low, high = float(shrink_clip[0]), float(shrink_clip[1])
        shrink = max(low, min(high, shrink))
        cov = shrinkage_covariance_torch(cov, shrink=shrink)

    # Portfolio variance and marginal contributions
    w_col = weights.unsqueeze(2)  # (B, N, 1)
    sigma2 = (w_col.transpose(1,2).bmm(cov).bmm(w_col)).squeeze(-1).squeeze(-1)  # (B,)
    # guard against zero variance
    sigma2 = sigma2.clamp(min=eps)

    mcontrib = cov.bmm(w_col).squeeze(-1)  # (B, N)
    rc = weights * mcontrib                # (B, N)

    # target equal contribution
    target = sigma2.unsqueeze(1) / float(N)  # (B, 1)

    # squared deviations summed per batch
    loss_per_batch = ((rc - target)**2).sum(dim=1)  # (B,)

    # scale-invariant normalization
    if scale_invariant:
        scaled = loss_per_batch / (sigma2**2 + eps)
    else:
        scaled = loss_per_batch

    return scaled.mean()