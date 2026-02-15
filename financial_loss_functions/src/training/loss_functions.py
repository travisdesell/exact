import math
import torch
from torch import Tensor
from typing import Callable
from torch.nn.functional import softmax, softplus

#### All Functions MUST get a decorator with the category and/or sub-category.
#### Objectives do not need a subcategory (as of now). Regularizer sub-categories are required.
#### High level keys are [objectives, regularizers, custom]

Registry = dict[str, dict[str, dict[str, Callable]]]  # category -> subcategory -> name -> fn

class LossLibrary:
    """
    Central registry class for loss functions.

    Usage pattern:
      1) Define this class first (in a module).
      2) Define functions in the same module (or other modules), and decorate them:
           @LossCollection.register("regularizers", "diversification", "herfindahl")
           def herfindahl(weights): ...
      3) Use LossCollection.get(...) or LossCollection.items() to retrieve.
    """
    _registry: Registry = {}

    @classmethod
    def register(
        cls,
        category: str = 'objectives',
        subcategory: str | None = None,
        name: str | None = None
    ):
        """
        Decorator to register a standalone function into the class registry.

        Example:
            @LossCollection.register("regularizers", "diversification", "herfindahl_index")
            def herfindahl_index(weights):
                return (weights**2).sum(dim=-1).mean()
        """
        def decorator(fn: Callable):
            cat = category
            sub = subcategory or '__default__'
            nm = name or fn.__name__
            cls._registry.setdefault(cat, {}).setdefault(sub, {})[nm] = fn
            return fn
        return decorator

    # --- query helpers ---
    @classmethod
    def items(cls) -> Registry:
        """Return the nested registry (live view)."""
        return cls._registry

    @classmethod
    def list_categories(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_subcategories(cls, category: str) -> list[str]:
        return list(cls._registry.get(category, {}).keys())

    @classmethod
    def list_functions(cls, category: str, subcategory: str|None = None) -> list[str]:
        sub = subcategory or '__default__'
        return list(cls._registry.get(category, {}).get(sub, {}).keys())

    @classmethod
    def get(cls, category: str,  name: str, subcategory: str|None = None) -> Callable:
        sub = subcategory or '__default__'
        return cls._registry[category][sub][name]

# -------------------- Sharpe -------------------- #
@LossLibrary.register(category='objectives')
def raw_sharpe_objective(
        weights: Tensor, returns: Tensor, eps: float = 1e-8
    ) -> Tensor:
    """
    Raw Sharpe ratio using standard deviation directly.

    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    # portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    mean_ret = port.mean(dim=1)                          # (B,)
    # population std: set unbiased=False
    port_std = port.std(dim=1, unbiased=False) + eps     # (B,)
    sharpe = mean_ret / port_std                         # (B,)
    # maximize Sharpe -> minimize negative Sharpe
    return -sharpe.mean()

@LossLibrary.register(category='objectives')
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

@LossLibrary.register(category='objectives')
def rms_sharpe_objective(weights: Tensor, returns: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Sharpe ratio where we use RMS instead of standard deviation.
    RMS is the population standard deviation.
    
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T)
    mean_ret = port.mean(dim=1, keepdim=True)            # (B,1)
    rms = torch.sqrt(torch.mean((port - mean_ret)**2, dim=1) + eps)  # (B,)
    sharpe = mean_ret.squeeze(1) / rms
    return -sharpe.mean()

@LossLibrary.register(category='objectives')
def smooth_neglog_sharpe_loss(
    weights: Tensor,
    returns: Tensor,
    eps: float = 1e-8,
    unbiased: bool = False,
    beta: float = 1.0,
) -> Tensor:
    """
    Smooth, always-differentiable Sharpe loss.
    Uses softplus to map Sharpe -> positive before log.

    Minimizing this maximizes Sharpe.
    """
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T)
    mean_ret = port.mean(dim=1)

    var = port.var(dim=1, unbiased=unbiased)
    std = torch.sqrt(var + eps)
    sharpe = mean_ret / (std + eps)

    # smooth positive mapping (always > 0)
    sharpe_pos = softplus(sharpe, beta=beta)

    loss = torch.log(sharpe_pos + eps)
    return -loss.mean()

# -------------------- Sortino -------------------- #
@LossLibrary.register(category='objectives')
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

@LossLibrary.register(category='objectives')
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

@LossLibrary.register(category='objectives')
def smooth_neglog_sortino_objective(
    weights: Tensor,
    returns: Tensor,
    target: float = 0.0,
    use_soft_downside: bool = True,
    beta: float = 10.0,                # sharpness for softplus; larger -> closer to clamp
    eps: float = 1e-8
) -> Tensor:
    """
    Returns a loss to MINIMIZE. Minimizing this increases Sortino.

    transform:
      - "neglog": loss = -log( softpos(sortino) + eps )  (recommended)
      - "neg":    loss = -sortino
      - "raw":    returns sortino.mean()  (rare; treat as reward)
    """
    # prepare weights and portfolio
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B,T)

    # downside: smooth or hard
    if use_soft_downside:
        # softplus approximates clamp(target - port, min=0)
        # we feed (target - port) so positive means downside
        downside = softplus(target - port, beta=beta)
    else:
        downside = torch.clamp(target - port, min=0.0)

    # RMS downside (population)
    downside_rms = torch.sqrt(torch.mean(downside**2, dim=1) + eps)  # (B,)
    mean_ret = port.mean(dim=1)  # (B,)
    sortino = mean_ret / (downside_rms + eps)  # (B,)

    s_pos = softplus(sortino)  # > 0
    return -torch.log(s_pos + eps).mean()

# -------------------- Max Drawdown -------------------- #
@LossLibrary.register(category='regularizers', subcategory='tail_risk')
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
@LossLibrary.register(category='regularizers', subcategory='tail_risk')
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

@LossLibrary.register(category='regularizers', subcategory='tail_risk')
def smooth_cvar_regularizer(
    weights: Tensor,
    returns: Tensor,
    temp: float = 1e-2,
    eps: float = 1e-8,
    scale_by_std: bool = True,
    normalize_by_port_std: bool = True,
    port_std_floor: float = 1e-3
) -> torch.Tensor:
    """
    Smooth differentiable approximation to CVaR using soft-selection (softmax) over losses.
    It uses Soft-Max/Minimax (Soft Attention Mechanism) for a CVaR-like regularizer.
    It tells the model: 
    "I don't care about anything else, just make sure our absolute worst day isn't a catastrophe."
    It minimizes the single worst-case scenario.

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
    # port: (B, T)
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)
    losses = -port  # (B, T)

    if scale_by_std:
        scores_std = losses.std(dim=1, keepdim=True) + eps   # (B,1)
        scores = losses / (scores_std * (temp + eps))
    else:
        scores = losses / (temp + eps)

    sel = softmax(scores, dim=1)           # (B, T) sums to 1
    weighted_mean = (sel * losses).sum(dim=1)  # (B,) -- already an average-like quantity

    approx_cvar = weighted_mean  # NOT dividing by alpha here

    if normalize_by_port_std:
        port_std = port.std(dim=1)           # (B,)
        port_std = torch.clamp(port_std, min=port_std_floor)
        approx_cvar = approx_cvar / (port_std + eps)

    return approx_cvar.mean()

@LossLibrary.register(category='regularizers', subcategory='tail_risk')
def smooth_rockafellar_cvar_regularizer(
    weights: torch.Tensor,
    returns: torch.Tensor,
    alpha: float = 0.05,
    temp: float = 1e-2, # In R&U, temp controls the Softplus "smoothness"
    eps: float = 1e-8,
    normalize_by_port_std: bool = True,
    port_std_floor: float = 1e-3
) -> torch.Tensor:
    """
    Differentiable CVaR using the Rockafellar & Uryasev formula. 
    Uses alpha to get average of 5% worst case scenarios.
    
    @param weights: Tensor[B, N] - Portfolio weights
    @param returns: Tensor[B, T, N] - Asset returns
    @param alpha: float - The tail probability (e.g., 0.05 for 95% CVaR)
    @param temp: float - Smoothness of the ReLU approximation (Softplus)

    @return Tensor smooth CVaR approx (minimize)
    """
    # 1. Calculate Portfolio Returns and Losses
    # port: (B, T), losses: (B, T)
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)
    losses = -port 

    # 2. Estimate VaR (zeta) for the batch
    # We take the (1-alpha) quantile of the losses as our starting point for zeta
    # This is the "threshold" where the tail begins.
    with torch.no_grad():
        zeta = torch.quantile(losses, 1 - alpha, dim=1, keepdim=True) # (B, 1)

    # 3. Rockafellar & Uryasev Formula
    # Instead of max(0, losses - zeta), we use Softplus for a smooth gradient.
    # soft_excess = temp * log(1 + exp((losses - zeta) / temp))
    excess_losses = (losses - zeta)
    soft_excess = softplus(excess_losses, beta=1/temp)
    
    # CVaR = zeta + (1 / alpha) * Average(excess_losses)
    # (B,)
    approx_cvar = zeta.squeeze(1) + (1.0 / alpha) * soft_excess.mean(dim=1)

    # 4. Normalization (The "Tail Ratio" approach)
    if normalize_by_port_std:
        port_std = port.std(dim=1)
        port_std = torch.clamp(port_std, min=port_std_floor)
        # Final value is dimensionless: how many STDs is the average tail loss?
        approx_cvar = approx_cvar / (port_std + eps)

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

@LossLibrary.register(category='regularizers', subcategory='structural')
def risk_parity_regularizer(
    weights: Tensor,
    returns: Tensor,
    shrink: float = 0.1,
    use_shrink: bool = True,
    shrink_clip: tuple = (0.0, 0.9),
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

# -------------------- Omega Ratio -------------------- #
@LossLibrary.register(category='objectives')
def raw_omega_ratio(
    weights: Tensor,
    returns: Tensor,
    theta: float = 0.0,
    eps: float = 1e-8
) -> Tensor:
    """
    Exact empirical Omega ratio (batch mean).
    
    @param weights Tensor(B, N)
    @param returns Tensor (B, T, N) simple returns
    @param theta float threshold (same units as returns)
    
    @return Tensor scalar (batch mean Omega to minimize)
    """

    port = (weights.unsqueeze(1) * returns).sum(dim=-1)   # (B, T)
    # positives = (R - theta)_+, negatives = (theta - R)_+
    pos = torch.clamp(port - theta, min=0.0)        # (B, T)
    neg = torch.clamp(theta - port, min=0.0)        # (B, T)

    # expectations are simple means over time
    pos_mean = pos.mean(dim=1)   # (B,)
    neg_mean = neg.mean(dim=1)   # (B,)

    # avoid divide-by-zero
    omega_per_batch = pos_mean / (neg_mean + eps)  # (B,)

    return -omega_per_batch.mean()

@LossLibrary.register(category='objectives')
def smooth_omega_objective(
    weights: Tensor,
    returns: Tensor,
    theta: float = 0.0,
    beta: float = 10.0,
    eps: float = 1e-8,
    use_log_loss: bool = True,
    cap_omega: float | None = None
) -> Tensor:
    """
    Smooth Omega objective (LOSS TO MINIMIZE).
    Minimizing this is equivalent to maximizing Omega.

    Can be used as:
      - primary objective: loss = smooth_omega_objective(...)
      - regularizer: loss += lambda * smooth_omega_objective(...)

    Args:
    @param weights Tensor(B, N) allocation weights
    @param returns Tensor(B, T, N)
    @param theta float Omega threshold (per-period)
    @param beta float softplus sharpness (>0)
    @param eps float numerical stability
    @param cap_omega bool optional cap to limit extreme ratios

    @return Tensor scalar (batch mean Omega to minimize)
    """

    # portfolio returns per timestep
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T)

    # smoothed positive / negative parts
    pos = softplus(port - theta, beta=beta)      # (R - theta)_+
    neg = softplus(theta - port, beta=beta)      # (theta - R)_+

    # expectations over time
    pos_mean = pos.mean(dim=1)
    neg_mean = neg.mean(dim=1)

    # omega per batch
    omega = pos_mean / (neg_mean + eps)

    if cap_omega is not None:
        omega = torch.clamp(omega, max=float(cap_omega))

    # loss per batch
    if use_log_loss:
        # canonical loss: -log(Omega)
        loss_per_batch = torch.log(omega + eps)
    else:
        loss_per_batch = omega

    return -loss_per_batch.mean()

# -------------------- Herfindahl–Hirschman Index (HHI) -------------------- #
@LossLibrary.register(category='regularizers', subcategory='structural')
def hhi_regularizer(
    weights: Tensor,
    scale_to_unit: bool = True,
    eps: float = 1e-12
) -> Tensor:
    """
    HHI concentration penalty (batch-mean).
    - weights: (B, N) model allocations (logits or already normalized).
    - scale_to_unit: if True, scales HHI to [0,1] using (HHI - 1/N) / (1 - 1/N).
                     This makes penalty interpretable and easier to combine with other losses.
    - returns: scalar Tensor (mean over batch) representing HHI penalty to add to loss.

    Example usage:
      weights = torch.softmax(logits, dim=-1)  # or pass normalize_weights=True
      loss = main_loss + lambda_hhi * hhi_penalty(weights)
    """

    # squared weights and HHI per sample
    hhi = (weights * weights).sum(dim=1)  # (B,)

    if scale_to_unit:
        N = weights.shape[-1]
        min_hhi = 1.0 / float(N)  # equal-weight HHI
        max_hhi = 1.0
        # scaled in [0,1]
        hhi_scaled = (hhi - min_hhi) / (max_hhi - min_hhi + eps)
        # numerical safety clamp
        hhi_scaled = torch.clamp(hhi_scaled, min=0.0, max=1.0)
        return hhi_scaled.mean()
    else:
        return hhi.mean()

@LossLibrary.register(category='regularizers', subcategory='structural')
def hhi_signed_regularizer(
    weights: Tensor,
    *,
    normalize_by_gross: bool = False,
    scale_to_unit: bool = True,
    eps: float = 1e-12
) -> Tensor:
    """
    For signed weights. Measures concentration on absolute exposure.
    weights: (B, N) can have negative entries (shorting).
    """
    w_abs = weights.abs()  # (B, N)
    if normalize_by_gross:
        gross = w_abs.sum(dim=1, keepdim=True) + eps  # (B,1)
        w_norm = w_abs / gross  # treat relative exposures
    else:
        w_norm = w_abs

    hhi = (w_norm * w_norm).sum(dim=1)  # (B,)
    if scale_to_unit:
        N = w_norm.shape[-1]
        min_hhi = 1.0 / float(N)
        hhi_scaled = (hhi - min_hhi) / (1.0 - min_hhi + eps)
        return torch.clamp(hhi_scaled, 0.0, 1.0).mean()
    else:
        return hhi.mean()

# -------------------- Portfolio entropy (Shannon entropy) -------------------- #
@LossLibrary.register(category='regularizers', subcategory='structural')
def entropy_conc_regularizer(
    weights: Tensor,
    signed: bool = False,
    mode: str = 'scaled',
    eps: float = 1e-12,
) -> Tensor:
    """
    Entropy concentration penalty (no clustering). Returns a scalar loss to MINIMIZE.

    Args:
      weights: Tensor[B, N] -- either normalized allocations (simplex) or logits if normalize_weights=True
      normalize_weights: if True, apply softmax(weights) to get simplex weights inside the function
      signed: if True, convert weights -> abs(weights) and renormalize to gross exposure = 1
      mode: one of {"neg_entropy", "scaled", "kl"}:
        - "neg_entropy": return -H(w)  (minimize -> maximize entropy)
        - "scaled": return 1 - H(w)/log(N)  (in [0,1], 0 = uniform)
        - "kl": return log(N) - H(w)  (KL(uniform || w), >=0)
      eps: small constant to avoid log(0)

    Returns:
      scalar Tensor: batch-mean penalty (minimize).
    """
    if mode not in {'neg_entropy', 'scaled', 'kl'}:
        raise ValueError("mode must be one of {'neg_entropy','scaled','kl'}")

    if signed:
        # Use absolute exposure and renormalize to simplex
        weights = weights.abs()
        denom = weights.sum(dim=1, keepdim=True) + eps
        weights = weights / denom

    B, N = weights.shape

    # clamp for numeric stability then compute entropy H = -sum w log w
    w_safe = weights.clamp(min=eps)
    entropy = -(w_safe * torch.log(w_safe)).sum(dim=1)  # (B,)

    if mode == 'neg_entropy':
        penalty = -entropy                              # minimize -> maximize entropy
    elif mode == 'scaled':
        max_ent = float(torch.log(torch.tensor(float(N), device=weights.device)))
        penalty = 1.0 - (entropy / (max_ent + eps))
        penalty = penalty.clamp(min=0.0, max=1.0)
    else:  # mode == "kl"
        max_ent = float(torch.log(torch.tensor(float(N), device=weights.device)))
        penalty = max_ent - entropy                     # KL(uniform || w)

    return penalty.mean()

# -------------------- Calmar Ratio -------------------- #
@LossLibrary.register(category='objectives')
def raw_calmar_objective(
    weights: Tensor,
    returns: Tensor,
    theta: float = 0.0,
    apply_theta_to_return: bool = False,
    apply_theta_to_drawdown: bool = False,
    min_return: float = -0.999,
    eps: float = 1e-8
) -> Tensor:
    """
    Raw Calmar ratio computed on the provided window (no annualization).
    Returns the batch-mean Calmar (higher is better).

    Args:
      weights: (B, N) allocations (if logits, set normalize_weights=True)
      returns: (B, T, N) simple per-period returns (e.g., daily)
      theta: per-period MAR (optional); same units as returns
      apply_theta_to_return: if True, numerator uses (port - theta)
      apply_theta_to_drawdown: if True, drawdown path uses (port - theta) (uncommon)
      normalize_weights: if True, apply softmax(weights) inside
      min_return: lower clamp for each period (must be > -1)
      eps: small constant to avoid division by zero

    Returns:
      scalar Tensor: mean(Calmar_i) across batch
    """

    # portfolio returns per period (B, T)
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)

    # optionally apply theta to numerator and/or drawdown path
    port_for_return = port - theta if apply_theta_to_return else port
    port_for_dd     = port - theta if apply_theta_to_drawdown else port

    # clamp for log1p safety (for cumulative wealth / drawdown)
    port_for_return_clamped = torch.clamp(port_for_return, min=min_return)
    port_for_dd_clamped     = torch.clamp(port_for_dd,     min=min_return)

    # numerator = mean simple return over the window (B,)
    numerator = port_for_return_clamped.mean(dim=1)

    # compute empirical max drawdown (exact) from cumulative wealth path
    #  cumulative log-wealth path -> cum_log -> running peak -> drawdown
    log_ret_dd = torch.log1p(port_for_dd_clamped)   # (B, T)
    cum_log = torch.cumsum(log_ret_dd, dim=1)      # (B, T)
    running_peak = torch.cummax(cum_log, dim=1).values  # (B, T)
    drawdown_log = running_peak - cum_log          # (B, T), >= 0
    drawdown_frac = 1.0 - torch.exp(-drawdown_log) # (B, T), in [0,1)
    max_dd, _ = drawdown_frac.max(dim=1)           # (B,)

    # Calmar per-sample (guard denom)
    calmar_per_sample = numerator / (max_dd + eps)  # (B,)

    # return batch-mean Calmar (no sign flip; higher is better)
    return calmar_per_sample.mean()

@LossLibrary.register(category='objectives')
def smooth_calmar_objective(
    weights: Tensor,
    returns: Tensor,
    mdd_temp: float = 50.0,
    theta: float = 0.0,
    apply_theta_to_return: bool = False,
    apply_theta_to_drawdown: bool = False,
    eps: float = 1e-8,
    use_log_loss: bool = True,
    min_return: float = -0.999
) -> Tensor:
    """
    Smooth Calmar computed directly on the model output horizon (no annualization).
    Minimizing this loss -> maximizes Calmar on the window.

    Args:
      weights: (B, N) allocations (if logits, set normalize_weights=True)
      returns: (B, T, N) simple per-period returns (e.g., daily returns)
      mdd_temp: temperature for the smooth max-drawdown surrogate (higher -> closer to max)
      theta: optional per-period threshold (MAR). If apply_theta_to_return=True, subtract theta from
             portfolio returns before computing numerator (mean). If apply_theta_to_drawdown=True, subtract
             theta from path used to compute drawdown (uncommon).
      apply_theta_to_return: bool
      apply_theta_to_drawdown: bool
      eps: numeric epsilon for stability
      normalize_weights: if True apply softmax(weights) inside
      use_log_loss: if True return -log(clamped_calmar + eps), else return -calmar
      min_return: lower clamp for per-step port returns to keep log1p safe

    Returns:
      scalar Tensor: batch-mean loss to MINIMIZE (so optimizer minimizing this increases Calmar)
    """
    B, T, N = returns.shape

    # 2) portfolio per-step returns (B, T)
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)

    # 3) apply theta where requested
    port_for_return = port - theta if apply_theta_to_return else port
    port_for_dd = port - theta if apply_theta_to_drawdown else port

    # clamp for log1p safety (for drawdown path)
    port_for_return_clamped = torch.clamp(port_for_return, min=min_return)
    port_for_dd_clamped     = torch.clamp(port_for_dd,     min=min_return)

    # 4) numerator: mean simple return over the window (per-batch)
    mean_return = port_for_return_clamped.mean(dim=1)  # (B,)

    # 5) smooth max drawdown on the same window (log-space path)
    log_ret_dd = torch.log1p(port_for_dd_clamped)        # (B, T)
    cum_log = torch.cumsum(log_ret_dd, dim=1)           # (B, T)
    running_peak = torch.cummax(cum_log, dim=1).values  # (B, T)
    drawdown_log = running_peak - cum_log               # (B, T) >= 0

    # smooth max via log-sum-exp with bias correction subtract log(T)
    s = mdd_temp * drawdown_log
    lse = torch.logsumexp(s, dim=1)                     # (B,)
    smooth_max_log = (lse - math.log(T)) / (mdd_temp + eps)
    mdd = 1.0 - torch.exp(-smooth_max_log)              # (B,) in [0,1)

    # 6) compute Calmar (per-batch) and loss
    denom = mdd + eps
    calmar = mean_return / denom   # note: mean_return can be negative -> calmar negative

    # stable loss: -log(calmar) if calmar>0 else penalize strongly
    if use_log_loss:
        loss_per_batch = -torch.log(torch.clamp(calmar, min=eps) + eps)
    else:
        loss_per_batch = -calmar

    return loss_per_batch.mean()

# -------------------- Combination Loss Functions -------------------- #
@LossLibrary.register(category='custom')
def custom_loss_1(weights: Tensor, returns: Tensor, lambda1: float):
    """
    loss = differentiable sharpe + lambda1 * smooth CVar
    """
    sharpe = differentiable_sharpe_loss(weights, returns)
    cvar = smooth_rockafellar_cvar_regularizer(weights, returns)

    # print('Sharpe:',sharpe)
    # print('CVaR:', cvar * lambda1)
    return sharpe + lambda1 * cvar 

def combined_loss_2():
    pass

def combined_loss_3():
    pass

def combined_loss_4():
    pass