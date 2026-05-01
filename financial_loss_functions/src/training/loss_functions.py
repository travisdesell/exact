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

    - Define this class first (in a module).
    - Define functions in the same module (or other modules), and decorate them:
        @LossCollection.register("regularizers", "diversification", "herfindahl")
        def herfindahl(weights): ...
    - Use LossCollection.get(...) or LossCollection.items() to retrieve.
    """
    _registry: Registry = {}

    @classmethod
    def register(
        cls,
        category: str,
        subcategory: str | None = None,
        name: str | None = None
    ):
        """
        Decorator to register a standalone function into the class registry.

        Example::
        
            @LossCollection.register('regularizers', 'diversification', 'herfindahl_index')
            def herfindahl_index(weights):
                return (weights**2).sum(dim=-1).mean()
        
        Args:
            category (str): category of the loss functions. 'objectives', 'regularizers' or 'custom'.
            subcategory (str | None): Subcategory that the function/term belongs too. 
                Default = None, '__default__' will be set as subcategory.
            name (str | None): Name of the loss function. Default = None. 
                If None, name of the function will be used as default.
        """
        def decorator(fn: Callable):
            cat = category
            sub = subcategory or '__default__'
            nm = name or fn.__name__
            # Prevent duplicate registration
            if nm in cls._registry.get(cat, {}).get(sub, {}):
                raise KeyError(f"Function '{nm}' already registered in category '{cat}', subcategory '{sub}'")
            cls._registry.setdefault(cat, {}).setdefault(sub, {})[nm] = fn
            return fn
        return decorator

    # --- query helpers ---
    @classmethod
    def items(cls) -> Registry:
        """
        Get the entire nesed dictionary of the library of loss functions.

        Returns:
            Registry: Dictionary of all loss terms and functions.
        """
        return cls._registry

    @classmethod
    def list_categories(cls) -> list[str]:
        """
        Get a list of all available categories of loss functions/terms.        
        
        Returns:
            list[str]: List of all available categories of loss functions.
        """
        return list(cls._registry.keys())

    @classmethod
    def list_subcategories(cls, category: str) -> list[str]:
        """
        Get a list of all available subcategories of loss terms under a given category.
        Here, objectives and regularizers have only one subcategory called '__default__'.        
        
        Returns:
            list[str]: List of all available subcategories of loss functions.
        """
        return list(cls._registry.get(category, {}).keys())

    @classmethod
    def list_functions(cls, category: str, subcategory: str|None = None) -> list[str]:
        """
        Get a list of available functions under category (with or without a subcategory).

        Args:
            category (str): Category that the loss term belongs to.
            subcategory (str | None): Subcategory that the required loss functions/terms belong to.
        """
        sub = subcategory or '__default__'
        return list(cls._registry.get(category, {}).get(sub, {}).keys())

    @classmethod
    def get(cls, category: str,  name: str, subcategory: str|None = None) -> Callable:
        """
        Get a particular model that belongs to a subcategory (with or without subcategory).

        Args:
            category (str): Category that the loss function/term belongs to.
            name (str): Name of the required loss function/term.
            subcategory (str | None): Subcategory of the loss function/term.

        Returns:
            Callable: Callable loss function from the library.
        """
        sub = subcategory or '__default__'
        return cls._registry[category][sub][name]

# -------------------- Returns -------------------- #
@LossLibrary.register(category='objectives')
def log_return_objective(port_returns: Tensor, eps: float = 1e-8, **kwargs
)-> Tensor:
    """
    PyTorch loss function to calculate the log returns for the 
    given portfolio allocation loss objective.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8
    
    Returns:
        Tensor: Batch mean negative log returns
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    # Convert to log returns: log(1 + R)
    # We add a tiny epsilon to avoid log(0) if a portfolio hits -100%
    log_returns = torch.log(1.0 + port_returns + eps)
    
    # Sum of log returns = Cumulative log growth
    cum_log_return = log_returns.sum(dim=1)
    
    return -cum_log_return.mean()

# -------------------- Sharpe -------------------- #
@LossLibrary.register(category='objectives')
def raw_sharpe_objective(pf_returns: Tensor, eps: float = 1e-8, **kwargs
    ) -> Tensor:
    """
    Raw Sharpe ratio using standard deviation directly.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean raw Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    # portfolio returns per step
    mean_ret = pf_returns.mean(dim=1)                          # (B,)
    # population std: set unbiased=False
    port_std = pf_returns.std(dim=1, unbiased=False) + eps     # (B,)
    sharpe = mean_ret / port_std                         # (B,)
    # maximize Sharpe -> minimize negative Sharpe
    return -sharpe.mean()

@LossLibrary.register(category='objectives')
def differentiable_sharpe_objective(pf_returns: Tensor, eps: float = 1e-6, **kwargs
    ):
    """
    Differentiable Sharpe ratio where we calculate square root of variance 
    instead of direct standard deviation. Helps with numerical stabilty with 
    PyTorch backprop as its broken down into 2 steps.
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    mean_ret = pf_returns.mean(dim=1)
    var  = pf_returns.var(dim=1)          # variance, not std
    # Avoiding the std entirely
    return -(mean_ret / (var.sqrt() + eps)).mean()

@LossLibrary.register(category='objectives')
def rms_sharpe_objective(pf_returns: Tensor, eps: float = 1e-8, **kwargs
) -> Tensor:
    """
    Sharpe ratio where we use RMS instead of standard deviation.
    RMS is the population standard deviation.
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean RMS Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss).
    """
    mean_ret = pf_returns.mean(dim=1, keepdim=True)            # (B,1)
    rms = torch.sqrt(torch.mean((pf_returns - mean_ret)**2, dim=1) + eps)  # (B,)
    sharpe = mean_ret.squeeze(1) / rms
    return -sharpe.mean()

@LossLibrary.register(category='objectives')
def smooth_neglog_sharpe_loss(
    pf_returns: Tensor,
    eps: float = 1e-8,
    beta: float = 1.0,
    **kwargs
) -> Tensor:
    """
    Smooth, always-differentiable Sharpe loss.
    Uses softplus to map Sharpe -> positive before log.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
        beta (float): Beta sharpness for softplus; larger -> closer to clamp. Default = 1.0

    Returns:
        Tensor: Batch mean smooth negative log Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss).
    """
    mean_ret = pf_returns.mean(dim=1)

    var = pf_returns.var(dim=1)
    std = torch.sqrt(var + eps)
    sharpe = mean_ret / (std + eps)

    # smooth positive mapping (always > 0)
    sharpe_pos = softplus(sharpe, beta=beta)

    loss = torch.log(sharpe_pos + eps)
    return -loss.mean()

@LossLibrary.register(category='objectives')
def log_sharpe_objective(pf_returns: Tensor, eps: float = 1e-8, **kwargs) -> Tensor: 
    """
    Differentiable Sharpe using mean log returns instead of mean returns. 
    This essential changes the ratio from arthimetic mean to geometric mean.
    The model will be made to focus on compunded returns over the given window.
    Variance and Standard Deviation is calculated on log of portfolio returns.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean log Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss). 
    """

    log_returns = torch.log(1.0 + pf_returns + eps)
    
    mean_log_ret = log_returns.mean(dim=1)
    var_log  = log_returns.var(dim=1)          # variance, not std
    
    return -(mean_log_ret / (var_log.sqrt() + eps)).mean()

# -------------------- Sortino -------------------- #
@LossLibrary.register(category='objectives')
def raw_sortino_objective(
    pf_returns: Tensor,
    target: float = 0.0,
    eps: float = 1e-8,
    **kwargs
    ):
    """
    PyTorch loss function to calcuate raw Sortino Ratio for the objective.
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        target (float): Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
            Default = 0.0.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean raw Sortino ratio objective 
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - pf_returns, min=0.0)  # (B, T_out), only positive for downside
    downside_std = downside.std(dim=1) + eps  # (B,)
    
    mean_return = pf_returns.mean(dim=1)  # (B,)
    sortino = mean_return / downside_std  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

@LossLibrary.register(category='objectives')
def differentiable_sortino_objective(
        pf_returns: Tensor, target: float = 0.0, eps: float = 1e-8, **kwargs
    ):
    """
    Differentiable Sortino ratio where we calculate square root of variance 
    instead of direct standard deviation for variance below the target. 
    Helps with numerical stabilty with PyTorch backprop as its broken down into 2 steps.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        target (float): Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
            Default = 0.0.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean differntiable Sortino ratio objective 
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - pf_returns, min=0.0)  # (B, T_out), only positive for downside
    downside_var = downside.var(dim=1) 
    
    mean_return = pf_returns.mean(dim=1)  # (B,)
    sortino = mean_return / (downside_var.sqrt() + eps)  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

@LossLibrary.register(category='objectives')
def rms_sortino_loss(
        pf_returns: Tensor, target: float = 0.0, eps: float = 1e-8, **kwargs
    ):
    """
    Sortino ratio where we use RMS of variance below target value, instead of standard deviation.
    RMS is the population standard deviation.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        target (float): Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
            Default = 0.0.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean RMS Sortino ratio objective 
            (-ve because NN should maximize +ve returns but minimize loss). 
    """
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - pf_returns, min=0.0)  # (B, T_out), only positive for downside
    
    # RMS downside: sqrt(mean(downside**2) + eps) -> for stable gradients
    downside_rms = torch.sqrt(torch.mean(downside ** 2, dim=1) + eps)  # (B,)
    
    mean_return = pf_returns.mean(dim=1)  # (B,)
    sortino = mean_return / downside_rms  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

@LossLibrary.register(category='objectives')
def smooth_neglog_sortino_objective(
    pf_returns: Tensor,
    target: float = 0.0,
    use_soft_downside: bool = True,
    beta: float = 10.0,                # sharpness for softplus; larger -> closer to clamp
    eps: float = 1e-8,
    **kwargs
) -> Tensor:
    """
    Smooth, always-differentiable negative log Sortino loss.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        target (float): Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
            Default = 0.0.
        use_soft_downside (bool): Uses softplus on downward variance if True. Uses clamp if False. 
            Default = True
        beta (float): Beta sharpness for softplus; larger -> closer to clamp. Default = 1.0
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean smooth negative log Sharpe ratio objective
            (-ve because NN should maximize +ve returns but minimize loss).
    """
    # downside: smooth or hard
    if use_soft_downside:
        # softplus approximates clamp(target - port, min=0)
        # we feed (target - port) so positive means downside
        downside = softplus(target - pf_returns, beta=beta)
    else:
        downside = torch.clamp(target - pf_returns, min=0.0)

    # RMS downside (population)
    downside_rms = torch.sqrt(torch.mean(downside**2, dim=1) + eps)  # (B,)
    mean_ret = pf_returns.mean(dim=1)  # (B,)
    sortino = mean_ret / (downside_rms + eps)  # (B,)

    sortino_loss = torch.log(softplus(sortino) + eps) 
    return -sortino_loss.mean()

@LossLibrary.register(category='objectives')
def log_sortino_objective(
        pf_returns: Tensor,
        target: float = 0.0, use_soft_downside: bool = True, 
        beta: float = 10.0, eps: float = 1e-8, **kwargs
    ):
    """
    Differentiable Sortino using mean log returns instead of mean returns. 
    This essential changes the ratio from arthimetic mean to geometric mean.
    The model will be made to focus on compunded returns over the given window.
    Variance and Standard Deviation is calculated on log of portfolio returns.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        target (float): Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
            Default = 0.0.
        use_soft_downside (bool): Uses softplus on downward variance if True. Uses clamp if False. 
            Default = True
        beta (float): Beta sharpness for softplus; larger -> closer to clamp. Default = 1.0
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8

    Returns:
        Tensor: Batch mean log Sortino ratio objective
            (-ve because NN should maximize +ve returns but minimize loss).
    """
    # Log portfolio returns
    log_returns = torch.log(1.0 + pf_returns + eps)

    # Downside deviation: std of negative deviations from target
    # downside: smooth or hard
    if use_soft_downside:
        # softplus approximates clamp(target - port, min=0)
        # we feed (target - port) so positive means downside
        downside = softplus(target - log_returns, beta=beta)
    else:
        downside = torch.clamp(target - log_returns, min=0.0)

    downside_var = downside.var(dim=1) 
    
    mean_return = log_returns.mean(dim=1)  # (B,)
    sortino = mean_return / (downside_var.sqrt() + eps)  # (B,)
    
    # Maximize Sortino -> minimize negative Sortino
    return -sortino.mean()

# -------------------- Max Drawdown -------------------- #
@LossLibrary.register(category='regularizers', subcategory='tail_risk')
def smooth_mdd_regularizer(
    pf_returns: Tensor,
    temp: float = 50.0,
    eps: float = 1e-8,
    min_return: float = -0.999,   # to keep log1p safe
    use_percent: bool = True,
    **kwargs
) -> Tensor:
    """
    Differentiable smooth Max Drawdown regularizer (to MINIMIZE).

    Assumptions:
      - `weights` are already normalized (e.g., via softmax) and shaped (B, N).
      - `returns` are simple returns shaped (B, T, N) (e.g., 0.01 => +1%).
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        temp (float): Temperature for the log-sum-exp smoothing. Higher -> closer to true max.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8
        min_return (float): Minimum returns clamp. Lower clamp for per-period returns (must be > -1).
            Default = -0.999, to keeo log1p safe.
        use_percent (bool) if True returns fractional drawdown in [0,1); else returns log-units.
            Default = True

    Return:
        Tensor: batch mean smooth max drawdown regularizer.
    """
    B, T = pf_returns.shape

    # clamp port to > -1 for log1p safety
    port_clamped = torch.clamp(pf_returns, min=min_return)

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
def cvar_topk_regularizer(pf_returns: Tensor, alpha: float = 0.05, **kwargs) -> Tensor:
    """
    Empirical CVaR (expected shortfall) over the worst alpha fraction of losses.
    Uses torch.topk to compute average of top-k losses.

    Args:
        weights (Tensor): (B, N) Portfolio allocation weights (normalized).
        returns (Tensor): (B, T, N) Output (future) window of raw returns to calculate the loss term on.
        alpha (float): alpha tail fraction (0 < alpha <= 1). Default = 0.05.

    Returns: 
        Tensor: Batch mean Empirical CVaR regularizer term
    """
    B, T = pf_returns.shape
    
    # losses = -returns (higher = worse)
    losses = -pf_returns  # (B, T)

    # number of tail points to average (at least 1)
    k = max(1, math.ceil(alpha * T))

    # top-k largest losses per batch
    topk_vals, _ = torch.topk(losses, k, dim=1, largest=True, sorted=False)  # (B, k)
    cvar_per_batch = topk_vals.mean(dim=1)  # (B,)
    return cvar_per_batch.mean()  # scalar

@LossLibrary.register(category='regularizers', subcategory='tail_risk')
def smooth_cvar_regularizer(
    pf_returns: Tensor,
    temp: float = 1e-2,
    eps: float = 1e-8,
    scale_by_std: bool = True,
    normalize_by_port_std: bool = True,
    port_std_floor: float = 1e-3,
    **kwargs
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

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        alpha (float): alpha tail fraction (0 < alpha <= 1). Default = 0.05.
        temp (float): Temperature for the log-sum-exp smoothing. Higher -> closer to true max.
            smaller temp -> more concentrated on worst losses.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
        scale_by_std (bool) whether to standardize losses per batch for numeric stability. 
            Default = true.
        normalize_by_port_std (bool): Normalize by portfolio standard deviation to make it a 
            'Tail Ratio' to convert CVaR into a unitless ratio.
        port_std_floor (float): Minimum floor clamp and portfolio standard deviation.
            
    Returns: 
        Tensor: Batch mean smooth CVaR regularizer term
    """
    losses = -pf_returns  # (B, T)

    if scale_by_std:
        scores_std = losses.std(dim=1, keepdim=True) + eps   # (B,1)
        scores = losses / (scores_std * (temp + eps))
    else:
        scores = losses / (temp + eps)

    sel = softmax(scores, dim=1)           # (B, T) sums to 1
    weighted_mean = (sel * losses).sum(dim=1)  # (B,) -- already an average-like quantity

    approx_cvar = weighted_mean  # NOT dividing by alpha here

    if normalize_by_port_std:
        port_std = pf_returns.std(dim=1)           # (B,)
        port_std = torch.clamp(port_std, min=port_std_floor)
        approx_cvar = approx_cvar / (port_std + eps)

    return approx_cvar.mean()

@LossLibrary.register(category='regularizers', subcategory='tail_risk')
def smooth_rockafellar_cvar_regularizer(
    pf_returns: torch.Tensor,
    alpha: float = 0.05,
    temp: float = 1e-2, # In R&U, temp controls the Softplus "smoothness"
    eps: float = 1e-8,
    normalize_by_port_std: bool = True,
    port_std_floor: float = 1e-3,
    **kwargs
) -> torch.Tensor:
    """
    Differentiable CVaR using the Rockafellar & Uryasev formula. 
    Uses alpha tail risk to get average of 5% (Default) worst case scenarios.
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        alpha (float): alpha tail fraction (0 < alpha <= 1). Default = 0.05.
        temp (float): Temperature for the log-sum-exp smoothing. Higher -> closer to true max.
            smaller temp -> more concentrated on worst losses.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
        scale_by_std (bool) whether to standardize losses per batch for numeric stability. 
            Default = true.
        normalize_by_port_std (bool): Normalize by portfolio standard deviation to make it a 
            'Tail Ratio' to convert CVaR into a unitless ratio.
        port_std_floor (float): Minimum floor clamp and portfolio standard deviation.
            
    Returns: 
        Tensor: Batch mean differentiable Rockafellar & Uryasev CVaR regularizer term
    """
    # Calculate Portfolio Returns and Losses
    # port: (B, T), losses: (B, T)
    losses = -pf_returns 

    # Estimate VaR (zeta) for the batch
    # We take the (1-alpha) quantile of the losses as our starting point for zeta
    # This is the "threshold" where the tail begins.
    with torch.no_grad():
        zeta = torch.quantile(losses, 1 - alpha, dim=1, keepdim=True) # (B, 1)

    # Rockafellar & Uryasev Formula
    # Instead of max(0, losses - zeta), we use Softplus for a smooth gradient.
    # soft_excess = temp * log(1 + exp((losses - zeta) / temp))
    excess_losses = (losses - zeta)
    soft_excess = softplus(excess_losses, beta=1/temp)
    
    # CVaR = zeta + (1 / alpha) * Average(excess_losses)
    # (B,)
    approx_cvar = zeta.squeeze(1) + (1.0 / alpha) * soft_excess.mean(dim=1)

    # Normalization (The 'Tail Ratio')
    if normalize_by_port_std:
        port_std = pf_returns.std(dim=1)
        port_std = torch.clamp(port_std, min=port_std_floor)
        # Final value is dimensionless: how many STDs is the average tail loss?
        approx_cvar = approx_cvar / (port_std + eps)

    return approx_cvar.mean()

# -------------------- Risk Parity -------------------- #
def sample_covariance(returns: Tensor, unbiased: bool = True):
    """
    Calculate covariance matrix for the given returns matrix.
    
    Args:
        returns (Tensor): (B, T, N) Output (future) window of raw returns.
        unbiased (bool): Use Bessels correction with True or MLE for False.
            Default = True.
    
    Returns:
        cov (Tensor): sample covariance per batch -> (B, N, N)
    """
    B, T, N = returns.shape
    mean = returns.mean(dim=1, keepdim=True)  # (B, 1, N)
    X = returns - mean  # (B, T, N)
    # cov = X^T X / (T-1) if unbiased else / T
    denom = (T - 1) if unbiased and T > 1 else T
    cov = X.transpose(1, 2).bmm(X) / float(max(denom, 1))
    return cov

def shrinkage_covariance_torch(cov: Tensor, shrink: float = 0.1) -> Tensor:
    """
    Linear shrinkage toward scaled identity:
      cov_shrunk = (1 - shrink) * cov + shrink * (trace(cov)/N) * I
    
    Args:
        cov (Tensor): covariance matrix (B, N, N).
    
    Returns:
        Tensor: Linear skrunk covaraince matrix.
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

    Args:
        weights (Tensor): (B, N) Portfolio allocation weights (normalized).
        returns (Tensor): (B, T, N) Output (future) window of raw returns to calculate the loss term on.
        shrink (float): in [0,1] shrinkage intensity. Default = 0.1.
        use_shrink (bool): whether to apply shrinkage. Default = True.
        shrink_clip (tuple): allowed range for shrink (safety). Default = (0.0, 0.9).
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
        scale_invariant (bool)L if True divide squared-deviation by 
            sigma2^2 to be scale-invariant. Default = True.

    Returns:
        Tensor: Batch mean risk contribution.
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
    pf_returns: Tensor,
    theta: float = 0.0,
    eps: float = 1e-8,
    **kwargs
) -> Tensor:
    """
    Exact empirical Omega ratio (batch mean).
    
    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        theta (float): threshold for omega ration (same units as returns). Default = 0.0.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
    
    Returns:
        Tensor: Batch mean raw Omega ratio.
    """

    # positives = (R - theta)_+, negatives = (theta - R)_+
    pos = torch.clamp(pf_returns - theta, min=0.0)        # (B, T)
    neg = torch.clamp(theta - pf_returns, min=0.0)        # (B, T)

    # expectations are simple means over time
    pos_mean = pos.mean(dim=1)   # (B,)
    neg_mean = neg.mean(dim=1)   # (B,)

    # avoid divide-by-zero
    omega_per_batch = pos_mean / (neg_mean + eps)  # (B,)

    return -omega_per_batch.mean()

@LossLibrary.register(category='objectives')
def smooth_omega_objective(
    pf_returns: Tensor,
    theta: float = 0.0,
    beta: float = 10.0,
    eps: float = 1e-8,
    use_log_loss: bool = True,
    cap_omega: float | None = None,
    **kwargs
) -> Tensor:
    """
    Smooth Omega objective (LOSS TO MINIMIZE).

    Can be used as:
      - primary objective: loss = smooth_omega_objective(...)
      - regularizer: loss += lambda * smooth_omega_objective(...)

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        theta (float): threshold for omega ration (same units as returns). Default = 0.0.
        beta (float) softplus sharpness (>0).
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.

        cap_omega (bool | None) optional cap to limit extreme ratios. Default = None.

    Returns:
        Tensor: Batch mean smooth omega objective.
    """
    # smoothed positive / negative parts
    pos = softplus(pf_returns - theta, beta=beta)      # (R - theta)_+
    neg = softplus(theta - pf_returns, beta=beta)      # (theta - R)_+

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
    eps: float = 1e-8
) -> Tensor:
    """
    HHI concentration penalty (batch-mean).
    
    Args:
        weights (Tensor): (B, N) Portfolio allocation weights (normalized).
        scale_to_unit (bool): if True, scales HHI to [0,1] using (HHI - 1/N) / (1 - 1/N). 
            This makes penalty interpretable and easier to combine with other losses.
            Default = True.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
    
    Returns: 
        Tensor: Batch mean HHI penalty.
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
    normalize_by_gross: bool = False,
    scale_to_unit: bool = True,
    eps: float = 1e-8
) -> Tensor:
    """

    Args:
        weights (Tensor): (B, N) Portfolio allocation weights (normalized).
        normalize_by_gross (bool): Normalize by gross sum of weights. Default = False.
        scale_to_unit (bool): if True, scales HHI to [0,1] using (HHI - 1/N) / (1 - 1/N). 
            This makes penalty interpretable and easier to combine with other losses.
            Default = True.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
    
    Returns: 
        Tensor: Batch mean HHI penalty.
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
    eps: float = 1e-8,
) -> Tensor:
    """
    Entropy concentration penalty (no clustering).

    Args:
        weights (Tensor): (B, N) Portfolio allocation weights (normalized).
        signed (bool): If True, convert weights -> abs(weights) and renormalize to gross exposure = 1.
            Default = False
        mode (str): one of {'neg_entropy', 'scaled', 'kl'}:
            - 'neg_entropy': return -H(w)  (minimize -> maximize entropy)
            - 'scaled': return 1 - H(w)/log(N)  (in [0,1], 0 = uniform)
            - 'kl': return log(N) - H(w)  (KL(uniform || w), >=0).
            Default = 'scaled'
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-12.

    Returns:
      Tensor: Batch mean entropy penalty.
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
    pf_returns: Tensor,
    theta: float = 0.0,
    apply_theta_to_return: bool = False,
    apply_theta_to_drawdown: bool = False,
    min_return: float = -0.999,
    eps: float = 1e-8,
    **kwargs
) -> Tensor:
    """
    Raw Calmar ratio computed on the provided window (no annualization).
    Returns the batch-mean Calmar (higher is better).

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        theta (float): per-period MAR; same units as returns. Default = 0.0.
        apply_theta_to_return (bool): If True, numerator uses (port - theta). Default = False.
        apply_theta_to_drawdown (bool): if True, drawdown path uses (port - theta) (uncommon). 
            Deafult = False.
        min_return (float): lower clamp for per-step port returns to keep log1p safe.
            Default = -0.999.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.

    Returns:
        Tensor: Batch mean raw Calmar ratio.
    """

    # optionally apply theta to numerator and/or drawdown path
    port_for_return = pf_returns - theta if apply_theta_to_return else pf_returns
    port_for_dd     = pf_returns - theta if apply_theta_to_drawdown else pf_returns

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
    return -calmar_per_sample.mean()

@LossLibrary.register(category='objectives')
def smooth_calmar_objective(
    pf_returns: Tensor,
    mdd_temp: float = 50.0,
    theta: float = 0.0,
    apply_theta_to_return: bool = False,
    apply_theta_to_drawdown: bool = False,
    eps: float = 1e-8,
    use_log_loss: bool = True,
    min_return: float = -0.999,
    beta: float = 1.0,
    **kwargs
) -> Tensor:
    """
    Smooth Calmar computed directly on the model output horizon (no annualization).
    Minimizing this loss -> maximizes Calmar on the window.

    Args:
        pf_returns (Tensor): (B, T_out) Portfolio returns calculated by 
            weights * returns of all stocks.
        mdd_temp (float): temperature for the smooth max-drawdown surrogate (higher -> closer to max).
        theta (float): Per-period threshold (MAR). If apply_theta_to_return=True, subtract theta from
            portfolio returns before computing numerator (mean). If apply_theta_to_drawdown=True, subtract
            theta from path used to compute drawdown (uncommon).
        apply_theta_to_return (bool): Apply theta to returns.
        apply_theta_to_drawdown (bool): Apply theta to drawdown.
        eps (float): Epsilon value to avoid divide by zero errors for numerical stability.
            Default = 1e-8.
        use_log_loss (bool): If True return -log(clamped_calmar + eps), else return -calmar.
            Default = False.
        min_return (float): lower clamp for per-step port returns to keep log1p safe.
            Default = -0.999.

    Returns:
        Tensor: Batch mean smooth calmar ratio.
    """
    B, T = pf_returns.shape

    # apply theta where requested
    port_for_return = pf_returns - theta if apply_theta_to_return else pf_returns
    port_for_dd = pf_returns - theta if apply_theta_to_drawdown else pf_returns

    # clamp for log1p safety (for drawdown path)
    port_for_return_clamped = torch.clamp(port_for_return, min=min_return)
    port_for_dd_clamped     = torch.clamp(port_for_dd,     min=min_return)

    # numerator: mean simple return over the window (per-batch)
    mean_return = port_for_return_clamped.mean(dim=1)  # (B,)

    # smooth max drawdown on the same window (log-space path)
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
        calmar_pos = softplus(calmar, beta=beta)
        loss_per_batch = torch.log(torch.clamp(calmar_pos, min=eps) + eps)
    else:
        loss_per_batch = calmar

    return -loss_per_batch.mean()

# -------------------- Custom Loss Functions -------------------- #
# @LossLibrary.register(category='custom')
def custom_loss_1(pf_returns: Tensor, lambda1: float, **kwargs) -> Tensor:
    """Combines differentiable Sharpe ratio with a smooth CVaR regulariser.

    Args:
        pf_returns (Tensor): Portfolio daily returns (B, T_out).
        lambda1 (float): Weight for the CVaR term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = differentiable_sharpe + lambda1 * smooth_CVaR.
    """
    sharpe = differentiable_sharpe_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)

    # print('Sharpe:',sharpe)
    # print('CVaR:', cvar * lambda1)
    return sharpe + (lambda1 * cvar) 

# @LossLibrary.register(category='custom')
def custom_loss_2(pf_returns: Tensor, lambda1: float, **kwargs) -> Tensor:
    """Combines RMS-based Sharpe ratio with a smooth CVaR regulariser.

    Args:
        pf_returns (Tensor): Portfolio daily returns (B, T_out).
        lambda1 (float): Weight for the CVaR term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = rms_sharpe + lambda1 * smooth_CVaR.
    """
    sharpe = rms_sharpe_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)

    # print('Sharpe:',sharpe)
    # print('CVaR:', cvar * lambda1)
    return sharpe + (lambda1 * cvar) 

# @LossLibrary.register(category='custom')
def custom_loss_3(pf_returns: Tensor, lambda1: float, **kwargs) -> Tensor:
    """Combines raw Sortino ratio with a smooth CVaR regulariser.

    Args:
        pf_returns (Tensor): Portfolio daily returns (B, T_out).
        lambda1 (float): Weight for the CVaR term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = rms_sortino + lambda1 * smooth_CVaR.
    """
    sortino = rms_sortino_loss(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)

    return sortino + (lambda1 * cvar)

# @LossLibrary.register(category='custom')
def custom_loss_4(pf_returns: Tensor, lambda1: float, **kwargs) -> Tensor:
    """Combines differentiable Sortino ratio with a smooth CVaR regulariser.

    Args:
        pf_returns (Tensor): Portfolio daily returns (B, T_out).
        lambda1 (float): Weight for the CVaR term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = differentiable_sortino + lambda1 * smooth_CVaR.
    """
    sortino = differentiable_sortino_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)

    return sortino + lambda1 * cvar

# @LossLibrary.register(category='custom')
def custom_loss_5(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor, lambda1: float, **kwargs
) -> Tensor:
    """Combines differentiable Sharpe ratio with a risk parity regulariser.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns over the holding period (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        lambda1 (float): Weight for the risk parity term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = differentiable_sharpe + lambda1 * risk_parity.
    """
    sharpe = differentiable_sharpe_objective(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:',sharpe)
    # print('RP:', risk_parity * lambda1)
    return sharpe + lambda1 * risk_parity 

# @LossLibrary.register(category='custom')
def custom_loss_7(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor, 
    lambda1: float, lambda2: float, **kwargs
) -> Tensor: 
    """Combines log Sharpe ratio, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        lambda1 (float): Weight for CVaR term.
        lambda2 (float): Weight for risk parity term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = log_sharpe + lambda1 * smooth_CVaR + lambda2 * risk_parity.
    """
    log_sharpe = log_sharpe_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:', sharpe)
    # print('Log Ret:', log_returns)
    # print('CVaR:', cvar)
    # print('RP:', risk_parity)
    return log_sharpe + (lambda1 * cvar) + (lambda2 * risk_parity)

# @LossLibrary.register(category='custom')
def custom_loss_8(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor, log_ret_lambda: float,
    cvar_lambda: float, risk_p_lambda: float, **kwargs
) -> Tensor:
    """Combines differentiable Sharpe, log return, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        log_ret_lambda (float): Weight for log return term.
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = differentiable_sharpe + log_ret_lambda * log_return +
                cvar_lambda * smooth_CVaR + risk_p_lambda * risk_parity.
    """
    ### 2nd Best
    sharpe = differentiable_sharpe_objective(pf_returns)
    log_returns = log_return_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar)
    # print('RP:', risk_parity)
    loss = sharpe + \
        (log_ret_lambda * log_returns) + \
            (cvar_lambda * cvar) + \
                (risk_p_lambda * risk_parity)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_9(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    lambda1: float, lambda2: float, **kwargs
) -> Tensor:
    """Combines log Sortino ratio, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        lambda1 (float): Weight for CVaR term.
        lambda2 (float): Weight for risk parity term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = log_sortino + lambda1 * smooth_CVaR + lambda2 * risk_parity.
    """
    log_sortino = log_sortino_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar)
    # print('RP:', risk_parity)
    return log_sortino + (lambda1 * cvar) + (lambda2 * risk_parity)

# @LossLibrary.register(category='custom')
def custom_loss_6(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor, 
    cvar_lambda: float, risk_p_lambda: float, **kwargs
) -> Tensor:
    """Combines differentiable Sharpe ratio, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        **kwargs: Additional unused arguments.

    Returns:
        Tensor: loss = differentiable_sharpe + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity.
    """
    #### 2ND BEST
    sharpe = differentiable_sharpe_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar)
    # print('RP:', risk_parity)
    return sharpe + (cvar_lambda * cvar) + (risk_p_lambda * risk_parity)


@LossLibrary.register(category='custom')
def custom_loss_10(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float
) -> Tensor:
    """Combines smooth negative-log Sharpe, smooth CVaR and risk parity.
    This is the best-performing loss according to empirical tests.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.

    Returns:
        Tensor: loss = smooth_neglog_sharpe + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity.
    """
    sharpe = smooth_neglog_sharpe_loss(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar * cvar_lambda)
    # print('RP:', risk_parity * risk_p_lambda)
    loss = sharpe + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity)
    return loss

@LossLibrary.register(category='custom')
def custom_loss_11(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float
) -> Tensor:
    """Combines smooth Omega ratio, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.

    Returns:
        Tensor: loss = smooth_omega + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity.
    """
    ### BEST!
    omega = smooth_omega_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Omega:', omega)
    # print('CVaR:', cvar* cvar_lambda)
    # print('RP:', risk_parity* risk_p_lambda)
    loss = omega + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_12(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float
) -> Tensor:
    """Combines raw Omega ratio, smooth CVaR and risk parity.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.

    Returns:
        Tensor: loss = raw_omega + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity.
    """
    omega = raw_omega_ratio(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)

    # print('Omega:', omega)
    # print('CVaR:', cvar* cvar_lambda)
    # print('RP:', risk_parity* risk_p_lambda)
    loss = omega + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_13(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float, ent_lambda: float
) -> Tensor:
    """Combines smooth negative-log Sharpe, smooth CVaR, risk parity and entropy regulariser.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        ent_lambda (float): Weight for entropy regulariser.

    Returns:
        Tensor: loss = smooth_neglog_sharpe + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity + ent_lambda * entropy.
    """
    sharpe = smooth_neglog_sharpe_loss(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)
    entropy = entropy_conc_regularizer(weights)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar * cvar_lambda)
    # print('RP:', risk_parity * risk_p_lambda)
    # print('Entropy:', entropy)

    loss = sharpe + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity) + \
                (ent_lambda * entropy)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_14(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float, ent_lambda: float
) -> Tensor:
    """Combines smooth Omega ratio, smooth CVaR, risk parity and entropy regulariser.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        ent_lambda (float): Weight for entropy regulariser.

    Returns:
        Tensor: loss = smooth_omega + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity + ent_lambda * entropy.
    """
    omega = smooth_omega_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)
    entropy = entropy_conc_regularizer(weights)

    # print('Omega:', omega)
    # print('CVaR:', cvar* cvar_lambda)
    # print('RP:', risk_parity* risk_p_lambda)
    loss = omega + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity) + \
                (ent_lambda * entropy)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_15(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float, hhi_lambda: float
) -> Tensor:
    """Combines smooth negative-log Sharpe, smooth CVaR, risk parity and HHI regulariser.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        hhi_lambda (float): Weight for Herfindahl-Hirschman Index regulariser.

    Returns:
        Tensor: loss = smooth_neglog_sharpe + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity + hhi_lambda * hhi.
    """
    sharpe = smooth_neglog_sharpe_loss(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)
    hhi = hhi_regularizer(weights)

    # print('Sharpe:', sharpe)
    # print('CVaR:', cvar * cvar_lambda)
    # print('RP:', risk_parity * risk_p_lambda)
    # print('Entropy:', entropy)

    loss = sharpe + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity) + \
                (hhi_lambda * hhi)
    return loss

# @LossLibrary.register(category='custom')
def custom_loss_16(
    weights: Tensor, all_returns: Tensor, pf_returns: Tensor,
    cvar_lambda: float, risk_p_lambda: float, hhi_lambda: float
) -> Tensor:
    """Combines smooth Omega ratio, smooth CVaR, risk parity and HHI regulariser.

    Args:
        weights (Tensor): Portfolio weights (B, N).
        all_returns (Tensor): Asset returns (B, T_out, N).
        pf_returns (Tensor): Portfolio returns (B, T_out).
        cvar_lambda (float): Weight for CVaR term.
        risk_p_lambda (float): Weight for risk parity term.
        hhi_lambda (float): Weight for Herfindahl-Hirschman Index regulariser.

    Returns:
        Tensor: loss = smooth_omega + cvar_lambda * smooth_CVaR +
                risk_p_lambda * risk_parity + hhi_lambda * hhi.
    """
    omega = smooth_omega_objective(pf_returns)
    cvar = smooth_rockafellar_cvar_regularizer(pf_returns)
    risk_parity = risk_parity_regularizer(weights, all_returns)
    hhi = hhi_regularizer(weights)

    # print('Omega:', omega)
    # print('CVaR:', cvar* cvar_lambda)
    # print('RP:', risk_parity* risk_p_lambda)
    loss = omega + \
        (cvar_lambda * cvar) + \
            (risk_p_lambda * risk_parity) + \
                (hhi_lambda * hhi)
    return loss