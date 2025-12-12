from torch import clamp

def raw_sharpe_loss(weights, returns, eps=1e-8):
    """
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    # portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    mean = port.mean(dim=1)          # (B,)
    std = port.std(dim=1) + eps      # (B,)
    sharpe = mean / std              # (B,)
    # maximize Sharpe → minimize negative Sharpe
    return -sharpe.mean()

def differentiable_sharpe_loss(weights, returns, eps=1e-6):
    port_ret = (weights.unsqueeze(1) * returns).sum(-1)   # (B, T)
    mean = port_ret.mean(dim=1)
    var  = port_ret.var(dim=1)          # variance, not std
    # Avoiding the sqrt entirely
    return -(mean / (var.sqrt() + eps)).mean()
    # even more stable:
    # return -(mean**2 / (var + eps)).mean()

def raw_sortino_loss(weights, returns, target=0.0, eps=1e-8):
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
    downside = clamp(target - port, min=0.0)  # (B, T_out), only positive for downside
    downside_std = downside.std(dim=1) + eps  # (B,)
    
    mean = port.mean(dim=1)  # (B,)
    sortino = mean / downside_std  # (B,)
    
    # Maximize Sortino → minimize negative Sortino
    return -sortino.mean()