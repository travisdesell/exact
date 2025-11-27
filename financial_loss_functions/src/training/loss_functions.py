import torch
def sharpe_loss(weights, returns, eps=1e-8):
    """
    weights: (B, N)
    returns: (B, T_out, N) -- raw returns
    """
    # portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    mean = port.mean(dim=1)          # (B,)
    std = port.std(dim=1) + eps      # (B,)
    sharpe = mean / std              # (B,)
    # maximize Sharpe → minimize negative Sharpe
    return -sharpe.mean()

def sortino_loss(weights, returns, target=0.0, eps=1e-8):
    """
    weights: (B, N)
    returns: (B, T_out, N) -- raw returns
    target: Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
    """
    # Portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    # Downside deviation: std of negative deviations from target
    downside = torch.clamp(target - port, min=0.0)  # (B, T_out), only positive for downside
    downside_std = downside.std(dim=1) + eps  # (B,)
    
    mean = port.mean(dim=1)  # (B,)
    sortino = mean / downside_std  # (B,)
    
    # Maximize Sortino → minimize negative Sortino
    return -sortino.mean()