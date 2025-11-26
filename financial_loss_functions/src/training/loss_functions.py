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
