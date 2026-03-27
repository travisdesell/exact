import torch.nn as nn
from torch import Tensor

class FeatureAttention(nn.Module):
    """Calculates dependencies between different stocks (Neural Covariance)."""
    def __init__(self, max_seq_len: int, hidden_size: int, nheads: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=max_seq_len, 
            num_heads=nheads,
            batch_first=True
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, H) -> (64, 120, 16)
        
        # 1. Flip to (B, H, T) -> (64, 16, 120)
        x_inverted = x.transpose(1, 2).contiguous()
        
        # 2. Attention over the 16 features
        attn_out, _ = self.attn(x_inverted, x_inverted, x_inverted) # (64, 16, 120)

        # 3. Flip BACK to (B, T, H) -> (64, 120, 16)
        attn_out = attn_out.transpose(1, 2).contiguous()
        
        # 4. Now shapes match: (B, T, H) + (B, T, H)
        # rel_x = x + self.dropout(attn_out) 
        return self.dropout(attn_out)


class VariableSelectionLayer(nn.Module):
    """
    Gating mechanism that learns per-feature importance based on the input.
    - Aggregates input over time (mean) to produce a global context.
    - Passes the context through a small MLP to generate feature weights (sigmoid).
    - Applies weights to each feature channel across all time steps.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.1):
        """
        Args:
            feature_dim (int): Number of input features (F).
            hidden_dim (int): Hidden dimension for the gating MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.feature_dim = input_size
        # MLP to produce feature weights
        self.gate_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        Returns:
            weighted_x: same shape as x, with features scaled by learned importance.
        """
        # Aggregate over time to get a global context per batch
        context = x.mean(dim=1)                # (B, F)
        # Compute feature‑wise weights
        weights = self.gate_net(context)       # (B, F)
        weights = weights.unsqueeze(1)         # (B, 1, F)
        # Apply weights to all time steps
        weighted_x = x * weights
        return self.dropout(weighted_x)