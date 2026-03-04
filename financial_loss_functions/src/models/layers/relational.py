from torch import nn

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