from torch import nn
from torch import Tensor

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, nheads: int, dropout: float):
        super().__init__()

        # Attention layer components
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=nheads,
            batch_first=True
        ) 

        self.dropout = nn.Dropout(dropout) # Branch dropout
        
        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output

    def forward(self, x: Tensor) -> Tensor:
        
        attn_out, _ = self.attn(x, x, x)  # (B, T, H)

        # Residual Connection + Norm (Standard Transformer Block trick)
        # We add the input (x) to the output (attn_out) to help gradients flow
        attn_out = x + self.dropout(attn_out) 
        attn_out = self.ln_attn(attn_out)

        return attn_out