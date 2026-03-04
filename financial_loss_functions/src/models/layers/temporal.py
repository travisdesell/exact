import torch
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

        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output
        self.dropout = nn.Dropout(dropout) # Branch dropout

    def forward(self, x: Tensor) -> Tensor:
        
        attn_out, _ = self.attn(x, x, x)  # (B, T, H)

        # Residual Connection + Norm (Standard Transformer Block trick)
        # We add the input (x) to the output (attn_out) to help gradients flow
        # attn_out = x + self.dropout(attn_out) 
        # attn_out = self.ln_attn(attn_out)
        return self.dropout(attn_out) 

class ContextualGate(nn.Module):
    """Uses macro signals (SP500) to gate the asset-specific features."""
    def __init__(
            self, context_in: int,context_hidden: int, context_layers: int, hidden_size: int
        ):
        super().__init__()
        self.context_gru = nn.GRU(
            input_size=context_in,
            hidden_size=context_hidden,
            num_layers=context_layers,
            batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(context_hidden, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, global_data):
        # market_data: (B, T, 1)
        _, h_n = self.context_gru(global_data)
        return self.gate(h_n[-1]).unsqueeze(1) # (B, 1, H)
    
# class ContextualGate(nn.Module):
#     def __init__(self, context_in: int, hidden_size: int, kernel_size: int):
#         super().__init__()
        
#         # Match out_channels to hidden_size directly to avoid the Linear layer bottleneck
#         self.conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=context_in, 
#                 out_channels=hidden_size, # Output 16 features
#                 kernel_size=kernel_size, 
#                 padding=kernel_size // 2
#             ),
#             nn.BatchNorm1d(hidden_size), # Better than LayerNorm for CNNs on MPS
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1), 
#             nn.Flatten()
#         )
        
#         # Use a single linear projection to get the gating scale
#         self.gate_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Sigmoid()
#         )

#     def forward(self, global_data):
#         # global_data: (B, T, C) -> (B, C, T)
#         x = global_data.transpose(1, 2).contiguous()
        
#         context_features = self.conv(x) # (B, H)
#         gate = self.gate_proj(context_features) # (B, H)
        
#         return gate.unsqueeze(1) # (B, 1, H)

class TemporalEncoder(nn.Module):
    """Encodes the 120-day history of a single stock into a feature vector."""
    def __init__(
            self, input_size: int, hidden_size:int, lstm_layers: int,
            trans_layers: int, nhead:int, dropout:float
        ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*N, T, F)
        out, _ = self.lstm(x)
        out = torch.relu(out)
        
        out = self.transformer(out)
        
        return self.ln(out) # Summary of the stock's 'vibe'