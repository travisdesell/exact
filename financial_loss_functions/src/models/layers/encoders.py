import torch
# import numpy as np
import torch.nn as nn
from torch import Tensor

class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # 1. LSTM Layer (Local Temporal Smoothing)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 2. Layer normalization & Dropout
        self.lstm_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x) # (B, T, H)
        x = self.lstm_ln(x)
        # x = torch.relu(x)
        x = nn.functional.gelu(x)
        return self.dropout(x)

class GlobalAttentionProcessor(nn.Module):
    def __init__(
            self, hidden_size: int, num_layers: int, attention_heads: int,
            expansion_factor: int, max_seq_len: int, dropout: float
        ):
        super().__init__()
        # 1. Position Encoding (Crucial for the Transformer part)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 2. Transformer Block (Global Context)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * expansion_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        # Add Positional Information
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        x = self.transformer(x) # (B, T, H)

        return x
    
class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Captures 3-day, 5-day, and 7-day micro-trends
        self.conv3 = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3)
        self.proj = nn.Linear((out_channels//3) * 3, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C) -> needs (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1).transpose(1, 2)
        return self.proj(out)
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # A learnable 'ideal' summary of a trading window
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

    def forward(self, x):
        # x: (B, T, H)
        # We attend to the sequence using our learnable query
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        # context: (B, 1, H)
        context, _ = self.attn(query, x, x)
        return context.squeeze(1)

class LightweightConvStem(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Depthwise: each feature is convolved independently (very few parameters)
        self.depthwise = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        # Pointwise: mixes the features
        self.pointwise = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        return self.pointwise(x)