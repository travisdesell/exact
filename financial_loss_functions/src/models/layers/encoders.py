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