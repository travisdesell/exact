import torch
# import numpy as np
import math
import torch.nn as nn
from torch import Tensor

class LearnableSpectralFilter(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.seq_len = seq_len
        # Learnable filter in frequency domain (real-valued)
        self.filter = nn.Parameter(torch.ones(seq_len // 2 + 1, hidden_size))
        self.alpha = nn.Parameter(torch.tensor(0.1))  # scaling

    def forward(self, x):
        # x: (B, T, H)
        x_fft = torch.fft.rfft(x, dim=1)               # (B, T/2+1, H)
        x_fft = x_fft * (self.filter.unsqueeze(0))     # apply filter
        x_filtered = torch.fft.irfft(x_fft, n=self.seq_len, dim=1)
        return x + self.alpha * x_filtered              # residual

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

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Generates encodings of shape (1, max_seq_len, hidden_size).
    """
    def __init__(self, hidden_size: int, max_seq_len: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_size % 2 == 1:
            # if hidden_size is odd, the last dimension gets sin (0::2 covers up to last-1)
            pe[:, 1::2] = torch.cos(position * div_term)[:, :(hidden_size-1)//2]  # adjust for odd
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_seq_len, hidden_size)
        self.register_buffer('pe', pe)  # not a parameter, not updated during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            x + positional encoding (truncated to seq_len)
        """
        return x + self.pe[:, :x.size(1), :]

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
        
class DenoisingConv1d(nn.Module):
    """
    Lightweight depthwise 1D convolution for temporal smoothing/denoising.
    Preserves input shape (B, T, H) and uses residual connection + layer norm.
    """
    def __init__(self, hidden_size: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size  # depthwise: each channel processed independently
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, H)
        x_conv = self.conv(x.transpose(1, 2))  # (B, H, T)
        x_conv = x_conv.transpose(1, 2)        # (B, T, H)
        x = x + self.dropout(self.activation(x_conv))
        return self.norm(x)
    
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