import torch
# import numpy as np
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary

@NNModelLibrary.register(category='gru')
class AttentionGRU(nn.Module):
    """AttentionLSTM Model"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_stocks: int,
        attention_heads: int,
        dropout: float = 0.2,
        equal_prior: bool = False
    ):
        """
        Initialize Attention LSTM object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input window.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            attention_heads (int): Number of attention heads.
            dropout (float): Dropout rate. Default = 0.2.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.equal_prior = equal_prior
        self.ln_lstm = nn.LayerNorm(hidden_size) # Normalizes LSTM output
        
        # Attention layer components
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=attention_heads,
            batch_first=True
        )
        self.gelu_act = nn.GELU()
        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for the neural network.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        out, _ = self.gru(x)  # (B, T, hidden)
        
        out = self.ln_lstm(out)
        out = self.gelu_act(out)
        out = self.dropout(out)
        
        attn_out, _ = self.attn(out, out, out)  # (B, T, H)
        
        # Residual Connection + Norm (Standard Transformer Block trick)
        # We add the input (out) to the output (attn_out) to help gradients flow
        attn_out = out + attn_out 
        attn_out = self.ln_attn(attn_out)
        
        # Pooling
        context = attn_out.mean(dim=1)
        context = self.dropout(context)
        
        logits = self.fc(context)  # (B, N)
        
        pf_weights = torch.softmax(logits, dim=-1)
        return pf_weights