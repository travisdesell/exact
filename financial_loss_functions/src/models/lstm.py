import torch
import numpy as np
import torch.nn as nn

class BaseLSTM(nn.Module):
    """
    Implementation BaseLSTM Model
    Base line LSTM model
    """
    def __init__(
            self, input_size, hidden_size, num_layers, num_stocks, dropout=0.2
        ):
        """
        Initialize BaseLSTM model which inherits from `torch.nn.Module`

        @param input_size int Size of input window
        @param hidden_size int Number of nodes in hidden layers
        @param num_layers int Number of hidden layers
        @param num_stocks int Number of stocks in dataset. 
            It is the number of output nodes.
        @param dropout float Dropout rate. Default = 0.2
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.ln = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

        # nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass method
        @param x torch.tensor Input window for forward pass. Shape = (B, T, E)

        @return torch.tensor Portfolio allocation weights calcuated from the forward pass
        """
        out, _ = self.lstm(x)      # (B, T, hidden)
        last = out[:, -1, :]      # (B, hidden)

        # last = self.ln(last) # Layer Norm
        
        last = torch.relu(last)
        last = self.dropout(last)
        
        logits = self.fc(last)     # (B, N)
        # Strong equal-weight prior that never goes away
        equal_prior = torch.full_like(
            logits,
            fill_value=np.log(1.0 / logits.shape[-1]),
            device=logits.device
        )
        logits = logits + equal_prior
        weights = torch.softmax(logits, dim=-1)
        return weights

class AttentionLSTM(nn.Module):
    """AttentionLSTM Model"""
    def __init__(
        self, input_size, hidden_size, num_layers, num_stocks, dropout=0.2
    ):
        """
        Initialize Attention LSTM object which inherits from torch.nn.Module

        @param input_size int Size of input window
        @param hidden_size int Number of nodes in hidden layers
        @param num_layers int Number of hidden layers
        @param num_stocks int Number of stocks in dataset. 
            It is the number of output nodes.
        @param dropout float Dropout rate. Default = 0.2
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.ln_lstm = nn.LayerNorm(hidden_size) # Normalizes LSTM output
        
        # Attention layer components
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)
        
        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

        # nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass method
        @param x torch.tensor Input window for forward pass. Shape = (B, T, E)

        @return torch.tensor Portfolio allocation weights calcuated from the forward pass
        """
        out, _ = self.lstm(x)  # (B, T, hidden)
        
        out = self.ln_lstm(out)
        out = torch.relu(out)
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
        # Strong equal-weight prior that never goes away
        equal_prior = torch.full_like(
            logits,
            fill_value=np.log(1.0 / logits.shape[-1]),
            device=logits.device
        )
        logits = logits + equal_prior
        weights = torch.softmax(logits, dim=-1)
        return weights