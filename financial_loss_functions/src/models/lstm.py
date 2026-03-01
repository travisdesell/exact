import torch
# import numpy as np
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary

#### All models MUST get a registration decorator with a category.
#### Here category will mostly be the file name.

@NNModelLibrary.register(category='lstm')
class BaseLSTM(nn.Module):
    """
    Implementation BaseLSTM Model with an initial equal prior option.
    """
    def __init__(
            self,
            input_size: int, 
            hidden_size: int, 
            num_layers: int,
            num_stocks: int,
            dropout: float = 0.2,
            equal_prior: bool = False
        ):
        """
        Initialize BaseLSTM model which inherits from `torch.nn.Module`

        Args:
            input_size (int): Size of input window.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            dropout (float): Dropout rate. Default = 0.2.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.equal_prior = equal_prior
        # self.ln = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
            # nn.init.constant_(self.fc.weight, 0.0)
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3) 
            
            # 2. Initialize bias to zero
            # Softmax(0) = 1/N
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for the neural network.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        out, _ = self.lstm(x)      # (B, T, hidden)
        last = out[:, -1, :]      # (B, hidden)

        # last = self.ln(last) # Layer Norm
        
        last = torch.relu(last)
        last = self.dropout(last)
        
        logits = self.fc(last)     # (B, N)
        
        pf_weights = torch.softmax(logits, dim=-1)
        return pf_weights

@NNModelLibrary.register(category='lstm')
class AttentionLSTM(nn.Module):
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
        self.lstm = nn.LSTM(
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
        
        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)
        
        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
            # nn.init.constant_(self.fc.weight, 0.0)
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3) 
            
            # 2. Initialize bias to zero
            # Softmax(0) = 1/N
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for the neural network.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
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
        
        pf_weights = torch.softmax(logits, dim=-1)
        return pf_weights

@NNModelLibrary.register(category='lstm')
class LSTMTransformer(nn.Module):
    """
    Hybrid Model: LSTM for local temporal features + Transformer for global attention.
    """
    def __init__(
        self,
        input_size: int,       # 251 features
        hidden_size: int,      # Embedding dimension
        num_layers: int,       # LSTM layers
        num_stocks: int,       # 50 stocks
        attention_heads: int,
        dropout: float,
        expansion_factor: int,
        max_seq_len: int
    ):
        super().__init__()
        
        # 1. Feature Projection (Initial step to clean up features)
        self.feature_proj = nn.Linear(input_size, hidden_size)
        
        # 2. LSTM Layer (Local Temporal Smoothing)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Position Encoding (Crucial for the Transformer part)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 4. Transformer Block (Global Context)
        # Replacing simple Attention with a full Encoder Layer (includes FFN + Norms)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * expansion_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. Output Head
        self.ln_final = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, 251)
        
        # Initial Projection
        x = self.feature_proj(x)
        
        # Step 1: LSTM local processing
        # This helps the Transformer 'see' the sequence as a flow
        x, _ = self.lstm(x) # (B, T, H)
        
        # Step 2: Add Positional Information
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Step 3: Transformer Global Attention
        # Every day now looks at every other day through the lens of the LSTM output
        x = self.transformer(x) # (B, T, H)
        
        # Step 4: Pooling
        # Mean pooling the context of the whole 120-day window
        context = x.mean(dim=1)
        context = self.ln_final(context)
        context = self.dropout(context)
        
        # Step 5: Portfolio Allocation
        logits = self.fc(context)  # (B, N)
        return torch.softmax(logits, dim=-1)