import torch
# import numpy as np
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary
from src.models.layers.encoders import LSTMEncoder, GlobalAttentionProcessor
from src.models.layers.TFT_vsn import (
    VariableSelectionNetwork,
    GatedResidualNetwork
)

@NNModelLibrary.register(category='transformer')
class TemporalTransformer(nn.Module):
    """
    Hybrid Model: LSTM for local temporal features + Transformer for global attention.
    """
    def __init__(
        self,
        input_size: int,       # 251 features
        hidden_size: int,      # Embedding dimension
        lstm_layers: int,       # LSTM layers
        trans_layers: int,      # Transformer layers
        num_stocks: int,       # 50 stocks
        nheads: int,
        dropout: float,
        expansion_factor: int,
        max_seq_len: int
    ):
        super().__init__()
        
        # 1. Feature Projection (Initial step to clean up features), kind of denoising
        self.feature_proj = nn.Linear(input_size, hidden_size)
        
        self.lstm_encoder = LSTMEncoder(hidden_size, lstm_layers, dropout)
        
        self.glob_attn = GlobalAttentionProcessor(
            hidden_size, trans_layers, nheads, expansion_factor, max_seq_len, dropout
        )
        # Output Head
        # self.ln_final = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.ones(hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, 251)
        
        # Initial Projection
        x = self.feature_proj(x)
        
        # Step 1: LSTM local processing
        # This helps the Transformer 'see' the sequence as a flow
        x = self.lstm_encoder(x)
        
        # Step 2: Transformer Global Attention
        # Every day now looks at every other day through the lens of the LSTM output
        x = self.glob_attn(x)
        
        # Step 3: Pooling
        # Mean pooling the context of the whole 120-day window
        context = x.mean(dim=1)
        # context = self.ln_final(context)

        # STep 4: Scaling
        context = context * self.alpha # Scale it without centering or standardizing
        context = self.dropout(context)
        
        # Step 5: Portfolio Allocation
        logits = self.fc(context)  # (B, N)
        return torch.softmax(logits, dim=-1)


@NNModelLibrary.register(category='transformer')
class TFT(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_stocks: int,
            attention_heads: int,
            dropout: float,
            expansion_factor: int,
            max_seq_len: int
        ):
        super().__init__()
        
        # 1. Feature Selection Layer (VSN)
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)
        
        # 2. Position Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Temporal Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * expansion_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Gating & Output
        self.post_attention_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, 251)
        
        # Filter 251 features down to hidden_size
        x = self.vsn(x) 
        
        # Add position context
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Temporal context (Attention)
        x = self.transformer(x)
        
        # Regulate attention output with gating
        x = self.post_attention_grn(x)
        
        # Mean Pooling for stability
        context = x.mean(dim=1)
        
        logits = self.fc(context)
        return torch.softmax(logits, dim=-1)


# @NNModelLibrary.register(category='transformer')
class PatchTST(nn.Module):
    def __init__(
            self, 
            input_size: int,
            hidden_size: int,
            num_stocks: int,
            patch_size: int,
            stride: int, 
            seq_len: int
        ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // stride
        
        # 1. Patching Linear Layer
        # Takes a patch of 5 days across 251 features and projects it
        self.patch_projection = nn.Linear(input_size * patch_size, hidden_size)
        
        # 2. Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, 60, 251)
        B, T, C = x.shape
        
        # Create patches: (Batch, Num_Patches, Patch_Size * Features)
        x = x.unfold(1, self.patch_size, self.patch_size) # Extract patches
        x = x.reshape(B, self.num_patches, -1) 
        
        x = self.patch_projection(x) # (B, 12, hidden_size)
        x = self.transformer(x)
        
        # Pooling: Use the context of the most recent patches
        context = x.mean(dim=1) 
        return torch.softmax(self.fc(context), dim=-1)