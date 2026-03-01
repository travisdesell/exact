import torch
# import numpy as np
import torch.nn as nn
from torch.nn.functional import elu
from torch import Tensor
from src.models.registry import NNModelLibrary


@NNModelLibrary.register(category='transformer')
class TemporalTransformerEncoder(nn.Module):
    """
    SOTA-style Transformer for Portfolio Optimization.
    Replaces LSTM with Learned Positional Encodings and Transformer Blocks.
    """
    def __init__(
        self,
        input_size: int,    # Number of features (251)
        hidden_size: int,   # d_model
        num_layers: int,    # Number of Transformer blocks
        num_stocks: int,    # Output size
        attention_heads: int,
        dropout: float,
        max_seq_len: int # Length of your lookback window
    ):
        super().__init__()
        
        # 1. Feature Projection: Projects 251 features to hidden_size
        self.feature_projection = nn.Linear(input_size, hidden_size)
        
        # 2. Learned Positional Encoding
        # Financial data is sequential; the model needs to know 'when' a bar happened
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu' # GELU is standard for SOTA Transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Global LayerNorm
        self.ln_final = nn.LayerNorm(hidden_size)
        
        # 5. Output Head
        self.fc = nn.Linear(hidden_size, num_stocks)
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (Batch, Time, Features)
        
        # Project features to embedding space
        x = self.feature_projection(x) # (B, T, H)
        
        # Add Positional Encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Pass through Transformer Blocks
        # Self-attention allows every day in the window to look at every other day
        out = self.transformer_encoder(x) # (B, T, H)
        
        # Mean pooling to average context of the whole window
        # context = out[:, -1, :] # # Pooling: The last time step's representation makes model sensitive to last day
        context = out.mean(dim=1)
        context = self.ln_final(context)
        
        # Generate Portfolio Logits
        logits = self.fc(context) # (B, N)
        
        # Softmax to ensure weights sum to 1
        return torch.softmax(logits, dim=-1)

class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float
        ):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(input_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GLU-style gating: (Residual + (Transformation * Gating))
        # This helps the model "ignore" noisy features
        residual = self.gate(x)
        x = elu(self.lin1(x)) # ELU is standard for TFT
        x = self.dropout(self.lin2(x))
        gate = torch.sigmoid(residual)
        return self.ln(residual + (x * gate))

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Correct shape for broadcasting: (1, 1, F, H)
        # This allows it to skip B and T and multiply directly against F
        self.feature_weights = nn.Parameter(torch.randn(1, 1, input_size, hidden_size))
        self.feature_bias = nn.Parameter(torch.zeros(1, 1, input_size, hidden_size))
        
        self.selector_grn = GatedResidualNetwork(
            input_size * hidden_size, 
            hidden_size, 
            input_size, 
            dropout
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, 120, 251)
        b, t, f = x.shape
        
        # 1. Project each feature to hidden_size
        # x.unsqueeze(-1) is (B, T, F, 1)
        # Multiplication now aligns correctly: (B, T, 251, 1) * (1, 1, 251, H)
        var_outputs = x.unsqueeze(-1) * self.feature_weights + self.feature_bias # (B, T, F, H)
        
        # 2. Variable Selection Weights
        flattened = var_outputs.view(b, t, -1) # (B, T, F*H)
        
        # selector_grn returns (B, T, F)
        sparse_weights = torch.softmax(self.selector_grn(flattened), dim=-1) 
        sparse_weights = sparse_weights.unsqueeze(-1) # (B, T, F, 1)
        
        # 3. Weighted Sum across the Feature dimension
        # (B, T, F, 1) * (B, T, F, H) -> sum over F -> (B, T, H)
        return torch.sum(sparse_weights * var_outputs, dim=-2)


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
            dim_feedforward=hidden_size * 4,
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


@NNModelLibrary.register(category='transformer')
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