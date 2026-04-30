import torch
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary
from src.models.layers.encoders import LSTMEncoder, GlobalAttentionProcessor
from src.models.layers.TFT_vsn import (
    VariableSelectionNetwork,
    GatedResidualNetwork
)

class LearnableTemporalWeight(nn.Module):
    """
    Learnable Temporal Weights that can be used for pooling.
    """
    def __init__(self, max_seq_len: int):
        """
        Initialize object to learn temporal weights for pooling.
        
        Args:
            max_seq_len (int): Maximum sequence length. Here, we use length of input window.
        """
        super().__init__()
        # Learnable weights, one for each day
        self.day_weights = nn.Parameter(torch.ones(max_seq_len))

    def forward(self, x: Tensor):
        # x: (B, T, H)
        # Apply weights to the T dimension
        w = torch.softmax(self.day_weights[:x.size(1)], dim=0)
        return torch.sum(x * w.view(1, -1, 1), dim=1)

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
        max_seq_len: int,
        equal_prior: bool = False,
        **kwargs
    ):
        """
        Initialize TemporalTransformer object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            lstm_layers (int): Number of hidden LSTM layers.
            trans_layers (int): Number of transformer encoder layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            expansion_factor (int): Expansion factor to calculate feedforward dimension of each
                transformer layer.
            max_seq_len (int): Maximum sequence length. Here, we use length of input window.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
        """
        super().__init__()
        
        # 1. Feature Projection (Initial step to clean up features), kind of denoising
        self.feature_proj = nn.Linear(input_size, hidden_size)
        
        self.lstm_encoder = LSTMEncoder(hidden_size, lstm_layers, dropout)
        
        self.glob_attn = GlobalAttentionProcessor(
            hidden_size, trans_layers, nheads, expansion_factor, max_seq_len, dropout
        )

        # self.temporal_pooler = LearnableTemporalWeight(max_seq_len)
        # self.attn_pooler = AttentionPooling(hidden_size)

        # Output Head
        # self.ln_final = nn.LayerNorm(hidden_size)
        # self.alpha = nn.Parameter(torch.ones(hidden_size))

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
        Forward pass method for the TemporalTransformer.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        # x: (B, T, 251)
        
        # Initial Projection
        x = self.feature_proj(x)
        
        # Step 1: LSTM local processing
        # This helps the Transformer 'see' the sequence as a flow
        x = self.lstm_encoder(x)
        
        # Step 2: Transformer Global Attention
        # Every day now looks at every other day through the lens of the LSTM output
        x = self.glob_attn(x)
        
        # x = lstm_x + trans_x
        # Step 3: Pooling
        # Mean pooling the context of the whole 120-day window
        context = x.mean(dim=1)

        # context = self.ln_final(context)

        # STep 4: Scaling
        # context = context * self.alpha # Scale it without centering or standardizing
        context = self.dropout(context)
        
        # Step 5: Portfolio Allocation
        logits = self.fc(context)  # (B, N)
        return torch.softmax(logits, dim=-1)

    def _recency_pooling(self, x: Tensor) -> Tensor:
        """
        Pooling that weights depending on how recent the data in the input tensor is.
        """
        # x shape: (B, T, H)
        T = x.size(1)
        # Create weights that increase linearly: [1, 2, 3, ... 120]
        weights = torch.linspace(0.5, 1.0, steps=T).to(x.device)
        weights = weights.view(1, T, 1) # Match dimensions
        
        # Weighted average
        return (x * weights).mean(dim=1)


# @NNModelLibrary.register(category='transformer')
class TFT(nn.Module):
    """Temporal Fusion Transformer"""
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_stocks: int,
            nheads: int,
            dropout: float,
            expansion_factor: int,
            max_seq_len: int,
            equal_prior: bool = False,
            **kwargs
        ):
        super().__init__()
        """
        Initialize TFT object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers for the transformer encoder.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            expansion_factor (int): Expansion factor to calculate feedforward dimension of each
                transformer layer.
            max_seq_len (int): Maximum sequence length. Here, we use length of input window.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
        """
        
        # 1. Feature Selection Layer (VSN)
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)
        
        # 2. Position Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Temporal Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dim_feedforward=hidden_size * expansion_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Gating & Output
        self.post_attention_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3) 
            
            # 2. Initialize bias to zero
            # Softmax(0) = 1/N
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for TFT.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
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
        input_size: int,          # 251 features
        hidden_size: int,         # d_model
        num_layers: int,
        num_stocks: int,          # output dimension
        patch_size: int,
        stride: int,
        nheads: int,
        dropout: float,
        expansion_factor: int,
        max_seq_len: int,
        equal_prior: bool = False,
        **kwargs
    ):
        """
        Initialize PatchTST object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers for the transformer encoder.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            patch_size (int): Patch size to break up the input window into patches.
            stride (int): Stride to slide the patching window to create overlapping or 
                non-overlapping patch windows.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            expansion_factor (int): Expansion factor to calculate feedforward dimension of each
                transformer layer.
            max_seq_len (int): Maximum sequence length. Here, we use length of input window.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        # Number of patches (non‑overlapping if stride == patch_size)
        self.num_patches = (max_seq_len - patch_size) // stride + 1

        # Patch projection
        self.patch_proj = nn.Linear(input_size * patch_size, hidden_size)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_size))

        # Transformer encoder with pre‑norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dim_feedforward=hidden_size * expansion_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True          # pre‑normalisation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, num_stocks)

        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
            # nn.init.constant_(self.fc.weight, 0.0)
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3) 
            
            # 2. Initialize bias to zero
            # Softmax(0) = 1/N
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method for PatchTST.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        B, T, C = x.shape

        # Extract patches: (B, num_patches, patch_size, C)
        patches = x.unfold(1, self.patch_size, self.stride)  # (B, num_patches, C, patch_size)
        patches = patches.permute(0, 1, 3, 2)                # (B, num_patches, patch_size, C)
        # Flatten patch into a vector
        patch_vec = patches.reshape(B, self.num_patches, -1) # (B, num_patches, patch_size*C)

        # Project to d_model
        x = self.patch_proj(patch_vec)                       # (B, num_patches, hidden_size)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :self.num_patches, :]

        # Transformer
        x = self.transformer(x)                              # (B, num_patches, hidden_size)

        # Pool (mean over patches) and output
        context = x.mean(dim=1)                              # (B, hidden_size)
        logits = self.fc(context)                            # (B, num_stocks)
        return torch.softmax(logits, dim=-1)