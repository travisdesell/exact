import torch
from torch import nn
from torch import Tensor

class TemporalAttention(nn.Module):
    """Temporal self-attention with residual connection and layer norm.

    Applies multi-head self-attention to the input sequence, adds the original input (residual),
    and normalises the output.
    """
    def __init__(self, hidden_size: int, nheads: int, dropout: float):
        """Initialises the temporal attention module.

        Args:
            hidden_size (int): Feature dimension (d_model) used for attention.
            nheads (int): Number of attention heads.
            dropout (float): Dropout probability (currently unused in forward).
        """
        super().__init__()

        # Attention layer components
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=nheads,
            batch_first=True
        ) 

        self.ln_attn = nn.LayerNorm(hidden_size) # Normalizes Attention output
        # self.dropout = nn.Dropout(dropout) # Branch dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through temporal attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, hidden_size).

        Returns:
            torch.Tensor: Output of the same shape (B, T, hidden_size) after attention,
                residual addition and layer normalisation.
        """
        attn_out, _ = self.attn(x, x, x)  # (B, T, H)

        # Residual Connection + Norm (Standard Transformer Block trick)
        # We add the input (x) to the output (attn_out) to help gradients flow
        attn_out = x + attn_out 
        attn_out = self.ln_attn(attn_out)
        return attn_out 

class ContextualGate(nn.Module):
    """
    GRU-based gate that learns a context vector from market-wide signals (e.g., S&P500)
    and outputs a sigmoid gating signal to be multiplied with asset-specific features.
    """
    def __init__(
            self, context_in: int,context_hidden: int, context_layers: int, hidden_size: int
        ):
        """Initialises the contextual gating network.

        Args:
            context_in (int): Number of input features for the context (e.g., 1 for S&P500 returns).
            context_hidden (int): Hidden size of the GRU.
            context_layers (int): Number of GRU layers.
            hidden_size (int): Output dimension of the gate (matches asset feature dimension).
        """
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
        """Forward pass through the contextual gate.

        Args:
            global_data (torch.Tensor): Contextual time series, shape (B, T, context_in).

        Returns:
            torch.Tensor: Gate signal of shape (B, 1, hidden_size), broadcastable to asset features.
        """
        # market_data: (B, T, 1)
        _, h_n = self.context_gru(global_data)
        return self.gate(h_n[-1]).unsqueeze(1) # (B, 1, H)
    
class ContextualCNNGate(nn.Module):
    """
    CNN-based gate that extracts context using a 1D convolution, global average pooling,
    and a sigmoid projection. Outputs a per-asset gating vector.
    """
    def __init__(self, context_in: int, hidden_size: int, kernel_size: int):
        """Initialises the CNN contextual gate.

        Args:
            context_in (int): Number of input features for the context.
            hidden_size (int): Output dimension of the gate (matches asset feature dimension).
            kernel_size (int): Kernel size for the 1D convolution; padding is automatically set
                to keep the sequence length unchanged before pooling.
        """
        super().__init__()
        
        # Match out_channels to hidden_size directly to avoid the Linear layer bottleneck
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=context_in, 
                out_channels=hidden_size, # Output 16 features
                kernel_size=kernel_size, 
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(hidden_size), # Better than LayerNorm for CNNs on MPS
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()
        )
        
        # Use a single linear projection to get the gating scale
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, global_data):
        """Forward pass through the CNN gate.

        Args:
            global_data (torch.Tensor): Contextual time series, shape (B, T, context_in).

        Returns:
            torch.Tensor: Gate signal of shape (B, 1, hidden_size).
        """
        # global_data: (B, T, C) -> (B, C, T)
        x = global_data.transpose(1, 2).contiguous()
        
        context_features = self.conv(x) # (B, H)
        gate = self.gate_proj(context_features) # (B, H)
        
        return gate.unsqueeze(1) # (B, 1, H)

class TemporalEncoder(nn.Module):
    """
    Encodes a single stock's 180-day history into a feature vector using an LSTM,
    a Transformer encoder, and final layer normalisation. Designed to be applied
    to a batch of (N stocks x batch size) simultaneously.
    """
    def __init__(
            self, input_size: int, hidden_size:int, lstm_layers: int,
            trans_layers: int, nhead:int, dropout:float
        ):
        """Initialises the temporal encoder.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Hidden dimension for LSTM and Transformer.
            lstm_layers (int): Number of LSTM layers.
            trans_layers (int): Number of Transformer encoder layers.
            nhead (int): Number of attention heads in the Transformer.
            dropout (float): Dropout probability (applied in LSTM between layers and in Transformer).
        """
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
        """Forward pass of the temporal encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B * N, T, input_size), where
                B is batch size, N is number of stocks, T = 180 days.

        Returns:
            torch.Tensor: Encoded features of shape (B * N, T, hidden_size).
        """
        # x: (B*N, T, F)
        out, _ = self.lstm(x)
        out = torch.relu(out)
        
        out = self.transformer(out)
        
        return self.ln(out) # Summary of the stock's 'vibe'