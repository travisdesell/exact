import torch
# import numpy as np
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary
from src.models.layers.relational import FeatureAttention
from src.models.layers.temporal import TemporalAttention, ContextualGate

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
            dropout: float,
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
        
        self.t_attn = TemporalAttention(hidden_size, attention_heads, dropout)
    
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
        
        attn_out = self.t_attn(out)
        
        # Pooling
        context = attn_out.mean(dim=1)
        context = self.dropout(context)
        
        logits = self.fc(context)  # (B, N)
        
        pf_weights = torch.softmax(logits, dim=-1)
        return pf_weights

@NNModelLibrary.register(category='lstm')
class InvertedAttentionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_stocks: int,
        attention_heads: int,
        dropout: float,
        max_seq_len: int, # Needed for the inverted Attention/Norm layers
    ):
        super().__init__()
        # 1. Temporal Extraction (Standard)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln_lstm = nn.LayerNorm(hidden_size)

        # 2. THE INVERSION: Attention now operates on the Time dimension (max_seq_len)
        # We treat each hidden node as a token, and its sequence over time as the 'embedding'
        self.attn = nn.MultiheadAttention(
            embed_dim=max_seq_len, 
            num_heads=attention_heads,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(max_seq_len)
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_stocks)

        # 3. Decision Space (Expansion FFN)
        # After inversion and pooling, we go from hidden_size -> stocks
        # self.ffn = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * expansion_factor),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size * expansion_factor, num_stocks)
        # )

    def forward(self, x: Tensor) -> Tensor:
        # Step 1: Standard LSTM processing
        # x shape: (Batch, Time, Features) -> (B, 120, 251)
        out, _ = self.lstm(x)  # (B, 120, hidden_size)
        out = self.ln_lstm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # Step 2: INVERT (Transpose)
        # Swap Time (120) and Hidden (32)
        # New shape: (Batch, hidden_size, Time) -> (B, 16, 120)
        out_inverted = out.transpose(1, 2)
        
        # Step 3: Feature-wise Attention
        # The model asks: "How do these hidden features correlate across the whole window?"
        attn_out, _ = self.attn(out_inverted, out_inverted, out_inverted)
        
        # Residual Connection on the inverted shape
        out_inverted = out_inverted + attn_out 
        out_inverted = self.ln_attn(out_inverted)
        
        # Step 4: Pooling across the temporal "embeddings"
        # We mean-pool the time dimension (dim=2) to get one vector per hidden feature
        context = out_inverted.mean(dim=-1) # (B, hidden_size)
        context = self.dropout(context)
        
        # Step 5: Final Portfolio Weights
        logits = self.fc(context) 
        return torch.softmax(logits, dim=-1)

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
        # x = torch.relu(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        
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

@NNModelLibrary.register(category='lstm')    
class BiAttentionLSTM(nn.Module):
    def __init__(
            self,
            num_stocks: int,
            feats_per_stock: int,
            num_global: int, 
            hidden_size: int, 
            lstm_layers: int,
            t_nheads: int,
            r_nheads: int,
            cont_hidden: int,
            cont_layers: int,
            dropout: float,
            max_seq_len: int,
            **kwargs
        ):
        super().__init__()

        self.N = num_stocks
        self.F = feats_per_stock
        self.hidden_size = hidden_size
        
        self.num_tick_feats = num_stocks * feats_per_stock
        
        self.C = num_global

        self.lstm = nn.LSTM(
            input_size=self.num_tick_feats,
            hidden_size=self.hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.ln = nn.LayerNorm(self.hidden_size) # Normalizes LSTM output
        
        self.t_attn = TemporalAttention(self.hidden_size, t_nheads, dropout)
        
        self.r_attn = FeatureAttention(max_seq_len, self.hidden_size, r_nheads, dropout)
        self.context_gate = ContextualGate(
            self.C, cont_hidden, cont_layers, self.hidden_size
        )
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, self.N)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for the neural network.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        B, T, _ = x.shape

        # # 1. Isolate and Fold
        # stock_data = x[:, :, :self.N*self.F].view(B, T, self.N, self.F).transpose(1, 2).reshape(B*self.N, T, self.F)
        # global_data = x[:, :, self.N*self.F:]

        # 1. Faster Folding
        # Ensure memory is contiguous after transpose for MPS speed
        stock_data = x[:, :, :self.num_tick_feats].contiguous() # (B, T, N*F)
        global_data = x[:, :, self.num_tick_feats:].contiguous() # (B, T, C)

        # 2. Extract Stock-Level Alpha (Time)
        stock_features, _ = self.lstm(stock_data) # (B*N, T, H)
        stock_features = self.ln(stock_features)
        stock_features = torch.relu(stock_features)
        stock_features = self.dropout(stock_features)

        
        t_out = self.t_attn(stock_features)
        r_out = self.r_attn(stock_features)

        attn_out = stock_features + t_out + r_out
        attn_out = self.ln(attn_out)
        # r_out = r_out.mean(dim=-1)

        # 6. Apply Market Regime Gate (Macro)
        gate = self.context_gate(global_data) # (B, 1, 16)
        final_rep = attn_out * gate # (B, N, 16)

        # 7. Final Allocation
        final_rep = attn_out.mean(dim=1) # (B, hidden_size)
        final_rep = self.dropout(final_rep)
        
        # fc maps (16 -> 1) for each of the 50 stocks
        logits = self.fc(final_rep)
        
        return torch.softmax(logits, dim=-1)

        