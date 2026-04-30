import torch
import torch.nn as nn
from torch import Tensor
from src.models.registry import NNModelLibrary
from src.models.layers.relational import (
    FeatureAttention, VariableSelectionLayer
)
from src.models.layers.temporal import (
    TemporalAttention,
    ContextualGate
)

#### All models MUST get a registration decorator with a category.
#### Here category will mostly be the file name.

@NNModelLibrary.register(category='lstm')
class BaseLSTM(nn.Module):
    """BaseLSTM Model"""
    def __init__(
            self,
            input_size: int, 
            hidden_size: int, 
            num_layers: int,
            num_stocks: int,
            dropout: float,
            equal_prior: bool = False,
            **kwargs
        ):
        """
        Initialize BaseLSTM model which inherits from `torch.nn.Module`

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            dropout (float): Dropout rate.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
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
        Forward pass method for the BaseLSTM.

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
        nheads: int,
        dropout: float = 0.2,
        equal_prior: bool = False,
        **kwargs
    ):
        """
        Initialize AttentionLSTM object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
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
        
        self.t_attn = TemporalAttention(hidden_size, nheads, dropout)
    
        self.dropout = nn.Dropout(dropout)
        
        # A learnable vector initialized to 1.0
        # self.alpha = nn.Parameter(torch.ones(hidden_size))
        
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
        Forward pass method for the AttentionLSTM.

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
        # context = self.final_ln(context)
        # context = context * self.alpha # Scale it without centering or standardizing
        context = self.dropout(context)
        
        logits = self.fc(context)  # (B, N)
        
        pf_weights = torch.softmax(logits, dim=-1)
        return pf_weights

@NNModelLibrary.register(category='lstm')
class InvertedAttentionLSTM(nn.Module):
    """InvertedAttentionLSTM"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_stocks: int,
        nheads: int,
        dropout: float,
        max_seq_len: int, # Needed for the inverted Attention/Norm layers,
        equal_prior: bool = False
    ):
        """
        Initialize InvertedAttentionLSTM object which inherits from `torch.nn.Module`.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            max_seq_len (int): Maximum sequence length. Here, we use length of input window.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
        """
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
            num_heads=nheads,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(max_seq_len)


        self.alpha = nn.Parameter(torch.ones(hidden_size))

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

        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3) 
            
            # 2. Initialize bias to zero
            # Softmax(0) = 1/N
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method for the InvertedAttentionLSTM.

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
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

        context = context * self.alpha # Scale it without centering or standardizing
        context = self.dropout(context)
        
        # Step 5: Final Portfolio Weights
        logits = self.fc(context) 
        return torch.softmax(logits, dim=-1)

# @NNModelLibrary.register(category='lstm')    
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
            cont_kernel: int,
            dropout: float,
            max_seq_len: int,
            equal_prior: bool,
            **kwargs
        ):
        super().__init__()

        self.hidden_size = hidden_size
        self.C = num_global
        
        self.num_tick_feats = num_stocks * feats_per_stock
        

        self.lstm = nn.LSTM(
            input_size=self.num_tick_feats,
            hidden_size=self.hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.lstm_ln = nn.LayerNorm(self.hidden_size) # Normalizes LSTM output
        
        self.t_attn = TemporalAttention(self.hidden_size, t_nheads, dropout)
        
        self.r_attn = FeatureAttention(max_seq_len, self.hidden_size, r_nheads, dropout)

        self.attn_ln = nn.LayerNorm(self.hidden_size)
        self.context_gate = ContextualGate(
            self.C, cont_hidden, cont_layers, self.hidden_size
        )
        # self.context_gate = ContextualCNNGate(self.C, cont_hidden, cont_kernel)
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_stocks)

        if equal_prior:
            # 1. Initialize weights to near-zero 
            # This makes the output independent of the hidden state at start
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
        stock_features = self.lstm_ln(stock_features)
        stock_features = nn.functional.gelu(stock_features)
        stock_features = self.dropout(stock_features)

        
        t_out = self.t_attn(stock_features)
        r_out = self.r_attn(stock_features)

        attn_out = stock_features + t_out + r_out
        attn_out = self.attn_ln(attn_out)
        # r_out = r_out.mean(dim=-1)

        # 6. Apply Market Regime Gate (Macro)
        gate = self.context_gate(global_data) # (B, 1, 16)
        final_rep = attn_out * gate # (B, N, 16)

        # 7. Final Allocation
        final_rep = final_rep.mean(dim=1) # (B, hidden_size)
        final_rep = self.dropout(final_rep)
        
        # fc maps (16 -> 1) for each of the 50 stocks
        logits = self.fc(final_rep)
        
        return torch.softmax(logits, dim=-1)

@NNModelLibrary.register(category='lstm')
class VLSTM(nn.Module):
    """
    Variable Selection Network + LSTM (VLSTM) for portfolio weight generation.
    - First, a variable selection layer learns per-feature importance.
    - Then an LSTM processes the weighted features.
    - Optional temporal attention aggregates over time.
    - Finally, a linear layer outputs portfolio weights.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_stocks: int,
        nheads: int,
        dropout: float = 0.2,
        equal_prior: bool = False,
        vsn_hidden_size: int | None = None,
        **kwargs
    ):
        """
        Initialize VSN-LSTM object which inherits from `torch.nn.Module`.
        In this project, to shorten the name, we call this VLSTM.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Number of nodes in hidden layers.
            num_layers (int): Number of hidden layers.
            num_stocks (int): Number of stocks in dataset.
                It is the number of output nodes.
            nheads (int): Number of attention heads.
            dropout (float): Dropout rate.
            equal_prior (bool): Initialize logits to 0 to start model with equal 
                portfolio allocation weights. This does not initialize internal
                models weights other than the final fully connected layer weights.
                Default = False.
            vsn_hidden_size (int): Hidden size of the Variable Selection Network. If None,
                the hidden size is calculate has 'hidden_size // 2'.
        """
        super().__init__()
        self.equal_prior = equal_prior

        if nheads != 0:
            self.use_attention = True
        else:
            self.use_attention = False

        # Set default vsn_hidden_dim to hidden_size // 2 if not provided
        if vsn_hidden_size is None:
            vsn_hidden_size = hidden_size // 2

        # Variable selection layer
        self.vsn = VariableSelectionLayer(
            input_size=input_size,
            hidden_size=vsn_hidden_size,
            dropout=dropout
        )

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        if self.use_attention:
            self.t_attn = TemporalAttention(hidden_size, nheads, dropout)

        self.ln_lstm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, num_stocks)

        if equal_prior:
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method for the VLSTM (VSN-LSTM).

        Args:
            x (Tensor): Input window for forward pass. Shape = (B, T, E)

        Returns:
            pf_weights (Tensor): Portfolio allocation weights calcuated from the forward pass.
        """
        # Step 1: Variable selection (feature weighting)
        x = self.vsn(x)                     # (B, T, F)

        # Step 2: LSTM
        out, _ = self.lstm(x)               # (B, T, hidden)

        # Step 3: Layer norm and activation
        out = self.ln_lstm(out)
        out = nn.functional.gelu(out)
        out = self.dropout(out)

        if self.use_attention:
            out = self.t_attn(out)       # (B, T, hidden)         

        # Step 5: Pooling (mean over time)
        context = out.mean(dim=1)           # (B, hidden)

        # Step 6: Final linear layer
        logits = self.fc(context)           # (B, num_stocks)
        return torch.softmax(logits, dim=-1)