import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import elu

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) as used in the Temporal Fusion Transformer (TFT).

    Applies a gating mechanism: output = LayerNorm( residual + transformed * gate ),
    where residual is a linear projection of the input, transformed is a two-layer MLP with ELU,
    and gate is the sigmoid of the residual.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float
        ):
        """Initialises the GRN.

        Args:
            input_size (int): Dimension of the input features.
            hidden_size (int): Hidden dimension of the MLP.
            output_size (int): Output dimension (matches residual projection).
            dropout (float): Dropout rate applied after the second linear layer.
        """
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(input_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the GRN.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, input_size) or (B, input_size).

        Returns:
            torch.Tensor: Output of shape (B, T, output_size) or (B, output_size), after layer norm.
        """
        # GLU-style gating: (Residual + (Transformation * Gating))
        # This helps the model "ignore" noisy features
        residual = self.gate(x)
        x = elu(self.lin1(x)) # ELU is standard for TFT
        x = self.dropout(self.lin2(x))
        gate = torch.sigmoid(residual)
        return self.ln(residual + (x * gate))

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network that learns a weighted combination of input features.

    For each time step, it projects each feature independently into a hidden space,
    computes a sparse softmax weight per feature, and returns a weighted sum over features.
    The weights are learned through a GRN that takes the concatenated hidden projections.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> Tensor:
        """Initialises the Variable Selection Network.

        Args:
            input_size (int): Number of input features (F).
            hidden_size (int): Hidden dimension for per-feature projection.
            dropout (float): Dropout rate used inside the GRN selector.
        """
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the variable selection network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_size), which is the
                weighted sum of per-feature hidden projections, weighted by the
                learned feature importance.
        """
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