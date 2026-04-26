import pytest
import torch
from src.models.layers.TFT_vsn import GatedResidualNetwork, VariableSelectionNetwork

@pytest.fixture
def sample_input_3d():
    """Batch, time, features: (4, 10, 16)"""
    torch.manual_seed(42)
    return torch.randn(4, 10, 16)

@pytest.fixture
def sample_input_2d():
    """Batch, features: (4, 16)"""
    torch.manual_seed(42)
    return torch.randn(4, 16)

# -------------------- Tests for GatedResidualNetwork -------------------- #
def test_grn_forward_shape_2d(sample_input_2d):
    input_size = sample_input_2d.shape[-1]
    hidden_size = 32
    output_size = 8
    grn = GatedResidualNetwork(input_size, hidden_size, output_size, dropout=0.1)
    out = grn(sample_input_2d)
    assert out.shape == (sample_input_2d.shape[0], output_size)

def test_grn_forward_shape_3d(sample_input_3d):
    B, T, F = sample_input_3d.shape
    input_size = F
    hidden_size = 32
    output_size = 16
    grn = GatedResidualNetwork(input_size, hidden_size, output_size, dropout=0.1)
    out = grn(sample_input_3d)
    assert out.shape == (B, T, output_size)

def test_grn_gradient_flow_3d(sample_input_3d):
    B, T, F = sample_input_3d.shape
    grn = GatedResidualNetwork(input_size=F, hidden_size=32, output_size=16, dropout=0.1)
    x = sample_input_3d.clone().requires_grad_(True)
    out = grn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in grn.parameters():
        assert param.grad is not None

def test_grn_output_range(sample_input_3d):
    # Output after layer norm can be any value, but we can check that it's not NaN.
    B, T, F = sample_input_3d.shape
    grn = GatedResidualNetwork(input_size=F, hidden_size=32, output_size=16, dropout=0.0)
    out = grn(sample_input_3d)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_grn_dropout(sample_input_2d):
    input_size = sample_input_2d.shape[-1]
    grn = GatedResidualNetwork(input_size, hidden_size=32, output_size=8, dropout=0.5)
    grn.train()
    out1 = grn(sample_input_2d)
    out2 = grn(sample_input_2d)
    # With dropout, outputs should differ
    assert not torch.allclose(out1, out2, atol=1e-5)
    grn.eval()
    out3 = grn(sample_input_2d)
    out4 = grn(sample_input_2d)
    assert torch.allclose(out3, out4, atol=1e-5)

# -------------------- Tests for VariableSelectionNetwork -------------------- #
def test_vsn_forward_shape(sample_input_3d):
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.1)
    out = vsn(sample_input_3d)
    # Output should be (B, T, hidden_size)
    assert out.shape == (B, T, hidden_size)

def test_vsn_gradient_flow(sample_input_3d):
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.1)
    x = sample_input_3d.clone().requires_grad_(True)
    out = vsn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in vsn.parameters():
        assert param.grad is not None

def test_vsn_sparse_weights_sum_to_one(sample_input_3d):
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.0)
    # To inspect the weights, we need to compute intermediate.
    # We'll compute the weights manually by calling the selector_grn.
    x = sample_input_3d
    b, t, f = x.shape
    # Compute var_outputs
    var_outputs = x.unsqueeze(-1) * vsn.feature_weights + vsn.feature_bias  # (B,T,F,H)
    flattened = var_outputs.view(b, t, -1)  # (B,T,F*H)
    weights = torch.softmax(vsn.selector_grn(flattened), dim=-1)  # (B,T,F)
    # Check that weights sum to 1 along feature dimension
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

def test_vsn_output_is_weighted_sum(sample_input_3d):
    # Verify that the output equals the weighted sum of var_outputs.
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.0)
    x = sample_input_3d
    b, t, f = x.shape
    var_outputs = x.unsqueeze(-1) * vsn.feature_weights + vsn.feature_bias
    flattened = var_outputs.view(b, t, -1)
    weights = torch.softmax(vsn.selector_grn(flattened), dim=-1).unsqueeze(-1)  # (B,T,F,1)
    expected = torch.sum(weights * var_outputs, dim=-2)  # (B,T,H)
    out = vsn(x)
    assert torch.allclose(out, expected, atol=1e-6)

def test_vsn_feature_weights_shape(sample_input_3d):
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.0)
    # Check that feature_weights has shape (1,1,F,H)
    assert vsn.feature_weights.shape == (1, 1, F, hidden_size)
    assert vsn.feature_bias.shape == (1, 1, F, hidden_size)

def test_vsn_dropout_in_selector_grn(sample_input_3d):
    B, T, F = sample_input_3d.shape
    hidden_size = 8
    vsn = VariableSelectionNetwork(input_size=F, hidden_size=hidden_size, dropout=0.5)
    vsn.train()
    out1 = vsn(sample_input_3d)
    out2 = vsn(sample_input_3d)
    # The selector_grn uses dropout, so outputs should differ
    assert not torch.allclose(out1, out2, atol=1e-5)
    vsn.eval()
    out3 = vsn(sample_input_3d)
    out4 = vsn(sample_input_3d)
    assert torch.allclose(out3, out4, atol=1e-5)