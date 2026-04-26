import pytest
import torch
import torch.nn as nn
from src.models.layers.relational import FeatureAttention, VariableSelectionLayer

@pytest.fixture
def sample_input_3d():
    """Batch, time, hidden: (4, 10, 8)"""
    torch.manual_seed(42)
    return torch.randn(4, 10, 8)

# -------------------- Tests for FeatureAttention -------------------- #
def test_feature_attention_shape(sample_input_3d):
    B, T, H = sample_input_3d.shape
    # max_seq_len = T (time dimension), hidden_size = H
    attn = FeatureAttention(max_seq_len=T, hidden_size=H, nheads=2, dropout=0.1)
    out = attn(sample_input_3d)
    assert out.shape == sample_input_3d.shape

def test_feature_attention_dropout(sample_input_3d):
    B, T, H = sample_input_3d.shape
    attn = FeatureAttention(max_seq_len=T, hidden_size=H, nheads=2, dropout=0.5)
    attn.train()
    out1 = attn(sample_input_3d)
    out2 = attn(sample_input_3d)
    # With dropout, outputs should differ (stochastic)
    assert not torch.allclose(out1, out2, atol=1e-5)

    attn.eval()
    out3 = attn(sample_input_3d)
    out4 = attn(sample_input_3d)
    # In eval mode, dropout disabled, so outputs deterministic
    assert torch.allclose(out3, out4, atol=1e-5)

def test_feature_attention_gradient_flow(sample_input_3d):
    B, T, H = sample_input_3d.shape
    attn = FeatureAttention(max_seq_len=T, hidden_size=H, nheads=2, dropout=0.1)
    x = sample_input_3d.clone().requires_grad_(True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in attn.parameters():
        assert param.grad is not None

def test_feature_attention_divisible_heads():
    # Test that nheads must divide max_seq_len
    B, T, H = 2, 12, 8
    # 12 divisible by 3 -> ok
    attn = FeatureAttention(max_seq_len=T, hidden_size=H, nheads=3, dropout=0.0)
    x = torch.randn(B, T, H)
    out = attn(x)
    assert out.shape == x.shape
    # 12 not divisible by 5 -> should raise error
    with pytest.raises(AssertionError, match="embed_dim must be divisible by num_heads"):
        attn = FeatureAttention(max_seq_len=T, hidden_size=H, nheads=5, dropout=0.0)
        _ = attn(x)

# -------------------- Tests for VariableSelectionLayer --------------------
def test_variable_selection_layer_shape(sample_input_3d):
    B, T, F = sample_input_3d.shape
    layer = VariableSelectionLayer(input_size=F, hidden_size=16, dropout=0.1)
    out = layer(sample_input_3d)
    assert out.shape == sample_input_3d.shape

def test_variable_selection_layer_weights_between_0_and_1(sample_input_3d):
    B, T, F = sample_input_3d.shape
    layer = VariableSelectionLayer(input_size=F, hidden_size=16, dropout=0.0)
    # Compute weights from context manually
    context = sample_input_3d.mean(dim=1)
    weights = layer.gate_net(context)  # (B, F)
    assert (weights >= 0).all() and (weights <= 1).all()

def test_variable_selection_layer_gradient_flow(sample_input_3d):
    B, T, F = sample_input_3d.shape
    layer = VariableSelectionLayer(input_size=F, hidden_size=16, dropout=0.1)
    x = sample_input_3d.clone().requires_grad_(True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in layer.parameters():
        assert param.grad is not None

def test_variable_selection_layer_dropout(sample_input_3d):
    B, T, F = sample_input_3d.shape
    layer = VariableSelectionLayer(input_size=F, hidden_size=16, dropout=0.5)
    layer.train()
    out1 = layer(sample_input_3d)
    out2 = layer(sample_input_3d)
    # Dropout in gate_net and final dropout should cause variation
    assert not torch.allclose(out1, out2, atol=1e-5)

    layer.eval()
    out3 = layer(sample_input_3d)
    out4 = layer(sample_input_3d)
    assert torch.allclose(out3, out4, atol=1e-5)

def test_variable_selection_layer_identity_when_all_weights_one():
    # If the gate_net outputs all ones, then weighted_x = x.
    # We can force it by initializing the last linear layer to produce ones.
    F = 4
    layer = VariableSelectionLayer(input_size=F, hidden_size=8, dropout=0.0)
    # Override the last linear layer to output ones
    layer.gate_net[-2] = nn.Linear(8, F)
    with torch.no_grad():
        layer.gate_net[-2].weight.fill_(0.0)
        layer.gate_net[-2].bias.fill_(1.0)
        # Set bias large so sigmoid ~1
        layer.gate_net[-2].bias.fill_(10.0)
    x = torch.randn(2, 5, F)
    out = layer(x)
    # Since weights are all ~1, out should be close to x
    assert torch.allclose(out, x, atol=1e-4)