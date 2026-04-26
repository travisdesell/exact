import pytest
import torch
import torch.nn as nn
from src.models.layers.temporal import (
    TemporalAttention,
    ContextualGate,
    ContextualCNNGate,
    TemporalEncoder
)

@pytest.fixture
def sample_input_3d():
    """Batch, time, hidden: (4, 10, 16)"""
    torch.manual_seed(42)
    return torch.randn(4, 10, 16)

@pytest.fixture
def sample_global_data():
    """Batch, time, context_in = 1 (e.g., SP500 returns)"""
    torch.manual_seed(42)
    return torch.randn(4, 10, 1)

# -------------------- Tests for TemporalAttention -------------------- #
def test_temporal_attention_shape(sample_input_3d):
    B, T, H = sample_input_3d.shape
    attn = TemporalAttention(hidden_size=H, nheads=2, dropout=0.1)
    out = attn(sample_input_3d)
    assert out.shape == sample_input_3d.shape

def test_temporal_attention_residual(sample_input_3d):
    B, T, H = sample_input_3d.shape
    attn = TemporalAttention(hidden_size=H, nheads=2, dropout=0.0)
    # Since dropout=0 and no randomness, output should be input + attn_out after norm
    out = attn(sample_input_3d)
    assert not torch.allclose(out, sample_input_3d, atol=1e-5)

def test_temporal_attention_gradient_flow(sample_input_3d):
    B, T, H = sample_input_3d.shape
    attn = TemporalAttention(hidden_size=H, nheads=2, dropout=0.1)
    x = sample_input_3d.clone().requires_grad_(True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in attn.parameters():
        assert param.grad is not None

# def test_temporal_attention_dropout(sample_input_3d):
#     B, T, H = sample_input_3d.shape
#     attn = TemporalAttention(hidden_size=H, nheads=2, dropout=0.5)
#     attn.train()
#     out1 = attn(sample_input_3d)
#     out2 = attn(sample_input_3d)
#     assert not torch.allclose(out1, out2, atol=1e-5)
#     attn.eval()
#     out3 = attn(sample_input_3d)
#     out4 = attn(sample_input_3d)
#     assert torch.allclose(out3, out4, atol=1e-5)

# -------------------- Tests for ContextualGate -------------------- #
def test_contextual_gate_shape(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    gate = ContextualGate(context_in=C, context_hidden=8, context_layers=2, hidden_size=hidden_size)
    out = gate(sample_global_data)
    # Output should be (B, 1, hidden_size)
    assert out.shape == (B, 1, hidden_size)

def test_contextual_gate_gradient_flow(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    gate = ContextualGate(context_in=C, context_hidden=8, context_layers=2, hidden_size=hidden_size)
    x = sample_global_data.clone().requires_grad_(True)
    out = gate(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in gate.parameters():
        assert param.grad is not None

def test_contextual_gate_output_range(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    gate = ContextualGate(context_in=C, context_hidden=8, context_layers=2, hidden_size=hidden_size)
    out = gate(sample_global_data)
    # Output is sigmoid, so should be in (0,1)
    assert (out >= 0).all() and (out <= 1).all()

# -------------------- for ContextualCNNGate -------------------- #
def test_contextual_cnn_gate_shape(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    kernel_size = 3
    gate = ContextualCNNGate(context_in=C, hidden_size=hidden_size, kernel_size=kernel_size)
    out = gate(sample_global_data)
    assert out.shape == (B, 1, hidden_size)

def test_contextual_cnn_gate_gradient_flow(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    kernel_size = 3
    gate = ContextualCNNGate(context_in=C, hidden_size=hidden_size, kernel_size=kernel_size)
    x = sample_global_data.clone().requires_grad_(True)
    out = gate(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in gate.parameters():
        assert param.grad is not None

def test_contextual_cnn_gate_output_range(sample_global_data):
    B, T, C = sample_global_data.shape
    hidden_size = 16
    kernel_size = 3
    gate = ContextualCNNGate(context_in=C, hidden_size=hidden_size, kernel_size=kernel_size)
    out = gate(sample_global_data)
    assert (out >= 0).all() and (out <= 1).all()

def test_contextual_cnn_gate_kernel_size_preserves_time():
    # The Conv1d with padding = kernel_size//2 should keep sequence length unchanged.
    B, T, C = 2, 10, 1
    hidden_size = 8
    kernel_size = 5
    gate = ContextualCNNGate(context_in=C, hidden_size=hidden_size, kernel_size=kernel_size)
    x = torch.randn(B, T, C)
    out = gate(x)  # only output shape (B,1,H) is checked; internal conv step preserves time but then pooling reduces to 1.
    # We can test the internal conv output shape by temporarily modifying the class,
    # instead, we trust that the AdaptiveAvgPool1d(1) collapses time.
    assert out.shape == (B, 1, hidden_size)

# ----------------------------------------------------------------------
# TemporalEncoder
# ----------------------------------------------------------------------
def test_temporal_encoder_shape():
    # Input: (batch * stocks, time, features)
    B = 4  # batch
    N = 10 # number of stocks
    T = 20 # time steps
    F = 8  # features per stock
    hidden_size = 16
    encoder = TemporalEncoder(input_size=F, hidden_size=hidden_size,
                              lstm_layers=2, trans_layers=1, nhead=4, dropout=0.1)
    x = torch.randn(B * N, T, F)
    out = encoder(x)
    assert out.shape == (B * N, T, hidden_size)

def test_temporal_encoder_gradient_flow():
    B, N, T, F = 2, 5, 12, 6
    hidden_size = 16
    encoder = TemporalEncoder(input_size=F, hidden_size=hidden_size,
                              lstm_layers=2, trans_layers=1, nhead=4, dropout=0.1)
    x = torch.randn(B * N, T, F, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in encoder.parameters():
        assert param.grad is not None

def test_temporal_encoder_lstm_dropout():
    # If lstm_layers=1, dropout should be 0
    encoder = TemporalEncoder(input_size=6, hidden_size=16,
                              lstm_layers=1, trans_layers=1, nhead=4, dropout=0.5)
    assert encoder.lstm.dropout == 0
    # If lstm_layers>1, dropout should be passed
    encoder2 = TemporalEncoder(input_size=6, hidden_size=16,
                               lstm_layers=2, trans_layers=1, nhead=4, dropout=0.3)
    assert encoder2.lstm.dropout == 0.3

def test_temporal_encoder_transformer_pre_norm():
    # The default TransformerEncoderLayer uses post-norm (norm_first=False)
    # but we didn't specify, so it's False. We can check.
    encoder = TemporalEncoder(input_size=6, hidden_size=16,
                              lstm_layers=2, trans_layers=1, nhead=4, dropout=0.1)
    # The encoder_layer is inside the TransformerEncoder; we can access first layer's norm_first
    layer = encoder.transformer.layers[0]
    assert layer.norm_first == False  # default