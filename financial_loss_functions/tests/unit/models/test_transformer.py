import pytest
import torch
import torch.nn as nn
from src.models.transformer import (
    LearnableTemporalWeight,
    TemporalTransformer,
    TFT,
    PatchTST
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_input_3d():
    """Batch, time, features: (4, 120, 251)"""
    torch.manual_seed(42)
    return torch.randn(4, 120, 251)

@pytest.fixture
def default_kwargs():
    """Common hyperparameters for all models."""
    return {
        'input_size': 251,
        'hidden_size': 32,
        'num_layers': 2,           # for models that use num_layers (TFT, PatchTST)
        'lstm_layers': 2,          # for TemporalTransformer
        'trans_layers': 1,
        'num_stocks': 50,
        'nheads': 4,
        'dropout': 0.2,
        'expansion_factor': 4,
        'max_seq_len': 120,
        'equal_prior': False,
        # Additional for PatchTST
        'patch_size': 12,
        'stride': 12,
    }

# ----------------------------------------------------------------------
# LearnableTemporalWeight
# ----------------------------------------------------------------------
def test_learnable_temporal_weight_shape():
    B, T, H = 4, 10, 16
    layer = LearnableTemporalWeight(max_seq_len=20)
    x = torch.randn(B, T, H)
    out = layer(x)
    assert out.shape == (B, H)

def test_learnable_temporal_weight_weights_softmax():
    max_seq_len = 20
    layer = LearnableTemporalWeight(max_seq_len)
    # Initially all ones -> softmax gives uniform distribution
    w = torch.softmax(layer.day_weights[:10], dim=0)
    assert torch.allclose(w, torch.full((10,), 1/10, device=w.device), atol=1e-6)

def test_learnable_temporal_weight_gradient():
    B, T, H = 2, 5, 4
    layer = LearnableTemporalWeight(max_seq_len=10)
    x = torch.randn(B, T, H, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.day_weights.grad is not None
    assert x.grad is not None

# ----------------------------------------------------------------------
# TemporalTransformer
# ----------------------------------------------------------------------
def test_temporal_transformer_forward_shape(sample_input_3d, default_kwargs):
    model = TemporalTransformer(**default_kwargs)
    out = model(sample_input_3d)
    assert out.shape == (sample_input_3d.shape[0], default_kwargs['num_stocks'])

def test_temporal_transformer_equal_prior_init(default_kwargs):
    kwargs = default_kwargs.copy()
    kwargs['equal_prior'] = True
    model = TemporalTransformer(**kwargs)
    assert torch.allclose(model.fc.weight, torch.zeros_like(model.fc.weight), atol=1e-3)
    assert torch.allclose(model.fc.bias, torch.zeros_like(model.fc.bias), atol=1e-6)

def test_temporal_transformer_gradient_flow(sample_input_3d, default_kwargs):
    model = TemporalTransformer(**default_kwargs)
    x = sample_input_3d.clone().requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None

def test_temporal_transformer_output_softmax(sample_input_3d, default_kwargs):
    model = TemporalTransformer(**default_kwargs)
    out = model(sample_input_3d)
    assert torch.allclose(out.sum(dim=-1), torch.ones(sample_input_3d.shape[0]), atol=1e-6)
    assert (out >= 0).all() and (out <= 1).all()

# ----------------------------------------------------------------------
# TFT
# ----------------------------------------------------------------------
def test_tft_forward_shape(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    # TFT uses num_layers for transformer layers
    model = TFT(**kwargs)
    out = model(sample_input_3d)
    assert out.shape == (sample_input_3d.shape[0], kwargs['num_stocks'])

def test_tft_equal_prior_init(default_kwargs):
    kwargs = default_kwargs.copy()
    kwargs['equal_prior'] = True
    model = TFT(**kwargs)
    assert torch.allclose(model.fc.weight, torch.zeros_like(model.fc.weight), atol=1e-3)
    assert torch.allclose(model.fc.bias, torch.zeros_like(model.fc.bias), atol=1e-6)

def test_tft_gradient_flow(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    model = TFT(**kwargs)
    x = sample_input_3d.clone().requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None

def test_tft_output_softmax(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    model = TFT(**kwargs)
    out = model(sample_input_3d)
    assert torch.allclose(out.sum(dim=-1), torch.ones(sample_input_3d.shape[0]), atol=1e-6)
    assert (out >= 0).all() and (out <= 1).all()

# ----------------------------------------------------------------------
# PatchTST
# ----------------------------------------------------------------------
def test_patchtst_forward_shape(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    model = PatchTST(**kwargs)
    out = model(sample_input_3d)
    assert out.shape == (sample_input_3d.shape[0], kwargs['num_stocks'])

def test_patchtst_num_patches_calculation(default_kwargs):
    max_seq_len = 120
    patch_size = 12
    stride = 12
    expected_patches = (max_seq_len - patch_size) // stride + 1  # = 10
    kwargs = default_kwargs.copy()
    kwargs['patch_size'] = patch_size
    kwargs['stride'] = stride
    model = PatchTST(**kwargs)
    assert model.num_patches == expected_patches

def test_patchtst_equal_prior_init(default_kwargs):
    kwargs = default_kwargs.copy()
    kwargs['equal_prior'] = True
    model = PatchTST(**kwargs)
    assert torch.allclose(model.fc.weight, torch.zeros_like(model.fc.weight), atol=1e-3)
    assert torch.allclose(model.fc.bias, torch.zeros_like(model.fc.bias), atol=1e-6)

def test_patchtst_gradient_flow(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    model = PatchTST(**kwargs)
    x = sample_input_3d.clone().requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None

def test_patchtst_output_softmax(sample_input_3d, default_kwargs):
    kwargs = default_kwargs.copy()
    model = PatchTST(**kwargs)
    out = model(sample_input_3d)
    assert torch.allclose(out.sum(dim=-1), torch.ones(sample_input_3d.shape[0]), atol=1e-6)
    assert (out >= 0).all() and (out <= 1).all()

def test_patchtst_different_patch_stride(default_kwargs):
    kwargs = default_kwargs.copy()
    kwargs['patch_size'] = 8
    kwargs['stride'] = 4  # overlapping patches
    model = PatchTST(**kwargs)
    x = torch.randn(2, 120, 251)
    out = model(x)
    assert out.shape == (2, kwargs['num_stocks'])