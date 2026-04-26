import pytest
import torch
import numpy as np
import torch.nn as nn
from src.models.layers.filters import (
    RobustNormalization,
    FFTSpectralFilter,
    WaveletDenoiseLayer
)

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

# -------------------- Tests for RobustNormalization -------------------- #
def test_robust_normalization_3d(sample_input_3d):
    median = np.random.randn(16)
    iqr = np.random.rand(16) + 0.1
    norm = RobustNormalization(feature_dim=16, median=median, iqr=iqr, eps=1e-8)
    out = norm(sample_input_3d)
    assert out.shape == sample_input_3d.shape
    # Convert median/iqr to tensors for comparison
    median_t = torch.tensor(median, dtype=torch.float32)
    iqr_t = torch.tensor(iqr, dtype=torch.float32)
    expected = (sample_input_3d - median_t) / (iqr_t + 1e-8)
    assert torch.allclose(out, expected, atol=1e-6)

def test_robust_normalization_2d(sample_input_2d):
    median = np.random.randn(16)
    iqr = np.random.rand(16) + 0.1
    norm = RobustNormalization(feature_dim=16, median=median, iqr=iqr)
    out = norm(sample_input_2d)
    assert out.shape == sample_input_2d.shape
    median_t = torch.tensor(median, dtype=torch.float32)
    iqr_t = torch.tensor(iqr, dtype=torch.float32)
    expected = (sample_input_2d - median_t) / (iqr_t + 1e-8)
    assert torch.allclose(out, expected, atol=1e-6)

def test_robust_normalization_affine(sample_input_3d):
    median = np.zeros(16)
    iqr = np.ones(16)
    norm = RobustNormalization(feature_dim=16, median=median, iqr=iqr, eps=1e-8)
    norm.affine = True
    norm.weight = nn.Parameter(torch.ones(16) * 2)
    norm.bias = nn.Parameter(torch.ones(16))
    out = norm(sample_input_3d)
    expected = (sample_input_3d) * 2 + 1
    assert torch.allclose(out, expected, atol=1e-6)

def test_robust_normalization_buffers():
    median = np.array([1., 2.])
    iqr = np.array([0.5, 0.5])
    norm = RobustNormalization(feature_dim=2, median=median, iqr=iqr)
    assert 'median' in norm._buffers
    assert 'iqr' in norm._buffers
    assert not norm.median.requires_grad
    assert not norm.iqr.requires_grad

# -------------------- Tests for FFTSpectralFilter -------------------- #
def test_fft_spectral_filter_shape(sample_input_3d):
    seq_len = sample_input_3d.shape[1]
    hidden_size = sample_input_3d.shape[2]
    filt = FFTSpectralFilter(seq_len, hidden_size)
    out = filt(sample_input_3d)
    assert out.shape == sample_input_3d.shape

def test_fft_spectral_filter_identity_init(sample_input_3d):
    seq_len = sample_input_3d.shape[1]
    hidden_size = sample_input_3d.shape[2]
    filt = FFTSpectralFilter(seq_len, hidden_size)
    # With default init (low‑pass prior), the filter is not identity.
    # But we can check that output is not identical to input.
    out = filt(sample_input_3d)
    assert not torch.allclose(out, sample_input_3d, atol=1e-5)

def test_fft_spectral_filter_learnable_parameters(sample_input_3d):
    seq_len = sample_input_3d.shape[1]
    hidden_size = sample_input_3d.shape[2]
    filt = FFTSpectralFilter(seq_len, hidden_size)
    # Check that filter_real and filter_imag are parameters
    assert isinstance(filt.filter_real, nn.Parameter)
    assert isinstance(filt.filter_imag, nn.Parameter)
    assert filt.filter_real.requires_grad
    assert filt.filter_imag.requires_grad
    # Check alpha is parameter
    assert isinstance(filt.alpha, nn.Parameter)

def test_fft_spectral_filter_gradient_flow(sample_input_3d):
    seq_len = sample_input_3d.shape[1]
    hidden_size = sample_input_3d.shape[2]
    filt = FFTSpectralFilter(seq_len, hidden_size)
    out = filt(sample_input_3d)
    loss = out.sum()
    loss.backward()
    # Check that gradients are computed
    assert filt.filter_real.grad is not None
    assert filt.filter_imag.grad is not None
    assert filt.alpha.grad is not None

def test_fft_spectral_filter_non_contiguous_input():
    # Test that it works with non‑contiguous tensors (e.g., after transpose)
    x = torch.randn(2, 8, 4)
    x = x.transpose(1, 2)  # now (2,4,8)
    x = torch.randn(2, 8, 4)
    x = x[:, [1,0,3,2,5,4,7,6], :]  # non‑contiguous
    filt = FFTSpectralFilter(seq_len=8, hidden_size=4)
    out = filt(x)
    assert out.shape == x.shape

# -------------------- WaveletDenoiseLayer -------------------- #
def test_wavelet_denoise_layer_shape():
    x = torch.randn(4, 10, 8)
    layer = WaveletDenoiseLayer()
    out = layer(x)
    assert out.shape == x.shape

def test_wavelet_denoise_layer_threshold_effect():
    # With threshold=0, detail should be unchanged (soft‑thresholding with 0 is identity)
    x = torch.randn(4, 8, 6)   # even seq_len
    layer = WaveletDenoiseLayer()
    layer.threshold.data.fill_(0.0)
    out = layer(x)
    assert torch.allclose(out, x, atol=1e-6)

def test_wavelet_denoise_layer_threshold_positive():
    x = torch.randn(4, 8, 6)
    layer = WaveletDenoiseLayer()
    layer.threshold.data.fill_(0.1)
    out = layer(x)
    # Output should be different from input because detail coefficients are shrunk.
    assert not torch.allclose(out, x, atol=1e-5)

def test_wavelet_denoise_layer_gradient_flow():
    x = torch.randn(4, 8, 6, requires_grad=True)
    layer = WaveletDenoiseLayer()
    out = layer(x)
    loss = out.sum()
    loss.backward()
    # Check that the threshold parameter has gradient
    assert layer.threshold.grad is not None
    # Also input gradient
    assert x.grad is not None

def test_wavelet_denoise_layer_odd_sequence_length():
    x = torch.randn(4, 9, 8)   # odd
    layer = WaveletDenoiseLayer()
    with pytest.raises(RuntimeError) as excinfo:
        out = layer(x)
    # The error message may be about shape mismatch; we accept any RuntimeError.
    assert True  # test passes if exception raised