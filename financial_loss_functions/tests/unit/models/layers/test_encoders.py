import math
import torch
import pytest
import torch.nn as nn
from src.models.layers.encoders import (
    LSTMEncoder,
    SinusoidalPositionalEncoding,
    GlobalAttentionProcessor,
    DenoisingConv1d,
    ConvStem,
    AttentionPooling,
    LightweightConvStem
)

@pytest.fixture
def sample_input():
    """Creates a random tensor of shape (batch_size, seq_len, hidden_size)"""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 10
    hidden_size = 16
    return torch.randn(batch_size, seq_len, hidden_size)

# ------------------- Tests for LSTMEncoder ------------------- #
@pytest.fixture
def lstm_encoder():
    """Creates an LSTMEncoder instance with default parameters"""
    return LSTMEncoder(hidden_size=16, num_layers=2, dropout=0.2)

def test_lstm_encoder_forward_shape(lstm_encoder, sample_input):
    """Test that output shape matches input shape"""
    output = lstm_encoder(sample_input)
    assert output.shape == sample_input.shape, f'Expected {sample_input.shape}, got {output.shape}'

def test_lstm_encoder_forward_no_error(lstm_encoder, sample_input):
    """Test that forward pass runs without exceptions"""
    try:
        output = lstm_encoder(sample_input)
        assert output is not None
    except Exception as e:
        pytest.fail(f'Forward pass failed with exception: {e}')

def test_lstm_encoder_dropout_train_vs_eval(lstm_encoder, sample_input):
    """Test that dropout behaves differently in train vs eval mode"""
    lstm_encoder.train()
    out1 = lstm_encoder(sample_input)
    out2 = lstm_encoder(sample_input)
    # With dropout, outputs should be different (due to randomness)
    assert not torch.allclose(out1, out2, atol=1e-6), 'Dropout not active in train mode'

    lstm_encoder.eval()
    out3 = lstm_encoder(sample_input)
    out4 = lstm_encoder(sample_input)
    # In eval mode, dropout should be deterministic, so outputs should be identical
    assert torch.allclose(out3, out4, atol=1e-6), 'Dropout not deterministic in eval mode'

def test_lstm_encoder_gradient_flow(lstm_encoder, sample_input):
    """Test that gradients can flow through the encoder"""
    lstm_encoder.train()
    output = lstm_encoder(sample_input)
    loss = output.sum()
    loss.backward()
    # Check that at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in lstm_encoder.parameters())
    assert has_grad, 'No gradients computed'

def test_lstm_encoder_single_layer_dropout():
    """Test that dropout is correctly set to 0 when num_layers=1"""
    encoder_single = LSTMEncoder(hidden_size=16, num_layers=1, dropout=0.3)
    # The LSTM's dropout parameter should be 0 because dropout only applies between layers
    assert encoder_single.lstm.dropout == 0, 'Dropout should be 0 for single layer LSTM'

def test_lstm_encoder_multi_layer_dropout():
    """Test that dropout is correctly passed to LSTM for multi-layer"""
    dropout_val = 0.4
    encoder_multi = LSTMEncoder(hidden_size=16, num_layers=3, dropout=dropout_val)
    assert encoder_multi.lstm.dropout == dropout_val, f'Expected dropout {dropout_val}, got {encoder_multi.lstm.dropout}'

# ------------------- Tests for SinusoidalPositionalEncoding ------------------- #
def test_sinusoidal_encoding_shape(sample_input):
    hidden_size = sample_input.shape[-1]
    max_seq_len = 20
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    output = encoding(sample_input)
    assert output.shape == sample_input.shape, f'Output shape {output.shape} != input shape {sample_input.shape}'

def test_sinusoidal_encoding_buffer_not_trainable():
    hidden_size = 8
    max_seq_len = 10
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    # The positional encoding should be a buffer, not a parameter
    assert 'pe' in encoding._buffers, 'Positional encoding not registered as buffer'
    assert not any(p.requires_grad for p in encoding.parameters()), 'Encoding has trainable parameters'

def test_sinusoidal_encoding_values():
    hidden_size = 4
    max_seq_len = 3
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    pe = encoding.pe.squeeze(0).cpu().numpy()  # (3,4)
    # Expected: For even indices, sin; for odd, cos
    # We can manually compute for a specific position and dimension
    # For hidden_size=4, indices: 0: sin, 1: cos, 2: sin, 3: cos
    position = 1
    dim0 = 0
    # Compute manually
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
    sin_val = torch.sin(torch.tensor(position).float() * div_term[0])
    # Compare with encoding
    assert abs(pe[1, 0] - sin_val.item()) < 1e-6, "Sin value mismatch"
    # For dim1 (cos)
    cos_val = torch.cos(torch.tensor(position).float() * div_term[0])
    assert abs(pe[1, 1] - cos_val.item()) < 1e-6, "Cos value mismatch"

def test_sinusoidal_encoding_odd_hidden_size():
    hidden_size = 5  # odd
    max_seq_len = 4
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    pe = encoding.pe.squeeze(0)  # (4,5)
    # The last dimension (index 4) should be filled with sin (since even indices cover 0,2,4)
    # The odd indices (1,3) get cos, but for odd size the cos slice is adjusted.
    assert pe.shape == (4, 5), 'Shape mismatch for odd hidden_size'
    # No NaN or inf
    assert not torch.isnan(pe).any(), 'NaN in encoding for odd hidden_size'

def test_sinusoidal_encoding_truncation(sample_input):
    hidden_size = sample_input.shape[-1]
    max_seq_len = 20
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    # Input seq_len is 10, which is less than max_seq_len
    output = encoding(sample_input)
    # Should only add the first 10 positions of the encoding
    expected_pe = encoding.pe[:, :10, :]
    # Check that output = input + expected_pe
    assert torch.allclose(output, sample_input + expected_pe), 'Positional encoding not added correctly'

def test_sinusoidal_encoding_longer_input_than_max():
    hidden_size = 4
    max_seq_len = 5
    encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len)
    input_long = torch.randn(2, 7, hidden_size)
    with pytest.raises(RuntimeError, match='The size of tensor a .* must match the size of tensor b'):
        _ = encoding(input_long)

# ------------------- Tests for GlobalAttentionProcessor ------------------- #
@pytest.fixture
def processor(sample_input):
    """Creates a GlobalAttentionProcessor instance with parameters matching sample_input."""
    hidden_size = sample_input.shape[-1]
    max_seq_len = 20  # larger than seq_len to allow truncation
    return GlobalAttentionProcessor(
        hidden_size=hidden_size,
        num_layers=2,
        attention_heads=4,
        expansion_factor=4,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

def test_global_attention_processor_forward_shape(processor, sample_input):
    output = processor(sample_input)
    assert output.shape == sample_input.shape, f'Output shape {output.shape} != input shape {sample_input.shape}'

def test_global_attention_processor_forward_no_error(processor, sample_input):
    try:
        output = processor(sample_input)
        assert output is not None
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

def test_global_attention_processor_positional_embedding_trainable(processor):
    # The positional embedding should be a trainable parameter
    assert hasattr(processor, 'pos_embedding'), "Missing pos_embedding"
    assert processor.pos_embedding.requires_grad, 'Positional embedding not trainable'

def test_global_attention_processor_transformer_applied(processor, sample_input):
    # Run forward once, then modify the transformer's parameters and see if output changes
    # A simpler check: output should not be equal to input (since transformer modifies)
    output = processor(sample_input)
    # Input and output should not be identical (unless transformer is identity, which it's not)
    assert not torch.allclose(output, sample_input, atol=1e-5), 'Transformer seems to have no effect'

def test_global_attention_processor_layer_norm_present(processor, sample_input):
    output = processor(sample_input)
    # Layer norm should have normalized the output (mean near 0, std near 1 along feature dim)
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    # Check that mean is close to 0 and std close to 1 for each token (tolerance is loose due to randomness)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1), 'Layer norm not applied correctly'
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1), 'Layer norm not applied correctly'

def test_global_attention_processor_gradient_flow(processor, sample_input):
    processor.train()
    output = processor(sample_input)
    loss = output.sum()
    loss.backward()
    # Check that at least one parameter has gradient (e.g., transformer parameters)
    has_grad = any(p.grad is not None for p in processor.parameters())
    assert has_grad, 'No gradients computed'

def test_global_attention_processor_truncates_positional_embedding(sample_input):
    hidden_size = sample_input.shape[-1]
    max_seq_len = 5  # smaller than sample_input seq_len (10)
    processor = GlobalAttentionProcessor(
        hidden_size=hidden_size,
        num_layers=1,
        attention_heads=2,
        expansion_factor=2,
        max_seq_len=max_seq_len,
        dropout=0.0
    )
    # Input has seq_len=10, but max_seq_len=5. The forward method will attempt to slice pos_embedding up to x.size(1)=10,
    # which will cause an error because pos_embedding only has 5 positions. This is expected behaviour.
    with pytest.raises(RuntimeError, match='The size of tensor a .* must match the size of tensor b'):
        _ = processor(sample_input)

# ------------------- Tests for DenoisingConv1d ------------------- #
@pytest.fixture
def denoiser(sample_input):
    """Create DenoisingConv1d with default parameters matching input."""
    hidden_size = sample_input.shape[-1]
    return DenoisingConv1d(hidden_size=hidden_size, kernel_size=3, dropout=0.1)

def test_denoising_conv1d_forward_shape(denoiser, sample_input):
    output = denoiser(sample_input)
    assert output.shape == sample_input.shape, f'Output shape {output.shape} != input shape {sample_input.shape}'

def test_denoising_conv1d_residual_connection(denoiser, sample_input):
    # The forward pass includes x + something. We can test that output is not equal to input.
    output = denoiser(sample_input)
    assert not torch.allclose(output, sample_input, atol=1e-5), 'Residual connection or convolution not active'

def test_denoising_conv1d_depthwise_convolution(denoiser, sample_input):
    # Depthwise convolution has groups = hidden_size, so each input channel is processed separately.
    # Check that the conv layer is indeed depthwise.
    conv_layer = denoiser.conv
    assert conv_layer.groups == conv_layer.in_channels == conv_layer.out_channels, \
        'Not a depthwise convolution (groups != in_channels)'

def test_denoising_conv1d_gradient_flow(denoiser, sample_input):
    denoiser.train()
    output = denoiser(sample_input)
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in denoiser.parameters())
    assert has_grad, 'No gradients computed'

def test_denoising_conv1d_dropout_train_vs_eval(denoiser, sample_input):
    denoiser.train()
    out1 = denoiser(sample_input)
    out2 = denoiser(sample_input)
    # With dropout, outputs should differ (stochastic)
    assert not torch.allclose(out1, out2, atol=1e-5), 'Dropout not active in train mode'

    denoiser.eval()
    out3 = denoiser(sample_input)
    out4 = denoiser(sample_input)
    # In eval mode, dropout should be deterministic
    assert torch.allclose(out3, out4, atol=1e-5), 'Dropout not deterministic in eval mode'

def test_denoising_conv1d_odd_kernel_size():
    # Test with kernel_size=5, hidden_size=8
    hidden_size = 8
    kernel_size = 5
    conv = DenoisingConv1d(hidden_size, kernel_size, dropout=0.0)
    x = torch.randn(2, 6, hidden_size)
    out = conv(x)
    assert out.shape == x.shape, 'Shape mismatch with odd kernel size'

def test_denoising_conv1d_even_kernel_size_raises_error():
    hidden_size = 8
    kernel_size = 4
    conv = DenoisingConv1d(hidden_size, kernel_size, dropout=0.0)
    x = torch.randn(2, 6, hidden_size)
    with pytest.raises(RuntimeError):
        conv(x)


# ------------------- Tests for ConvStem ------------------- #
# @pytest.fixture
# def sample_input_cnn():
#     """Fixed input tensor of shape (batch_size, seq_len, in_channels)"""
#     torch.manual_seed(42)
#     batch_size = 4
#     seq_len = 10
#     in_channels = 8
#     return torch.randn(batch_size, seq_len, in_channels)

def test_conv_stem_forward_shape(sample_input):
    in_channels = sample_input.shape[-1]
    out_channels = 24  # must be divisible by 3
    stem = ConvStem(in_channels, out_channels)
    output = stem(sample_input)
    # Output should have shape (batch_size, seq_len, out_channels)
    expected_shape = (sample_input.shape[0], sample_input.shape[1], out_channels)
    assert output.shape == expected_shape, f'Expected {expected_shape}, got {output.shape}'

def test_conv_stem_out_channels_divisible_by_3():
    in_channels = 8
    out_channels = 32  # not divisible by 3
    stem = ConvStem(in_channels, out_channels)
    x = torch.randn(2, 5, 8)
    out = stem(x)
    assert out.shape[-1] == 32

def test_conv_stem_gradient_flow(sample_input):
    in_channels = sample_input.shape[-1]
    out_channels = 18
    stem = ConvStem(in_channels, out_channels)
    stem.train()
    output = stem(sample_input)
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in stem.parameters())
    assert has_grad, 'No gradients computed'

def test_conv_stem_padding_keeps_sequence_length(sample_input):
    in_channels = sample_input.shape[-1]
    out_channels = 12
    stem = ConvStem(in_channels, out_channels)
    seq_len = sample_input.shape[1]
    output = stem(sample_input)
    assert output.shape[1] == seq_len, f'Sequence length changed from {seq_len} to {output.shape[1]}'

def test_conv_stem_with_different_in_out_channels():
    # Test with in_channels=1, out_channels=9
    stem = ConvStem(1, 9)
    x = torch.randn(3, 8, 1)
    out = stem(x)
    assert out.shape == (3, 8, 9), 'Shape mismatch for in=1, out=9'

# ------------------- Tests for AttentionPooling ------------------- #
def test_attention_pooling_forward_shape(sample_input):
    hidden_size = sample_input.shape[-1]
    pool = AttentionPooling(hidden_size)
    output = pool(sample_input)
    # Output should be (batch_size, hidden_size)
    expected_shape = (sample_input.shape[0], hidden_size)
    assert output.shape == expected_shape, f'Expected {expected_shape}, got {output.shape}'

def test_attention_pooling_query_expands(sample_input):
    hidden_size = sample_input.shape[-1]
    pool = AttentionPooling(hidden_size)
    # Access the query parameter
    query = pool.query
    assert query.shape == (1, 1, hidden_size), 'Query shape incorrect'
    # After forward, the query should be expanded to batch size internally; we can't directly check, but test shape passes.
    output = pool(sample_input)
    assert output.shape[0] == sample_input.shape[0], 'Batch dimension not preserved'

def test_attention_pooling_gradient_flow(sample_input):
    hidden_size = sample_input.shape[-1]
    pool = AttentionPooling(hidden_size)
    pool.train()
    output = pool(sample_input)
    loss = output.sum()
    loss.backward()
    # Check that the query parameter and attention parameters have gradients
    assert pool.query.grad is not None, 'Query parameter has no gradient'
    assert any(p.grad is not None for p in pool.attn.parameters()), 'Attention parameters have no gradients'

def test_attention_pooling_not_simple_mean(sample_input):
    hidden_size = sample_input.shape[-1]
    pool = AttentionPooling(hidden_size)
    output = pool(sample_input)
    # Compute simple mean over time dimension
    mean_out = sample_input.mean(dim=1)
    # The attention output should not be equal to the mean (since it's a learned weighted sum)
    assert not torch.allclose(output, mean_out, atol=1e-5), 'Attention pooling output equals mean, likely not attending'

def test_attention_pooling_deterministic(sample_input):
    hidden_size = sample_input.shape[-1]
    pool = AttentionPooling(hidden_size)
    pool.eval()
    out1 = pool(sample_input)
    out2 = pool(sample_input)
    # In eval mode, with no randomness, outputs should be identical
    assert torch.allclose(out1, out2, atol=1e-6), 'Attention pooling not deterministic in eval mode'

def test_attention_pooling_batch_size_one():
    hidden_size = 8
    pool = AttentionPooling(hidden_size)
    x = torch.randn(1, 5, hidden_size)
    output = pool(x)
    assert output.shape == (1, hidden_size), 'Shape mismatch for batch size 1'

# ------------------- Tests for LightweightConvStem ------------------- #
def test_lightweight_conv_stem_forward_shape(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = 32
    stem = LightweightConvStem(in_dim, out_dim)
    output = stem(sample_input)
    expected_shape = (sample_input.shape[0], sample_input.shape[1], out_dim)
    assert output.shape == expected_shape, f'Expected {expected_shape}, got {output.shape}'

def test_lightweight_conv_stem_preserves_sequence_length(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = 16
    stem = LightweightConvStem(in_dim, out_dim)
    seq_len = sample_input.shape[1]
    output = stem(sample_input)
    assert output.shape[1] == seq_len, f'Sequence length changed from {seq_len} to {output.shape[1]}'

def test_lightweight_conv_stem_depthwise_configuration(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = 32
    stem = LightweightConvStem(in_dim, out_dim)
    depthwise = stem.depthwise
    # Should be a depthwise conv: groups == in_channels == out_channels
    assert depthwise.groups == depthwise.in_channels == depthwise.out_channels == in_dim, \
        'Depthwise convolution not correctly configured'

def test_lightweight_conv_stem_gradient_flow(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = 32
    stem = LightweightConvStem(in_dim, out_dim)
    stem.train()
    output = stem(sample_input)
    loss = output.sum()
    loss.backward()
    # Check that both depthwise and pointwise parameters have gradients
    assert stem.depthwise.weight.grad is not None, 'Depthwise conv has no gradient'
    assert stem.pointwise.weight.grad is not None, 'Pointwise linear has no gradient'

def test_lightweight_conv_stem_output_not_equal_to_input(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = in_dim  # same dimension
    stem = LightweightConvStem(in_dim, out_dim)
    output = stem(sample_input)
    # Even with same dimensions, output should differ due to convolution and linear projection
    assert not torch.allclose(output, sample_input, atol=1e-5), 'Output equals input, transformation not applied'

def test_lightweight_conv_stem_with_different_in_out(sample_input):
    in_dim = sample_input.shape[-1]
    out_dim = 8  # smaller output dimension
    stem = LightweightConvStem(in_dim, out_dim)
    output = stem(sample_input)
    assert output.shape[-1] == out_dim, f'Expected output dim {out_dim}, got {output.shape[-1]}'

def test_lightweight_conv_stem_batch_size_one():
    in_dim = 8
    out_dim = 16
    stem = LightweightConvStem(in_dim, out_dim)
    x = torch.randn(1, 5, in_dim)
    output = stem(x)
    assert output.shape == (1, 5, out_dim), 'Shape mismatch for batch size 1'