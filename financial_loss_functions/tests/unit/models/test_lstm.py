import torch
import pytest
import numpy as np
import torch.nn.functional as F
from src.models.lstm import BaseLSTM, AttentionLSTM
from src.utils.device import set_seed

# reuse seed helper from previous file if present; otherwise:
def seed_everything(seed=0):
    set_seed(seed)


def assert_probability_vector(weights: torch.Tensor, atol=1e-6):
    assert torch.isfinite(weights).all().item()
    assert (weights >= 0).all().item()
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=atol)

# -------------------- Tests for BaseLSTM class -------------------- #
def test_baselstm_constructor_attributes():
    """Constructor should create expected submodules and shapes."""
    input_size, hidden_size, num_layers, num_stocks = 4, 8, 2, 5
    model = BaseLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks, 
        dropout=0.1
    )

    # core modules present
    assert hasattr(model, 'lstm'), 'missing lstm'
    assert hasattr(model, 'fc'), 'missing fc'
    assert hasattr(model, 'dropout'), 'missing dropout'

    # shape properties on modules
    assert model.lstm.input_size == input_size
    assert model.lstm.hidden_size == hidden_size
    assert model.lstm.num_layers == num_layers
    assert model.fc.out_features == num_stocks
    assert model.fc.in_features == hidden_size

def test_equal_prior_yields_uniform_when_fc_zero_baselstm():
    """
    If fc weights and biases are zero, outputs should equal uniform distribution
    because logits = 0 + equal_prior (log(1/N)) -> softmax -> 1/N.
    """
    seed_everything(42)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 3, 7, 4, 8, 1, 6
    model = BaseLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        dropout=0.0
    )
    model.eval()

    # zero the linear layer so logits come only from equal_prior
    model.fc.weight.data.zero_()
    model.fc.bias.data.zero_()

    x = torch.zeros(batch, seq_len, input_size)
    with torch.no_grad():
        weights = model(x)

    expected = torch.full_like(weights, fill_value=1.0 / num_stocks)
    assert torch.allclose(weights, expected, atol=1e-6), f'expected uniform prob, got {weights}'

def test_dropout_train_vs_eval_behavior_baselstm():
    """
    In eval mode outputs should be deterministic for identical inputs.
    In train mode (with dropout>0) outputs should vary across calls.
    """
    seed_everything(7)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 2, 6, 4, 8, 2, 3
    model = BaseLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        dropout=0.5
    )

    x = torch.randn(batch, seq_len, input_size)

    # deterministic in eval()
    model.eval()
    with torch.no_grad():
        w1 = model(x)
        w2 = model(x)
    assert torch.allclose(w1, w2, atol=1e-7), "eval() forward should be deterministic"

    # in train(), dropout should introduce variability over calls
    model.train()
    # Do not reseed between calls so dropout RNG moves
    w_train_1 = model(x)
    w_train_2 = model(x)
    # It's possible (very unlikely) masks are identical; assert not identical to catch most regressions:
    assert not torch.allclose(w_train_1, w_train_2), "train() forward with dropout should usually produce different outputs"

@pytest.mark.parametrize("batch,seq_len,input_size,hidden_size,num_layers,num_stocks", [
    (1, 10, 6, 8, 2, 5),
    (4, 15, 6, 8, 2, 7),
])
def test_baselstm_output_shape_and_probability(batch, seq_len, input_size, hidden_size, num_layers, num_stocks):
    """BaseLSTM should return (B, num_stocks) probability vectors that sum to 1"""
    seed_everything(0)
    model = BaseLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_stocks=num_stocks, dropout=0.2)
    model.eval()  # deterministic (dropout off)
    x = torch.randn(batch, seq_len, input_size, dtype=torch.float32)

    with torch.no_grad():
        weights = model(x)

    assert weights.shape == (batch, num_stocks)
    assert_probability_vector(weights)


def test_baselstm_grad_flow_and_input_sensitivity():
    """Checks grads flow back to params and outputs change for different inputs."""
    seed_everything(1)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 3, 12, 4, 8, 1, 5
    model = BaseLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_stocks=num_stocks, dropout=0.0)
    model.train()

    x1 = torch.randn(batch, seq_len, input_size, requires_grad=False)
    x2 = x1 + 0.1 * torch.randn_like(x1)  # slightly different input

    out1 = model(x1)
    out2 = model(x2)

    # outputs should be probability vectors
    assert_probability_vector(out1)
    assert_probability_vector(out2)

    # ensure outputs are not identical for different inputs
    assert not torch.allclose(out1, out2), "model outputs identical for different inputs"

    # --- Use a loss that depends on logits/outputs (not a constant) ---
    # Option A: MSE to a random target probability vector
    target = torch.rand_like(out1)
    target = target / target.sum(dim=-1, keepdim=True)  # normalize to valid probs
    loss = F.mse_loss(out1, target)

    # Option B (alternative): negative log-prob of class 0
    # eps = 1e-9
    # loss = -torch.log(out1[:, 0] + eps).sum()

    loss.backward()

    grads_present = any((p.grad is not None and torch.any(p.grad != 0).item()) for p in model.parameters())
    assert grads_present, "No non-zero gradients found on model parameters after backward()"


def test_baselstm_last_timestep_dependency_and_zero_input_stability():
    """
    Sanity test for BaseLSTM that mirrors 'residual_and_pooling_effects' idea:
      - Zero input produces a valid probability vector (numerical stability).
      - The model output depends on sequence order / last timestep: an input with
        a non-zero first timestep should not equal an input with a non-zero last timestep.
      - Deterministic in eval() mode for identical inputs.
    """
    seed_everything(5)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 2, 6, 4, 8, 1, 3
    model = BaseLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_stocks=num_stocks, dropout=0.0)
    model.eval()

    # zero input stability
    x_zero = torch.zeros(batch, seq_len, input_size, dtype=torch.float32)
    with torch.no_grad():
        w_zero = model(x_zero)
    assert_probability_vector(w_zero)

    # last-timestep dependency: only-first vs only-last non-zero
    x_first = torch.zeros_like(x_zero)
    x_first[:, 0, :] = 1.0
    x_last = torch.zeros_like(x_zero)
    x_last[:, -1, :] = 1.0

    with torch.no_grad():
        w_first = model(x_first)
        w_last = model(x_last)

    # outputs must be valid probabilities
    assert_probability_vector(w_first)
    assert_probability_vector(w_last)

    # model should be sensitive to where the signal occurs in the sequence (last hidden is used)
    assert not torch.allclose(w_first, w_last), "BaseLSTM appears insensitive to last-timestep — expected different outputs"

    # deterministic in eval for identical inputs
    with torch.no_grad():
        w_last2 = model(x_last)
    assert torch.allclose(w_last, w_last2, atol=1e-7), "eval() forward should be deterministic"


# -------------------- Tests for AttentionLSTM class -------------------- #
def test_attentionlstm_constructor_attributes():
    """Constructor should create expected submodules including attention and layer norms."""
    input_size, hidden_size, num_layers, num_stocks = 6, 8, 1, 4
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.0,
        equal_prior = False
    )

    assert hasattr(model, 'lstm')
    assert hasattr(model, 'ln_lstm')
    assert hasattr(model, 't_attn')
    assert hasattr(model, 'fc')
    assert model.lstm.input_size == input_size
    assert model.fc.out_features == num_stocks

def test_equal_prior_yields_uniform_when_fc_zero_attentionlstm():
    """Same equal_prior invariant test for AttentionLSTM."""
    seed_everything(0)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 2, 5, 6, 8, 1, 4
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.0,
        equal_prior = True
    )
    model.eval()

    model.fc.weight.data.zero_()
    model.fc.bias.data.zero_()

    x = torch.zeros(batch, seq_len, input_size)
    with torch.no_grad():
        weights = model(x)

    expected = torch.full_like(weights, fill_value=1.0 / num_stocks)
    assert torch.allclose(weights, expected, atol=1e-6)

def test_dropout_train_vs_eval_behavior_attentionlstm():
    """Same dropout deterministic check for AttentionLSTM."""
    seed_everything(8)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 2, 6, 6, 8, 2, 3
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.5,
        equal_prior = True
    )

    x = torch.randn(batch, seq_len, input_size)

    model.eval()
    with torch.no_grad():
        w1 = model(x)
        w2 = model(x)
    assert torch.allclose(w1, w2, atol=1e-7)

    model.train()
    w_train_1 = model(x)
    w_train_2 = model(x)
    assert not torch.allclose(w_train_1, w_train_2)

@pytest.mark.parametrize("batch,seq_len,input_size,hidden_size,num_layers,num_stocks", [
    (2, 8, 6, 8, 2, 6),  # hidden_size (8) is divisible by num_heads (2) used in model
    (3, 12, 4, 4, 2, 4),
])
def test_attentionlstm_output_shape_and_probability(batch, seq_len, input_size, hidden_size, num_layers, num_stocks):
    """
    AttentionLSTM should produce (B, num_stocks) probability vectors.
    Note: AttentionLSTM sets num_heads=2, so hidden_size must be divisible by 2.
    """
    seed_everything(2)
    # ensure hidden_size is divisible by 2 for multihead attention
    if hidden_size % 2 != 0:
        pytest.skip("hidden_size must be divisible by num_heads (2) for MultiheadAttention in this model")

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.2,
        equal_prior = False
    )
    model.eval()  # deterministic for testing dropout
    x = torch.randn(batch, seq_len, input_size, dtype=torch.float32)

    with torch.no_grad():
        weights = model(x)

    assert weights.shape == (batch, num_stocks)
    assert_probability_vector(weights)


def test_attentionlstm_residual_and_pooling_effects():
    """
    A simple sanity test: if the input is all zeros the LSTM+attention pipeline should still produce a
    valid probability vector (because of learned bias / equal_prior). This checks model's numerical stability.
    """
    seed_everything(3)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 2, 5, 6, 8, 1, 5
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.0,
        equal_prior = False
    )
    model.eval()
    x = torch.zeros(batch, seq_len, input_size, dtype=torch.float32)

    with torch.no_grad():
        weights = model(x)

    assert weights.shape == (batch, num_stocks)
    assert_probability_vector(weights)

    # All-equal input should not produce NaNs or Inf and should be somewhat consistent across repeated calls
    with torch.no_grad():
        weights2 = model(x)
    assert torch.allclose(weights, weights2, atol=1e-6), "Deterministic eval() forward pass should produce same results for identical inputs"

def test_attentionlstm_grad_flow_and_input_sensitivity():
    """
    Mirror of the BaseLSTM grad test for AttentionLSTM:
      - outputs differ for slightly different inputs
      - gradients flow to model parameters after backward()
    """
    seed_everything(6)
    batch, seq_len, input_size, hidden_size, num_layers, num_stocks = 3, 10, 4, 8, 1, 5
    assert hidden_size % 2 == 0  # needed for num_heads=2

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        nheads = 2,
        dropout=0.0,
        equal_prior = False
    )
    model.train()

    x1 = torch.randn(batch, seq_len, input_size, dtype=torch.float32)
    x2 = x1 + 1e-2 * torch.randn_like(x1)  # small perturbation

    out1 = model(x1)
    out2 = model(x2)

    assert_probability_vector(out1)
    assert_probability_vector(out2)

    assert not torch.allclose(out1, out2), "AttentionLSTM outputs identical for perturbed inputs"

    # Use a non-constant loss
    target = torch.rand_like(out1)
    target = target / target.sum(dim=-1, keepdim=True)
    loss = F.mse_loss(out1, target)

    loss.backward()

    grads_present = any((p.grad is not None and torch.any(p.grad != 0).item()) for p in model.parameters())
    assert grads_present, "No non-zero gradients found on AttentionLSTM parameters after backward()"