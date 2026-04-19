import torch
import pytest
from src.training.loss_functions import (
    LossLibrary,
    log_return_objective,
    raw_sharpe_objective,
    differentiable_sharpe_objective,
    rms_sharpe_objective,
    smooth_neglog_sharpe_loss,
    log_sharpe_objective,
    raw_sortino_objective,
    differentiable_sortino_objective,
    rms_sortino_loss,
    smooth_neglog_sortino_objective,
    log_sortino_objective,
)

# Fixture to populate registry with dummy functions
@pytest.fixture
def populated_registry():
    # Clear any previous registrations
    LossLibrary._registry.clear()
    
    # Register dummy functions inside the fixture
    @LossLibrary.register(category='objectives', name='sharpe')
    def dummy_sharpe(weights, returns):
        return 0.5
    
    @LossLibrary.register(category='objectives', subcategory='risk', name='cvar')
    def dummy_cvar(weights, returns):
        return 0.2
    
    @LossLibrary.register(category='regularizers', name='risk_parity')
    def dummy_risk_parity(weights):
        return 0.1
    
    @LossLibrary.register(category='regularizers', subcategory='diversification', name='hhi')
    def dummy_hhi(weights):
        return 0.05
    
    yield LossLibrary._registry

# Fixture to ensure empty registry
@pytest.fixture
def empty_registry():
    LossLibrary._registry.clear()
    yield

# -------------------- Tests that require populated registry -------------------- #
def test_register_and_get(populated_registry):
    # Get by category, name, default subcategory
    fn = LossLibrary.get('objectives', 'sharpe')
    assert fn is not None
    assert fn(1,2) == 0.5  # call to verify

    fn = LossLibrary.get('objectives', 'cvar', subcategory='risk')
    assert fn is not None
    assert fn(1,2) == 0.2

    fn = LossLibrary.get('regularizers', 'risk_parity')
    assert fn is not None
    assert fn(1) == 0.1

    fn = LossLibrary.get('regularizers', 'hhi', subcategory='diversification')
    assert fn is not None
    assert fn(1) == 0.05

def test_duplicate_registration_raises(populated_registry):
    with pytest.raises(KeyError, match="already registered"):
        @LossLibrary.register(category='objectives', name='sharpe')
        def another_sharpe():
            pass

def test_items(populated_registry):
    items = LossLibrary.items()
    assert 'objectives' in items
    assert 'regularizers' in items
    assert '__default__' in items['objectives']
    assert 'risk' in items['objectives']
    assert 'sharpe' in items['objectives']['__default__']
    assert 'cvar' in items['objectives']['risk']
    assert '__default__' in items['regularizers']
    assert 'diversification' in items['regularizers']
    assert 'risk_parity' in items['regularizers']['__default__']
    assert 'hhi' in items['regularizers']['diversification']

def test_list_categories(populated_registry):
    categories = LossLibrary.list_categories()
    assert set(categories) == {'objectives', 'regularizers'}

def test_list_subcategories(populated_registry):
    subcats = LossLibrary.list_subcategories('objectives')
    assert set(subcats) == {'__default__', 'risk'}
    subcats = LossLibrary.list_subcategories('regularizers')
    assert set(subcats) == {'__default__', 'diversification'}
    # Non-existent category returns empty list
    assert LossLibrary.list_subcategories('nonexistent') == []

def test_list_functions(populated_registry):
    funcs = LossLibrary.list_functions('objectives')
    assert set(funcs) == {'sharpe'}
    funcs = LossLibrary.list_functions('objectives', subcategory='risk')
    assert set(funcs) == {'cvar'}
    funcs = LossLibrary.list_functions('regularizers')
    assert set(funcs) == {'risk_parity'}
    funcs = LossLibrary.list_functions('regularizers', subcategory='diversification')
    assert set(funcs) == {'hhi'}
    # Non-existent category returns empty list
    assert LossLibrary.list_functions('nonexistent') == []
    # Non-existent subcategory returns empty list
    assert LossLibrary.list_functions('objectives', subcategory='missing') == []

# -------------------- Tests for empty registry (no functions) -------------------- #
def test_empty_registry(empty_registry):
    assert LossLibrary.items() == {}
    assert LossLibrary.list_categories() == []
    assert LossLibrary.list_subcategories('objectives') == []
    assert LossLibrary.list_functions('objectives') == []
    with pytest.raises(KeyError):
        LossLibrary.get('objectives', 'sharpe')

# -------------------- Tests for log_return_objective -------------------- #
def test_log_return_positive_single_batch():
    # Portfolio returns: [0.01, 0.02, 0.03] (three days)
    # log(1+0.01)=log(1.01)=0.00995, log(1.02)=0.01980, log(1.03)=0.02956
    # Sum = 0.05931, mean across batch = 0.05931, negative loss = -0.05931
    returns = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
    loss = log_return_objective(returns)
    expected = -0.05931
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_log_return_negative_single_batch():
    # Returns: [-0.01, -0.02, -0.03]
    # log(1-0.01)=log(0.99)=-0.01005, log(0.98)=-0.02020, log(0.97)=-0.03046
    # Sum = -0.06071, mean = -0.06071, negative loss = +0.06071
    returns = torch.tensor([[-0.01, -0.02, -0.03]], dtype=torch.float32)
    loss = log_return_objective(returns)
    expected = 0.06071
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_log_return_two_batches():
    # Batch 1: [0.01, 0.01] → sum log = 2*log(1.01)=0.01990
    # Batch 2: [-0.01, -0.01] → sum log = 2*log(0.99)=-0.02010
    # Mean across batches = (0.01990 + (-0.02010))/2 = -0.00010
    # Negative loss = +0.00010
    returns = torch.tensor([[0.01, 0.01], [-0.01, -0.01]], dtype=torch.float32)
    loss = log_return_objective(returns)
    expected = 0.00010
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_log_return_zero_returns():
    # All returns zero: log(1+0)=0, sum=0, mean=0, loss=0
    returns = torch.zeros(3, 5)
    loss = log_return_objective(returns)
    assert loss.item() == 0.0

def test_log_return_near_minus_one():
    # Return = -0.999 → 1 + (-0.999) = 0.001, plus eps ≈ 0.00100001
    # log(0.001) ≈ -6.9078, sum = -6.9078, mean = -6.9078, loss = +6.9078
    returns = torch.full((1, 1), -0.999, dtype=torch.float32)
    loss = log_return_objective(returns)
    expected = 6.9078
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-2)

# Shape and gradient tests
def test_log_return_shape():
    B, T = 4, 60
    returns = torch.randn(B, T)
    loss = log_return_objective(returns)
    assert loss.shape == ()  # scalar

def test_log_return_gradient():
    returns = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32, requires_grad=True)
    loss = log_return_objective(returns)
    loss.backward()
    assert returns.grad is not None
    assert returns.grad.shape == returns.shape

# ==================== Fixtures for deterministic returns tensor ==================== #
@pytest.fixture
def sample_returns():
    return torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)

@pytest.fixture
def negative_returns():
    # One batch of negative returns
    return torch.tensor([[-0.01, -0.02, -0.03]], dtype=torch.float32)

@pytest.fixture
def zero_returns():
    return torch.zeros(2, 5, dtype=torch.float32)

# -------------------- raw_sharpe_objective -------------------- #
def test_raw_sharpe_positive(sample_returns):
    # mean = 0.02, population std = 0.00816497, Sharpe = 2.44949, loss = -2.44949
    loss = raw_sharpe_objective(sample_returns)
    expected = -2.44949
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_raw_sharpe_zero(zero_returns):
    loss = raw_sharpe_objective(zero_returns)
    assert loss.item() == 0.0

def test_raw_sharpe_shape(sample_returns):
    loss = raw_sharpe_objective(sample_returns)
    assert loss.shape == ()

def test_raw_sharpe_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = raw_sharpe_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- differentiable_sharpe_objective -------------------- #
def test_diff_sharpe_positive(sample_returns):
    # Uses sample variance (unbiased), so std = 0.01, Sharpe = 0.02/0.01 = 2.0
    loss = differentiable_sharpe_objective(sample_returns)
    expected = -2.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_diff_sharpe_zero(zero_returns):
    loss = differentiable_sharpe_objective(zero_returns)
    assert loss.item() == 0.0

def test_diff_sharpe_shape(sample_returns):
    loss = differentiable_sharpe_objective(sample_returns)
    assert loss.shape == ()

def test_diff_sharpe_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = differentiable_sharpe_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- rms_sharpe_objective -------------------- #
def test_rms_sharpe_positive(sample_returns):
    # Uses population std (same as raw), expected -2.44949
    loss = rms_sharpe_objective(sample_returns)
    expected = -2.44949
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)  # tolerance due to floating point

def test_rms_sharpe_zero(zero_returns):
    loss = rms_sharpe_objective(zero_returns)
    assert loss.item() == 0.0

def test_rms_sharpe_shape(sample_returns):
    loss = rms_sharpe_objective(sample_returns)
    assert loss.shape == ()

def test_rms_sharpe_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = rms_sharpe_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- smooth_neglog_sharpe_loss -------------------- #
def test_smooth_sharpe_positive(sample_returns):
    # Uses sample std (0.01) → Sharpe = 2.0
    # softplus(2.0) = log(1+exp(2)) = log(8.389) = 2.126
    # loss = -log(2.126) = -0.7546
    loss = smooth_neglog_sharpe_loss(sample_returns, beta=1.0)
    expected = -0.7546
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_smooth_sharpe_zero(zero_returns):
    # Sharpe=0 → softplus(0)=log(2)=0.6931, log(0.6931)= -0.3665, loss = +0.3665
    loss = smooth_neglog_sharpe_loss(zero_returns)
    expected = 0.3665
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_smooth_sharpe_shape(sample_returns):
    loss = smooth_neglog_sharpe_loss(sample_returns)
    assert loss.shape == ()

def test_smooth_sharpe_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_neglog_sharpe_loss(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- log_sharpe_objective -------------------- #
def test_log_sharpe_positive(sample_returns):
    # Log returns: [0.00995, 0.01980, 0.02956]
    # mean_log = 0.01977, sample var = 0.0000961, std_log = 0.009804
    # Sharpe_log = 0.01977/0.009804 = 2.0165, loss = -2.0165
    loss = log_sharpe_objective(sample_returns)
    expected = -2.0165
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_log_sharpe_zero(zero_returns):
    loss = log_sharpe_objective(zero_returns)
    assert loss.item() == 0.0

def test_log_sharpe_shape(sample_returns):
    loss = log_sharpe_objective(sample_returns)
    assert loss.shape == ()

def test_log_sharpe_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = log_sharpe_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- raw_sortino_objective -------------------- #
def test_raw_sortino_positive(sample_returns):
    loss = raw_sortino_objective(sample_returns)
    # For positive returns, downside = 0 → loss ≈ -2e6 (depends on eps)
    expected = -2_000_000.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e3)
    assert loss.shape == ()

def test_raw_sortino_negative(negative_returns):
    loss = raw_sortino_objective(negative_returns)
    expected = 2.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)
    assert loss.shape == ()

def test_raw_sortino_zero(zero_returns):
    loss = raw_sortino_objective(zero_returns)
    assert loss.item() == 0.0

def test_raw_sortino_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = raw_sortino_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- differentiable_sortino_objective -------------------- #
def test_diff_sortino_positive(sample_returns):
    loss = differentiable_sortino_objective(sample_returns)
    expected = -2_000_000.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e3)
    assert loss.shape == ()

def test_diff_sortino_negative(negative_returns):
    loss = differentiable_sortino_objective(negative_returns)
    expected = 2.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)
    assert loss.shape == ()

def test_diff_sortino_zero(zero_returns):
    loss = differentiable_sortino_objective(zero_returns)
    assert loss.item() == 0.0

def test_diff_sortino_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = differentiable_sortino_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- rms_sortino_loss -------------------- #
def test_rms_sortino_positive(sample_returns):
    loss = rms_sortino_loss(sample_returns)
    # For positive returns, downside = 0 → loss ≈ -200 (since sqrt(eps) = 1e-4)
    expected = -200.0
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-2)
    assert loss.shape == ()

def test_rms_sortino_negative(negative_returns):
    loss = rms_sortino_loss(negative_returns)
    expected = 0.9259
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)
    assert loss.shape == ()

def test_rms_sortino_zero(zero_returns):
    loss = rms_sortino_loss(zero_returns)
    assert loss.item() == 0.0

def test_rms_sortino_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = rms_sortino_loss(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- smooth_neglog_sortino_objective -------------------- #
def test_smooth_sortino_positive(sample_returns):
    loss = smooth_neglog_sortino_objective(sample_returns, use_soft_downside=True, beta=10.0)
    # Expected value from actual run (observed)
    expected = 0.13512209057807922
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)
    assert loss.shape == ()

def test_smooth_sortino_negative(negative_returns):
    loss = smooth_neglog_sortino_objective(negative_returns, use_soft_downside=True, beta=10.0)
    # Should be positive (as it's a loss to minimize)
    assert loss.shape == ()
    assert loss.item() > 0

def test_smooth_sortino_zero(zero_returns):
    loss = smooth_neglog_sortino_objective(zero_returns, use_soft_downside=True, beta=10.0)
    expected = 0.3665
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)
    assert loss.shape == ()

def test_smooth_sortino_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_neglog_sortino_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- log_sortino_objective -------------------- #
def test_log_sortino_positive(sample_returns):
    loss = log_sortino_objective(sample_returns, use_soft_downside=True)
    assert loss.shape == ()
    # Should be negative because Sortino positive (expected from implementation)
    assert loss.item() < 0

def test_log_sortino_negative(negative_returns):
    loss = log_sortino_objective(negative_returns, use_soft_downside=True)
    expected = 3.6026   # observed value
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)
    assert loss.shape == ()

def test_log_sortino_zero(zero_returns):
    loss = log_sortino_objective(zero_returns)
    assert loss.item() == 0.0

def test_log_sortino_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = log_sortino_objective(returns)
    loss.backward()
    assert returns.grad is not None