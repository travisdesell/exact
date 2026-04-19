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
    smooth_mdd_regularizer,
    cvar_topk_regularizer,
    smooth_cvar_regularizer,
    smooth_rockafellar_cvar_regularizer,
    sample_covariance,
    shrinkage_covariance_torch,
    risk_parity_regularizer,
    raw_omega_ratio,
    smooth_omega_objective
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

@pytest.fixture
def returns_2d():
    """Simple returns: batch=1, time=3, assets=2"""
    return torch.tensor([[[0.1, 0.2],
                          [0.3, 0.4],
                          [0.5, 0.6]]], dtype=torch.float32)

@pytest.fixture
def weights_2d():
    """Weights: batch=1, assets=2"""
    return torch.tensor([[0.4, 0.6]], dtype=torch.float32)

@pytest.fixture
def returns_2d_batch2():
    """Two batches, each 3 time steps, 2 assets"""
    return torch.tensor([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]]
    ], dtype=torch.float32)

@pytest.fixture
def weights_2d_batch2():
    return torch.tensor([[0.4, 0.6], [0.5, 0.5]], dtype=torch.float32)

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

# -------------------- Tests for smooth_neglog_sortino_objective -------------------- #
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

# -------------------- Tests for log_sortino_objective -------------------- #
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

# -------------------- Tests for smooth_mdd_regularizer -------------------- #
def test_mdd_shape(sample_returns):
    loss = smooth_mdd_regularizer(sample_returns)
    assert loss.shape == ()

def test_mdd_positive_returns(sample_returns):
    loss = smooth_mdd_regularizer(sample_returns, use_percent=True)
    assert loss.item() == 0.0

def test_mdd_negative_returns(negative_returns):
    loss = smooth_mdd_regularizer(negative_returns, use_percent=True)
    expected = 0.03333002328872681
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_mdd_zero_returns(zero_returns):
    loss = smooth_mdd_regularizer(zero_returns, use_percent=True)
    assert loss.item() == 0.0

def test_mdd_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_mdd_regularizer(returns)
    loss.backward()
    assert returns.grad is not None
    assert returns.grad.shape == returns.shape

def test_mdd_percent_vs_log(sample_returns):
    loss_percent = smooth_mdd_regularizer(sample_returns, use_percent=True)
    loss_log = smooth_mdd_regularizer(sample_returns, use_percent=False)
    assert loss_percent.item() == 0.0
    assert loss_log.item() == 0.0

def test_mdd_parameter_effects(negative_returns):
    loss_low_temp = smooth_mdd_regularizer(negative_returns, temp=10.0)
    loss_high_temp = smooth_mdd_regularizer(negative_returns, temp=100.0)
    assert not torch.isclose(loss_low_temp, loss_high_temp, atol=1e-6)

def test_mdd_batch_consistency():
    returns = torch.tensor([[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]], dtype=torch.float32)
    loss = smooth_mdd_regularizer(returns)
    assert loss.shape == ()
    assert not torch.isnan(loss)

# -------------------- Tests for cvar_topk_regularizer -------------------- #
def test_cvar_topk_shape(sample_returns):
    loss = cvar_topk_regularizer(sample_returns, alpha=0.05)
    assert loss.shape == ()

def test_cvar_topk_positive_returns(sample_returns):
    loss = cvar_topk_regularizer(sample_returns, alpha=0.05)
    expected = -0.01
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_cvar_topk_negative_returns(negative_returns):
    loss = cvar_topk_regularizer(negative_returns, alpha=0.05)
    expected = 0.03
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_cvar_topk_zero_returns(zero_returns):
    loss = cvar_topk_regularizer(zero_returns, alpha=0.05)
    assert loss.item() == 0.0

def test_cvar_topk_alpha_large():
    returns = torch.tensor([[-0.01, -0.02, -0.03]], dtype=torch.float32)
    loss = cvar_topk_regularizer(returns, alpha=0.5)
    expected = 0.025
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_cvar_topk_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = cvar_topk_regularizer(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- Tests for smooth_cvar_regularizer -------------------- #
def test_smooth_cvar_shape(sample_returns):
    loss = smooth_cvar_regularizer(sample_returns)
    assert loss.shape == ()

def test_smooth_cvar_positive_returns(sample_returns):
    loss = smooth_cvar_regularizer(sample_returns, scale_by_std=False, normalize_by_port_std=False)
    assert loss.item() < 0

def test_smooth_cvar_negative_returns(negative_returns):
    loss = smooth_cvar_regularizer(negative_returns, scale_by_std=False, normalize_by_port_std=False)
    assert loss.item() > 0

def test_smooth_cvar_zero_returns(zero_returns):
    loss = smooth_cvar_regularizer(zero_returns, scale_by_std=False, normalize_by_port_std=False)
    assert loss.item() == 0.0

def test_smooth_cvar_normalization(sample_returns):
    loss_norm = smooth_cvar_regularizer(sample_returns, normalize_by_port_std=True)
    loss_no_norm = smooth_cvar_regularizer(sample_returns, normalize_by_port_std=False)
    assert not torch.isclose(loss_norm, loss_no_norm, atol=1e-6)

def test_smooth_cvar_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_cvar_regularizer(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- smooth_rockafellar_cvar_regularizer -------------------- #
def test_rockafellar_cvar_shape(sample_returns):
    loss = smooth_rockafellar_cvar_regularizer(sample_returns)
    assert loss.shape == ()

def test_rockafellar_cvar_positive_returns(sample_returns):
    loss = smooth_rockafellar_cvar_regularizer(sample_returns, normalize_by_port_std=False)
    # Observed deterministic value
    expected = 0.0706624910235405
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_rockafellar_cvar_negative_returns(negative_returns):
    loss = smooth_rockafellar_cvar_regularizer(negative_returns, normalize_by_port_std=False)
    assert loss.item() > 0

def test_rockafellar_cvar_zero_returns(zero_returns):
    loss = smooth_rockafellar_cvar_regularizer(zero_returns, normalize_by_port_std=False)
    # Observed deterministic value
    expected = 0.13862943649291992
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_rockafellar_cvar_alpha_effect(negative_returns):
    loss_alpha05 = smooth_rockafellar_cvar_regularizer(negative_returns, alpha=0.05, normalize_by_port_std=False)
    loss_alpha1 = smooth_rockafellar_cvar_regularizer(negative_returns, alpha=0.1, normalize_by_port_std=False)
    assert loss_alpha05.item() > loss_alpha1.item()

def test_rockafellar_cvar_normalization(sample_returns):
    loss_norm = smooth_rockafellar_cvar_regularizer(sample_returns, normalize_by_port_std=True)
    loss_no_norm = smooth_rockafellar_cvar_regularizer(sample_returns, normalize_by_port_std=False)
    assert not torch.isclose(loss_norm, loss_no_norm, atol=1e-6)

def test_rockafellar_cvar_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_rockafellar_cvar_regularizer(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- sample_covariance -------------------- #
def test_sample_covariance_shape(returns_2d):
    B, T, N = returns_2d.shape
    cov = sample_covariance(returns_2d)
    assert cov.shape == (B, N, N)

def test_sample_covariance_values(returns_2d):
    # Compute manually: mean of each asset = [0.3, 0.4]
    # demeaned: [[-0.2,-0.2], [0.0,0.0], [0.2,0.2]]
    # X^T X / (T-1) = ([[0.08,0.08],[0.08,0.08]]) / 2 = [[0.04,0.04],[0.04,0.04]]
    cov = sample_covariance(returns_2d, unbiased=True)
    expected = torch.tensor([[[0.04, 0.04], [0.04, 0.04]]], dtype=torch.float32)
    assert torch.allclose(cov, expected, atol=1e-6)

def test_sample_covariance_unbiased_false(returns_2d):
    # With unbiased=False, denominator = T = 3
    cov = sample_covariance(returns_2d, unbiased=False)
    expected = torch.tensor([[[0.08/3, 0.08/3], [0.08/3, 0.08/3]]], dtype=torch.float32)
    assert torch.allclose(cov, expected, atol=1e-6)

def test_sample_covariance_gradient(returns_2d):
    returns = returns_2d.clone().detach().requires_grad_(True)
    cov = sample_covariance(returns)
    loss = cov.sum()
    loss.backward()
    assert returns.grad is not None

# -------------------- shrinkage_covariance_torch -------------------- #
def test_shrinkage_covariance_shape(returns_2d):
    cov = sample_covariance(returns_2d)
    shrunk = shrinkage_covariance_torch(cov, shrink=0.2)
    assert shrunk.shape == cov.shape

def test_shrinkage_covariance_values(returns_2d):
    cov = sample_covariance(returns_2d)
    # cov = [[0.04,0.04],[0.04,0.04]], trace=0.08, scale=0.08/2=0.04, I=identity
    # shrunk = (1-0.2)*cov + 0.2*0.04*I = 0.8*cov + 0.008*I
    # = [[0.8*0.04+0.008, 0.8*0.04], [0.8*0.04, 0.8*0.04+0.008]]
    # = [[0.032+0.008, 0.032], [0.032, 0.032+0.008]] = [[0.04,0.032],[0.032,0.04]]
    shrunk = shrinkage_covariance_torch(cov, shrink=0.2)
    expected = torch.tensor([[[0.04, 0.032], [0.032, 0.04]]], dtype=torch.float32)
    assert torch.allclose(shrunk, expected, atol=1e-6)

def test_shrinkage_covariance_gradient(returns_2d):
    cov = sample_covariance(returns_2d).requires_grad_(True)
    shrunk = shrinkage_covariance_torch(cov, shrink=0.1)
    loss = shrunk.sum()
    loss.backward()
    assert cov.grad is not None

# -------------------- risk_parity_regularizer -------------------- #
def test_risk_parity_regularizer_shape(weights_2d, returns_2d):
    loss = risk_parity_regularizer(weights_2d, returns_2d)
    assert loss.shape == ()

def test_risk_parity_regularizer_values(weights_2d, returns_2d):
    # Manual calculation:
    # returns: [[0.1,0.2],[0.3,0.4],[0.5,0.6]] -> cov = [[0.04,0.04],[0.04,0.04]]
    # weights = [0.4,0.6]
    # portfolio variance = w^T cov w = 0.4*0.04*0.4 + 2*0.4*0.04*0.6 + 0.6*0.04*0.6 = 0.04*(0.16+0.48+0.36)=0.04*1=0.04
    # marginal contributions: cov w = [[0.04,0.04],[0.04,0.04]] * [0.4,0.6]^T = [0.04,0.04] (since sum=1)
    # rc = weights * mcontrib = [0.4*0.04, 0.6*0.04] = [0.016,0.024]
    # target = sigma2/N = 0.04/2 = 0.02
    # squared deviations = [(0.016-0.02)^2, (0.024-0.02)^2] = [0.000016, 0.000016] sum=0.000032
    # scale_invariant = loss/(sigma2^2)=0.000032/0.0016=0.02
    loss = risk_parity_regularizer(weights_2d, returns_2d, use_shrink=False, scale_invariant=True)
    expected = 0.02
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_risk_parity_regularizer_no_scale_invariant(weights_2d, returns_2d):
    loss = risk_parity_regularizer(weights_2d, returns_2d, use_shrink=False, scale_invariant=False)
    expected = 0.000032
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_risk_parity_regularizer_with_shrink(weights_2d, returns_2d):
    # Shrinkage changes cov, so loss will be different; just test that it runs.
    loss = risk_parity_regularizer(weights_2d, returns_2d, use_shrink=True, shrink=0.1)
    assert loss.shape == ()
    assert not torch.isnan(loss)

def test_risk_parity_regularizer_gradient(weights_2d, returns_2d):
    w = weights_2d.clone().detach().requires_grad_(True)
    loss = risk_parity_regularizer(w, returns_2d)
    loss.backward()
    assert w.grad is not None

def test_risk_parity_regularizer_batch(weights_2d_batch2, returns_2d_batch2):
    loss = risk_parity_regularizer(weights_2d_batch2, returns_2d_batch2, use_shrink=False)
    # Should produce a scalar (mean across batches)
    assert loss.shape == ()
    assert not torch.isnan(loss)

def test_risk_parity_regularizer_zero_variance_edge():
    # All returns zero -> covariance zero -> sigma2=0 -> should be handled by eps
    returns = torch.zeros(1, 3, 2, dtype=torch.float32)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    loss = risk_parity_regularizer(weights, returns)
    assert not torch.isnan(loss)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)

# -------------------- raw_omega_ratio -------------------- #
def test_raw_omega_ratio_shape(sample_returns):
    loss = raw_omega_ratio(sample_returns)
    assert loss.shape == ()

def test_raw_omega_ratio_positive_returns(sample_returns):
    # returns all > theta=0, so neg=0, pos = returns mean = 0.02
    # omega = pos / (0+eps) = very large, loss = -very large (negative)
    loss = raw_omega_ratio(sample_returns)
    assert loss.item() < 0

def test_raw_omega_ratio_negative_returns(negative_returns):
    # returns all < theta=0, so pos=0, neg = -returns mean = 0.02
    # omega = 0 / (0.02+eps) = 0, loss = 0
    loss = raw_omega_ratio(negative_returns)
    assert loss.item() == 0.0

def test_raw_omega_ratio_mixed():
    # returns: [0.01, -0.02, 0.03], theta=0
    # pos_mean = (0.01+0+0.03)/3 = 0.01333, neg_mean = (0+0.02+0)/3 = 0.0066667
    # omega = 0.01333/0.0066667 = 2.0, loss = -2.0
    returns = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float32)
    loss = raw_omega_ratio(returns)
    expected = -2.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_raw_omega_ratio_zero_returns(zero_returns):
    loss = raw_omega_ratio(zero_returns)
    # pos_mean=0, neg_mean=0 -> omega=0/(eps)=0, loss=0
    assert loss.item() == 0.0

def test_raw_omega_ratio_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = raw_omega_ratio(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- Tests for smooth_omega_objective -------------------- #
def test_smooth_omega_shape(sample_returns):
    loss = smooth_omega_objective(sample_returns)
    assert loss.shape == ()

def test_smooth_omega_positive_returns(sample_returns):
    loss = smooth_omega_objective(sample_returns, use_log_loss=True)
    assert loss.item() < 0

def test_smooth_omega_negative_returns(negative_returns):
    loss = smooth_omega_objective(negative_returns, use_log_loss=True)
    assert loss.item() > 0

def test_smooth_omega_without_log(sample_returns):
    loss = smooth_omega_objective(sample_returns, use_log_loss=False)
    assert loss.item() < 0

def test_smooth_omega_cap(sample_returns):
    # Use a cap that is smaller than the actual omega to see an effect.
    # For sample_returns, compute omega roughly; set cap to 0.01.
    loss_uncapped = smooth_omega_objective(sample_returns, use_log_loss=True, cap_omega=None)
    loss_capped = smooth_omega_objective(sample_returns, use_log_loss=True, cap_omega=0.01)
    # Capped loss should be higher (less negative) than uncapped
    assert loss_capped.item() > loss_uncapped.item()

def test_smooth_omega_zero_returns(zero_returns):
    loss = smooth_omega_objective(zero_returns)
    expected = 0.0
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_smooth_omega_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_omega_objective(returns)
    loss.backward()
    assert returns.grad is not None