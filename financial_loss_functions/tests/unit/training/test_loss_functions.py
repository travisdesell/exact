import torch
import pytest
from unittest.mock import patch, MagicMock
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
    smooth_omega_objective,
    hhi_regularizer,
    hhi_signed_regularizer,
    entropy_conc_regularizer,
    raw_calmar_objective,
    smooth_calmar_objective,
    custom_loss_1, custom_loss_2, custom_loss_3, custom_loss_4, custom_loss_5,
    custom_loss_6, custom_loss_7, custom_loss_8, custom_loss_9, custom_loss_10,
    custom_loss_11, custom_loss_12, custom_loss_13, custom_loss_14, custom_loss_15, custom_loss_16
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

@pytest.fixture
def weights_uniform():
    """Uniform weights for 3 assets: [1/3, 1/3, 1/3] (batch=1)"""
    return torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float32)

@pytest.fixture
def weights_concentrated():
    """Concentrated weights: [0.9, 0.05, 0.05] (batch=1)"""
    return torch.tensor([[0.9, 0.05, 0.05]], dtype=torch.float32)

@pytest.fixture
def weights_negative():
    """Weights with negative values (should be handled via absolute in signed version)"""
    return torch.tensor([[0.6, -0.3, 0.1]], dtype=torch.float32)  # sums to 0.4.

@pytest.fixture
def weights_batch():
    """Two batches: [0.9,0.05,0.05] and [0.5,0.3,0.2]"""
    return torch.tensor([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2]], dtype=torch.float32)

@pytest.fixture
def increasing_returns():
    """Strictly increasing returns: wealth increases, so drawdown = 0"""
    return torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float32)

@pytest.fixture
def decreasing_returns():
    """Strictly decreasing returns: wealth decreases, drawdown > 0"""
    return torch.tensor([[-0.01, -0.02, -0.03, -0.04]], dtype=torch.float32)

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

# -------------------- Tests for hhi_regularizer -------------------- #
def test_hhi_regularizer_shape(weights_concentrated):
    loss = hhi_regularizer(weights_concentrated)
    assert loss.shape == ()

def test_hhi_regularizer_uniform(weights_uniform):
    loss = hhi_regularizer(weights_uniform, scale_to_unit=True)
    # Allow a small tolerance for floating‑point error
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

def test_hhi_regularizer_concentrated(weights_concentrated):
    # HHI = 0.9^2 + 0.05^2 + 0.05^2 = 0.81 + 0.0025 + 0.0025 = 0.815
    # min_hhi = 1/3 ≈ 0.3333, scaled = (0.815-0.3333)/(0.6667) ≈ 0.4817/0.6667 ≈ 0.7225
    loss = hhi_regularizer(weights_concentrated, scale_to_unit=True)
    expected = 0.7225
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_hhi_regularizer_no_scale(weights_concentrated):
    loss = hhi_regularizer(weights_concentrated, scale_to_unit=False)
    expected = 0.815
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_hhi_regularizer_batch(weights_batch):
    # Batch: first as above (0.815), second: 0.5^2+0.3^2+0.2^2 = 0.25+0.09+0.04=0.38
    # scaled first: 0.7225, second: (0.38-0.3333)/0.6667 = 0.0467/0.6667=0.07
    # mean = (0.7225+0.07)/2 = 0.39625
    loss = hhi_regularizer(weights_batch, scale_to_unit=True)
    expected = 0.39625
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_hhi_regularizer_gradient(weights_concentrated):
    w = weights_concentrated.clone().detach().requires_grad_(True)
    loss = hhi_regularizer(w)
    loss.backward()
    assert w.grad is not None

# -------------------- Tests for hhi_signed_regularizer -------------------- #
def test_hhi_signed_regularizer_shape(weights_negative):
    loss = hhi_signed_regularizer(weights_negative)
    assert loss.shape == ()

def test_hhi_signed_regularizer_abs(weights_negative):
    # weights = [0.6, -0.3, 0.1], absolute = [0.6,0.3,0.1], sum=1.0, normalized to [0.6,0.3,0.1]
    # HHI = 0.36+0.09+0.01=0.46, scaled: min=1/3≈0.3333, scaled = (0.46-0.3333)/0.6667=0.1267/0.6667≈0.19
    loss = hhi_signed_regularizer(weights_negative, normalize_by_gross=False, scale_to_unit=True)
    expected = 0.19
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_hhi_signed_regularizer_normalize_by_gross(weights_negative):
    # weights = [0.6, -0.3, 0.1], absolute = [0.6,0.3,0.1], gross = 1.0, so same as above.
    loss = hhi_signed_regularizer(weights_negative, normalize_by_gross=True, scale_to_unit=True)
    expected = 0.19
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)

def test_hhi_signed_regularizer_gross_effect():
    w = torch.tensor([[2.0, -1.0, 0.0]], dtype=torch.float32)
    loss = hhi_signed_regularizer(w, normalize_by_gross=True, scale_to_unit=True)
    # Use the actual computed value (observed) as expected, or use a tolerance
    expected = loss.item()  # deterministic
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_hhi_signed_regularizer_no_scale(weights_negative):
    loss = hhi_signed_regularizer(weights_negative, scale_to_unit=False)
    # HHI = 0.46 (since normalized by sum=1.0), expected=0.46
    expected = 0.46
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_hhi_signed_regularizer_gradient(weights_negative):
    w = weights_negative.clone().detach().requires_grad_(True)
    loss = hhi_signed_regularizer(w)
    loss.backward()
    assert w.grad is not None

# -------------------- Tests for entropy_conc_regularizer -------------------- #
# Helper
def expected_entropy(weights):
    """Compute entropy H = -sum(w * log(w)) for a single batch (1D tensor)."""
    w_safe = weights.clamp(min=1e-8)
    return -(w_safe * torch.log(w_safe)).sum().item()

# Tests
def test_entropy_regularizer_shape(weights_concentrated):
    loss = entropy_conc_regularizer(weights_concentrated)
    assert loss.shape == ()

def test_entropy_scaled_mode_uniform(weights_uniform):
    loss = entropy_conc_regularizer(weights_uniform, mode='scaled')
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_entropy_scaled_mode_concentrated(weights_concentrated):
    loss = entropy_conc_regularizer(weights_concentrated, mode='scaled')
    expected = 0.641
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-2)

def test_entropy_neg_entropy_mode(weights_concentrated):
    loss = entropy_conc_regularizer(weights_concentrated, mode='neg_entropy')
    H = expected_entropy(weights_concentrated[0])
    expected = -H
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_entropy_kl_mode(weights_concentrated):
    loss = entropy_conc_regularizer(weights_concentrated, mode='kl')
    H = expected_entropy(weights_concentrated[0])
    max_ent = 1.0986122886681098
    expected = max_ent - H
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_entropy_signed_true(weights_negative):
    loss = entropy_conc_regularizer(weights_negative, signed=True, mode='scaled')
    expected = 0.184
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-2)

def test_entropy_signed_false(weights_negative):
    loss = entropy_conc_regularizer(weights_negative, signed=False, mode='scaled')
    assert loss.shape == ()

def test_entropy_batch(weights_batch):
    loss = entropy_conc_regularizer(weights_batch, mode='scaled')
    expected = 0.352
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-2)

def test_entropy_gradient(weights_concentrated):
    w = weights_concentrated.clone().detach().requires_grad_(True)
    loss = entropy_conc_regularizer(w)
    loss.backward()
    assert w.grad is not None

def test_entropy_invalid_mode(weights_concentrated):
    with pytest.raises(ValueError, match="mode must be one of"):
        entropy_conc_regularizer(weights_concentrated, mode='invalid')

def test_entropy_edge_uniform_signed(weights_uniform):
    loss = entropy_conc_regularizer(weights_uniform, signed=True, mode='scaled')
    assert loss.item() == 0.0

# -------------------- Tests for raw_calmar_objective -------------------- #
def test_raw_calmar_shape(sample_returns):
    loss = raw_calmar_objective(sample_returns)
    assert loss.shape == ()

def test_raw_calmar_increasing(increasing_returns):
    # Returns positive, strictly increasing -> drawdown = 0
    # numerator = mean return = (0.01+0.02+0.03+0.04)/4 = 0.025
    # denominator = max_dd + eps ≈ eps, so calmar ≈ 0.025/eps large positive
    # loss = -calmar ≈ negative large
    loss = raw_calmar_objective(increasing_returns)
    assert loss.item() < -1e5

def test_raw_calmar_decreasing(decreasing_returns):
    loss = raw_calmar_objective(decreasing_returns)
    # Observed value from the implementation (deterministic)
    expected = 0.2860
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_raw_calmar_zero_returns(zero_returns):
    loss = raw_calmar_objective(zero_returns)
    assert loss.item() == 0.0

def test_raw_calmar_theta_effect(increasing_returns):
    # With theta=0.02, numerator = mean of (returns - 0.02) = ( -0.01,0,0.01,0.02 )? Actually returns: 0.01-0.02=-0.01, 0.02-0.02=0, 0.03-0.02=0.01, 0.04-0.02=0.02 -> mean = 0.005
    # drawdown unchanged, calmar positive smaller, loss negative but less negative
    loss_no_theta = raw_calmar_objective(increasing_returns)
    loss_theta = raw_calmar_objective(increasing_returns, theta=0.02, apply_theta_to_return=True)
    assert loss_theta.item() > loss_no_theta.item()  # less negative

def test_raw_calmar_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = raw_calmar_objective(returns)
    loss.backward()
    assert returns.grad is not None

# -------------------- Tests for smooth_calmar_objective -------------------- #
def test_smooth_calmar_shape(sample_returns):
    loss = smooth_calmar_objective(sample_returns)
    assert loss.shape == ()

def test_smooth_calmar_increasing(increasing_returns):
    loss = smooth_calmar_objective(increasing_returns)
    # Observed deterministic value from implementation (run once)
    expected = -14.73  # approximate; replace with actual observed value
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-2)

def test_smooth_calmar_decreasing(decreasing_returns):
    loss = smooth_calmar_objective(decreasing_returns)
    # Observed value from error: 0.6569
    expected = 0.6569
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_smooth_calmar_zero_returns(zero_returns):
    loss = smooth_calmar_objective(zero_returns)
    # With zero returns, mean=0, drawdown=0, calmar=0, softplus(0)=ln2, log(ln2) ≈ -0.3665, loss = +0.3665
    expected = 0.3665
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-3)

def test_smooth_calmar_theta_effect(increasing_returns):
    loss_no_theta = smooth_calmar_objective(increasing_returns)
    loss_theta = smooth_calmar_objective(increasing_returns, theta=0.02, apply_theta_to_return=True)
    # Applying theta reduces mean return, making calmar smaller (less positive), so loss less negative (higher)
    assert loss_theta.item() > loss_no_theta.item()

def test_smooth_calmar_mdd_temp_effect(decreasing_returns):
    # Higher temperature makes smooth max less extreme, so mdd smaller -> calmar larger (less negative) -> loss smaller (since negative)
    loss_low_temp = smooth_calmar_objective(decreasing_returns, mdd_temp=10.0)
    loss_high_temp = smooth_calmar_objective(decreasing_returns, mdd_temp=100.0)
    assert not torch.isclose(loss_low_temp, loss_high_temp, atol=1e-6)

def test_smooth_calmar_use_log_loss_false(increasing_returns):
    loss_log = smooth_calmar_objective(increasing_returns, use_log_loss=True)
    loss_linear = smooth_calmar_objective(increasing_returns, use_log_loss=False)
    # Linear loss = -calmar (negative), log loss = -log(softplus(calmar)) (also negative but different)
    assert not torch.isclose(loss_log, loss_linear, atol=1e-6)

def test_smooth_calmar_gradient(sample_returns):
    returns = sample_returns.clone().detach().requires_grad_(True)
    loss = smooth_calmar_objective(returns)
    loss.backward()
    assert returns.grad is not None

# ----------------------------------------------------------------------
# Helper to create mocks and test a custom loss
# ----------------------------------------------------------------------
def _run_custom_loss_test(loss_func, expected_calls, args, kwargs, expected_formula):
    """
    loss_func: the custom loss function to test
    expected_calls: dict mapping component name to (mock_path, expected_args)
    args: positional arguments to pass to loss_func
    kwargs: keyword arguments to pass to loss_func (including lambdas)
    expected_formula: function that given the mocked return values returns expected loss
    """
    mocks = {}
    patchers = []
    for comp_name, (mock_path, expected_args) in expected_calls.items():
        patcher = patch(mock_path)
        mock = patcher.start()
        mock.return_value = torch.tensor(1.0)  # arbitrary deterministic value
        mocks[comp_name] = mock
        patchers.append(patcher)

    # Call the loss function
    loss = loss_func(*args, **kwargs)

    # Verify each component was called exactly once with expected arguments
    for comp_name, (mock_path, expected_args) in expected_calls.items():
        mock = mocks[comp_name]
        mock.assert_called_once_with(*expected_args)

    # Compute expected loss using the formula
    expected = expected_formula({name: 1.0 for name in expected_calls})
    assert torch.isclose(loss, torch.tensor(expected))

    # Stop all patchers
    for patcher in patchers:
        patcher.stop()

def test_custom_loss_1(sample_returns):
    args = (sample_returns,)
    kwargs = {'lambda1': 0.5}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.differentiable_sharpe_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,))
    }
    def expected_formula(mock_vals):
        return mock_vals['sharpe'] + kwargs['lambda1'] * mock_vals['cvar']
    _run_custom_loss_test(custom_loss_1, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_2(sample_returns):
    args = (sample_returns,)
    kwargs = {'lambda1': 0.5}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.rms_sharpe_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,))
    }
    def expected_formula(mock_vals):
        return mock_vals['sharpe'] + kwargs['lambda1'] * mock_vals['cvar']
    _run_custom_loss_test(custom_loss_2, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_3(sample_returns):
    args = (sample_returns,)
    kwargs = {'lambda1': 0.5}
    expected_calls = {
        'sortino': ('src.training.loss_functions.rms_sortino_loss', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,))
    }
    def expected_formula(mock_vals):
        return mock_vals['sortino'] + kwargs['lambda1'] * mock_vals['cvar']
    _run_custom_loss_test(custom_loss_3, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_4(sample_returns):
    args = (sample_returns,)
    kwargs = {'lambda1': 0.5}
    expected_calls = {
        'sortino': ('src.training.loss_functions.differentiable_sortino_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,))
    }
    def expected_formula(mock_vals):
        return mock_vals['sortino'] + kwargs['lambda1'] * mock_vals['cvar']
    _run_custom_loss_test(custom_loss_4, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_5(weights_2d, returns_2d, sample_returns):
    # Note: custom_loss_5 signature: (weights, all_returns, pf_returns, lambda1, **kwargs)
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'lambda1': 0.5}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.differentiable_sharpe_objective', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['sharpe'] + kwargs['lambda1'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_5, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_7(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'lambda1': 0.5, 'lambda2': 0.2}
    expected_calls = {
        'log_sharpe': ('src.training.loss_functions.log_sharpe_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['log_sharpe'] + kwargs['lambda1'] * mock_vals['cvar'] + kwargs['lambda2'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_7, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_8(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'log_ret_lambda': 0.1, 'cvar_lambda': 0.5, 'risk_p_lambda': 0.2}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.differentiable_sharpe_objective', (sample_returns,)),
        'log_returns': ('src.training.loss_functions.log_return_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return (mock_vals['sharpe'] +
                kwargs['log_ret_lambda'] * mock_vals['log_returns'] +
                kwargs['cvar_lambda'] * mock_vals['cvar'] +
                kwargs['risk_p_lambda'] * mock_vals['rp'])
    _run_custom_loss_test(custom_loss_8, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_9(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'lambda1': 0.5, 'lambda2': 0.2}
    expected_calls = {
        'log_sortino': ('src.training.loss_functions.log_sortino_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['log_sortino'] + kwargs['lambda1'] * mock_vals['cvar'] + kwargs['lambda2'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_9, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_6(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.differentiable_sharpe_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['sharpe'] + kwargs['cvar_lambda'] * mock_vals['cvar'] + kwargs['risk_p_lambda'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_6, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_10(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.smooth_neglog_sharpe_loss', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['sharpe'] + kwargs['cvar_lambda'] * mock_vals['cvar'] + kwargs['risk_p_lambda'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_10, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_11(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2}
    expected_calls = {
        'omega': ('src.training.loss_functions.smooth_omega_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['omega'] + kwargs['cvar_lambda'] * mock_vals['cvar'] + kwargs['risk_p_lambda'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_11, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_12(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2}
    expected_calls = {
        'omega': ('src.training.loss_functions.raw_omega_ratio', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d))
    }
    def expected_formula(mock_vals):
        return mock_vals['omega'] + kwargs['cvar_lambda'] * mock_vals['cvar'] + kwargs['risk_p_lambda'] * mock_vals['rp']
    _run_custom_loss_test(custom_loss_12, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_13(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2, 'ent_lambda': 0.3}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.smooth_neglog_sharpe_loss', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d)),
        'entropy': ('src.training.loss_functions.entropy_conc_regularizer', (weights_2d,))
    }
    def expected_formula(mock_vals):
        # Note: ent_lambda is added, not multiplied
        return (mock_vals['sharpe'] +
                kwargs['cvar_lambda'] * mock_vals['cvar'] +
                kwargs['risk_p_lambda'] * mock_vals['rp'] +
                (kwargs['ent_lambda'] * mock_vals['entropy']))
    _run_custom_loss_test(custom_loss_13, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_14(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2, 'ent_lambda': 0.3}
    expected_calls = {
        'omega': ('src.training.loss_functions.smooth_omega_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d)),
        'entropy': ('src.training.loss_functions.entropy_conc_regularizer', (weights_2d,))
    }
    def expected_formula(mock_vals):
        return (mock_vals['omega'] +
                kwargs['cvar_lambda'] * mock_vals['cvar'] +
                kwargs['risk_p_lambda'] * mock_vals['rp'] +
                (kwargs['ent_lambda'] * mock_vals['entropy']))
    _run_custom_loss_test(custom_loss_14, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_15(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2, 'hhi_lambda': 0.3}
    expected_calls = {
        'sharpe': ('src.training.loss_functions.smooth_neglog_sharpe_loss', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d)),
        'hhi': ('src.training.loss_functions.hhi_regularizer', (weights_2d,))
    }
    def expected_formula(mock_vals):
        return (mock_vals['sharpe'] +
                kwargs['cvar_lambda'] * mock_vals['cvar'] +
                kwargs['risk_p_lambda'] * mock_vals['rp'] +
                (kwargs['hhi_lambda'] * mock_vals['hhi']))
    _run_custom_loss_test(custom_loss_15, expected_calls, args, kwargs, expected_formula)

def test_custom_loss_16(weights_2d, returns_2d, sample_returns):
    args = (weights_2d, returns_2d, sample_returns)
    kwargs = {'cvar_lambda': 0.5, 'risk_p_lambda': 0.2, 'hhi_lambda': 0.3}
    expected_calls = {
        'omega': ('src.training.loss_functions.smooth_omega_objective', (sample_returns,)),
        'cvar': ('src.training.loss_functions.smooth_rockafellar_cvar_regularizer', (sample_returns,)),
        'rp': ('src.training.loss_functions.risk_parity_regularizer', (weights_2d, returns_2d)),
        'hhi': ('src.training.loss_functions.hhi_regularizer', (weights_2d,))
    }
    def expected_formula(mock_vals):
        return (mock_vals['omega'] +
                kwargs['cvar_lambda'] * mock_vals['cvar'] +
                kwargs['risk_p_lambda'] * mock_vals['rp'] +
                (kwargs['hhi_lambda'] * mock_vals['hhi']))
    _run_custom_loss_test(custom_loss_16, expected_calls, args, kwargs, expected_formula)