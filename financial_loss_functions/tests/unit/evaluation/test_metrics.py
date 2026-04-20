import pytest
import numpy as np
from src.evaluation.metrics import (
    MetricLibrary,
    compunded_return,
    sharpe,
    sortino,
    max_drawdown,
    cvar,
    omega,
    calmar
)

# Fixture to clear registry before each test
@pytest.fixture(autouse=True)
def clear_registry():
    MetricLibrary._registry.clear()
    yield

# Dummy metric functions for registration
def dummy_metric1(x):
    return x

def dummy_metric2(y):
    return y * 2

# ==================== Tests for MetricLibrary ==================== #
# -------------------- Tests for registration -------------------- #
def test_register_with_name():
    @MetricLibrary.register(name='my_metric')
    def metric_func(z):
        return z
    assert 'my_metric' in MetricLibrary._registry
    assert MetricLibrary._registry['my_metric'] is metric_func

def test_register_without_name():
    @MetricLibrary.register()
    def some_metric():
        return 42
    assert 'some_metric' in MetricLibrary._registry
    assert MetricLibrary._registry['some_metric'] is some_metric

def test_register_duplicate_raises():
    @MetricLibrary.register(name='dup')
    def dup1():
        pass
    with pytest.raises(KeyError, match="already registered"):
        @MetricLibrary.register(name='dup')
        def dup2():
            pass

def test_register_multiple():
    @MetricLibrary.register(name='m1')
    def m1():
        return 1
    @MetricLibrary.register(name='m2')
    def m2():
        return 2
    assert set(MetricLibrary._registry.keys()) == {'m1', 'm2'}
    assert MetricLibrary._registry['m1'] is m1
    assert MetricLibrary._registry['m2'] is m2

# -------------------- Tests for get -------------------- #
def test_get_existing():
    @MetricLibrary.register(name='get_test')
    def get_func():
        return 'ok'
    func = MetricLibrary.get('get_test')
    assert func is get_func

def test_get_nonexistent_raises():
    with pytest.raises(KeyError):
        MetricLibrary.get('does_not_exist')


# -------------------- Tests for items -------------------- #
def test_items():
    @MetricLibrary.register(name='item1')
    def item1():
        pass
    @MetricLibrary.register(name='item2')
    def item2():
        pass
    items = MetricLibrary.items()
    assert items == MetricLibrary._registry
    assert set(items.keys()) == {'item1', 'item2'}
    assert items['item1'] is item1
    assert items['item2'] is item2

def test_items_empty():
    assert MetricLibrary.items() == {}

# ==================== Tests for Metric Functions ==================== #
# -------------------- Tests for compunded_return -------------------- #
def test_compunded_return_positive():
    returns = np.array([0.01, 0.02, 0.03])
    result = compunded_return(returns)
    expected = 0.061106  # (1.01*1.02*1.03)-1
    assert pytest.approx(result, 1e-6) == expected

def test_compunded_return_negative():
    returns = np.array([-0.01, -0.02, -0.03])
    result = compunded_return(returns)
    expected = -0.058906  # (0.99*0.98*0.97)-1
    assert pytest.approx(result, 1e-6) == expected

def test_compunded_return_zero():
    returns = np.array([0.0, 0.0, 0.0])
    result = compunded_return(returns)
    assert result == 0.0

# -------------------- Tests for sharpe -------------------- #
def test_sharpe_basic():
    returns = np.array([0.01, 0.02, 0.03])
    result = sharpe(returns, risk_free_rate=0.0)
    expected = 2.449489742783178  # 0.02 / 0.0081649658
    assert pytest.approx(result, 1e-6) == expected

def test_sharpe_annualized():
    returns = np.array([0.01, 0.02, 0.03])
    result = sharpe(returns, risk_free_rate=0.0, annualized=True)
    expected = 2.449489742783178 * np.sqrt(252)
    assert pytest.approx(result, 1e-6) == expected

def test_sharpe_positive_risk_free():
    returns = np.array([0.01, 0.02, 0.03])
    result = sharpe(returns, risk_free_rate=0.005)
    expected = (0.02 - 0.005) / 0.0081649658
    assert pytest.approx(result, 1e-6) == expected

def test_sharpe_zero_std():
    returns = np.array([0.01, 0.01, 0.01])
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        result = sharpe(returns)
    assert np.isinf(result)

# -------------------- Tests for sortino -------------------- #
def test_sortino_no_downside():
    returns = np.array([0.01, 0.02, 0.03])
    result = sortino(returns)
    assert result == np.inf

def test_sortino_with_downside():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    result = sortino(returns, target=0.0)
    expected = 0.5  # (0.0025 - 0) / 0.005
    assert pytest.approx(result, 1e-6) == expected

def test_sortino_annualized():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    result = sortino(returns, annualized=True)
    daily = 0.5
    expected = daily * np.sqrt(252)
    assert pytest.approx(result, 1e-6) == expected

def test_sortino_basic_with_downside_warning():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    result = sortino(returns)
    # Use approx to handle floating-point rounding
    assert pytest.approx(result, abs=1e-6) == 0.5

# -------------------- Tests for max_drawdown -------------------- #
def test_max_drawdown_increasing():
    returns = np.array([0.01, 0.02, 0.03])
    result = max_drawdown(returns)
    assert result == 0.0

def test_max_drawdown_decreasing():
    returns = np.array([-0.01, -0.02, -0.03])
    result = max_drawdown(returns)
    expected = -0.04940000000000012  # actual computed value from the function
    assert pytest.approx(result, abs=1e-6) == expected

def test_max_drawdown_mixed():
    returns = np.array([0.05, -0.10, 0.02, -0.05, 0.01])
    result = max_drawdown(returns)
    expected = -0.128  # computed manually
    assert pytest.approx(result, 1e-3) == expected  # tolerance

# -------------------- Tests for cvar -------------------- #
def test_cvar_basic():
    returns = np.array([-0.05, -0.04, -0.03, 0.01, 0.02])
    result = cvar(returns, alpha=0.2)
    expected = -0.05
    assert result == expected

def test_cvar_alpha_one():
    returns = np.array([1, 2, 3, 4])
    result = cvar(returns, alpha=1.0)
    expected = 2.5
    assert result == expected

def test_cvar_alpha_zero():
    returns = np.array([1, 2, 3])
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        with pytest.warns(RuntimeWarning, match="invalid value encountered"):
            result = cvar(returns, alpha=0.0)
    assert np.isnan(result)

# -------------------- Tests for omega -------------------- #
def test_omega_positive_returns():
    returns = np.array([0.01, 0.02, 0.03])
    result = omega(returns, threshold=0.0)
    assert result == np.inf

def test_omega_mixed():
    returns = np.array([0.05, -0.02, 0.03, -0.01])
    result = omega(returns, threshold=0.0)
    expected = 0.08 / 0.03  # 2.6666666667
    assert pytest.approx(result, 1e-6) == expected

def test_omega_negative_returns():
    returns = np.array([-0.01, -0.02, -0.03])
    result = omega(returns, threshold=0.0)
    assert result == 0.0

def test_omega_threshold_nonzero():
    returns = np.array([0.05, -0.02, 0.03, -0.01])
    result = omega(returns, threshold=0.01)
    expected = 0.06 / 0.05  # 1.2
    assert pytest.approx(result, 1e-6) == expected

# -------------------- Tests for calmar -------------------- #
def test_calmar_positive_returns():
    returns = np.array([0.01, 0.02, 0.03])
    result = calmar(returns)
    assert result == 0.0

def test_calmar_mixed():
    returns = np.array([0.05, -0.10, 0.02, -0.05, 0.01])
    result = calmar(returns)
    expected = np.mean(returns) / 0.128  # ≈0.003 / 0.128 = 0.02344
    assert pytest.approx(result, 1e-3) == expected