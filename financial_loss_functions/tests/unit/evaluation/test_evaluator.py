import pytest
import numpy as np
import pandas as pd
from src.evaluation.evaluator import Evaluator, EqualWeightCalculator

# =================== Tests for Evaluator =================== #

# Dummy metric functions for testing
def dummy_sharpe(returns):
    return np.mean(returns) / (np.std(returns) + 1e-8)

def dummy_cvar(returns):
    return -np.mean(returns[returns < 0]) if np.any(returns < 0) else 0.0

@pytest.fixture
def evaluator_with_returns():
    # Creating dummy returns: 2 windows, each with 3 days, 2 models
    returns = {
        'modelA': np.array([[0.01, 0.02, 0.03],
                            [0.04, 0.05, 0.06]]),
        'modelB': np.array([[0.02, 0.03, 0.04],
                            [0.05, 0.06, 0.07]])
    }
    evaluator = Evaluator(eval_returns=None, all_daily_returns=returns)
    evaluator.metrics_lib = {'sharpe': dummy_sharpe, 'cvar': dummy_cvar}
    evaluator.rounding_digits = 4
    return evaluator

# -------------------- Tests for __init__ -------------------- #
def test_init_with_eval_returns():
    eval_returns = np.random.randn(2, 3, 4)
    evaluator = Evaluator(eval_returns)
    assert evaluator.eval_returns is eval_returns
    assert evaluator.ba_eval is None
    assert evaluator.all_daily_returns == {}

def test_init_with_eval_returns_and_ba():
    eval_returns = np.random.randn(2, 3, 4)
    ba_eval = np.random.randn(2, 4)
    evaluator = Evaluator(eval_returns, ba_eval)
    assert evaluator.ba_eval is ba_eval

def test_init_without_eval_returns_but_with_all_daily_returns():
    all_daily_returns = {'model1': np.array([[0.1,0.2],[0.3,0.4]])}
    evaluator = Evaluator(eval_returns=None, all_daily_returns=all_daily_returns)
    assert evaluator.eval_returns is None
    assert evaluator.ba_eval is None
    assert evaluator.all_daily_returns == all_daily_returns

def test_init_without_eval_returns_and_no_all_daily_returns_raises():
    with pytest.raises(ValueError, match="If out-of-sample evaluation data is not provided"):
        Evaluator(eval_returns=None, all_daily_returns=None)

def test_init_invalid_eval_returns_ndim_raises():
    eval_returns = np.random.randn(2, 3)  # 2D instead of 3D
    with pytest.raises(ValueError, match="Evaluation Returns must have 3 dim"):
        Evaluator(eval_returns)

def test_init_ba_eval_wrong_shape():
    eval_returns = np.random.randn(2, 3, 4)
    ba_eval = np.random.randn(2, 3, 4)  # 3D instead of 2D
    evaluator = Evaluator(eval_returns, ba_eval)
    # Should print warning and set ba_eval to None
    assert evaluator.ba_eval is None

# -------------------- Tests for _calc_step_ba_costs -------------------- #
def test_calc_step_ba_costs_first_step():
    evaluator = Evaluator(np.random.randn(1,1,1))  # dummy instance
    curr_weights = np.array([0.4, 0.6])
    first_d_bas = np.array([0.001, 0.002])
    cost = evaluator._calc_step_ba_costs(None, curr_weights, first_d_bas)
    expected = 0.5 * (0.4*0.001 + 0.6*0.002)
    assert cost == expected

def test_calc_step_ba_costs_subsequent_step():
    evaluator = Evaluator(np.random.randn(1,1,1))
    prev_weights = np.array([0.3, 0.7])
    curr_weights = np.array([0.4, 0.6])
    first_d_bas = np.array([0.001, 0.002])
    cost = evaluator._calc_step_ba_costs(prev_weights, curr_weights, first_d_bas)
    expected = 0.5 * (0.1*0.001 + 0.1*0.002)  # 0.00015
    assert cost == pytest.approx(expected)

def test_calc_step_ba_costs_spread_cost_factor():
    evaluator = Evaluator(np.random.randn(1,1,1))
    original_factor = evaluator.spread_cost_factor
    evaluator.spread_cost_factor = 1.0
    curr_weights = np.array([0.5, 0.5])
    first_d_bas = np.array([0.001, 0.001])
    cost = evaluator._calc_step_ba_costs(None, curr_weights, first_d_bas)
    expected = 1.0 * (0.5*0.001 + 0.5*0.001)
    assert cost == expected
    evaluator.spread_cost_factor = original_factor

# -------------------- Tests for _calc_net_returns -------------------- #
def test_calc_net_returns():
    daily_rets = np.array([0.02, 0.01, 0.03])
    cost = 0.001
    result = Evaluator._calc_net_returns(daily_rets.copy(), cost)
    expected_first = (1 + 0.02) * (1 - cost) - 1
    assert result[0] == expected_first
    assert result[1] == 0.01
    assert result[2] == 0.03

# -------------------- Tests for calc_pf_daily_rets (without BA spreads) -------------------- #
def test_calc_pf_daily_rets_no_ba():
    eval_returns = np.array([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]]
    ])
    eval_weights = np.array([
        [0.4, 0.6],
        [0.5, 0.5]
    ])
    evaluator = Evaluator(eval_returns)
    evaluator.calc_pf_daily_rets(eval_weights, 'test_model')
    expected = np.array([
        [0.16, 0.36, 0.56],
        [0.15, 0.35, 0.55]
    ])
    np.testing.assert_array_almost_equal(evaluator.all_daily_returns['test_model'], expected)

# -------------------- Tests for calc_pf_daily_rets (with BA spreads) -------------------- #
def test_calc_pf_daily_rets_with_ba():
    eval_returns = np.array([
        [[0.01, 0.02], [0.03, 0.04]],
        [[0.02, 0.01], [0.04, 0.03]]
    ])
    ba_eval = np.array([
        [0.001, 0.002],
        [0.002, 0.001]
    ])
    eval_weights = np.array([
        [0.4, 0.6],
        [0.5, 0.5]
    ])
    evaluator = Evaluator(eval_returns, ba_eval)
    evaluator.calc_pf_daily_rets(eval_weights, 'test_model')
    result = evaluator.all_daily_returns['test_model']
    # Manual calculation (as before)
    expected = np.array([
        [0.0151872, 0.036],
        [0.01484775, 0.035]
    ])
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_calc_pf_daily_rets_weights_ndim_error(capsys):
    eval_returns = np.random.randn(2, 3, 2)
    evaluator = Evaluator(eval_returns)
    eval_weights = np.random.randn(2, 3, 2)
    evaluator.calc_pf_daily_rets(eval_weights, 'test')
    captured = capsys.readouterr()
    assert "Evaluation weights array must have only 2 dims" in captured.out

# -------------------- Tests for get_rets_for_one, update_rets_for_one, add_benchmark_rets -------------------- #
def test_get_rets_for_one():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={'modelA': np.array([1,2])})
    assert np.array_equal(evaluator.get_rets_for_one('modelA'), np.array([1,2]))
    assert evaluator.get_rets_for_one('modelB') is None

def test_update_rets_for_one():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={'modelA': np.array([1,2])})
    new_returns = np.array([3,4])
    evaluator.update_rets_for_one('modelA', new_returns)
    assert np.array_equal(evaluator.all_daily_returns['modelA'], new_returns)

def test_update_rets_for_one_missing_model():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={})
    with pytest.warns(UserWarning, match="Returns for modelX do not exist"):
        evaluator.update_rets_for_one('modelX', np.array([1]))

def test_add_benchmark_rets():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={})
    bench_rets = np.array([0.1,0.2])
    evaluator.add_benchmark_rets('SP500', bench_rets)
    assert np.array_equal(evaluator.all_daily_returns['SP500'], bench_rets)

# -------------------- Tests for calc_metric_performance -------------------- #
def test_calc_metric_performance_mean_false():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={
        'modelA': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'modelB': np.array([[0.5, 0.6], [0.7, 0.8]])
    })
    result = evaluator.calc_metric_performance(np.mean, mean=False)
    expected = pd.DataFrame({
        'modelA': [0.15, 0.35],
        'modelB': [0.55, 0.75]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_calc_metric_performance_mean_true():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={
        'modelA': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'modelB': np.array([[0.5, 0.6], [0.7, 0.8]])
    })
    result = evaluator.calc_metric_performance(np.mean, mean=True)
    expected = pd.Series([0.25, 0.65], index=['modelA', 'modelB'], name=None)
    pd.testing.assert_series_equal(result, expected)

def test_calc_metric_performance_no_daily_returns_raises():
    evaluator = Evaluator(eval_returns=np.random.randn(1,1,1))
    with pytest.raises(ValueError, match="No daily returns calculated"):
        evaluator.calc_metric_performance(np.mean)

# -------------------- Tests for calc_avg_performance -------------------- #
def test_calc_avg_performance():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={
        'modelA': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'modelB': np.array([[0.5, 0.6], [0.7, 0.8]])
    })
    metrics_lib = {'sharpe': dummy_sharpe, 'cvar': dummy_cvar}
    evaluator.metrics_lib = metrics_lib
    result = evaluator.calc_avg_performance()
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'sharpe', 'cvar'}
    assert set(result.index) == {'modelA', 'modelB'}

def test_calc_avg_performance_no_metrics_lib(capsys):
    evaluator = Evaluator(eval_returns=None, all_daily_returns={'modelA': np.array([1])})
    evaluator.metrics_lib = None
    result = evaluator.calc_avg_performance()
    assert result is None
    captured = capsys.readouterr()
    assert "No metrics library or dict provided" in captured.out

# -------------------- Tests for _combine_rets_winds -------------------- #
def test_combine_rets_winds(evaluator_with_returns):
    combined = evaluator_with_returns._combine_rets_winds()
    assert isinstance(combined, dict)
    assert set(combined.keys()) == {'modelA', 'modelB'}
    # modelA: [0.01,0.02,0.03,0.04,0.05,0.06]
    expected_A = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    np.testing.assert_array_equal(combined['modelA'], expected_A)
    # modelB: [0.02,0.03,0.04,0.05,0.06,0.07]
    expected_B = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    np.testing.assert_array_equal(combined['modelB'], expected_B)

# -------------------- Tests for _calc_overall_metric_perf -------------------- #
def test_calc_overall_metric_perf(evaluator_with_returns):
    combined = evaluator_with_returns._combine_rets_winds()
    # Test with dummy_sharpe
    result = evaluator_with_returns._calc_overall_metric_perf(dummy_sharpe, combined)
    expected = {
        'modelA': round(dummy_sharpe(np.array([0.01,0.02,0.03,0.04,0.05,0.06])), 4),
        'modelB': round(dummy_sharpe(np.array([0.02,0.03,0.04,0.05,0.06,0.07])), 4)
    }
    assert result == expected

def test_calc_overall_metric_perf_with_cvar(evaluator_with_returns):
    combined = evaluator_with_returns._combine_rets_winds()
    result = evaluator_with_returns._calc_overall_metric_perf(dummy_cvar, combined)
    # Manual compute dummy_cvar for modelA: negative returns? none -> 0.0
    expected_A = 0.0
    expected_B = 0.0  # no negative returns in dummy data
    assert result['modelA'] == expected_A
    assert result['modelB'] == expected_B

# -------------------- Tests for calc_pf_performances -------------------- #
def test_calc_pf_performances(evaluator_with_returns):
    df = evaluator_with_returns.calc_pf_performances()
    assert isinstance(df, pd.DataFrame)
    # Expected columns: metrics ('sharpe', 'cvar')
    assert set(df.columns) == {'sharpe', 'cvar'}
    # Expected index: models
    assert set(df.index) == {'modelA', 'modelB'}
    # Check a value
    sharpe_A = dummy_sharpe(np.array([0.01,0.02,0.03,0.04,0.05,0.06]))
    assert df.loc['modelA', 'sharpe'] == round(sharpe_A, 4)

def test_calc_pf_performances_no_metrics_lib(capsys):
    evaluator = Evaluator(eval_returns=None, all_daily_returns={'modelA': np.array([[1,2],[3,4]])})
    evaluator.metrics_lib = None
    result = evaluator.calc_pf_performances()
    assert result is None
    captured = capsys.readouterr()
    assert "No metrics library or dict provided" in captured.out

def test_calc_pf_performances_empty_returns_raises():
    evaluator = Evaluator(eval_returns=None, all_daily_returns={})
    evaluator.metrics_lib = {'sharpe': dummy_sharpe}
    with pytest.raises(ValueError, match="No daily returns calculated"):
        evaluator.calc_pf_performances()

# -------------------- Tests for get_all_daily_returns -------------------- #
def test_get_all_daily_returns():
    all_rets = {'modelA': np.array([1,2])}
    evaluator = Evaluator(eval_returns=None, all_daily_returns=all_rets)
    assert evaluator.get_all_daily_returns() is all_rets

# -------------------- Tests for update_spread_cost_factor -------------------- #
def test_update_spread_cost_factor():
    evaluator = Evaluator(eval_returns=np.random.randn(1,1,1))
    evaluator.update_spread_cost_factor(0.3)
    assert evaluator.spread_cost_factor == 0.3

def test_update_spread_cost_factor_invalid():
    evaluator = Evaluator(eval_returns=np.random.randn(1,1,1))
    with pytest.raises(ValueError, match="Spread Cost factor cannot be greater than 1"):
        evaluator.update_spread_cost_factor(1.5)

# =================== Tests for EqualWeightCalculator =================== #
# -------------------- Tests for _equal_weight_pf -------------------- #
def test_equal_weight_pf():
    weights = EqualWeightCalculator._equal_weight_pf(5)
    expected = np.full(5, 0.2)
    np.testing.assert_array_equal(weights, expected)

def test_equal_weight_pf_single_ticker():
    weights = EqualWeightCalculator._equal_weight_pf(1)
    expected = np.array([1.0])
    np.testing.assert_array_equal(weights, expected)

def test_equal_weight_pf_zero_tickers():
    with pytest.raises(ZeroDivisionError):
        EqualWeightCalculator._equal_weight_pf(0)

# -------------------- Tests for calc_eq_wt_daily_rets -------------------- #
def test_calc_eq_wt_daily_rets_basic():
    # Create eval_returns shape: (2 windows, 3 time steps, 2 stocks)
    eval_returns = np.array([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],   # window 0
        [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]]    # window 1
    ])
    calc = EqualWeightCalculator(eval_returns)
    result = calc.calc_eq_wt_daily_rets()
    # Equal weights: [0.5, 0.5]
    # Window 0: daily returns = [0.1*0.5+0.2*0.5=0.15, 0.3*0.5+0.4*0.5=0.35, 0.5*0.5+0.6*0.5=0.55]
    # Window 1: [0.2*0.5+0.1*0.5=0.15, 0.4*0.5+0.3*0.5=0.35, 0.6*0.5+0.5*0.5=0.55]
    expected = np.array([
        [0.15, 0.35, 0.55],
        [0.15, 0.35, 0.55]
    ])
    np.testing.assert_array_almost_equal(result, expected)
    # Check that eq_weights and eq_weights_rets are set
    np.testing.assert_array_equal(calc.eq_weights, np.array([0.5, 0.5]))
    assert calc.eq_weights_rets is result

def test_calc_eq_wt_daily_rets_single_window():
    eval_returns = np.array([[[0.1, 0.2], [0.3, 0.4]]])  # (1,2,2)
    calc = EqualWeightCalculator(eval_returns)
    result = calc.calc_eq_wt_daily_rets()
    expected = np.array([[0.15, 0.35]])
    np.testing.assert_array_almost_equal(result, expected)

def test_calc_eq_wt_daily_rets_single_stock():
    eval_returns = np.array([[[0.1], [0.2], [0.3]]])  # (1,3,1)
    calc = EqualWeightCalculator(eval_returns)
    result = calc.calc_eq_wt_daily_rets()
    # weights = [1.0]
    expected = np.array([[0.1, 0.2, 0.3]])
    np.testing.assert_array_almost_equal(result, expected)

# -------------------- Tests for get_eq_weights -------------------- #
def test_get_eq_weights_before_calculation(capsys):
    eval_returns = np.random.randn(2, 3, 2)
    calc = EqualWeightCalculator(eval_returns)
    weights = calc.get_eq_weights()
    assert weights is None
    captured = capsys.readouterr()
    assert "WARNING: No equal weights calculated." in captured.out
    assert "Run `EqualWeightCalculator.calc_eq_wt_daily_rets()` first." in captured.out

def test_get_eq_weights_after_calculation():
    eval_returns = np.random.randn(2, 3, 2)
    calc = EqualWeightCalculator(eval_returns)
    calc.calc_eq_wt_daily_rets()
    weights = calc.get_eq_weights()
    expected = np.array([0.5, 0.5])
    np.testing.assert_array_equal(weights, expected)