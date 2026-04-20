import pytest
import pandas as pd
from src.evaluation.selection import (
    high_corr_with_each_metric,
    pareto_dominance,
    filter_models
)

# -------------------- Tests for filter_models -------------------- #
def test_filter_models_basic():
    data = {
        'sharpe': [0.10, 0.08, 0.12, 0.05],
        'calmar': [0.2, 0.1, 0.3, 0.0]
    }
    index = ['ModelA', 'ModelB', 'ModelC', 'Equal_Weight']
    df = pd.DataFrame(data, index=index)
    keep = ['Equal_Weight']
    filtered_df, filtered_models = filter_models(df, 'Equal_Weight', 'sharpe', keep)
    # All models (A, B, C) have Sharpe > 0.05, so all should be kept, plus the benchmark
    expected_df = df.loc[['ModelA', 'ModelB', 'ModelC', 'Equal_Weight']]
    pd.testing.assert_frame_equal(filtered_df, expected_df)
    assert filtered_models == ['ModelA', 'ModelB', 'ModelC']

def test_filter_models_no_models_beat():
    data = {'sharpe': [0.04, 0.03, 0.05]}
    index = ['ModelX', 'ModelY', 'Equal_Weight']
    df = pd.DataFrame(data, index=index)
    keep = ['Equal_Weight']
    filtered_df, filtered_models = filter_models(df, 'Equal_Weight', 'sharpe', keep)
    expected_df = df.loc[['Equal_Weight']]  # only benchmark kept
    pd.testing.assert_frame_equal(filtered_df, expected_df)
    assert filtered_models == []

def test_filter_models_multiple_benchmarks():
    data = {'sharpe': [0.10, 0.06, 0.08, 0.07]}
    index = ['ModelA', 'ModelB', 'S&P500', 'Equal_Weight']
    df = pd.DataFrame(data, index=index)
    keep = ['S&P500', 'Equal_Weight']
    # Benchmark for comparison is 'Equal_Weight' (sharpe=0.07)
    filtered_df, filtered_models = filter_models(df, 'Equal_Weight', 'sharpe', keep)
    # Models beating 0.07: ModelA (0.10), ModelB (0.06 does NOT beat, so only ModelA)
    expected_df = df.loc[['ModelA', 'S&P500', 'Equal_Weight']]
    pd.testing.assert_frame_equal(filtered_df, expected_df)
    assert filtered_models == ['ModelA']  # ModelB excluded

def test_filter_models_benchmark_not_in_index():
    data = {'sharpe': [0.10, 0.08]}
    index = ['ModelA', 'ModelB']
    df = pd.DataFrame(data, index=index)
    keep = ['S&P500']
    # 'Equal_Weight' not in index -> KeyError
    with pytest.raises(KeyError):
        filter_models(df, 'Equal_Weight', 'sharpe', keep)

def test_filter_models_metric_column_missing():
    data = {'sharpe': [0.10, 0.08], 'calmar': [0.2,0.1]}
    index = ['ModelA', 'Equal_Weight']
    df = pd.DataFrame(data, index=index)
    keep = ['Equal_Weight']
    with pytest.raises(KeyError):
        filter_models(df, 'Equal_Weight', 'nonexistent', keep)

def test_filter_models_empty_keep_list():
    data = {'sharpe': [0.10, 0.08, 0.05]}
    index = ['ModelA', 'ModelB', 'Equal_Weight']
    df = pd.DataFrame(data, index=index)
    keep = []  # no benchmarks to keep
    filtered_df, filtered_models = filter_models(df, 'Equal_Weight', 'sharpe', keep)
    # Only models that beat 0.05 (ModelA and ModelB) should be kept; 
    # Equal_Weight is not kept because keep empty.
    expected_df = df.loc[['ModelA', 'ModelB']]
    pd.testing.assert_frame_equal(filtered_df, expected_df)
    assert filtered_models == ['ModelA', 'ModelB']

# -------------------- Tests for high_corr_with_each_metric -------------------- #
def test_high_corr_with_each_metric(capsys):
    # Create a correlation matrix
    corr = pd.DataFrame({
        'A': [1.0, 0.9, 0.2],
        'B': [0.9, 1.0, 0.1],
        'C': [0.2, 0.1, 1.0]
    }, index=['A', 'B', 'C'])
    high_corr_with_each_metric(corr, threshold=0.8)
    captured = capsys.readouterr()
    output = captured.out
    assert "A is highly correlated with:" in output
    assert "  B: 0.900" in output
    assert "B is highly correlated with:" in output
    assert "  A: 0.900" in output
    assert "C is highly correlated with:" not in output

def test_high_corr_with_each_metric_no_high(capsys):
    corr = pd.DataFrame({
        'A': [1.0, 0.5, 0.3],
        'B': [0.5, 1.0, 0.4],
        'C': [0.3, 0.4, 1.0]
    }, index=['A', 'B', 'C'])
    high_corr_with_each_metric(corr, threshold=0.8)
    captured = capsys.readouterr()
    assert captured.out == ""

def test_high_corr_with_each_metric_threshold_0(capsys):
    corr = pd.DataFrame({
        'A': [1.0, 0.9, 0.2],
        'B': [0.9, 1.0, 0.1],
        'C': [0.2, 0.1, 1.0]
    }, index=['A', 'B', 'C'])
    high_corr_with_each_metric(corr, threshold=0.0)
    captured = capsys.readouterr()
    # Should print all non‑self correlations
    assert "A is highly correlated with:" in captured.out
    assert "  B: 0.900" in captured.out
    assert "  C: 0.200" in captured.out
    assert "B is highly correlated with:" in captured.out
    assert "  A: 0.900" in captured.out
    assert "  C: 0.100" in captured.out
    assert "C is highly correlated with:" in captured.out
    assert "  A: 0.200" in captured.out
    assert "  B: 0.100" in captured.out

# -------------------- Tests for pareto_dominance -------------------- #
def test_pareto_dominance_no_domination():
    df = pd.DataFrame({
        'metric1': [0.8, 0.7, 0.6],
        'metric2': [0.5, 0.6, 0.7]
    }, index=['A', 'B', 'C'])
    dominated = pareto_dominance(df, columns=['metric1', 'metric2'])
    # No model dominates another (A best in metric1, C best in metric2, B in between)
    assert (dominated == False).all()

def test_pareto_dominance_with_domination():
    # Model A dominates B and C (both metrics higher)
    df = pd.DataFrame({
        'metric1': [0.9, 0.5, 0.7],
        'metric2': [0.8, 0.4, 0.6]
    }, index=['A', 'B', 'C'])
    dominated = pareto_dominance(df, columns=['metric1', 'metric2'])
    assert dominated['A'] == False
    assert dominated['B'] == True
    assert dominated['C'] == True

def test_pareto_dominance_ties():
    df = pd.DataFrame({
        'metric1': [1.0, 1.0, 0.9],
        'metric2': [0.5, 0.5, 0.6]
    }, index=['A', 'B', 'C'])
    dominated = pareto_dominance(df, columns=['metric1', 'metric2'])
    # A and B are equal, neither dominates the other.
    assert (dominated == False).all()

def test_pareto_dominance_single_row():
    df = pd.DataFrame({'metric1': [0.5], 'metric2': [0.6]}, index=['X'])
    dominated = pareto_dominance(df, columns=['metric1', 'metric2'])
    assert dominated['X'] == False

def test_pareto_dominance_returns_series():
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    result = pareto_dominance(df, ['a','b'])
    assert isinstance(result, pd.Series)
    assert result.index.equals(df.index)