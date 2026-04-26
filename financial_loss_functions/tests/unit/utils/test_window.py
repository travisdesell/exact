import pytest
import numpy as np
import pandas as pd
from src.utils.window import (
    calc_current_idxs,
    get_date_index_col,
    extract_oos_dates,
    extract_sp500_winds,
    calc_in_out_idx
)

# -------------------- Tests for calc_current_idxs -------------------- #
def test_calc_current_idxs_normal():
    start, end = calc_current_idxs(step=3, stride=10)
    assert start == 20
    assert end == 30

def test_calc_current_idxs_step_one():
    start, end = calc_current_idxs(step=1, stride=5)
    assert start == 0
    assert end == 5

def test_calc_current_idxs_step_zero_raises():
    with pytest.raises(ValueError, match='Got step 0'):
        calc_current_idxs(step=0, stride=10)

# ------------------- Tests for get_date_index_col -------------------- #
def test_get_date_index_col():
    index = pd.date_range('2020-01-01', periods=20, freq='D')
    df = pd.DataFrame(index=index, data={'val': range(20)})
    wind_strt_stops = [(2,5), (10,15)]
    result = get_date_index_col(df, wind_strt_stops)
    expected = [df.index[2:5], df.index[10:15]]
    assert len(result) == 2
    pd.testing.assert_index_equal(result[0], expected[0])
    pd.testing.assert_index_equal(result[1], expected[1])

def test_get_date_index_col_empty():
    df = pd.DataFrame(index=pd.DatetimeIndex([]))
    result = get_date_index_col(df, [])
    assert result == []

# -------------------- Tests for extract_oos_dates -------------------- #
def test_extract_oos_dates():
    index = pd.date_range('2020-01-01', periods=30, freq='D')
    df = pd.DataFrame(index=index, data={'x': 1})
    in_wind_idxs = [(0,5), (10,15)]
    out_wind_idxs = [(5,10), (15,20)]
    in_dates, out_dates = extract_oos_dates(df, in_wind_idxs, out_wind_idxs)
    assert len(in_dates) == 2
    assert len(out_dates) == 2
    pd.testing.assert_index_equal(in_dates[0], df.index[0:5])
    pd.testing.assert_index_equal(out_dates[0], df.index[5:10])

# -------------------- Tests for extract_sp500_winds ---------------- #
def test_extract_sp500_winds_multiple_windows():
    df = pd.DataFrame({'sp500': np.arange(20)})
    out_win_idxs = [(0,5), (5,10)]
    result = extract_sp500_winds(df, 'sp500', out_win_idxs)
    expected = np.array([df['sp500'].iloc[0:5].values, df['sp500'].iloc[5:10].values])
    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result, expected)

def test_extract_sp500_winds_single_window():
    df = pd.DataFrame({'sp500': np.arange(10)})
    out_win_idxs = [(3,7)]
    result = extract_sp500_winds(df, 'sp500', out_win_idxs)
    expected = np.array([df['sp500'].iloc[3:7].values])
    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result, expected)

def test_extract_sp500_winds_different_column_name():
    df = pd.DataFrame({'returns': np.arange(15)})
    out_win_idxs = [(1,4), (8,11)]
    result = extract_sp500_winds(df, 'returns', out_win_idxs)
    expected = np.array([df['returns'].iloc[1:4].values, df['returns'].iloc[8:11].values])
    np.testing.assert_array_equal(result, expected)

# -------------------- Tests for calc_in_out_idx -------------------- #
def test_calc_in_out_idx_typical():
    # Create a DataFrame with 100 rows, no NaNs
    df = pd.DataFrame(np.random.randn(100, 5))
    in_size = 20
    out_size = 10
    stride = 5
    in_idxs, out_idxs = calc_in_out_idx(df, in_size, out_size, stride)
    # Number of windows: floor((100 - (20+10)) / 5) + 1 = floor(70/5)+1 = 15
    assert len(in_idxs) == 15
    assert len(out_idxs) == 15
    # Check first window
    assert in_idxs[0] == (0, 20)
    assert out_idxs[0] == (20, 30)
    # Check second window
    assert in_idxs[1] == (5, 25)
    assert out_idxs[1] == (25, 35)

def test_calc_in_out_idx_stride_larger_than_window():
    df = pd.DataFrame(np.random.randn(200, 3))
    in_size = 30
    out_size = 10
    stride = 50   # larger than in_size, but possible
    in_idxs, out_idxs = calc_in_out_idx(df, in_size, out_size, stride)
    # Only windows that fit: start = 0, 50, 100, 150 150+30+10=190 <=200
    expected_starts = [0, 50, 100, 150]
    assert len(in_idxs) == len(expected_starts)
    for i, start in enumerate(expected_starts):
        assert in_idxs[i] == (start, start+in_size)
        assert out_idxs[i] == (start+in_size, start+in_size+out_size)

def test_calc_in_out_idx_missing_data_raises():
    df = pd.DataFrame([[1,2], [np.nan, 4], [5,6]])
    with pytest.raises(ValueError, match='Split has missing data'):
        calc_in_out_idx(df, in_size=1, out_size=1, stride=1)

def test_calc_in_out_idx_zero_stride():
    df = pd.DataFrame(np.random.randn(50, 2))
    # stride 0 would cause infinite loop; but function expects positive stride.
    # Doesn't check stride > 0, but the range(0, ..., stride) with stride=0 raises ValueError.
    with pytest.raises(ValueError):
        calc_in_out_idx(df, in_size=10, out_size=5, stride=0)

def test_calc_in_out_idx_insufficient_length():
    df = pd.DataFrame(np.random.randn(15, 2))
    in_size=10
    out_size=10
    stride=1
    # length=15, need in+out=20, so no windows.
    in_idxs, out_idxs = calc_in_out_idx(df, in_size, out_size, stride)
    assert in_idxs == []
    assert out_idxs == []