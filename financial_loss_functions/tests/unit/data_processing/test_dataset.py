import torch
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

# Adjust the import path/module name if your Reshaper class is in a different file
from src.data_processing.dataset import (
    Reshaper,
    ReshapeStyle,
    WindowDataset
)

# ---------- Tests for Rehsaper ---------- #
def test_split_col_valid_and_invalid():
    r = Reshaper(in_size=1, out_size=1, stride=1)
    # valid split
    t, f = r._split_col('ABC_feature_name')
    assert t == 'ABC' and f == 'feature_name'

    # invalid format (no separator) should raise
    with pytest.raises(ValueError):
        r._split_col('noseparator')

def test_extract_features_orders_tickers_and_features_and_builds_cols_per_ticker():
    # Columns intentionally shuffled to test sorting & deterministic order
    cols = ['B_y', 'A_x', 'A_y', 'B_x']
    df = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)

    r = Reshaper(in_size=1, out_size=1, stride=1)
    r.extract_features(df)

    # tickers sorted
    assert r.tickers == ['A', 'B']
    # features sorted
    assert r.features == ['x', 'y']
    # deterministic cols_per_ticker (t then features)
    assert r.cols_per_ticker == ['A_x', 'A_y', 'B_x', 'B_y']

def test_extract_features_raises_when_feature_sets_mismatch():
    # A has x,y but B only has x -> should raise
    df = pd.DataFrame(np.zeros((2, 3)), columns=['A_x', 'A_y', 'B_x'])
    r = Reshaper(in_size=1, out_size=1, stride=1)
    with pytest.raises(ValueError):
        r.extract_features(df)

@pytest.mark.parametrize("layout, expected_shape", [
    (ReshapeStyle.T_NxF, (2, 4)),    # T=2, N=2, F=2 -> (T, N*F)
    (ReshapeStyle.T_N_F, (2, 2, 2)), # (T, N, F)
    (ReshapeStyle.T_F_N, (2, 2, 2)), # (T, F, N)
])

def test_transform_one_window_for_all_layouts(layout, expected_shape):
    # Construct tickers and features in a shuffled column order to ensure extract_features sorts them
    cols = ['B_y', 'A_x', 'A_y', 'B_x']
    # T = 2 time steps, values chosen to be easily traceable
    values = {
        'A_x': [1.0, 2.0],
        'A_y': [3.0, 4.0],
        'B_x': [5.0, 6.0],
        'B_y': [7.0, 8.0],
    }
    df_window = pd.DataFrame(values, index=[0, 1])[cols]  # keep col order shuffled initially

    r = Reshaper(in_size=1, out_size=1, stride=1, layout=layout)
    # Need to set tickers/features/cols_per_ticker via extract_features on a dataframe with same columns
    r.extract_features(pd.DataFrame(columns=cols))

    out = r._transform_one_window(df_window)

    assert out.shape == expected_shape

    # Validate that values are placed in deterministic positions.
    # For sorted tickers ['A','B'] and sorted features ['x','y']:
    # order for T_NxF: [A_x, A_y, B_x, B_y]
    if layout == ReshapeStyle.T_NxF:
        # out is (T, 4)
        expected = np.array([
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ])
        assert_array_equal(out, expected)
    elif layout == ReshapeStyle.T_N_F:
        # out is (T, N, F) with out[:, 0, 0] == A_x, out[:,0,1]==A_y, out[:,1,0]==B_x, out[:,1,1]==B_y
        expected = np.array([
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ])
        assert_array_equal(out, expected)
    elif layout == ReshapeStyle.T_F_N:
        # out is (T, F, N) with out[:,0,0]==A_x, out[:,0,1]==B_x, out[:,1,0]==A_y, out[:,1,1]==B_y
        expected = np.array([
            [[1.0, 5.0], [3.0, 7.0]],
            [[2.0, 6.0], [4.0, 8.0]],
        ])
        assert_array_equal(out, expected)

def test_features_check_raises_if_extract_not_called():
    r = Reshaper(in_size=1, out_size=1, stride=1)
    with pytest.raises(ValueError):
        r._features_check()

def test_reshape_success_and_contents():
    # Build a small dataset:
    # time steps = 4, in_size=2, out_size=1, stride=1 => starts = [0,1] -> 2 windows
    cols = ['A_x', 'A_y', 'B_x', 'B_y']
    # features_data: time x columns
    features_vals = np.array([
        [1.0, 3.0, 5.0, 7.0],  # t=0
        [2.0, 4.0, 6.0, 8.0],  # t=1
        [9.0, 11.0, 13.0, 15.0],# t=2
        [10.0, 12.0, 14.0, 16.0],# t=3
    ])
    features_data = pd.DataFrame(features_vals, columns=cols)

    # raw_returns must have columns tickers ['A','B'] matching extract_features ordering
    # We'll give returns for each time step for both tickers
    returns_vals = np.array([
        [0.1, 0.2],  # t=0
        [0.3, 0.4],  # t=1
        [0.5, 0.6],  # t=2
        [0.7, 0.8],  # t=3
    ])
    raw_returns = pd.DataFrame(returns_vals, columns=['A', 'B'])

    r = Reshaper(in_size=2, out_size=1, stride=1, layout=ReshapeStyle.T_NxF)
    r.extract_features(features_data)  # populate tickers/features/cols_per_ticker

    X, y, starts = r.reshape(features_data, raw_returns)

    # Two windows -> X.shape[0] == 2, each X window is (T=2, N*F=4)
    assert X.shape == (2, 2, 4)
    # y should be (2, out_size=1, N=2)
    assert y.shape == (2, 1, 2)
    # starts array should be [0,1]
    assert_array_equal(starts, np.array([0, 1]))

    # Check first y equals raw_returns at time index 2 (s=0: out window starts at in_size=2 -> returns at row 2)
    assert_array_equal(y[0], returns_vals[2:3])  # shape (1,2)
    # Check second y equals raw_returns at time index 3 (s=1)
    assert_array_equal(y[1], returns_vals[3:4])

def test_reshape_raises_when_window_size_too_large():
    cols = ['A_x', 'A_y', 'B_x', 'B_y']
    features_data = pd.DataFrame(np.zeros((3, 4)), columns=cols)
    raw_returns = pd.DataFrame(np.zeros((3, 2)), columns=['A', 'B'])

    # in_size + out_size > number of rows -> should raise
    r = Reshaper(in_size=2, out_size=2, stride=1)
    r.extract_features(features_data)
    with pytest.raises(ValueError):
        r.reshape(features_data, raw_returns)

def test_reshape_raises_when_y_window_has_nan():
    # Build data so that one y-window contains NaN
    cols = ['A_x', 'A_y', 'B_x', 'B_y']
    features_data = pd.DataFrame(np.zeros((4, 4)), columns=cols)
    # Put a NaN in raw_returns at the y-window for the first start (s=0, in_size=2 -> y at index 2)
    raw_returns = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8]], columns=['A', 'B'])

    r = Reshaper(in_size=2, out_size=1, stride=1)
    r.extract_features(features_data)

    with pytest.raises(ValueError):
        r.reshape(features_data, raw_returns)

def test_get_tickers_and_features_before_and_after_extract():
    cols = ['A_x', 'A_y', 'B_x', 'B_y']
    df = pd.DataFrame(columns=cols)

    r = Reshaper(in_size=1, out_size=1, stride=1)
    # before extract_features, methods print and return None
    assert r.get_tickers() is None
    assert r.get_features() is None

    # after extract_features, lists returned
    r.extract_features(df)
    assert r.get_tickers() == ['A', 'B']
    assert r.get_features() == ['x', 'y']

def test_constructor_raises_on_invalid_layout_type():
    # Passing a plain string should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        Reshaper(in_size=1, out_size=1, stride=1, layout='T_NxF')

    # The constructor raises TypeError with a message tuple; ensure the message mentions ReshapeStyle
    assert 'ReshapeStyle' in str(excinfo.value)

# ---------- Tests for WindowDataset ---------- #

def test_len_and_getitem_returns_correct_tensors():
    X = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ])  # shape (3,2,2)
    y = np.array([[0.1], [0.2], [0.3]])  # shape (3,1)

    ds = WindowDataset(X, y)

    assert len(ds) == 3

    x0, y0 = ds[0]
    # types and dtypes
    assert isinstance(x0, torch.Tensor)
    assert isinstance(y0, torch.Tensor)
    assert x0.dtype == torch.float32
    assert y0.dtype == torch.float32

    # shapes
    assert tuple(x0.shape) == (2, 2)
    assert tuple(y0.shape) == (1,)

    # values equal to original arrays (within float precision)
    assert torch.allclose(x0, torch.tensor(X[0], dtype=torch.float32))
    assert torch.allclose(y0, torch.tensor(y[0], dtype=torch.float32))


def test_negative_indexing_and_out_of_range_raises():
    X = np.arange(6).reshape(3, 2, 1).astype(float)  # (3,2,1)
    y = np.array([[1.0], [2.0], [3.0]])

    ds = WindowDataset(X, y)

    # negative indexing should work (last element)
    x_last, y_last = ds[-1]
    assert torch.allclose(x_last, torch.tensor(X[-1], dtype=torch.float32))
    assert torch.allclose(y_last, torch.tensor(y[-1], dtype=torch.float32))

    # positive out-of-range raises IndexError
    with pytest.raises(IndexError):
        _ = ds[3]

    # too-negative index (less than -len) raises IndexError
    with pytest.raises(IndexError):
        _ = ds[-4]


def test_original_numpy_modification_does_not_mutate_dataset():
    X = np.zeros((2, 2, 2), dtype=float)
    y = np.zeros((2, 1), dtype=float)

    ds = WindowDataset(X, y)

    # Modify original numpy arrays after construction
    X[0, 0, 0] = 999.0
    y[1, 0] = 888.0

    # Dataset tensors should remain unchanged (copied at construction)
    x0, y0 = ds[0]
    assert float(x0[0, 0].item()) != 999.0
    assert float(y0[0].item()) != 888.0


def test_zero_length_dataset_len_zero_and_indexing_raises():
    X = np.empty((0, 2))  # zero samples
    y = np.empty((0, 1))

    ds = WindowDataset(X, y)
    assert len(ds) == 0

    with pytest.raises(IndexError):
        _ = ds[0]