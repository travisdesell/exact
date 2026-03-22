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
def test_extract_features_excludes_common_and_date():
    # Columns: Ticker features first, then common features
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    
    # Note: common_features must be passed to constructor now
    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
    
    # New extract_features takes the column list directly
    r.extract_features(cols)

    # Check that 'sprtrn' was excluded from tickers/features
    assert 'sprtrn' not in r.tickers
    assert 'sprtrn' not in r.features
    # Verify counts (A, B and x, y)
    assert r.num_tickers == 2
    assert r.num_features == 2
    assert r.num_common_feats == 1

# def test_extract_features_raises_when_feature_sets_mismatch():
#     # A has x,y but B only has x -> should raise
#     df = pd.DataFrame(np.zeros((2, 3)), columns=['A_x', 'A_y', 'B_x'])
#     r = Reshaper(in_size=1, out_size=1, stride=1)
#     with pytest.raises(ValueError):
#         r.extract_features(df)

@pytest.mark.parametrize('layout, expected_shape', [
    (ReshapeStyle.T_NxF, (2, 5)),           # (T, N*F + C) -> (2, 4 + 1)
    (ReshapeStyle.T_N_F_plus_C, (2, 2, 3)), # (T, N, F + C) -> (2, 2, 2 + 1)
    (ReshapeStyle.CNN_2D, (3, 2, 2)),       # (F + C, T, N) -> (2 + 1, 2, 2)
])

def test_transform_one_window_new_layouts(layout, expected_shape):
    # CRITICAL: Ticker columns must come BEFORE common features for your slicing logic
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    values = {
        'A_x': [1.0, 2.0], 'A_y': [10.0, 20.0],
        'B_x': [5.0, 6.0], 'B_y': [50.0, 60.0],
        'sprtrn': [0.5, 0.9] # Common feature
    }
    df_window = pd.DataFrame(values)[cols]

    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'], layout=layout)
    r.extract_features(cols)

    out = r.transform_one_window(df_window)

    assert out.shape == expected_shape

    if layout == ReshapeStyle.T_N_F_plus_C:
        # Check that 'sprtrn' (0.5) was broadcasted to both tickers A and B at T=0
        # Result at T=0, Ticker=0 should be [1.0, 10.0, 0.5]
        assert_array_equal(out[0, 0, :], [1.0, 10.0, 0.5])
        assert_array_equal(out[0, 1, :], [5.0, 50.0, 0.5])

def test_reshape_success_with_common_features():
    cols = ['A_x', 'B_x', 'sprtrn'] # N=2, F=1, C=1
    features_vals = np.array([
        [1.0, 5.0, 0.1], # t0
        [2.0, 6.0, 0.2], # t1
        [3.0, 7.0, 0.3], # t2
        [4.0, 8.0, 0.4], # t3
    ])
    features_df = pd.DataFrame(features_vals, columns=cols)
    returns_df = pd.DataFrame([[0.1, 0.2]]*4, columns=['A', 'B'])

    r = Reshaper(in_size=2, out_size=1, stride=1, common_features=['sprtrn'], layout=ReshapeStyle.T_NxF)
    r.extract_features(cols)

    X, y, starts = r.reshape(features_df, returns_df)

    # X shape: (NumWindows, T, N*F + C) -> (2, 2, 3)
    assert X.shape == (2, 2, 3)
    assert_array_equal(starts, [0, 1])

def test_constructor_raises_on_invalid_layout_type():
    with pytest.raises(TypeError) as excinfo:
        # Missing common_features would also raise, so we provide it
        Reshaper(in_size=1, out_size=1, stride=1, common_features=[], layout='T_NxF')
    
    assert 'ReshapeStyle' in str(excinfo.value)

def test_features_check_raises_if_extract_not_called():
    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
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

    r = Reshaper(
        in_size=2, out_size=1, stride=1, 
        common_features=['sprtrn'], layout=ReshapeStyle.T_NxF
    )
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
    r = Reshaper(in_size=2, out_size=2, stride=1, common_features=['sprtrn'])
    r.extract_features(features_data)
    with pytest.raises(ValueError):
        r.reshape(features_data, raw_returns)

def test_reshape_raises_when_y_window_has_nan():
    # Build data so that one y-window contains NaN
    cols = ['A_x', 'A_y', 'B_x', 'B_y']
    features_data = pd.DataFrame(np.zeros((4, 4)), columns=cols)
    # Put a NaN in raw_returns at the y-window for the first start (s=0, in_size=2 -> y at index 2)
    raw_returns = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8]], columns=['A', 'B'])

    r = Reshaper(in_size=2, out_size=1, stride=1, common_features=['sprtrn'])
    r.extract_features(features_data)

    with pytest.raises(ValueError):
        r.reshape(features_data, raw_returns)

# def test_get_tickers_and_features_before_and_after_extract():
#     cols = ['A_x', 'A_y', 'B_x', 'B_y']
#     df = pd.DataFrame(columns=cols)

#     r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
#     # before extract_features, methods print and return None
#     assert r.get_tickers() is None
#     assert r.get_features() is None

#     # after extract_features, lists returned
#     r.extract_features(df)
#     assert r.get_tickers() == ['A', 'B']
#     assert r.get_features() == ['x', 'y']

def test_constructor_raises_on_invalid_layout_type():
    # Passing a plain string should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        Reshaper(in_size=1, out_size=1, stride=1, layout='T_NxF', common_features=['sprtrn'])

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