import torch
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

# Adjust the import path/module name if your Reshaper class is in a different file
from src.data_processing.dataset import (
    Reshaper,
    ReshapeStyle,
    WindowDataset,
    WFUtilities
)

# -------------------- Tests for Reshaper -------------------- #
def test_extract_features_excludes_common_and_date():
    """
    Test for Reshaper.extract_features(). Should extract all tickers and features in each column,
    as well as the common features.
    Column names are built to be in the format <ticker>_<feature?
    """
    # Columns: Ticker features first, then common features
    cols = ['date', 'A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    
    # Note: common_features must be passed to constructor now
    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
    
    # New extract_features takes the column list directly
    r.extract_features(cols)

    # Check that 'sprtrn' was excluded from tickers/features
    assert 'sprtrn' not in r.tickers
    assert 'sprtrn' not in r.features
    
    # Verify counts (A, B and x, y)
    assert r.total_num_cols == 6
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
    """
    Test to transform and reshape one window. Each window is a numpy array.
    CRITICAL: Ticker columns must come BEFORE common features for the slicing logic
    """
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    window_array = np.array([
        [1.0, 10.0, 5.0, 50.0, 0.5],
        [2.0, 20.0, 6.0, 60.0, 0.9]
    ])

    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'], layout=layout)
    r.extract_features(cols)

    out = r.transform_one_window(window_array)

    assert out.shape == expected_shape

    if layout == ReshapeStyle.T_N_F_plus_C:
        # Check that 'sprtrn' (0.5) was broadcasted to both tickers A and B at T=0
        # Result at T=0, Ticker=0 should be [1.0, 10.0, 0.5]
        assert_array_equal(out[0, 0, :], [1.0, 10.0, 0.5])
        assert_array_equal(out[0, 1, :], [5.0, 50.0, 0.5])

def test_reshape_success_with_common_features():
    cols = ['A_x', 'B_x', 'sprtrn'] # N=2, F=1, C=1
    features_array = np.array([
        [1.0, 5.0, 0.1], # t0
        [2.0, 6.0, 0.2], # t1
        [3.0, 7.0, 0.3], # t2
        [4.0, 8.0, 0.4], # t3
    ])

    returns_array = np.array([[0.1, 0.2]] * 4)

    r = Reshaper(
        in_size=2, out_size=1, stride=1,
        common_features=['sprtrn'], layout=ReshapeStyle.T_NxF
    )
    r.extract_features(cols)

    X, y, starts = r.reshape(features_array, returns_array)

    # X shape: (NumWindows, T, N*F + C) -> (2, 2, 3)
    assert X.shape == (2, 2, 3)
    assert_array_equal(starts, [0, 1])

def test_constructor_raises_on_invalid_layout_type():
    with pytest.raises(TypeError) as excinfo:
        # Missing common_features would also raise, so we provide it
        Reshaper(
            in_size=1, out_size=1, stride=1, common_features=['sprtrn'], layout='T_NxF'
        )
    
    assert 'ReshapeStyle' in str(excinfo.value)

def test_features_check_raises_if_extract_not_called():
    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
    with pytest.raises(ValueError) as excinfo:
        r._features_check()
    
    assert 'extract_features' in str(excinfo)

def test_reshape_success_and_contents():
    # Build a small dataset:
    # time steps = 4, in_size=2, out_size=1, stride=1 => starts = [0,1] -> 2 windows
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    # features_data: time x columns
    features_array = np.array([
        [1.0, 3.0, 5.0, 7.0, 1.0],  # t=0
        [2.0, 4.0, 6.0, 8.0, 2.0],  # t=1
        [9.0, 11.0, 13.0, 15.0, 3.0],# t=2
        [10.0, 12.0, 14.0, 16.0, 4.0],# t=3
    ])

    # raw_returns must have columns tickers ['A','B'] matching extract_features ordering
    # We'll give returns for each time step for both tickers
    returns_array = np.array([
        [0.1, 0.2],  # t=0
        [0.3, 0.4],  # t=1
        [0.5, 0.6],  # t=2
        [0.7, 0.8],  # t=3
    ])

    r = Reshaper(
        in_size=2, out_size=1, stride=1, 
        common_features=['sprtrn'], layout=ReshapeStyle.T_NxF
    )
    r.extract_features(cols)  # populate tickers/features/cols_per_ticker

    X, y, starts = r.reshape(features_array, returns_array)

    # Two windows -> X.shape[0] == 2, each X window is (T=2, N*F=4)
    assert X.shape == (2, 2, 5)
    # y should be (2, out_size=1, N=2)
    assert y.shape == (2, 1, 2)
    # starts array should be [0,1]
    assert_array_equal(starts, np.array([0, 1]))

    # Check first y equals raw_returns at time index 2 (s=0: out window starts at in_size=2 -> returns at row 2)
    assert_array_equal(y[0], returns_array[2:3])  # shape (1,2)
    # Check second y equals raw_returns at time index 3 (s=1)
    assert_array_equal(y[1], returns_array[3:4])

def test_reshape_raises_when_window_size_too_large():
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    features_array = np.zeros((3, 5))
    returns_array = np.zeros((3, 2))

    # in_size + out_size > number of rows -> should raise
    r = Reshaper(in_size=2, out_size=2, stride=1, common_features=['sprtrn'])
    r.extract_features(cols)
    with pytest.raises(ValueError) as excinfo:
        r.reshape(features_array, returns_array)
    
    assert 'Incorrect rolling window sizes' in str(excinfo) 

def test_reshape_raises_when_y_window_has_nan():
    # Build data so that one y-window contains NaN
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    features_array = np.zeros((4, 5))
    # Put a NaN in raw_returns at the y-window for the first start (s=0, in_size=2 -> y at index 2)
    returns_array = np.array([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8]])

    r = Reshaper(in_size=2, out_size=1, stride=1, common_features=['sprtrn'])
    r.extract_features(cols)

    with pytest.raises(ValueError) as excinfo:
        r.reshape(features_array, returns_array)
    
    assert 'Window has missing data' in str(excinfo.value)

def test_columns_array_mismatch():

    # For feature data mismatch
    # Build data so that one y-window contains NaN
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    features_array = np.zeros((4, 4)) # 4 columns provideed instead of 5
    # Put a NaN in raw_returns at the y-window for the first start (s=0, in_size=2 -> y at index 2)
    returns_array = np.array([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8]])

    r = Reshaper(in_size=2, out_size=1, stride=1, common_features=['sprtrn'])
    r.extract_features(cols)

    with pytest.raises(ValueError) as excinfo:
        r.reshape(features_array, returns_array)
    
    assert 'Extracted columns do not match' in str(excinfo)

    # For returns data mismatch
    # Build data so that one y-window contains NaN
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    features_array = np.zeros((4, 5))
    # Put a NaN in raw_returns at the y-window for the first start (s=0, in_size=2 -> y at index 2)
    returns_array = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.2],
            [0.1, 0.6, 0.4],
            [0.7, 0.8, 0.4]
        ]
    )

    r = Reshaper(in_size=2, out_size=1, stride=1, common_features=['sprtrn'])
    r.extract_features(cols)

    with pytest.raises(ValueError) as excinfo:
        r.reshape(features_array, returns_array)
    
    assert 'Number of tickers in extracted columns do not match' in str(excinfo)

def test_get_tickers_and_features_before_and_after_extract():
    cols = ['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']

    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
    # before extract_features, methods print and return None
    assert r.get_tickers() is None
    assert r.get_features() is None

    # after extract_features, lists returned
    r.extract_features(cols)
    assert set(r.get_tickers()) == {'A', 'B'}
    assert set(r.get_features()) == {'x', 'y'}

def test_update_stride():
    r = Reshaper(in_size=1, out_size=1, stride=1, common_features=['sprtrn'])
    initial_stride = r.stride
    expected_change = 2
    r.update_stride(expected_change)

    assert r.stride == expected_change
    assert r.stride != initial_stride


# -------------------- Tests for WindowDataset -------------------- #

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

def test_get_X_y_shapes():
    X = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ])  # shape (3,2,2)
    y = np.array([[0.1], [0.2], [0.3]])  # shape (3,1)

    ds = WindowDataset(X, y)

    X_tensor_shape, y_tensor_shape = ds.get_X_y_shapes()

    assert X_tensor_shape == X.shape
    assert y_tensor_shape == y.shape

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

# -------------------- Tests for WFUtilities -------------------- #
def test_calc_walk_steps():
    out_size = 60
    
    wf = WFUtilities(out_size)

    initial_num_rows = 501
    expected_extra_days = initial_num_rows % out_size
    expected_steps = initial_num_rows // out_size

    test_split = pd.DataFrame(
        1,
        index=range(initial_num_rows),
        columns=['A', 'B']
    )
    steps, extra_days = wf.calc_walk_steps(test_split)

    assert steps == expected_steps, 'Incorrect number of steps calculated'
    assert extra_days == expected_extra_days, 'Incorrect extra days calculated'

    # When len of dataframe is divisible by steps, hence extra days is 0
    initial_num_rows = 600
    expected_extra_days = initial_num_rows % out_size
    expected_steps = initial_num_rows // out_size
    
    test_split = pd.DataFrame(
        1,
        index=range(out_size*expected_steps),
        columns=['A', 'B']
    )
    steps, extra_days = wf.calc_walk_steps(test_split)

    assert steps == expected_steps, 'Incorrect number of steps calculated'
    assert extra_days == expected_extra_days, 'Incorrect extra days calculated'

def test_init_datasets():
    out_size = 60
    
    wf = WFUtilities(out_size)

    initial_num_rows = 501
    expected_extra_rows = initial_num_rows % out_size
    expected_steps = initial_num_rows // out_size

    test_train = pd.DataFrame(
        1,
        index=range(initial_num_rows),
        columns=['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    )

    test_split = pd.DataFrame(
        2,
        index=range(initial_num_rows),
        columns=['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    )

    wf = WFUtilities(out_size)

    adjusted_train, adjusted_split = wf.init_datasets(test_train, test_split)

    assert adjusted_train.shape[0]-expected_extra_rows == test_train.shape[0]
    assert adjusted_split.shape[0]+expected_extra_rows == test_split.shape[0]

    assert adjusted_train.shape[1] == test_train.shape[1]
    assert adjusted_split.shape[1] == test_split.shape[1]

    assert wf.num_steps == expected_steps
    assert wf.extra_days == expected_extra_rows

def test_get_num_steps_extra_days():
    out_size = 50
    
    wf = WFUtilities(out_size)

    # Before calculation
    assert wf.get_num_steps() is None
    assert wf.get_extra_days() is None


    # After calculation
    initial_num_rows = 620
    expected_extra_days = initial_num_rows % out_size
    expected_steps = initial_num_rows // out_size


    test_train = pd.DataFrame(
        1,
        index=range(initial_num_rows),
        columns=['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    )

    test_split = pd.DataFrame(
        2,
        index=range(initial_num_rows),
        columns=['A_x', 'A_y', 'B_x', 'B_y', 'sprtrn']
    )

    _, _ = wf.init_datasets(test_train, test_split)

    assert wf.get_num_steps() is expected_steps
    assert wf.get_extra_days() is expected_extra_days

# -------------------- Tests for build_eval_windows -------------------- #
def test_build_eval_windows_typical():
    out_size = 4
    
    wf = WFUtilities(out_size)
    wf.num_steps = 3
    
    split = pd.DataFrame(np.arange(100).reshape(20, 5))  # 20 rows, 5 cols

    windows, idxs = wf.build_eval_windows(split)
    # Expected shapes: 3 windows of shape (4,5)
    assert windows.shape == (3, 4, 5)
    # Check indices
    assert idxs == [(0,4), (4,8), (8,12)]
    # Check content: first window should be rows 0-3
    np.testing.assert_array_equal(windows[0], split[0:4])

def test_build_eval_windows_small_df():

    out_size = 10
    wf = WFUtilities(out_size)
    wf.num_steps = 3
    
    split = pd.DataFrame(np.arange(100).reshape(20, 5))  # 20 rows, 5 cols

    with pytest.raises(ValueError) as excinfo:
        windows, idxs = wf.build_eval_windows(split)
    
    assert 'smaller than num_steps * out_size' in str(excinfo)