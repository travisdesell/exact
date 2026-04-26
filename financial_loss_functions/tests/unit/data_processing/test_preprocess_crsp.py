import pytest
import numpy as np
import pandas as pd
from scipy.stats import skew
from numpy.testing import assert_allclose
from src.data_processing.preprocess_crsp import (
    clean_inplace,
    preprocessor2,
    Preprocessor,
    SSA
)

# -------------------- Clean inplace tests -------------------- #
def test_clean_inplace_uneq_cols():
    train = pd.DataFrame({'ABCD_RET': [0.1, 0.2, 0.3], 'ABCD_EFG':[100, 200, 300]})
    val = pd.DataFrame({'ABCD_RET': [0.4, 0.5, 0.6], 'ABCD_VOL':[150, 250, 350]})
    test = pd.DataFrame({'ABCD_RET': [0.7, 0.8, 0.9], 'ABCD_VOL':[400, 500, 600]})

    with pytest.raises(ValueError) as excinfo:
        clean_inplace(train, val, test)

    assert 'ERROR: Columns do not match!' in str(excinfo.value)

def test_clean_inplace_dup_cols():
    test_cols = ['date', 'ABCD_RET', 'ABCD_VOL', 'ABCD_sprtrn', 'EFG_sprtrn']
    train = pd.DataFrame(columns=test_cols)
    val = pd.DataFrame(columns=test_cols)
    test = pd.DataFrame(columns=test_cols)

    clean_inplace(train, val, test)

    assert 'EFG_sprtrn' not in train.columns, '2nd duplicate s&p500 column should be removed'
    assert 'EFG_sprtrn' not in val.columns, '2nd duplicate s&p500 column should be removed'
    assert 'EFG_sprtrn' not in test.columns, '2nd duplicate s&p500 column should be removed'

    assert 'sprtrn' in train.columns, 's&p500 return column should be renamed'
    assert 'sprtrn' in val.columns, 's&p500 return column should be renamed'
    assert 'sprtrn' in test.columns, 's&p500 return column should be renamed'

def test_clean_inplace_dup_dates():
    train = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-01', '2023-01-02'],
        'ABCD': [100, 200, 300]
    })
    val = pd.DataFrame({
        'date': ['2023-02-01', '2023-02-02', '2023-02-02'],
        'ABCD': [150, 250, 350]
    })
    test = pd.DataFrame({
        'date':  ['2023-03-01', '2023-03-02', '2023-03-03'],
        'ABCD': [400, 500, 600]
    })

    train, val, test = clean_inplace(train, val, test)

    assert train.shape[0] == 2
    assert val.shape[0] == 2
    assert test.shape[0] == 3

    assert not train.index.duplicated().any(), 'Duplicate index date found in the training set.'
    assert not val.index.duplicated().any(), 'Duplicate index date found in the validation set.'
    assert not test.index.duplicated().any(), 'Duplicate index date found in the test set.'

# -------------------- Tests for SSA -------------------- #

@pytest.fixture
def sample_data_array():
    """Numpy array of shape (50, 5) – 50 time steps, 5 features."""
    np.random.seed(42)
    return np.random.randn(50, 5)

@pytest.fixture
def sample_data_df(sample_data_array):
    """DataFrame version with column names."""
    return pd.DataFrame(sample_data_array, columns=['A', 'B', 'C', 'D', 'E'])

def test_fit_transform_numpy(sample_data_array):
    ssa = SSA(window_len=10, variance_thres=0.9)
    ssa.ssa_fit(sample_data_array)
    denoised = ssa.ssa_transform(sample_data_array)
    # Check shape
    assert denoised.shape == sample_data_array.shape
    # Check that denoised is not identical to input (should reduce noise)
    assert not np.allclose(denoised, sample_data_array, atol=1e-6)
    # Check that U_dict contains entries for each column
    assert len(ssa.U_dict) == sample_data_array.shape[1]

def test_fit_transform_dataframe(sample_data_df):
    ssa = SSA(window_len=10, variance_thres=0.9)
    ssa.ssa_fit(sample_data_df)
    denoised = ssa.ssa_transform(sample_data_df)
    # Check output is DataFrame with same index/columns
    assert isinstance(denoised, pd.DataFrame)
    assert denoised.shape == sample_data_df.shape
    assert (denoised.columns == sample_data_df.columns).all()
    assert (denoised.index == sample_data_df.index).all()
    # Check internal flags
    assert ssa._input_is_df is True
    assert ssa._column_names == sample_data_df.columns.tolist()

def test_fit_before_transform_raises():
    ssa = SSA(window_len=10)
    with pytest.raises(ValueError, match="Run `ssa_fit` before"):
        ssa.ssa_transform(np.random.randn(20, 3))

def test_transform_missing_column_raises(sample_data_array):
    ssa = SSA(window_len=10)
    ssa.ssa_fit(sample_data_array)
    # Create array with more features
    larger = np.random.randn(20, sample_data_array.shape[1] + 1)
    with pytest.raises(KeyError):
        ssa.ssa_transform(larger)

# -------------------- Tests for Kalman Filter -------------------- #

# -------------------- Tests for Preprocessor -------------------- #
@pytest.fixture
def preprocessor():
    return Preprocessor(common_features=['sprtrn'], broadcast=False)

def test_normalize_fit_median_and_iqr(preprocessor):
    # small deterministic dataframe whose median and IQR are known
    df = pd.DataFrame({
        'col1': np.array([0, 1, 2, 3, 4], dtype=float),   # median=2, IQR=2 (3-1)
        'col': np.array([10, 20, 30, 40, 50], dtype=float), # median=30, IQR=20 (40-20)
    })

    scaled = preprocessor._normalize(df.copy(), mode='fit')

    # For RobustScaler, after fitting:
    # - column medians should be ~ 0
    # - IQR should be ~ 1
    for col in scaled.columns:
        col_vals = scaled[col].values
        median = np.median(col_vals)
        iqr = np.percentile(col_vals, 75) - np.percentile(col_vals, 25)

        assert np.isclose(median, 0.0, atol=1e-8), f'median for {col} not ~0 (got {median})'
        assert np.isclose(iqr, 1.0, atol=1e-8), f'IQR for {col} not ~1 (got {iqr})'

def test_broadcast_common_features(preprocessor):
    # Small df with macro common feature 'sprtrn'
    n = 10
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=n)
    df = pd.DataFrame({
        'A_VOL_CHANGE': rng.normal(size=n),
        'A_RET': rng.normal(size=n),
        'A_TURNOVER': rng.uniform(0.1, 5.0, size=n),
        'B_VOL_CHANGE': rng.normal(loc=0.5, size=n),
        'B_RET': rng.normal(size=n),
        'B_TURNOVER': rng.uniform(0.2, 8.0, size=n),
        'sprtrn': np.linspace(2.0, 3.0, n),
    }, index=dates)
    preprocessor.broadcast = True # Set to true, since default is false
    processed, _ = preprocessor.process_train_data(df.copy())
    # processed

    # Original macro column 'GDP' must be removed
    assert 'sprtrn' not in processed.columns

    # Broadcasted columns for each ticker must exist
    assert 'A_sprtrn' in processed.columns
    assert 'B_sprtrn' in processed.columns

    # There should not be any leftover macro column names without ticker
    assert all(('_' in c) or c == 'date' for c in processed.columns)

    # A_GDP and B_GDP should be equal vectors (they are broadcast copies)
    assert_allclose(
        processed['A_sprtrn'].values,
        processed['B_sprtrn'].values,
        atol=1e-12
    )

def test_update_common_features_merges_and_deduplicates_preserve_order():
    # existing common features should stay in front, new ones appended,
    # and duplicates removed preserving first-seen order
    p = Preprocessor(common_features=['A', 'B'])
    p._update_common_features(['B', 'C', 'D'])
    assert p.common_features == ['A', 'B', 'C', 'D']

def test_update_common_features_empty_result_sets_none():
    """
    Test for update common features list method when either on or
    both lists are None or empty
    """
    p = Preprocessor(common_features=None)
    p._update_common_features([])   # no macro columns and no base -> None
    assert p.common_features is None

    p2 = Preprocessor(common_features=[])
    p2._update_common_features([])  # empty list treated same as None in constructor use
    assert p2.common_features is None

    p3 = Preprocessor(common_features=None)
    p3._update_common_features(['GDP', 'CPI'])
    assert p3.common_features == ['GDP', 'CPI']

def test_build_feats_order(preprocessor):
    unordered_cols = [
        'sprtrn', 'EFGH_RET', 'ABCD_RET', 'ABCD_VOL_CHANGE', 
        'EFGH_VOL_CHANGE', 'ABCD_BA_SPREAD', 'EFGH_BA_SPREAD'
    ]

    expected_ticker = ['ABCD', 'EFGH']
    expected_order = [
        'ABCD_BA_SPREAD', 'ABCD_RET', 'ABCD_VOL_CHANGE', 
        'EFGH_BA_SPREAD', 'EFGH_RET', 'EFGH_VOL_CHANGE', 'sprtrn'
    ]
    preprocessor.unordered_cols = unordered_cols

    all_ordered, tickers = preprocessor._build_feats_order()

    assert all_ordered == expected_order, 'All features must be sorted by ticker first,\
        then feature, and lastly sorted common features'
    assert tickers == expected_ticker, 'Only tickers should be extracted, and sorted alphabetically.'

def test_extract_only_returns(preprocessor):
    train = pd.DataFrame({'ABCD_RET': [0.1, 0.2, 0.3], 'ABCD_VOL':[100, 200, 300]})
    val = pd.DataFrame({'ABCD_RET': [0.4, 0.5, 0.6], 'ABCD_VOL':[150, 250, 350]})
    test = pd.DataFrame({'ABCD_RET': [0.7, 0.8, 0.9], 'ABCD_VOL':[400, 500, 600]})

    ret_train = preprocessor._extract_only_returns(train, 'fit')
    ret_val = preprocessor._extract_only_returns(val, 'split')
    ret_test = preprocessor._extract_only_returns(test, 'split')

    # Should keep only columns containing '_RET',
    # but with suffix removed
    assert list(ret_train.columns) == ['ABCD']
    assert list(ret_val.columns) == ['ABCD']
    assert list(ret_test.columns) == ['ABCD']

    # Check values remain the same
    assert ret_train.iloc[0,0] == 0.1
    assert ret_val.iloc[1,0] == 0.5
    assert ret_test.iloc[2,0] == 0.9

def test_extract_only_returns_incorrect_init(preprocessor):
    train = pd.DataFrame({'ABCD_RET': [0.1, 0.2, 0.3], 'ABCD_VOL':[100, 200, 300]})

    with pytest.raises(RuntimeError) as excinfo:
        preprocessor._extract_only_returns(train, 'split')
    
    assert 'Run `process_train_data` first.' in str(excinfo)

def test_extract_only_ba(preprocessor):
    train = pd.DataFrame({'ABCD_BA_SPREAD': [0.1, 0.2, 0.3], 'ABCD_RET':[100, 200, 300]})
    val = pd.DataFrame({'ABCD_BA_SPREAD': [0.4, 0.5, 0.6], 'ABCD_RET':[150, 250, 350]})
    test = pd.DataFrame({'ABCD_BA_SPREAD': [0.7, 0.8, 0.9], 'ABCD_RET':[400, 500, 600]})

    ret_train = preprocessor._extract_only_ba(train)
    ret_val = preprocessor._extract_only_ba(val)
    ret_test = preprocessor._extract_only_ba(test)

    # Should keep only columns containing '_BA_SPREAD',
    # but with suffix removed
    assert list(ret_train.columns) == ['ABCD_BA_SPREAD']
    assert list(ret_val.columns) == ['ABCD_BA_SPREAD']
    assert list(ret_test.columns) == ['ABCD_BA_SPREAD']

    # Check values remain the same
    assert ret_train.iloc[0,0] == 0.1
    assert ret_val.iloc[1,0] == 0.5
    assert ret_test.iloc[2,0] == 0.9

def test_build_ba_spread_cols(preprocessor):
    # Preprocessor must have all_tickers set
    preprocessor.all_tickers = ['A', 'B']
    result = preprocessor._build_ba_spread_cols()
    expected = ['A_BA_SPREAD', 'B_BA_SPREAD']
    assert result == expected

def test_build_ba_spread_cols_empty(preprocessor):
    preprocessor.all_tickers = []
    result = preprocessor._build_ba_spread_cols()
    assert result == []

@pytest.fixture
def sample_train_data():
    # Create a DataFrame with columns: date (ignored), ticker features, common feature, returns, BA spreads
    # Tickers: A, B; Features: x, y
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    data = {
        'A_x': np.random.randn(10),
        'A_y': np.random.randn(10),
        'B_x': np.random.randn(10),
        'B_y': np.random.randn(10),
        'sprtrn': np.random.randn(10),
        'A_RET': np.random.randn(10),
        'B_RET': np.random.randn(10),
        'A_BA_SPREAD': np.random.randn(10),
        'B_BA_SPREAD': np.random.randn(10),
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_macro_data():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    return pd.DataFrame({'macro1': np.random.randn(10), 'macro2': np.random.randn(10)}, index=dates)

@pytest.fixture
def sample_split_data(sample_train_data):
    # Create a similar DataFrame for validation/test
    dates = pd.date_range('2020-01-11', periods=5, freq='D')
    data = {
        'A_x': np.random.randn(5),
        'A_y': np.random.randn(5),
        'B_x': np.random.randn(5),
        'B_y': np.random.randn(5),
        'sprtrn': np.random.randn(5),
        'A_RET': np.random.randn(5),
        'B_RET': np.random.randn(5),
        'A_BA_SPREAD': np.random.randn(5),
        'B_BA_SPREAD': np.random.randn(5),
    }
    return pd.DataFrame(data, index=dates)

def test_process_train_data_basic(preprocessor, sample_train_data):
    train_processed, ret_train = preprocessor.process_train_data(sample_train_data)
    # Check shapes
    assert train_processed.shape[0] == sample_train_data.shape[0]
    # Check that return columns are extracted correctly
    assert ret_train.shape[1] == 2  # two tickers
    assert list(ret_train.columns) == ['A', 'B']
    # Check that scaler is fitted (should not raise)
    assert hasattr(preprocessor._robust_scaler, 'scale_')
    # Check that common_features are unchanged if no macro
    assert preprocessor.common_features == ['sprtrn']

def test_process_train_data_with_macro(preprocessor, sample_train_data, sample_macro_data):
    train_processed, ret_train = preprocessor.process_train_data(sample_train_data, macro_data=sample_macro_data)
    # Macro columns should be added to common_features
    assert 'macro1' in preprocessor.common_features
    assert 'macro2' in preprocessor.common_features
    # The processed data should contain the macro columns (broadcast? only if broadcast=True)
    # By default broadcast=False, so macro columns remain as single columns
    # Since we added macro columns, they are not part of ordered_cols? Actually they are added to common features, so they appear at the end.
    assert 'macro1' in train_processed.columns
    assert 'macro2' in train_processed.columns

def test_process_train_data_broadcast(preprocessor, sample_train_data):
    preprocessor.broadcast = True
    train_processed, ret_train = preprocessor.process_train_data(sample_train_data)
    # After broadcast, common feature 'sprtrn' should be broadcasted to each ticker
    expected_broadcast_cols = ['A_sprtrn', 'B_sprtrn']
    for col in expected_broadcast_cols:
        assert col in train_processed.columns
    # Original 'sprtrn' should be dropped
    assert 'sprtrn' not in train_processed.columns

def test_process_train_data_orders_columns(preprocessor, sample_train_data):
    train_processed, _ = preprocessor.process_train_data(sample_train_data)
    # Check that columns are sorted: first ticker features alphabetically, then common features
    # For tickers A,B and features x,y, the order should be A_x, A_y, B_x, B_y, then sprtrn
    expected_order = [
        'A_BA_SPREAD', 'A_RET', 'A_x', 'A_y', 
        'B_BA_SPREAD','B_RET', 'B_x', 'B_y', 'sprtrn'
    ]
    assert list(train_processed.columns) == expected_order

def test_process_split_data_no_macro(preprocessor, sample_train_data, sample_split_data):
    # First fit on training
    preprocessor.process_train_data(sample_train_data)
    # Then process split
    split_processed, ret_split, ba_split = preprocessor.process_split_data(sample_split_data)
    # Check shapes
    assert split_processed.shape[0] == sample_split_data.shape[0]
    assert ret_split.shape == (sample_split_data.shape[0], 2)
    assert ba_split.shape == (sample_split_data.shape[0], 2)
    # Check column order matches training
    expected_order = [
        'A_BA_SPREAD', 'A_RET', 'A_x', 'A_y', 
        'B_BA_SPREAD','B_RET', 'B_x', 'B_y', 'sprtrn'
    ]
    assert list(split_processed.columns) == expected_order
    # Check that returns columns are tickers only
    assert list(ret_split.columns) == ['A', 'B']
    # Check BA spreads columns are ordered
    assert list(ba_split.columns) == ['A_BA_SPREAD', 'B_BA_SPREAD']

def test_process_split_data_with_macro(preprocessor, sample_train_data, sample_macro_data, sample_split_data):
    # Training with macro
    preprocessor.process_train_data(sample_train_data, macro_data=sample_macro_data)
    # Split also needs macro aligned? For split, macro should be provided, but not necessary for test.
    # However, macro columns are now in common_features, so split must have those columns or else missing columns error.
    # We'll create a split macro data with same columns
    split_macro = sample_macro_data.iloc[:len(sample_split_data)].copy()
    split_processed, ret_split, ba_split = preprocessor.process_split_data(sample_split_data, macro_data=split_macro)
    # The split should have macro columns in the data
    assert 'macro1' in split_processed.columns
    assert 'macro2' in split_processed.columns

def test_process_split_data_missing_columns(preprocessor, sample_train_data, sample_split_data):
    preprocessor.process_train_data(sample_train_data)
    # Remove a column from split
    split_missing = sample_split_data.drop(columns=['A_x'])
    with pytest.raises(ValueError, match="Missing columns in split data"):
        preprocessor.process_split_data(split_missing)

def test_process_split_data_extra_columns(preprocessor, sample_train_data, sample_split_data):
    preprocessor.process_train_data(sample_train_data)
    # Add an extra column
    split_extra = sample_split_data.copy()
    split_extra['extra'] = 0
    # Should not raise, but extra column should be dropped
    split_processed, _, _ = preprocessor.process_split_data(split_extra)
    assert 'extra' not in split_processed.columns

def test_process_split_data_broadcast(preprocessor, sample_train_data, sample_split_data):
    preprocessor.broadcast = True
    preprocessor.process_train_data(sample_train_data)
    split_processed, _, _ = preprocessor.process_split_data(sample_split_data)
    # Broadcasted common features should appear
    expected_broadcast = ['A_sprtrn', 'B_sprtrn']
    for col in expected_broadcast:
        assert col in split_processed.columns
    assert 'sprtrn' not in split_processed.columns

def test_get_common_features_initial(preprocessor):
    assert preprocessor.get_common_features() == ['sprtrn']

def test_get_common_features_updated(preprocessor, sample_train_data, sample_macro_data):
    preprocessor.process_train_data(sample_train_data, macro_data=sample_macro_data)
    common = preprocessor.get_common_features()
    assert 'macro1' in common
    assert 'macro2' in common
    assert 'sprtrn' in common

# ---------- Cov Preprocessor tests ---------- #
def test_cov_preprocessor():
    train = pd.DataFrame({
        'RET1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.3],
        'RET2': [0.2, 0.1, 0.0, 0.2, 0.1, 0.0]
    })

    cov, corr = preprocessor2(train)

    # Check that returned objects are DataFrames
    assert isinstance(cov, pd.DataFrame)
    assert isinstance(corr, pd.DataFrame)

    # Check dimensions
    assert cov.shape == (2,2)
    assert corr.shape == (2,2)

    # Check that covariance is correct
    expected_cov = train.cov()
    pd.testing.assert_frame_equal(cov, expected_cov)