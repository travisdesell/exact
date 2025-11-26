import pytest
import numpy as np
import pandas as pd
from scipy.stats import skew
from numpy.testing import assert_allclose
from src.data_processing.preprocess import (
    clean_inplace,
    cov_preprocessor,
    get_only_returns,
    Preprocessor
)

# ---------- Clean inplace tests ---------- #
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

# ---------- Get only returns tests ---------- #
def test_get_only_data_returns():
    train = pd.DataFrame({'ABCD_RET': [0.1, 0.2, 0.3], 'ABCD_VOL':[100, 200, 300]})
    val = pd.DataFrame({'ABCD_RET': [0.4, 0.5, 0.6], 'ABCD_VOL':[150, 250, 350]})
    test = pd.DataFrame({'ABCD_RET': [0.7, 0.8, 0.9], 'ABCD_VOL':[400, 500, 600]})

    train_ret, val_ret, test_ret = get_only_returns(train, val, test)
    
    # Should keep only columns containing '_RET',
    # but with suffix removed
    assert list(train_ret.columns) == ['ABCD']
    assert list(val_ret.columns) == ['ABCD']
    assert list(test_ret.columns) == ['ABCD']

    # Check values remain the same
    assert train_ret.iloc[0,0] == 0.1
    assert val_ret.iloc[1,0] == 0.5
    assert test_ret.iloc[2,0] == 0.9

# ---------- Cov Preprocessor tests ---------- #
def test_cov_preprocessor():
    train = pd.DataFrame({
        'RET1': [0.1, 0.2, 0.3],
        'RET2': [0.2, 0.1, 0.0]
    })

    val = pd.DataFrame({
        'RET1': [0.4, 0.5, 0.3],
        'RET2': [0.2, 0.1, 0.0]
    })

    data = pd.concat([train, val], axis=0)

    cov, corr = cov_preprocessor(train, val)

    # Check that returned objects are DataFrames
    assert isinstance(cov, pd.DataFrame)
    assert isinstance(corr, pd.DataFrame)

    # Check dimensions
    assert cov.shape == (2,2)
    assert corr.shape == (2,2)

    # Check that covariance is correct
    expected_cov = data.cov()
    pd.testing.assert_frame_equal(cov, expected_cov)

# ---------- NN Preprocessor ---------- #
@pytest.fixture
def preprocessor():
    return Preprocessor(col_sep='_', common_features=['sp500r'])

def test_extract_tickers(preprocessor):
    test_list = ['ABCD','EFGH']
    preprocessor.all_col_names = [
        'EFGH_RET', 'ABCD_RET', 'ABCD_VOL_CHANGE', 'EFGH_VOL_CHANGE', 'sp500r'
    ]
    tickers_list = preprocessor._extract_tickers()

    assert tickers_list == test_list, 'Only tickers should be extracted, and sorted alphabetically.'
    assert 'sp500r' not in tickers_list, 'Common features should not be in all_tickers list.'

def test_extract_req_cols(preprocessor):
    test_columns = [
        'ABCD_RET', 'ABCD_VOL_CHANGE', 'EFGH_RET', 'EFGH_VOL_CHANGE', 'sp500r'
    ]
    suffix = '_VOL_CHANGE'
    test_required = [f'ABCD{suffix}', f'EFGH{suffix}']

    req_cols = preprocessor._extract_req_cols(test_columns, suffix)

    assert req_cols == test_required
    assert 'ABCD_RET' not in req_cols
    assert 'sp500r' not in req_cols

def make_sample_train(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n)
    # vol_change can be negative (Yeo-Johnson supports negatives)
    a_vol = rng.normal(loc=0.0, scale=2.0, size=n)
    b_vol = rng.normal(loc=0.5, scale=1.5, size=n)
    # turnover must be strictly positive for Box-Cox
    a_turn = rng.uniform(0.1, 10.0, size=n)
    b_turn = rng.uniform(0.2, 20.0, size=n)
    # a regular feature (not transformed by _transform)
    a_price = rng.normal(loc=100, scale=5, size=n)
    # macro common feature (no underscore in name)
    gdp = np.linspace(1.0, 2.0, n)

    df = pd.DataFrame({
        'ABCD_PRICE': a_price,
        'ABCD_VOL_CHANGE': a_vol,
        'ABCD_TURNOVER': a_turn,
        'BCDE_VOL_CHANGE': b_vol,
        'BCDE_TURNOVER': b_turn,
        'GDP': gdp,
    }, index=dates)
    return df

def test_transform_fit_and_split_applies_power_transforms(preprocessor):
    train = make_sample_train(n=50, seed=1)
    split = make_sample_train(n=20, seed=2)

    preprocessor.all_col_names = list(train.columns)

    # fit mode
    transformed_train = preprocessor._transform(train.copy(), mode='fit')

    # Ensure transformers were fitted and transformed data changed for the relevant cols
    vol_cols = [c for c in preprocessor.all_col_names if '_VOL_CHANGE' in c]
    turn_cols = [c for c in preprocessor.all_col_names if '_TURNOVER' in c]

    # At least one value should differ from original after fit-transform
    assert not np.allclose(train[vol_cols].values, transformed_train[vol_cols].values)
    assert not np.allclose(train[turn_cols].values, transformed_train[turn_cols].values)

    # split mode
    preprocessor.all_col_names = list(split.columns)
    # Manually compute expected transformed values
    expected_split = split.copy()
    expected_split[vol_cols] = preprocessor._yeo_john.transform(split[vol_cols])
    expected_split[turn_cols] = preprocessor._box_cox.transform(split[turn_cols])

    result_split = preprocessor._transform(split.copy(), mode='split')

    # Compare
    assert_allclose(
        result_split[vol_cols].values, 
        expected_split[vol_cols].values,
        atol=1e-8
    )
    assert_allclose(
        result_split[turn_cols].values,
        expected_split[turn_cols].values,
        atol=1e-8
    )

def test_power_transform_reduces_skewness(preprocessor):
    """
    Ensure that Yeo-Johnson and Box-Cox transformations reduce skewness
    in VOL_CHANGE and TURNOVER columns.
    """
    # Create an intentionally skewed dataset
    rng = np.random.default_rng(0)
    n = 300

    # VOL_CHANGE (can be negative): heavily right-skewed via exp() - 5
    vol_a = np.exp(rng.normal(1, 1, size=n)) - 5
    vol_b = np.exp(rng.normal(1.2, 1, size=n)) - 6

    # TURNOVER (strictly positive): lognormal = extremely skewed
    turn_a = rng.lognormal(mean=1, sigma=1, size=n)
    turn_b = rng.lognormal(mean=0.8, sigma=1, size=n)

    df = pd.DataFrame({
        'A_VOL_CHANGE': vol_a,
        'B_VOL_CHANGE': vol_b,
        'A_TURNOVER': turn_a,
        'B_TURNOVER': turn_b,
    })

    # Preprocessor needs all_col_names set before calling _transform
    preprocessor.all_col_names = list(df.columns)

    # Skewness BEFORE transformation
    before_skew = df.apply(lambda s: skew(s.dropna()))

    # Transform (fit mode)
    df_after = preprocessor._transform(df.copy(), mode="fit")

    # Skewness AFTER transformation
    after_skew = df_after.apply(lambda s: skew(s.dropna()))

    # Check skewness reduction for each transformed column
    for col in ['A_VOL_CHANGE', 'B_VOL_CHANGE', 'A_TURNOVER', 'B_TURNOVER']:
        assert abs(after_skew[col]) < abs(before_skew[col]), \
            f'Skewness did not decrease for {col}: before={before_skew[col]}, after={after_skew[col]}'

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
    # Small df with macro common feature 'sp500r'
    n = 10
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-01', periods=n)
    df = pd.DataFrame({
        'A_VOL_CHANGE': rng.normal(size=n),
        'A_TURNOVER': rng.uniform(0.1, 5.0, size=n),
        'B_VOL_CHANGE': rng.normal(loc=0.5, size=n),
        'B_TURNOVER': rng.uniform(0.2, 8.0, size=n),
        'sp500r': np.linspace(2.0, 3.0, n),
    }, index=dates)

    processed = preprocessor.process_train_data(df.copy())

    # Original macro column 'GDP' must be removed
    assert 'sp500r' not in processed.columns

    # Broadcasted columns for each ticker must exist
    assert 'A_sp500r' in processed.columns
    assert 'B_sp500r' in processed.columns

    # There should not be any leftover macro column names without ticker
    assert all(('_' in c) or c == 'date' for c in processed.columns)

    # A_GDP and B_GDP should be equal vectors (they are broadcast copies)
    assert_allclose(
        processed['A_sp500r'].values,
        processed['B_sp500r'].values,
        atol=1e-12
    )