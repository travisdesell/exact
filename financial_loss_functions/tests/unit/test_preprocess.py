import pytest
import pandas as pd
from src.data_processing.preprocess import (
    clean_inplace,
    cov_preprocessor,
    get_only_returns
)


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