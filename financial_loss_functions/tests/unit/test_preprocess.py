import pytest
<<<<<<< HEAD
from src.data_processing.preprocess import load_crsp_datasets, clean_data_returns, preprocess_cov 
=======
import pandas as pd
from preprocess import (
    load_crsp_datasets,
    get_only_returns,
    preprocess_cov,
    clean_inplace
) 
>>>>>>> ad232f2 (cleaning function added)

def test_load_crsp_datasets(tmp_path):
    # Create tiny CSV files for train, val, test
    train_file = tmp_path / 'combined_predictors_train.csv'
    val_file = tmp_path / 'combined_predictors_validation.csv'
    test_file = tmp_path / 'combined_predictors_test.csv'

    train_file.write_text('COL1,COL2\n0.1,0.2')
    val_file.write_text('COL1,COL2\n0.3,0.4')
    test_file.write_text('COL1,COL2\n0.5,0.6')

    train, val, test = load_crsp_datasets(tmp_path)

    # Check returned types
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    # Check columns
    assert list(train.columns) == ['COL1', 'COL2']
    assert list(val.columns) == ['COL1', 'COL2']
    assert list(test.columns) == ['COL1', 'COL2']

    # Check values
    assert train.iloc[0,0] == 0.1
    assert test.iloc[0,1] == 0.6

def test_load_crsp_datasets_file_not_found(tmp_path):
    # Not creating any files, passing the empty directory
    with pytest.raises(FileNotFoundError) as excinfo:
        load_crsp_datasets(tmp_path)
    
    assert 'Required file not found' in str(excinfo.value)

def test_clean_data_returns():
    train = pd.DataFrame({'ABCD_RET': [0.1, 0.2, 0.3], 'ABCD_VOL':[100, 200, 300]})
    val = pd.DataFrame({'ABCD_RET': [0.4, 0.5, 0.6], 'ABCD_VOL':[150, 250, 350]})
    test = pd.DataFrame({'ABCD_RET': [0.7, 0.8, 0.9], 'ABCD_VOL':[400, 500, 600]})

    train_ret, val_ret, test_ret = get_only_returns(train, val, test)
    
    # Should keep only columns containing 'RET'
    assert list(train_ret.columns) == ['ABCD_RET']
    assert list(val_ret.columns) == ['ABCD_RET']
    assert list(test_ret.columns) == ['ABCD_RET']

    # Check values remain the same
    assert train_ret.iloc[0,0] == 0.1
    assert val_ret.iloc[1,0] == 0.5
    assert test_ret.iloc[2,0] == 0.9

def test_preprocess_cov():
    data = pd.DataFrame({
        'RET1': [0.1, 0.2, 0.3],
        'RET2': [0.2, 0.1, 0.0]
    })

    cov, corr = preprocess_cov(data)

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
    test_cols = ['ABCD_RET', 'ABCD_VOL', 'ABCD_sprtrn', 'EFG_sprtrn']
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

    clean_inplace(train, val, test)

    assert train.shape[0] == 2
    assert val.shape[0] == 2
    assert test.shape[0] == 3

    assert not train.duplicated(subset='date').any()
    assert not val.duplicated(subset='date').any()
    assert not test.duplicated(subset='date').any()