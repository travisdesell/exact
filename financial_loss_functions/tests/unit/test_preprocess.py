import pandas as pd
import pytest
from preprocess import load_crsp_datasets, clean_data_returns, preprocess_cov 

def test_load_crsp_datasets(tmp_path):
    # Create tiny CSV files for train, val, test
    train_file = tmp_path / 'combined_parameters_train.csv'
    val_file = tmp_path / 'combined_parameters_validation.csv'
    test_file = tmp_path / 'combined_parameters_test.csv'

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

    train_ret, val_ret, test_ret = clean_data_returns(train, val, test)
    
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