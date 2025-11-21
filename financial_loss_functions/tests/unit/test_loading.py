import pytest
import pandas as pd
from src.data_processing.loading import load_raw_crsp_datasets

def test_load_crsp_datasets(tmp_path):
    # Create tiny CSV files for train, val, test
    train_file = tmp_path / 'combined_predictors_train.csv'
    val_file = tmp_path / 'combined_predictors_validation.csv'
    test_file = tmp_path / 'combined_predictors_test.csv'

    train_file.write_text('COL1,COL2\n0.1,0.2')
    val_file.write_text('COL1,COL2\n0.3,0.4')
    test_file.write_text('COL1,COL2\n0.5,0.6')

    train, val, test = load_raw_crsp_datasets(tmp_path)

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
        load_raw_crsp_datasets(tmp_path)
    
    assert 'Required file not found' in str(excinfo.value)