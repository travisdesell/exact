import pytest
import pandas as pd
from src.data_processing.loading import (
    load_raw_crsp_datasets,
    load_macro_data,
    load_csv_files
)

# ---------- Load CRSP tests ---------- #
def test_load_crsp_datasets(tmp_path):
    # Create tiny CSV files for train, val, test
    train_file = tmp_path / 'combined_predictors_train.csv'
    val_file = tmp_path / 'combined_predictors_validation.csv'
    test_file = tmp_path / 'combined_predictors_test.csv'

    train_file.write_text('COL1,COL2\n0.1,0.2')
    val_file.write_text('COL1,COL2\n0.3,0.4')
    test_file.write_text('COL1,COL2\n0.5,0.6')

    train, val, test = load_raw_crsp_datasets(train_file, val_file, test_file)

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

def test_load_crsp_datasets_file_not_found():
    # Not creating any files, passing the empty directory
    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw_crsp_datasets('train.csv', 'val.csv', 'test.csv')
    
    assert 'Required file not found' in str(excinfo.value)

# ---------- Load CSV files tests ---------- #
def test_load_csv_files(tmp_path):
    # Creating test files
    test_file1 = tmp_path / 'test_file1.csv'
    test_file2 = tmp_path / 'test_file2.csv'
    test_file3 = tmp_path / 'test_file3.csv'
    
    test_file1.write_text('INDEX,COL1,COL2\n1,0.1,0.2')
    test_file2.write_text('INDEX,COL1,COL2\n1,0.3,0.4')
    test_file3.write_text('INDEX,COL1,COL2\n1,0.5,0.6')
    
    test_paths = {
        'path1': test_file1,
        'path2': test_file2,
        'path3': test_file3
    }

    files_dict = load_csv_files(test_paths)

    # Check returned type
    assert isinstance(files_dict, dict)

    # Check returned dataframes
    assert isinstance(files_dict['path1'], pd.DataFrame)
    assert isinstance(files_dict['path2'], pd.DataFrame)
    assert isinstance(files_dict['path3'], pd.DataFrame)

    # Check columns
    assert list(files_dict['path1'].columns) == ['COL1', 'COL2']
    assert list(files_dict['path2'].columns) == ['COL1', 'COL2']
    assert list(files_dict['path3'].columns) == ['COL1', 'COL2']

    # Check values
    assert files_dict['path1'].iloc[0,0] == 0.1
    assert files_dict['path2'].iloc[0,1] == 0.4
    assert files_dict['path3'].iloc[0,0] == 0.5

def test_load_csv_files():
    # Not creating any files, passing the empty directory
    test_paths = {
        'path1': 'file1.csv',
        'path2': 'file2.csv',
        'path3': 'file2.csv'
    }
    with pytest.raises(FileNotFoundError) as excinfo:
        load_csv_files(test_paths)
    
    assert 'Required file not found' in str(excinfo.value)

# ---------- Load macro files tests ---------- #
def test_load_macro_data(tmp_path):
    test_file1 = tmp_path / 'test_file1.csv'
    test_file2 = tmp_path / 'test_file2.csv'
    test_file3 = tmp_path / 'test_file3.csv'
    
    test_file1.write_text('INDEX,COL1,COL2\n1,0.1,0.2')
    test_file2.write_text('INDEX,COL1,COL2\n1,0.3,0.4')
    test_file3.write_text('INDEX,COL1,COL2\n1,0.5,0.6')

    files_dict = load_macro_data(tmp_path)

    # Check returned type
    assert isinstance(files_dict, dict)

    # Check returned dataframes
    assert isinstance(files_dict['test_file1'], pd.DataFrame)
    assert isinstance(files_dict['test_file2'], pd.DataFrame)
    assert isinstance(files_dict['test_file3'], pd.DataFrame)

    # Check columns
    assert list(files_dict['test_file1'].columns) == ['COL1', 'COL2']
    assert list(files_dict['test_file2'].columns) == ['COL1', 'COL2']
    assert list(files_dict['test_file3'].columns) == ['COL1', 'COL2']

    # Check values
    assert files_dict['test_file1'].iloc[0,0] == 0.1
    assert files_dict['test_file2'].iloc[0,1] == 0.4
    assert files_dict['test_file3'].iloc[0,0] == 0.5

def test_load_macro_data(tmp_path):
    # Not creating any files, passing the empty directory

    with pytest.raises(FileNotFoundError) as excinfo:
        load_macro_data(tmp_path)
    
    assert f'No CSVs not found in directory: {tmp_path}' in str(excinfo.value)