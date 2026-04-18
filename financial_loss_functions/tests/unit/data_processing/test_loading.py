import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data_processing.loading import (
    load_raw_crsp_datasets,
    load_macro_data,
    load_single_csv,
    load_csv_files,
    ArtifactDataExtractor
)

# -------------------- Tests for Load Raw CRSP -------------------- #
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

# -------------------- Tests for Load Single CSV file -------------------- #
def test_load_single_csv_dt_false(tmp_path):
    """
    Test to load single csv file when index is not datetime.
    """
    test_file = tmp_path / 'test_file.csv'
    test_file.write_text('INDEX,COL1,COL2\n1,0.1,0.2')

    test_df = load_single_csv(test_file, index_dt=False)
    
    # Check returned type
    assert isinstance(test_df, pd.DataFrame)

    # Check columns
    assert list(test_df.columns) == ['COL1', 'COL2']

    # Check values
    assert test_df.iloc[0,0] == 0.1
    assert test_df.iloc[0,-1] == 0.2

def test_load_single_csv_dt_true(tmp_path):
    """
    Test to load single csv file when index is pd.DatetimeIndex).
    """
    test_file = tmp_path / 'test_file.csv'
    test_file.write_text('INDEX,COL1,COL2\n2026/04/18,0.1,0.2')

    test_df = load_single_csv(test_file, index_dt=True)
    
    # Check returned type
    assert isinstance(test_df, pd.DataFrame)

    # Check columns
    assert list(test_df.columns) == ['COL1', 'COL2']

    # Check values
    assert test_df.iloc[0,0] == 0.1
    assert test_df.iloc[0,-1] == 0.2

    assert isinstance(test_df.index, pd.DatetimeIndex), 'Index is not datetime index' 

# -------------------- Tests for Load CSV files -------------------- #
def test_load_csv_files_dt_false(tmp_path):
    """
    Test for load_csv_files. It loads multiple csv files as dataframes.
    """
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

    files_dict = load_csv_files(test_paths, index_dt=False)

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

def test_load_csv_files_dt_true(tmp_path):
    """
    Test for load_csv_files when indices are datetime. 
    It loads multiple csv files as dataframes.
    """
    # Creating test files
    test_file1 = tmp_path / 'test_file1.csv'
    test_file2 = tmp_path / 'test_file2.csv'
    test_file3 = tmp_path / 'test_file3.csv'
    
    test_file1.write_text('INDEX,COL1,COL2\n2026/04/18,0.1,0.2')
    test_file2.write_text('INDEX,COL1,COL2\n2026/04/18,0.3,0.4')
    test_file3.write_text('INDEX,COL1,COL2\n2026/04/18,0.5,0.6')
    
    test_paths = {
        'path1': test_file1,
        'path2': test_file2,
        'path3': test_file3
    }

    files_dict = load_csv_files(test_paths, index_dt=True)

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

    for df in files_dict.values():
        assert isinstance(df.index, pd.DatetimeIndex), 'Index is not datetime index' 

# -------------------- Tests for Load macro files -------------------- #
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

# -------------------- Tests for ArtifactDataExtractor -------------------- #
@pytest.fixture
def artifacts_paths(tmp_path):
    return {
        'avg_perf_dir': tmp_path / 'avg_perf',
        'hparams_dir': tmp_path / 'hparams',
        'wfv_rets_dir': tmp_path / 'daily_returns'
    }

@pytest.fixture
def extractor(artifacts_paths):
    return ArtifactDataExtractor(
        prev_grid_mode='one_model',
        artifacts_paths=artifacts_paths
    )

def test_find_artifact_files_all_exist(extractor, artifacts_paths, tmp_path):
    # Create dummy files
    avg_dir = artifacts_paths['avg_perf_dir']
    avg_dir.mkdir(parents=True, exist_ok=True)
    file1 = avg_dir / 'test_A.csv'
    file2 = avg_dir / 'test_B.csv'
    file1.touch()
    file2.touch()

    suffixes = ['A', 'B']
    result = extractor.find_artifact_files('test', suffixes, avg_dir, '.csv')
    assert result == {'A': file1, 'B': file2}

def test_find_artifact_files_missing(extractor, artifacts_paths, capsys):

    avg_dir = artifacts_paths['avg_perf_dir']
    avg_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ['A', 'B']
    # No files created
    result = extractor.find_artifact_files('test', suffixes, avg_dir, '.csv')
    assert result == {}
    captured = capsys.readouterr()
    assert "not found" in captured.out

# def test_build_avg_perf_paths_success(extractor, artifacts_paths):
#     avg_dir = artifacts_paths['avg_perf_dir']
#     avg_dir.mkdir(parents=True, exist_ok=True)
#     (avg_dir / 'test_model.csv').touch()
#     result = extractor._build_avg_perf_paths('test', ['model'])
#     assert 'model' in result
#     assert result['model'] == avg_dir / 'test_model.csv'

# def test_build_avg_perf_paths_failure(extractor, artifacts_paths):
#     avg_dir = artifacts_paths['avg_perf_dir']
#     avg_dir.mkdir(parents=True, exist_ok=True)
#     # No file
#     with pytest.raises(RuntimeError, match="No average Performance files found"):
#         extractor._build_avg_perf_paths('test', ['model'])

# def test_build_opti_hparams_paths_success(extractor, artifacts_paths):
#     hparams_dir = artifacts_paths['hparams_dir']
#     hparams_dir.mkdir(parents=True, exist_ok=True)
#     (hparams_dir / 'test_model.json').touch()
#     result = extractor._build_opti_hparams_paths('test', ['model'])
#     assert result == {'model': hparams_dir / 'test_model.json'}

# def test_build_opti_hparams_paths_missing(extractor, artifacts_paths, capsys):
#     hparams_dir = artifacts_paths['hparams_dir']
#     hparams_dir.mkdir(parents=True, exist_ok=True)
#     result = extractor._build_opti_hparams_paths('test', ['model'])
#     assert result is None
#     captured = capsys.readouterr()
#     assert "WARNING: Models not tuned!" in captured.out

def test_agg_avg_perf_one_model_mode(extractor, artifacts_paths, monkeypatch):
    avg_dir = artifacts_paths['avg_perf_dir']
    avg_dir.mkdir(parents=True, exist_ok=True)
    (avg_dir / 'avg_A.csv').touch()
    (avg_dir / 'avg_B.csv').touch()

    df_A = pd.DataFrame({'A': [1]}, index=['modelA'])
    df_B = pd.DataFrame({'B': [2]}, index=['modelB'])
    def mock_load_csv_files(paths, index_dt=False):
        return {'A': df_A, 'B': df_B}
    monkeypatch.setattr('src.data_processing.loading.load_csv_files', mock_load_csv_files)

    result = extractor.agg_avg_perf('avg', model_names=['A', 'B'])
    expected = pd.concat([df_A, df_B], axis=0)
    expected = expected[~expected.index.duplicated(keep='first')]
    pd.testing.assert_frame_equal(result, expected)

# def test_agg_avg_perf_one_model_no_names(extractor):
#     with pytest.raises(ValueError, match="List of model names must be provided"):
#         extractor.agg_avg_perf('avg', model_names=None)

def test_agg_avg_perf_one_mode(extractor):
    extractor.prev_grid_mode = 'one'
    with pytest.raises(ValueError, match="does not work for `one` mode"):
        extractor.agg_avg_perf('avg', model_names=None)

def test_agg_opti_hparams_one_model_mode(extractor, artifacts_paths):
    hparams_dir = artifacts_paths['hparams_dir']
    hparams_dir.mkdir(parents=True, exist_ok=True)

    # Write valid JSON content (not empty)
    (hparams_dir / 'opti_A.json').write_text('{"A": {"lr": 0.01}}')
    (hparams_dir / 'opti_B.json').write_text('{"B": {"lr": 0.02}}')

    result = extractor.agg_opti_hparams('opti', model_names=['A', 'B'])
    expected = {'A': {'lr': 0.01}, 'B': {'lr': 0.02}}
    assert result == expected

def test_agg_opti_hparams_one_mode(extractor, artifacts_paths):
    extractor.prev_grid_mode = 'one'
    hparams_dir = artifacts_paths['hparams_dir']
    hparams_dir.mkdir(parents=True, exist_ok=True)

    (hparams_dir / 'opti_one.json').write_text('{"model1": {"c": 3}}')

    result = extractor.agg_opti_hparams('opti', model_names=['one'])
    expected = {'model1': {'c': 3}}
    assert result == expected

def test_agg_opti_hparams_one_model_missing(extractor, capsys):
    # No files
    result = extractor.agg_opti_hparams('opti', model_names=['missing'])
    assert result is None
    captured = capsys.readouterr()
    assert "WARNING: Models not tuned!" in captured.out

def test_agg_opti_hparams_one_mode_2_models(extractor):
    extractor.prev_grid_mode = 'one'
    with pytest.raises(ValueError, match="more than one model-loss provided"):
        extractor.agg_opti_hparams('opti', model_names=['1', '2'])

# Mock find_artifact_files to return a dict of paths
def test_agg_daily_rets_success(extractor, capsys):
    model_names = ['modelA', 'modelB']
    # Mock find_artifact_files to return a dict with two files
    with patch.object(extractor, 'find_artifact_files') as mock_find:
        mock_find.return_value = {
            'modelA': Path('/fake/path/modelA.json'),
            'modelB': Path('/fake/path/modelB.json')
        }
        # Mock load_json to return sample data
        with patch('src.data_processing.loading.load_json') as mock_load:
            mock_load.side_effect = [
                {'modelA': {'returns': [0.1, 0.2]}},
                {'modelB': {'returns': [0.3, 0.4]}}
            ]
            result = extractor.agg_daily_rets('rets_prefix', model_names)
    
    # Verify the merged dict
    expected = {
        'modelA': {'returns': [0.1, 0.2]},
        'modelB': {'returns': [0.3, 0.4]}
    }
    assert result == expected
    # No warning printed
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

def test_agg_daily_rets_no_files(extractor, capsys):
    model_names = ['modelA', 'modelB']
    with patch.object(extractor, 'find_artifact_files') as mock_find:
        mock_find.return_value = {}   # no files found
        result = extractor.agg_daily_rets('rets_prefix', model_names)
    assert result is None
    captured = capsys.readouterr()
    assert "WARNING: No daily returns found." in captured.out

def test_agg_daily_rets_overlapping_keys(extractor):
    # When two files have the same top-level key, later one overwrites
    model_names = ['modelA', 'modelB']
    with patch.object(extractor, 'find_artifact_files') as mock_find:
        mock_find.return_value = {
            'modelA': Path('/fake/path/modelA.json'),
            'modelB': Path('/fake/path/modelB.json')
        }
        with patch('src.data_processing.loading.load_json') as mock_load:
            mock_load.side_effect = [
                {'modelX': {'returns': [1,2]}},  # key 'modelX'
                {'modelX': {'returns': [3,4]}}   # same key, will overwrite
            ]
            result = extractor.agg_daily_rets('rets_prefix', model_names)
    expected = {'modelX': {'returns': [3,4]}}
    assert result == expected