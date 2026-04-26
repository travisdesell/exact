import os
import json
import pickle
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from pandas.testing import assert_frame_equal
from src.utils.io import (
    data_dir_check,
    create_directory,
    delete_file,
    delete_directory,
    check_if_files_exist,
    raise_file_not_found,
    save_to_csv,
    save_to_json,
    save_pickle_temp,
    load_pickle_temp,
    reset_data_stage,
    load_path_config, 
    load_json,
    artifact_paths_setup
)

# ------------------- Tests for create_directory -------------------- #
def test_create_directory_creates_and_prints(tmp_path, capsys):
    d = tmp_path / 'new_dir'
    assert not d.exists()
    create_directory(str(d))
    assert d.exists() and d.is_dir()

    captured = capsys.readouterr()
    assert f'{str(d)} Directory Created!' in captured.out

    # calling again should not re-print (directory already exists)
    create_directory(str(d))
    captured2 = capsys.readouterr()
    assert captured2.out == ""  # no new output when directory already exists

# ---------- Tests for delete_file ---------- #
def test_delete_file_removes_file_and_error_if_missing(tmp_path):
    f = tmp_path / 'file.txt'
    f.write_text('hello')
    assert f.exists()

    delete_file(str(f))
    assert not f.exists()

# ---------- Tests for delete_directory ---------- #
def test_delete_directory_deletes_and_prints(tmp_path, capsys):
    d = tmp_path / 'dir_to_delete'
    d.mkdir()
    # create nested file
    (d / 'a.txt').write_text('x')
    assert d.exists()

    delete_directory(str(d))
    assert not d.exists()

    captured = capsys.readouterr()
    assert 'have been deleted' in captured.out or 'deleted' in captured.out

def test_delete_directory_nonexistent_prints_message(tmp_path, capsys):
    d = tmp_path / 'no_such_dir'
    assert not d.exists()

    delete_directory(str(d))
    out = capsys.readouterr().out
    assert 'does not exist' in out

def test_delete_directory_handles_unexpected_exception(monkeypatch, tmp_path, capsys):
    # simulate shutil.rmtree raising an unexpected exception
    d = tmp_path / 'dir_err'
    d.mkdir()
    target = str(d)

    # monkeypatch the module-level shutil.rmtree
    import src.utils.io as utils_test

    def fake_rmtree(path):
        raise Exception('boom')

    monkeypatch.setattr(utils_test.shutil, 'rmtree', fake_rmtree)

    delete_directory(target)
    out = capsys.readouterr().out
    assert 'An error occurred' in out
    assert 'boom' in out

# ---------- Tests for check_if_files_exist ---------- #
def test_check_if_files_exist_all_exist(tmp_path):
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.touch()
    file2.touch()
    paths = [file1, file2]
    result = check_if_files_exist(paths)
    assert result[file1] is True
    assert result[file2] is True

def test_check_if_files_exist_some_missing(tmp_path):
    exists = tmp_path / "exists.txt"
    missing = tmp_path / "missing.txt"
    exists.touch()
    paths = [exists, missing]
    result = check_if_files_exist(paths)
    assert result[exists] is True
    assert result[missing] is False

def test_check_if_files_exist_empty_list():
    result = check_if_files_exist([])
    assert result == {}

def test_check_if_files_exist_with_string_paths(tmp_path):
    file1 = tmp_path / "test.dat"
    file1.touch()
    path_str = str(file1)
    missing_str = str(tmp_path / "nonexistent.dat")
    result = check_if_files_exist([path_str, missing_str])
    assert result[path_str] is True
    assert result[missing_str] is False

# -------------------- Tests for raise_file_not_found --------------------
def test_raise_file_not_found_file_exists(tmp_path):
    file_path = tmp_path / "existing.txt"
    file_path.touch()
    # Should not raise
    raise_file_not_found(file_path)

def test_raise_file_not_found_file_missing(tmp_path):
    missing_path = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match=f"Required file not found: {missing_path}"):
        raise_file_not_found(missing_path)

def test_raise_file_not_found_with_string_path(tmp_path):
    missing_str = str(tmp_path / "does_not_exist.dat")
    with pytest.raises(FileNotFoundError, match=f"Required file not found: {missing_str}"):
        raise_file_not_found(missing_str)

# ---------- Tests for data_dir_check ---------- #
def test_data_dir_check_create(tmp_path):
    """
    Test data_dir_check when the directory does not exist.
    It should create the directory and return True.
    """
    temp_dir = tmp_path / 'macro'

    # Run the function
    result = data_dir_check(str(temp_dir))

    # Assertions
    assert os.path.exists(temp_dir)
    assert result is True

def test_data_dir_check_overwriting_yes(monkeypatch, tmp_path):
    """
    Test data_dir_check when the directory exists and user chooses 'Y'.
    It should delete and recreate the directory, returning True.
    """
    temp_dir = tmp_path / 'macro'
    temp_dir.mkdir()  # create directory

    # Monkeypatch input to always return 'Y'
    monkeypatch.setattr('builtins.input', lambda _: 'Y')

    result = data_dir_check(str(temp_dir))

    assert os.path.exists(temp_dir)
    assert result is True

def test_data_dir_check_overwriting_no(monkeypatch, tmp_path):
    """
    Test data_dir_check when the directory exists and user chooses 'N'.
    It should not modify the directory and return False.
    """
    temp_dir = tmp_path / 'macro'
    temp_dir.mkdir()  # create directory

    # Monkeypatch input to always return 'N'
    monkeypatch.setattr('builtins.input', lambda _: 'N')

    result = data_dir_check(str(temp_dir))

    # Directory should still exist
    assert os.path.exists(temp_dir)
    assert result is False

# -------------------- Tests for save_to_csv -------------------- #
def test_save_to_csv_writes_file_and_content(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    out = tmp_path / 'out.csv'

    save_to_csv(df, str(out))
    assert out.exists()

    # read back. function writes index by default; read with index_col=0 to restore
    df_read = pd.read_csv(str(out), index_col=0)
    # reset index to compare to original DataFrame (which had default RangeIndex)
    df_read = df_read.reset_index(drop=True)
    assert_frame_equal(df_read, df)

# -------------------- Tests for save_to_json -------------------- #
def test_save_to_json(tmp_path):
    data = {'a': 1, 'b': [2, 3], 'c': {'d': 4}}
    output_file = tmp_path / 'test.json'
    save_to_json(data, output_file)
    assert output_file.exists()
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    assert loaded == data

def test_save_to_json_empty_dict(tmp_path):
    data = {}
    output_file = tmp_path / 'empty.json'
    save_to_json(data, output_file)
    assert output_file.exists()
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    assert loaded == data

# -------------------- Tests for save_pickle_temp and load_pickle_temp -------------------- #
def test_save_and_load_pickle_temp(tmp_path):
    data = {'key': [1, 2, 3], 'nested': {'x': 10, 'y': 20}}
    pickle_file = tmp_path / 'temp.pkl'
    save_pickle_temp(data, pickle_file)
    assert pickle_file.exists()
    loaded = load_pickle_temp(pickle_file)
    assert loaded == data

def test_save_pickle_temp_empty_dict(tmp_path):
    data = {}
    pickle_file = tmp_path / 'empty.pkl'
    save_pickle_temp(data, pickle_file)
    assert pickle_file.exists()
    loaded = load_pickle_temp(pickle_file)
    assert loaded == data

def test_load_pickle_temp_nonexistent_file(tmp_path):
    missing_file = tmp_path / 'missing.pkl'
    with pytest.raises(FileNotFoundError):
        load_pickle_temp(missing_file)

def test_load_pickle_temp_corrupted(tmp_path):
    # Write invalid pickle data
    bad_file = tmp_path / 'bad.pkl'
    bad_file.write_text('not a pickle')
    with pytest.raises(pickle.UnpicklingError):
        load_pickle_temp(bad_file)

# ---------- Tests for reset_data_stage ---------- #
def test_reset_data_stage_creates_when_missing(tmp_path, capsys):
    d = tmp_path / 'stage_dir'
    assert not d.exists()

    reset_data_stage(str(d))
    assert d.exists() and d.is_dir()

    out = capsys.readouterr().out
    assert 'Directory created' in out or 'Directory created.' in out

def test_reset_data_stage_overwrites_existing(tmp_path, capsys):
    d = tmp_path / 'stage_dir2'
    d.mkdir()
    (d / 'old.txt').write_text('old')

    # Ensure directory contains a file
    assert any(d.iterdir())

    reset_data_stage(str(d))
    # Directory should exist
    assert d.exists() and d.is_dir()
    # and should be empty after deletion+recreation
    assert not any(d.iterdir())

    out = capsys.readouterr().out
    assert 'Directory exists. Overwriting.' in out

# ---------- Tests for load_json ---------- #
def write_json(path: str, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

def test_load_json(tmp_path):
    cfg = {'hello': 'world', 'n': 123}
    cfg_path = tmp_path / 'some' / 'config.json'
    write_json(cfg_path, cfg)

    loaded = load_json(str(cfg_path))
    assert isinstance(loaded, dict)
    assert loaded['hello'] == 'world'
    assert loaded['n'] == 123

# ---------- Tests for load_paths_config ---------- #
def make_basic_config(repo_root) -> dict:
    """
    Returns a minimal config dict with relative paths (as strings)
    consistent with load_path_config expectations.
    """
    return {
        'data': {
            'raw_dir': 'data/raw',                    # relative to repo_root
            'processed_dir': 'data/processed',        # relative to repo_root
            'raw_macro_dir': 'data/raw/macro'         # relative to repo_root
        },
        # processed_paths are relative to repo_root in typical setups
        'processed_paths': {
            'processed_train': 'outputs/processed_train.csv',
            'processed_val': 'outputs/processed_val.csv'
        }
    }

def test_load_path_config_resolves_relative_and_default_crsp(tmp_path):
    # repo_root
    repo_root = tmp_path / 'repo'
    config_dir = repo_root / 'configs'
    config_dir.mkdir(parents=True)

    # construct basic config file (using relative paths)
    config = make_basic_config(repo_root)
    config_path = config_dir / 'config.json'
    write_json(config_path, config)

    # create the raw directories and the default CRSP folder expected by function
    raw_root = repo_root / 'data' / 'raw'
    default_crsp_dir = raw_root / '2023_sp_500_select_50'
    default_crsp_dir.mkdir(parents=True)

    # create the macro dir referenced in config
    (repo_root / 'data' / 'raw' / 'macro').mkdir(parents=True)

    # Call without providing crsp_data_dir, should pick up default
    out = load_path_config(str(config_path))

    # data.processed_dir and raw_macro_dir should be converted to absolute strings under repo_root
    assert out['data']['processed_dir'] == str((repo_root / 'data' / 'processed').resolve())
    assert out['data']['raw_macro_dir'] == str((repo_root / 'data' / 'raw' / 'macro').resolve())

    # CRSP dir should be the default we created
    assert out['data']['crsp_dir'] == str(default_crsp_dir.resolve())

    # processed_paths should all be absolute and under repo_root
    for k, v in out['processed_paths'].items():
        assert os.path.isabs(v), 'processed_paths value must be absolute'
        assert str(repo_root.resolve()) in v
    
def test_load_path_config_with_relative_crsp_arg(tmp_path):
    # repo_root
    repo_root = tmp_path / 'repo2'
    config_dir = repo_root / 'configs'
    config_dir.mkdir(parents=True)

    config = make_basic_config(repo_root)
    config_path = config_dir / 'config.json'
    write_json(config_path, config)

    # create raw_root and a custom crsp folder inside it
    raw_root = repo_root / 'data' / 'raw'
    custom_crsp = raw_root / 'my_crsp_dir'
    custom_crsp.mkdir(parents=True)

    # create the macro dir referenced in config
    (repo_root / 'data' / 'raw' / 'macro').mkdir(parents=True)

    out = load_path_config(str(config_path), crsp_data_dir='my_crsp_dir')
    assert out['data']['crsp_dir'] == str(custom_crsp.resolve())


def test_load_path_config_with_absolute_crsp_arg(tmp_path):
    # repo_root
    repo_root = tmp_path / 'repo3'
    config_dir = repo_root / 'configs'
    config_dir.mkdir(parents=True)

    config = make_basic_config(repo_root)
    config_path = config_dir / 'config.json'
    write_json(config_path, config)

    # create raw_root but we will make an external absolute crsp dir
    raw_root = repo_root / 'data' / 'raw'
    raw_root.mkdir(parents=True)
    (repo_root / 'data' / 'raw' / 'macro').mkdir(parents=True)

    external_crsp = tmp_path / 'external_crsp_abs'
    external_crsp.mkdir(parents=True)

    out = load_path_config(str(config_path), crsp_data_dir=str(external_crsp))
    assert out['data']['crsp_dir'] == str(external_crsp.resolve())

def test_load_path_config_raises_when_default_missing(tmp_path):
    # repo_root
    repo_root = tmp_path / 'repo4'
    config_dir = repo_root / 'configs'
    config_dir.mkdir(parents=True)

    config = make_basic_config(repo_root)
    config_path = config_dir / 'config.json'
    write_json(config_path, config)

    # create raw_root but DO NOT create default crsp directory
    raw_root = repo_root / 'data' / 'raw'
    raw_root.mkdir(parents=True)
    (repo_root / 'data' / 'raw' / 'macro').mkdir(parents=True)

    # calling without crsp_data_dir when default doesn't exist should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        _ = load_path_config(str(config_path))
    
# -------------------- Tests for artifact_paths_setup -------------------- #
def test_artifact_paths_setup_creates_directories(tmp_path):
    # Create a config with a few artifact directories
    config = {
        'artifacts': {
            'avg_perf_dir': tmp_path / 'avg_perf',
            'hparams_dir': tmp_path / 'hparams',
            'temp_dir': tmp_path / 'temp'
        }
    }
    with patch('src.utils.io.create_directory') as mock_create_dir:
        result = artifact_paths_setup(config)
        # Check that create_directory was called for each path
        for name, path in config['artifacts'].items():
            mock_create_dir.assert_any_call(Path(path))
        # Check returned dict
        assert result['avg_perf_dir'] == Path(config['artifacts']['avg_perf_dir'])
        assert result['hparams_dir'] == Path(config['artifacts']['hparams_dir'])
        assert result['temp_dir'] == Path(config['artifacts']['temp_dir'])

def test_artifact_paths_setup_missing_artifacts_key():
    config = {'some_other_key': {}}
    with pytest.raises(KeyError):
        artifact_paths_setup(config)

def test_artifact_paths_setup_empty_artifacts():
    config = {'artifacts': {}}
    result = artifact_paths_setup(config)
    assert result == {}