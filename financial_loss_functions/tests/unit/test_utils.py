import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.utils import (
    data_dir_check,
    create_directory,
    delete_file,
    delete_directory,
    check_if_files_exist,
    save_to_csv,
    reset_data_stage    
)


# ---------- Tests for create_directory ---------- #
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

    with pytest.raises(FileNotFoundError) as excinfo:
    # calling again on non-existent file should raise exception
        delete_file(str(f))
    assert f'File does not exist at {f}' in str(excinfo.value)

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
    import src.utils as utils_test

    def fake_rmtree(path):
        raise Exception('boom')

    monkeypatch.setattr(utils_test.shutil, 'rmtree', fake_rmtree)

    delete_directory(target)
    out = capsys.readouterr().out
    assert 'An error occurred' in out
    assert 'boom' in out

# ---------- Tests for check_if_files_exist ---------- #
def test_check_if_files_exist_all_exist(tmp_path):
    f1 = tmp_path / 'a.txt'
    f2 = tmp_path / 'b.txt'
    f1.write_text('1')
    f2.write_text('2')

    # should not raise
    check_if_files_exist([str(f1), str(f2)])

def test_check_if_files_exist_missing_raises(tmp_path):
    f1 = tmp_path / 'a.txt'
    f1.write_text('1')
    missing = tmp_path / 'missing.txt'

    with pytest.raises(FileNotFoundError) as exc:
        check_if_files_exist([str(f1), str(missing)])

    assert str(missing) in str(exc.value)

# ---------- Tests for data_dir_check ---------- #
def test_data_dir_check_create(tmp_path):
    """
    Test data_dir_check when the directory does not exist.
    It should create the directory and return True.
    """
    # tmp_path is a pytest fixture that gives a temporary directory
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

# ---------- Tests for save_to_csv ---------- #
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