import os
from src.utils import data_dir_check

# ---------- Tests for data_dir_check ---------- #
# Tests for data_dir_check
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