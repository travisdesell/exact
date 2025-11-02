import os
import pytest
import pandas as pd
from data_collectors.macro_api import data_dir_check, FredAPI

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


# ---------- Fixtures ---------- #
@pytest.fixture
def fred_api_fixture(tmp_path):
    """Create a FredAPI instance with dummy parameters and a temporary data directory."""
    return FredAPI(
        api_key='test_key',
        category_name='test_cat',
        required_series={'test_series': 'test_id'},
        data_dir=str(tmp_path)
    )

# ---------- Tests for FredAPI class ---------- #
def test_set_default_start_date(fred_api_fixture):
    fred_api_fixture.set_default_start_date('2025-01-01')
    assert fred_api_fixture.default_start_date == '2025-01-01'

    fred_api_fixture.set_default_start_date('2000-01-01')
    assert fred_api_fixture.default_start_date == '2000-01-01'

def test_set_rate_limit(fred_api_fixture):
    fred_api_fixture.set_rate_limit(60)
    assert fred_api_fixture.requests_per_min == 60
    assert fred_api_fixture.interval == int(60 / 60)  # interval should update

    fred_api_fixture.set_rate_limit(100)
    assert fred_api_fixture.requests_per_min == 100
    assert fred_api_fixture.interval == float(60 / 100)

def test_combine_save_to_csv(fred_api_fixture, tmp_path):
    # Create dummy series list
    s1 = pd.Series([1,2,3], name='GDP')
    s2 = pd.Series([4,5,6], name='CPI')
    expected_df = pd.DataFrame({'GDP': [1, 2, 3], 'CPI': [4, 5, 6]})

    # Output path inside tmp_path
    output_path = tmp_path / 'macro_output'

    # Call the real function
    fred_api_fixture._combine_save_to_csv([s1, s2], str(output_path))

    # Now read the CSV and check
    df = pd.read_csv(str(output_path) + '.csv', index_col=0)
    
    pd.testing.assert_frame_equal(df, expected_df)

