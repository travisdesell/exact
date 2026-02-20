import pytest
from src.data_collection.macro_api import FredAPI


# ---------- Fixtures ---------- #
@pytest.fixture
def fred_api_fixture():
    """Create a FredAPI instance with dummy parameters and a temporary data directory."""
    return FredAPI(
        api_key='test_key',
        category_name='test_cat',
        required_series={'test_series': 'test_id'}
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

# def test_combine_save_to_csv(fred_api_fixture, tmp_path):
#     # Create dummy series list
#     s1 = pd.Series([1,2,3], name='GDP')
#     s2 = pd.Series([4,5,6], name='CPI')
#     expected_df = pd.DataFrame({'GDP': [1, 2, 3], 'CPI': [4, 5, 6]})

#     # Output path inside tmp_path
#     output_path = tmp_path / 'macro_output'

#     # Call the real function
#     fred_api_fixture._combine_save_to_csv([s1, s2], str(output_path))

#     # Now read the CSV and check
#     df = pd.read_csv(str(output_path) + '.csv', index_col=0)
    
#     pd.testing.assert_frame_equal(df, expected_df)

