import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from src.data_collection.macro_api import FredAPI, save_to_csv, run_macro_pipeline

load_dotenv()

# @pytest.mark.integration
# def test_pull_category_data_integration(tmp_path):
#     # Load real FRED API key from env
#     api_key = os.getenv('FRED_KEY')
#     assert api_key is not None, 'FRED_KEY env var must be set for integration tests'

#     # Minimal valid FRED series for quick test
#     test_series = {'GDP': 'GDP'}  # GDP is a known valid FRED series ID

#     # Category folder inside tmp_path
#     category_name = 'test_macro'
#     data_dir = tmp_path

#     macro_api = FredAPI(
#         api_key=api_key,
#         category_name=category_name,
#         required_series=test_series
#     )

#     # Run real API call
#     category_df = macro_api.pull_category_data()
#     save_to_csv(category_df, category_name, data_dir)

#     # Verify file saved
#     output_file = data_dir / f'{category_name}.csv'
#     assert output_file.exists(), 'No CSV output created by pull_category_data()'

#     # Load and verify content
#     df = pd.read_csv(output_file)
#     assert not df.empty, 'Output CSV is empty'
#     assert 'GDP' in df.columns, 'GDP column missing in output CSV'

@pytest.mark.integration
def test_run_macro_pipeline(tmp_path):
    # Load real FRED API key from env
    api_key = os.getenv('FRED_KEY')
    assert api_key is not None, 'FRED_KEY env var must be set for integration tests'

    data_dir = tmp_path
    category_name = 'test_macro'
    
    # Minimal valid FRED series for quick test
    test_series = {
        category_name: {'GDP': 'GDP'} # GDP is a known valid FRED series ID
        }  
    path_config = {
        'data': {'raw_macro_dir': f'{data_dir}'} # temp directory
    }
    run_macro_pipeline(api_key, path_config, test_series)

     # Verify file saved
    output_file = data_dir / f'{category_name}.csv'
    assert output_file.exists(), 'No CSV output created by pull_category_data()'

    # Load and verify content
    df = pd.read_csv(output_file)
    assert not df.empty, 'Output CSV is empty'
    assert 'GDP' in df.columns, 'GDP column missing in output CSV'

