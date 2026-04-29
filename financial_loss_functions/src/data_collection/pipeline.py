import sys
from pathlib import Path
from src.data_collection.macro_api import FredAPI
from src.data_collection.const import FRED_SERIES
from src.utils.io import reset_data_stage, save_to_csv


def run_macro_pipeline(
        api_key: str, paths_config: dict, fred_series: dict = FRED_SERIES
    ):
    """
    DEPRECATED
    Macro-economic Data Collection Pipeline Entry point

    @param api_key str API key for FRED API
    @param paths_config Dict Config dictionary containg paths to files and directories
    @param fred_series Dict 
        Dictionary containing required categories and their fred series ids
        Default = src.data_collection.const.FRED_SERIES
    """
    print('\n','=' * 20, ' Fred API Macro-Economic Data Pipeline ', '=' * 20)
    print('DEPRECATED: Macro-economic data pipeline is no longer in use.')
    sys.exit(0)
    
    macro_data_path = Path(paths_config['data']['raw_macro_dir'])
    
    # Reset directory
    reset_data_stage(macro_data_path)
 
    for category, series_ids in fred_series.items():
        macro_api = FredAPI(api_key, category, series_ids) 
        category_data = macro_api.pull_category_data()

        save_to_csv(category_data, macro_data_path / f'{category}.csv')