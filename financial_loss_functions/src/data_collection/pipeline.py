from typing import Dict
from pathlib import Path
from src.data_collection.macro_api import FredAPI
from src.data_collection.const import FRED_SERIES
from src.utils import reset_data_stage, save_to_csv


def run_macro_pipeline(
        api_key: str, paths_config: Dict, fred_series: Dict = FRED_SERIES
    ):
    """
    Macro-economic Data Collection Pipeline Entry point

    Parameters
    ----------
    api_key: str
        API key for FRED API
    paths_config: Dict
        Config dictionary containg paths to files and directories
    fred_series: Dict
        Dictionary containing required categories and their fred series ids
        Default = src.data_collection.const.FRED_SERIES
    """
    print('\n','=' * 20, ' Fred API Macro-Economic Data Pipeline ', '=' * 20)
    
    macro_data_path = Path(paths_config['data']['raw_macro_dir'])
    
    # Reset directory
    reset_data_stage(macro_data_path)
 
    for category, series_ids in fred_series.items():
        macro_api = FredAPI(api_key, category, series_ids) 
        category_data = macro_api.pull_category_data()

        save_to_csv(category_data, macro_data_path / f'{category}.csv')