import os
import pandas as pd
from typing import Dict, List
from fredapi import Fred
from dotenv import load_dotenv
from const import BASE_SERIES_DICT

load_dotenv("../../.env")


class FredAPI:
    default_start_date = '2007-01-01' # Match first available date from CRSP
    
    def __init__(
            self, api_key: str,
            category_name: str,
            required_series: Dict[str, str],
            data_dir: str
        ):
        """
        Initialize object to pull macro-economic data for a specific category from Fred API

        Parameters:
            api_key (str): API Key for the Fred API account
            category_name (str): Name of the category of macro-economic data being requested
            required_series (Dict[str, str]): A dictionary of required series data with their name and key
        """
        self.fred = Fred(api_key)
        self.category_name = category_name
        self.required_series = required_series
        self.data_dir = data_dir

    def _get_historical_data(self, series_id: str, from_date:str) -> pd.Series:
        data = self.fred.get_series(series_id, observation_start=from_date)
        data = data.rename(series_id)
        print(f'Historical data for {series_id} pulled.')
        return data
    
    def set_default_start_date(self, date: str):
        """
        Setter function to set a default start date for pulling macro-economic data
        
        Parameters:
            date (str): date string in ISO format. e.g., '2000-01-01'
        """
        self.default_start_date = date

    def _combine_save_to_csv(self, series_list: List[pd.Series], output_path: str):
        category_df = pd.concat(series_list, axis=1, sort=True)

        category_df.to_csv(output_path + '.csv', index=True)

    def pull_category_data(self):
        """
        Loops to pull all required series data from Fred API and stores them into file(s)
        """

        all_series_list = []
        
        for name, id in self.required_series.items():
            hist_data = self._get_historical_data(id, self.default_start_date)

            all_series_list.append(hist_data)
        
        output_path = os.path.join(self.data_dir, 'macro', self.category_name)
        self._combine_save_to_csv(all_series_list, output_path)
            

if __name__ == '__main__':
    
    # For testing
    series_ids = {
        'Consumer Price Index for All Urban Consumers': 'CPIAUCSL'
    }
    api_key = os.getenv('FRED_KEY')
    data_dir = os.getenv('DATA_DIR')
    
    macro_api = FredAPI(api_key, 'CPI', series_ids, data_dir) # CPI for test
    macro_api.pull_category_data()



    # For other indicators
    # for category, series_ids in BASE_SERIES_DICT.items():
    #     macro_api = FredAPI(os.getenv('FRED_KEY'), series_ids)
    #     macro_api.pull_category_data()