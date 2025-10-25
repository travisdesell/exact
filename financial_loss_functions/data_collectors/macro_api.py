import os
import pandas as pd
from typing import Dict
from fredapi import Fred
from dotenv import load_dotenv
from const import BASE_SERIES_DICT

load_dotenv("../../.env")


class FredAPI:
    default_start_date = '2007-01-01' # Match first available date from CRSP
    
    def __init__(self, api_key: str, category_name: str, required_series: Dict[str, str]):
        """
        Initialize object to pull macro-economic data from Fred API

        Parameters:
            api_key (str): API Key for the Fred API account
            category_name (str): Name of the category of macro-economic data being requested
            required_series (Dict[str, str]): A dictionary of required series data with their name and key
        """
        self.fred = Fred(api_key)
        self.required_series = required_series

    def _get_historical_data(self, series_id: str, from_date:str) -> pd.Series:
        data = self.fred.get_series(series_id, observation_start=from_date)
        print(f'Historical data for {series_id} pulled.')
        return data
    
    def set_default_start_date(self, date: str):
        """
        Setter function to set a default start date for pulling macro-economic data
        
        Parameters:
            date (str): date string in ISO format. e.g., '2000-01-01'
        """
        self.default_start_date = date

    def pull_category_data(self):
        """
        Loops to pull all required series data from Fred API and stores them into file(s)
        """

        all_series_list = []
        
        for name, id in self.required_series.items():
            hist_data = self._get_historical_data(id, self.default_start_date)

            all_series_list.append(all_series_list)
            print(name, "-" * 10)
            print(hist_data)

        # TODO:
        # 1. Combine all pd.Series
        # 2. Save the to file(s), save to data folder

if __name__ == '__main__':
    # TODO: Add other indicators and their series ids from fred api
    series_ids = {
        'Consumer Price Index for All Urban Consumers': 'CPIAUCSL'
    }

    macro_api = FredAPI(os.getenv('FRED_KEY'), None , series_ids) # None for test
    macro_api.pull_category_data()



    # For other indicators
    # for category, series_ids in BASE_SERIES_DICT.items():
    #     macro_api = FredAPI(os.getenv('FRED_KEY'), series_ids)
    #     macro_api.pull_category_data()