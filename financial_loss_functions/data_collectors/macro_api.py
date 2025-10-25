import os
from typing import Dict
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv("../../.env", override=True)


class FredAPI:
    default_start_date = '2000-01-01'
    
    def __init__(self, api_key: str, required_series: Dict[str, str]):
        """
        Initialize object to pull macro-economic data from Fred API

        Parameters:
            api_key (str): API Key for the Fred API account
            required_series (Dict[str, str]): A dictionary of required series data with their name and key
        """
        self.fred = Fred(api_key)

    def get_historical_data(self, series_id, from_date):
        data = self.fred.get_series(series_id, observation_start=from_date)
        print(f'Historical data for {series_id} pulled.')
        return data
    
    def pull_macro_data(self):

        all_series_list = []
        
        for name, id in self.series_ids.items():
            hist_data = self.get_historical_data(id, self.default_start_date)

            all_series_list.append(all_series_list)

    def set_default_start_date(self, date: str):
        """
        Setter function to set a default start date for pulling macro-economic data
        
        Parameters:
            date (str): date string in ISO format. e.g., '2000-01-01'
        """
        self.default_start_date = date

if __name__ == '__main__':
    series_ids = {'Consumer Price Index for All Urban Consumers': 'CPIAUCSL'}
    
    macro_api = FredAPI(os.getenv('FRED_KEY'), series_ids)
    macro_api.pull_macro_data()