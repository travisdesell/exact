import os
import pandas as pd
from fredapi import Fred
from typing import Dict, List
from dotenv import load_dotenv
from const import BASE_SERIES_DICT
from utils import create_directory, delete_directory

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
        print(f'\n---- Pulling data for {self.category_name} ----')
        all_series_list = []
        
        for name, id in self.required_series.items():
            hist_data = self._get_historical_data(id, self.default_start_date)
            print(f'Pulled data for {name}, {id}.')

            all_series_list.append(hist_data)
        
        output_path = os.path.join(self.data_dir, self.category_name)
        self._combine_save_to_csv(all_series_list, output_path)
        print(f'Data for {self.category_name} pulled and saved as csv at {output_path}!')
            
def data_dir_check(macro_path: str):
    run_permission = False
    if os.path.exists(macro_path):
        print(macro_path, ', Directory Exists!!!!')
        choice = input('Are you sure you want to overwrite it? (Y/N): ').strip()
        if choice == 'Y':
            delete_directory(macro_path)
            create_directory(macro_path)
            run_permission = True
        else:
            print('Aborted. Directory not modified.')
            run_permission = False
    else:
        create_directory(macro_path)
        run_permission = True
    
    return run_permission

if __name__ == '__main__':
    print('\n=' * 20, ' Fred API Macro-Economic Data Pipeline ', '=' * 20)
    api_key = os.getenv('FRED_KEY')
    macro_data_dir = os.path.join(os.getenv('DATA_DIR'), 'macro')

    # To ask user permission before overwriting data
    if data_dir_check(macro_data_dir):
        # series = {
        #     'category': {
        #     'Consumer Price Index for All Urban Consumers': 'CPIAUCSL'
        #     }
        # }
        for category, series_ids in BASE_SERIES_DICT.items():
            macro_api = FredAPI(api_key, category, series_ids, macro_data_dir) 
            macro_api.pull_category_data()
    else:
        print('Fred API Pipeline Aborted!')