import time
import pandas as pd
from fredapi import Fred


class FredAPI:
    default_start_date = '2007-01-01' # Match first available date from CRSP
    requests_per_min = 120 # Rate limit
    retry_wait = 30 # Retry wait time
    
    def __init__(
            self, api_key: str,
            category_name: str,
            required_series: dict[str, str]
        ):
        """
        Initialize object to pull macro-economic data for a specific category from Fred API

        @param api_key str API Key for the Fred API account
        @param category_name str Name of the category of macro-economic data being requested
        @param required_series Dict[str, str] A dictionary of required series data with their name and key
        """
        self.fred = Fred(api_key)
        self.category_name = category_name
        self.required_series = required_series
        # self.data_dir = data_dir

        self.interval = float(60 / self.requests_per_min)

    def set_default_start_date(self, date: str):
        """
        Setter function to set a default start date for pulling macro-economic data
        
        @param date str date string in ISO format. e.g., '2000-01-01'
        """
        self.default_start_date = date
    
    def set_rate_limit(self, requests_per_min: int):
        """
        Setter function to set number of requests per minute

        @param requests_per_minute int number of requests per minute allowed
        """
        self.requests_per_min = requests_per_min
        self.interval = float(60 / self.requests_per_min)

    def _get_historical_data(self, series_id: str, from_date:str) -> pd.Series:
        data = self.fred.get_series(series_id, observation_start=from_date)
        data = data.rename(series_id)
        return data

    def pull_category_data(self):
        """
        Loops to pull all required series data from Fred API and stores them into file(s)
        """
        print(f'\n---- Pulling data for {self.category_name} ----')
        all_series_list = []
        
        for name, id in self.required_series.items():
            try:
                hist_data = self._get_historical_data(id, self.default_start_date)

                # Retry if rate limit is hit
                if hist_data.empty or hist_data is None:
                    print(
                        f'Rate limit hit for {name}, {id}!! Waiting for {self.retry_wait} seconds...'
                    )
                    time.sleep(self.retry_wait)
                    hist_data = self._get_historical_data(id, self.default_start_date)
                
                time.sleep(self.interval) # Regular interval time between requests
            
            except Exception as e:
                print(f'Error while pulling data for: {name}, {id}. Exception:', e)
                continue

            if not hist_data.empty:
                print(f'Pulled data for {name}, {id}.')
                all_series_list.append(hist_data)
            else:
                print(f'Data for {name}, {id}, not pulled. Skipping!!')
                continue
        
        category_df = pd.concat(all_series_list, axis=1, sort=True)
        print(f'Data for {self.category_name} pulled!')

        return category_df        