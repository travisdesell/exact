import requests
from dotenv import load_dotenv
import os
import csv
from datetime import datetime, timedelta
import pytz

class StockAggregator:
    def __init__(self, api_key, ticker, api_url="https://api.polygon.io/v2/aggs/ticker", query_limit=50000, timespan="minute", multiplier=1):
        self.api_key = api_key
        self.base_url = api_url
        self.query_limit = query_limit
        self.stock_ticker=ticker
        self.timespan = timespan
        self.multiplier = multiplier
        self.dates = []
    
    def get_aggregate_bars(self, from_date, to_date, adjusted=True, sort="asc"):
        """
        Fetches aggregate bars for a stock over a given date range.
        
        Parameters:
            from_date (str): The start date in "YYYY-MM-DD" format.
            to_date (str): The end date in "YYYY-MM-DD" format.
            adjusted (bool): Whether or not the results are adjusted for splits.
            sort (str): Sort order of the results ("asc" or "desc").
        
        Returns:
            dict: The API response containing aggregate bars data.
        """
        print(f'From: {from_date}\tto: {to_date}')
        # Construct the API request URL
        request_url = f"{self.base_url}/{self.stock_ticker}/range/{self.multiplier}/{self.timespan}/{from_date}/{to_date}?adjusted={str(adjusted).lower()}&sort={sort}&limit={self.query_limit}&apiKey={self.api_key}"
        
        # Make the API request
        response = requests.get(request_url, timeout=30)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data: {response.status_code}. Error Message: ")
            print(f"\t => {response.text}")
            exit
            return None
        
    # def __to_csv(self, data, filename):
    #     """
    #     Saves the fetched data to a CSV file.
        
    #     Parameters:
    #         data (dict): The data returned by the API.
    #         filename (str): The filename of the CSV file to save the data.
    #     """
    #     if not data or 'results' not in data:
    #         print("No data to save.")
    #         return
        
    #     # Assuming 'results' contains the list of data points to be saved
    #     results = data['results']
        
    #     # Open the file and prepare to write
    #     with open(filename, 'w', newline='') as file:
    #         # Determine the fieldnames from the first item's keys
    #         fieldnames = results[0].keys()
    #         writer = csv.DictWriter(file, fieldnames=fieldnames)
            
    #         # Write the header and the rows
    #         writer.writeheader()
    #         for row in results:
    #             writer.writerow(row)
                
    #     print(f"Data successfully saved to {filename}.")
    
    def __to_csv(self, data, filename):
        if not data or 'results' not in data:
            print("No data to save.")
            return

        results = data['results']

        with open(filename, 'w', newline='') as file:
            fieldnames = list(results[0].keys()) + ['datetime_est']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in results:
                row['datetime_est'] = self.__convert_timestamp_to_est(row['t'])
                writer.writerow(row)
                
        print(f"Data successfully saved to {filename}.")
        
    # def __append_to_csv(self, data, filename):
    #     """
    #     Appends the fetched data to a CSV file. If the file doesn't exist, it creates it.
        
    #     Parameters:
    #         data (dict): The data returned by the API.
    #         filename (str): The filename of the CSV file to append the data.
    #     """
    #     if not data or 'results' not in data:
    #         print("No data to append.")
    #         return
        
    #     should_write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    #     results = data['results']
        
    #     with open(filename, 'a', newline='') as file:
    #         fieldnames = results[0].keys()
    #         writer = csv.DictWriter(file, fieldnames=fieldnames)
            
    #         if should_write_header:
    #             writer.writeheader()
    #         for row in results:
    #             writer.writerow(row)
                
    #     print(f"Data successfully appended to {filename}.")
    def __append_to_csv(self, data, filename):
        if not data or 'results' not in data:
            print("No data to append.")
            return

        should_write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
        results = data['results']

        with open(filename, 'a', newline='') as file:
            fieldnames = list(results[0].keys()) + ['datetime_est']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if should_write_header:
                writer.writeheader()
            for row in results:
                row['datetime_est'] = self.__convert_timestamp_to_est(row['t'])
                writer.writerow(row)
                
        print(f"Data successfully appended to {filename}.")
        
    def fetch_and_save_data(self, from_date, to_date, interval_days, filename):
        """
        Fetches data at specified interval days from from_date to to_date and saves it to a CSV file.
        """
        start_date = datetime.strptime(from_date, "%Y-%m-%d")
        end_date = datetime.strptime(to_date, "%Y-%m-%d")
        current_date = start_date
        first_fetch = True

        while current_date <= end_date:
            # Calculate the next date based on the interval
            next_date = current_date + timedelta(days=interval_days)
            # Adjust the next_date not to exceed the to_date
            if next_date > end_date:
                next_date = end_date + timedelta(days=1)  # to include the end_date in the range
            
            # Fetch data for the current interval
            data = self.get_aggregate_bars(from_date=current_date.strftime("%Y-%m-%d"),
                                           to_date=next_date.strftime("%Y-%m-%d"))

            if data and data['results']:
                if first_fetch:
                    self.__to_csv(data, filename)
                    first_fetch = False
                else:
                    self.__append_to_csv(data, filename)

            current_date = next_date
            
    def __convert_timestamp_to_est(self, timestamp):
        """
        Converts a UNIX timestamp (in milliseconds) to a datetime object in EST.
        """
        utc_time = datetime.utcfromtimestamp(timestamp / 1000.0)
        utc_time = utc_time.replace(tzinfo=pytz.utc)
        est_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
        return est_time.strftime('%Y-%m-%d %H:%M:%S')
            
    def generate_interval_dates (self, start_date_str, interval_days_int, end_date_str=None, print_dates=False):
        # Define the start date and interval
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        interval = timedelta(days=interval_days_int)
        if end_date_str is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # Generate dates from start_date + interval days until today
        self.dates = []
        current_date = start_date
        while current_date <= end_date:
            self.dates.append(current_date)
            current_date += interval

        if (print_dates):
            # Convert dates to string format for display
            dates_str = [date.strftime("%Y-%m-%d") for date in self.dates]
            print("Dates generated: ")
            for date_str in dates_str:
                print(f"\t => {date_str}")

def main():
    # Load environment variables from .env file
    load_dotenv('code/.env')
    api_key = os.getenv("API_KEY")
    print("API_KEY : ", api_key)
    
    for ticker in ['UNH', 'MSFT', 'GS', 'HD', 'CAT']:
        stock_aggregator = StockAggregator(api_key=api_key, ticker=ticker)
        stock_aggregator.fetch_and_save_data(from_date="2014-01-01",
                                             to_date="2024-01-01",
                                             interval_days=3,
                                             filename=f"data/{ticker}.csv")
            
    
main()
