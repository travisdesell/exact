import os
from dotenv import load_dotenv
from data_collectors.const import BASE_SERIES_DICT
from data_collectors.macro_api import FredAPI, data_dir_check

load_dotenv()


if __name__ == '__main__':
    print('\n','=' * 20, ' Fred API Macro-Economic Data Pipeline ', '=' * 20)
    api_key = os.getenv('FRED_KEY')
    macro_data_dir = os.path.join(os.getenv('DATA_DIR'), 'macro')

    # To ask user permission before overwriting data
    if data_dir_check(macro_data_dir):
        for category, series_ids in BASE_SERIES_DICT.items():
            macro_api = FredAPI(api_key, category, series_ids, macro_data_dir) 
            macro_api.pull_category_data()
    else:
        print('Fred API Pipeline Aborted!')