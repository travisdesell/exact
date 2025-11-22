import os
from dotenv import load_dotenv
from scripts.utils import load_path_config
from src.data_collection.macro_api import run_macro_pipeline


if __name__ == '__main__':
    load_dotenv()

    api_key = os.getenv('FRED_KEY')
    paths_config = load_path_config(os.path.join('config', 'paths.json'))
    
    run_macro_pipeline(api_key, paths_config)