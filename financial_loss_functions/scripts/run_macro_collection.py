import os
from dotenv import load_dotenv
from src.data_collection.macro_api import run_macro_pipeline

if __name__ == '__main__':
    load_dotenv()

    api_key = os.getenv('FRED_KEY')
    macro_data_path = os.path.join(
        os.getenv('DATA_DIR'),
        os.getenv('RAW_DATA_DIR'),
        os.getenv('MACRO_DIR')
    )
    
    run_macro_pipeline(api_key, macro_data_path)