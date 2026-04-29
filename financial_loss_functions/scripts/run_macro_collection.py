#### DEPRECATED ####
"""
DEPRECATED
This is a script file to run marco-economic data collection pipeline.
"""

import os
import sys
from dotenv import load_dotenv
from src.utils.io import load_path_config
from src.data_collection.pipeline import run_macro_pipeline


if __name__ == '__main__':
    print('DEPRECATED: Macro-economic data pipeline is no longer in use.')
    sys.exit(0)
    
    load_dotenv()

    api_key = os.getenv('FRED_KEY')
    paths_config = load_path_config(os.path.join('config', 'paths.json'))
    
    run_macro_pipeline(api_key, paths_config)