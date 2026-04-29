"""
This is a script file to run the Data Preprocessing Pipeline.
"""

import os
from dotenv import load_dotenv
from src.utils.io import load_path_config, load_json
from src.data_processing.pipeline import run_processing_pipeline

if __name__ == '__main__':
    load_dotenv()

    paths_config = load_path_config(
        os.path.join('config', 'paths.json'),
        os.getenv('CRSP_DIR')
    )
    
    hparams_config = load_json(os.path.join('config', 'hparams.json'))
    features_config = load_json(os.path.join('config', 'features.json'))

    run_processing_pipeline(paths_config, hparams_config, features_config)