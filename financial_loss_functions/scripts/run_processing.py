import os
from dotenv import load_dotenv
from scripts.utils import load_path_config
from src.data_processing.pipeline import run_processing_pipeline

if __name__ == '__main__':
    load_dotenv()

    paths_config = load_path_config(
        os.path.join('config', 'paths.json'),
        os.getenv('CRSP_DIR')
    )

    run_processing_pipeline(paths_config)