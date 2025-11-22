import os
# from dotenv import load_dotenv
from scripts.utils import load_path_config
from src.models.pipeline import run_training_pipeline

if __name__ == '__main__':
    # load_dotenv()

    paths_config = load_path_config(
        os.path.join('config', 'paths.json')
    )

    run_training_pipeline(paths_config)