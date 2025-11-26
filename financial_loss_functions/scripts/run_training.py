import os
# from dotenv import load_dotenv
from scripts.utils import load_path_config, load_config
from src.training.pipeline import run_training_pipeline

if __name__ == '__main__':
    # load_dotenv()

    paths_config = load_path_config(os.path.join('config', 'paths.json'))

    hparams_config = load_config(os.path.join('config', 'hparams.json'))

    run_training_pipeline(paths_config, hparams_config)