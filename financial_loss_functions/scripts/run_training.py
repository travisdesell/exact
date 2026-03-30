import argparse
import os
# from dotenv import load_dotenv
from scripts.utils import load_path_config, load_config
from src.training.pipeline import ALL_MODELS, run_training_pipeline

if __name__ == '__main__':
    # load_dotenv()

    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(ALL_MODELS.keys()),
        default=None,
        help=f"Models to train. Choices: {list(ALL_MODELS.keys())}. Default: all",
    )
    args = parser.parse_args()

    paths_config = load_path_config(os.path.join('config', 'paths.json'))
    hparams_config = load_config(os.path.join('config', 'hparams.json'))

    run_training_pipeline(paths_config, hparams_config, models=args.models)