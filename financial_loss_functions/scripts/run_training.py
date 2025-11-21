import os
from dotenv import load_dotenv
from src.models.pipeline import run_training_pipeline

if __name__ == '__main__':
    load_dotenv()

    processed_data_path = os.path.join(
        os.getenv('DATA_DIR'),
        os.getenv('PROCESSED_DATA_DIR')
    )

    run_training_pipeline(processed_data_path)