import os
from dotenv import load_dotenv
from src.data_processing.pipeline import run_processing_pipeline

if __name__ == '__main__':
    load_dotenv()
    
    crsp_data_path = os.path.join(
        os.getenv('DATA_DIR'),
        os.getenv('RAW_DATA_DIR'),
        os.getenv('CRSP_DIR')
    )

    processed_data_path = os.path.join(
        os.getenv('DATA_DIR'),
        os.getenv('PROCESSED_DATA_DIR'),
        os.getenv('CRSP_DIR')
    )

    run_processing_pipeline(crsp_data_path, processed_data_path)