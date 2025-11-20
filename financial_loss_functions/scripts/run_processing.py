from dotenv import load_dotenv
from src.data_processing.pipeline import run_processing_pipeline

if __name__ == '__main__':
    load_dotenv()
    run_processing_pipeline()