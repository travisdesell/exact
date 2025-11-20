from dotenv import load_dotenv
from src.data_collection.macro_api import run_marco_pipeline

if __name__ == '__main__':
    load_dotenv()
    run_marco_pipeline()