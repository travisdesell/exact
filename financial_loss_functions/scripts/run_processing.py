import os
import sys
from dotenv import load_dotenv
from src.utils import data_dir_check
from src.data_processing.preprocess import (
    load_crsp_datasets,
    get_only_returns,
    preprocess_cov,
    clean_inplace,
    Preprocessor
)

load_dotenv()

if __name__ == '__main__':
    print('\n','=' * 20, ' Data Processing Pipeline ', '=' * 20)
    crsp_path = os.path.join(
        os.getenv('DATA_DIR'),
        os.getenv('RAW_DATA_DIR'),
        '2023_sp_500_select_50'
    )

    if not data_dir_check(crsp_path):
        sys.exit('Data Processing Pipeline Aborted!')
    
    # -------------------- Data Loading -------------------- #
    # Loading CRSP Dataset
    train_data, val_data, test_data = load_crsp_datasets(crsp_path)

    # -------------------- Cleaning & Processing -------------------- #
    # Clean dataset inplace
    clean_inplace(train_data, val_data, test_data)

    nn_preprocessor = Preprocessor(252*3, 90, 90)
    nn_preprocessor.process_train_data(train_data)