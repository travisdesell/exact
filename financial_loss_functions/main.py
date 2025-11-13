import os
from dotenv import load_dotenv
from cov_models import HierarchialRiskParity
from preprocess import (
    load_crsp_datasets,
    get_only_returns,
    preprocess_cov,
    clean_inplace,
    Preprocessor
)

load_dotenv()

if __name__ == '__main__':
    # -------------------- Data Loading -------------------- #
    data_dir = os.getenv('DATA_DIR')

    crsp_path = os.path.join(data_dir, '2023_sp_500_select_50')
    train_data, val_data, test_data = load_crsp_datasets(crsp_path)

    # -------------------- Cleaning & Processing -------------------- #
    # Clean dataset inplace
    clean_inplace(train_data, val_data, test_data)

    nn_preprocessor = Preprocessor(252*3, 90, 90)
    nn_preprocessor.process_train_data(train_data)

    
    
    # train_ret, val_ret, test_ret = get_only_returns(
    #     train_data,
    #     val_data,
    #     test_data
    # )

    # cov, corr = preprocess_cov(train_ret)

    # # -------------------- Modeling -------------------- #
    # hrp = HierarchialRiskParity()
    # weights = hrp.calculate_weights(cov, corr)
    # print(weights * 100)