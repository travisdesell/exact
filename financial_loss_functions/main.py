import os
from dotenv import load_dotenv
from cov_models import HierarchialRiskParity
from preprocess import load_crsp_datasets, get_only_returns, preprocess_cov, clean_inplace

load_dotenv()

if __name__ == '__main__':
    # -------------------- Data Loading -------------------- #
    data_dir = os.getenv('DATA_DIR')

    crsp_path = os.path.join(data_dir, '2023_sp_500_select_50')
    train_data, val_data, test_data = load_crsp_datasets(crsp_path)

    # -------------------- Cleaning & Processing -------------------- #
    # Clean dataset inplace
    print(train_data.shape)    
    clean_inplace(train_data, val_data, test_data)
    print(train_data.shape)
    
    
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