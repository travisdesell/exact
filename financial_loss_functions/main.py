import os
from dotenv import load_dotenv
from cov_models import HierarchialRiskParity
from preprocess import load_crsp_datasets, clean_data_returns, preprocess_cov

load_dotenv()

if __name__ == '__main__':
    # -------------------- Data Loading -------------------- #
    data_dir = os.getenv('DATA_DIR')

    crsp_path = os.path.join(data_dir, '2023_sp_500_select_50')
    train_data, val_data, test_data = load_crsp_datasets(crsp_path)

    # -------------------- Cleaning & Processing -------------------- #
    train_ret, val_ret, test_ret = clean_data_returns(
        train_data,
        val_data,
        test_data
    )

    cov, corr = preprocess_cov(train_ret)

    # -------------------- Modeling -------------------- #
    hrp = HierarchialRiskParity()
    weights = hrp.calculate_weights(cov, corr)
    print(weights * 100)