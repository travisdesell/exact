import os
from dotenv import load_dotenv
from src.models.cov_models import HierarchialRiskParity
<<<<<<< HEAD
from src.data_processing.preprocess import load_crsp_datasets, clean_data_returns, preprocess_cov
=======
from cov_models import HierarchialRiskParity
<<<<<<< HEAD:financial_loss_functions/src/main.py
<<<<<<< HEAD:financial_loss_functions/src/main.py
from preprocess import load_crsp_datasets, get_only_returns, preprocess_cov, clean_inplace
>>>>>>> ad232f2 (cleaning function added):financial_loss_functions/main.py
=======
=======
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/main.py
from preprocess import (
=======
from src.data_processing.preprocess import (
>>>>>>> 96d6df7 (rebase done)
    load_crsp_datasets,
    get_only_returns,
    preprocess_cov,
    clean_inplace,
    Preprocessor
)
<<<<<<< HEAD
<<<<<<< HEAD:financial_loss_functions/src/main.py
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/main.py
=======
>>>>>>> 8e18d58 (Yeo Johnson Transformation implemeted for vol_change):financial_loss_functions/main.py
=======

>>>>>>> 96d6df7 (rebase done)

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