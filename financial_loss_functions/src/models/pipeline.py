from typing import Dict
from src.data_processing.loading import load_cov_processed
from src.models.cov_models import (
    HierarchialRiskParity
)

def run_training_pipeline(paths_config: Dict):
    cov_train_path = paths_config['processed_paths']['cov_train']
    corr_train_path = paths_config['processed_paths']['corr_train']
    
    cov, corr = load_cov_processed(cov_train_path, corr_train_path)

    hrp = HierarchialRiskParity()
    hrp_weights = hrp.calculate_weights(cov, corr)

    print(hrp_weights)