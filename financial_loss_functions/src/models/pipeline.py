from src.data_processing.loading import load_cov_processed
from src.models.cov_models import (
    HierarchialRiskParity
)

def run_training_pipeline(processed_data_path):
    cov, corr, test = load_cov_processed(processed_data_path)

    hrp = HierarchialRiskParity()
    hrp_weights = hrp.calculate_weights(cov, corr)

    print(hrp_weights)