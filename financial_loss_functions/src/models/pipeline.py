from typing import Dict
from pathlib import Path
from src.data_processing.loading import load_processed_files
from src.models.cov_models import (
    HierarchialRiskParity
)

def run_training_pipeline(paths_config: Dict):
    cov_files = {
        'cov_train': Path(paths_config['processed_paths']['cov_train']),
        'corr_train': Path(paths_config['processed_paths']['corr_train'])
    }

    cov_proc_dfs = load_processed_files(cov_files) 

    hrp = HierarchialRiskParity()
    hrp_weights = hrp.calculate_weights(
        cov_proc_dfs['cov_train'],
        cov_proc_dfs['corr_train']
    )

    print(hrp_weights)