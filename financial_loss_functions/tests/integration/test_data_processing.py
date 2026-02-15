# tests/test_processing_with_committed_sample.py
import os
import pytest
import shutil
import pandas as pd
from pathlib import Path
from src.utils import load_config
from src.data_processing.pipeline import run_processing_pipeline

# ---------- Integration test for Data Processing Pipeline ---------- #
def get_raw_files_names():
    raw_files_dict = load_config(
        os.path.join('config', 'paths.json')
    )['raw_files']
    return raw_files_dict

@pytest.mark.integration
def test_processing_with_committed_sample(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]  # adjust if tests nested differently
    sample_raw_dir = repo_root / 'data' / 'raw' / 'sample'
    assert sample_raw_dir.exists(), 'Please commit a small sample dataset at data/raw/sample'

    # create tmp raw and processed dirs for an isolated run
    tmp_raw = tmp_path / 'raw' / 'crsp'
    tmp_raw.mkdir(parents=True, exist_ok=True)
    tmp_processed = tmp_path / 'processed'
    tmp_processed.mkdir(parents=True, exist_ok=True)

    # copy sample files (assumes sample contains train.csv, val.csv, test.csv)
    sample_raw_files = get_raw_files_names()
    for fname in list(sample_raw_files.values()):
        src = sample_raw_dir / fname
        dst = tmp_raw / fname
        shutil.copy(src, dst)

    paths_config = {
        'data': {
            'crsp_dir': str(tmp_raw),
            'processed_dir': str(tmp_processed)
        },
        'raw_files': sample_raw_files,
        'processed_paths': {
            'returns_train': str(tmp_processed / 'returns_train.csv'),
            'returns_val': str(tmp_processed / 'returns_val.csv'),
            'returns_test': str(tmp_processed / 'returns_test.csv'),
            'processed_train': str(tmp_processed / 'processed_train.csv'),
            'processed_val': str(tmp_processed / 'processed_val.csv'),
            'processed_test': str(tmp_processed / 'processed_test.csv'),
        }
    }

    features_config = {'common_features': ['sprtrn']}

    # run pipeline (integration smoke test)
    run_processing_pipeline(paths_config, features_config)

    # Basic assertions: outputs exist
    processed_train = pd.read_csv(paths_config['processed_paths']['processed_train'])
    processed_val = pd.read_csv(paths_config['processed_paths']['processed_val'])
    processed_test = pd.read_csv(paths_config['processed_paths']['processed_test'])

    returns_train = pd.read_csv(paths_config['processed_paths']['returns_train'])
    returns_val = pd.read_csv(paths_config['processed_paths']['returns_val'])
    returns_test = pd.read_csv(paths_config['processed_paths']['returns_test'])
    
    assert not processed_train.empty, 'Processed train csv is empty'
    assert not processed_val.empty, 'Processed validation csv is empty'
    assert not processed_test.empty, 'Processed test csv is empty'

    assert not returns_train.empty, 'Returns only train csv is empty'
    assert not returns_val.empty, 'Returns only val csv is empty'
    assert not returns_test.empty, 'Returns only test csv is empty'