# tests/test_processing_with_committed_sample.py
import os
import pytest
import shutil
import pandas as pd
from pathlib import Path
from src.utils import load_config
from src.data_processing.pipeline import run_processing_pipeline

# ---------- Integration test for Data Processing Pipeline ---------- #
def _get_raw_files_names():
    raw_files_dict = load_config(
        os.path.join('config', 'paths.json')
    )['raw_files']
    return raw_files_dict

def _build_expected_feats(df: pd.DataFrame, common_feats: list[str]):
    """Extract tickers and features from full DataFrame column names."""
    tickers = []
    features = []

    def _split_col(col: str) -> tuple[str, str]:
        """Split column into (ticker, feature) using first underscore only."""
        parts = col.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Column '{col}' does not match <ticker>_<feature> format")
        return parts[0], parts[1]  # ticker, feature-with-underscores
    
    for col in df.columns:
        if col != 'date' and col not in common_feats:
            t, f = _split_col(col)
            tickers.append(t)
            features.append(f)

    tickers = sorted(set(tickers)) # Important to sort
    # features.extend(common_feats)
    features = set(features) # Important to sort
    

    expected_columns = []
    for ticker in tickers:
        for feat in features:
            expected_columns.append(f'{ticker}_{feat}')
    
    for ticker in tickers:
        for com_feat in common_feats:
            expected_columns.append(f'{ticker}_{com_feat}')

    print(tickers)
    print(features)
    return expected_columns

    # Deterministic order for reshaping
    # cols_per_ticker = [
    #     f'{t}{col_sep}{f}' for t in tickers for f in features
    # ]


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
    sample_raw_files = _get_raw_files_names()
    for fname in list(sample_raw_files.values()):
        src = sample_raw_dir / fname
        dst = tmp_raw / fname
        shutil.copy(src, dst)
    
    paths_config = {
        'data': {
            'crsp_dir': str(tmp_raw),
            'processed_dir': str(tmp_processed)
        },
        'raw_files': {
            'train': str(tmp_raw / sample_raw_files['train']),
            'val': str(tmp_raw / sample_raw_files['val']),
            'test': str(tmp_raw / sample_raw_files['test'])
        },
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

    # load processed files
    processed_files = {
        file_name: pd.read_csv(processed_path, index_col=0, parse_dates=True) 
        for file_name, processed_path in paths_config['processed_paths'].items()
    }

    raw_files = {
        file_name: pd.read_csv(raw_path) 
        for file_name, raw_path in paths_config['raw_files'].items()
    }

    # Loop over processed dataframes
    for file_name, processed_df in processed_files.items():
        # Check if outputs exist
        assert not processed_df.empty, f'{file_name} csv is empty'

        # Check for any datetime-like index
        is_datetime = pd.api.types.is_datetime64_any_dtype(processed_df.index)
        assert is_datetime, f'Expected DatetimeIndex, but got {processed_df.index.dtype}'

        # Check for chronlogical date ordering
        assert processed_df.index.is_monotonic_increasing, 'Data is not chronologically sorted'


    expected_cols = _build_expected_feats(
        raw_files['train'],
        features_config['common_features']
    )

    for file_name, raw_df in raw_files.items():
        expected_cols = _build_expected_feats(
            raw_df,
            features_config['common_features']
        )

        assert set(processed_files[f'processed_{file_name}'].columns) \
            == set(expected_cols), 'Feature columns are different'

        # pd.testing.assert_index_equal(processed_files['processed_train'].columns, pd.Index(expected_cols))
