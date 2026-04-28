# tests/test_processing_with_committed_sample.py
import os
import pytest
import shutil
import pandas as pd
from pathlib import Path
from src.utils.io import load_json
from src.data_processing.pipeline import run_processing_pipeline

# ---------- Integration test for Data Processing Pipeline ---------- #
def _get_raw_files_names():
    raw_files_dict = load_json(
        os.path.join('config', 'paths.json')
    )['raw_files']
    return raw_files_dict

def _build_expected_feats(df: pd.DataFrame, common_feats: list[str]) -> tuple[list, list]:
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
    features = sorted(set(features)) # Important to sort
    
    # Build list of <ticker>_<feature> in alphabetical order
    all_features = []
    for ticker in tickers:
        for feat in features:
            all_features.append(f'{ticker}_{feat}')
    
    # Append sorted common features, eg., sprtrn (s&p500)
    all_features.extend(sorted(common_feats))

    order_ba_spreads = [f'{ticker}_BA_SPREAD' for ticker in tickers]

    return all_features, tickers, order_ba_spreads

@pytest.mark.integration
def test_processing_pipeline_smoke(tmp_path):
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
    
    # ---------- SHOULD MATCH PATHS CONFIG ---------- #
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
            "benchmark_train": str(tmp_processed / 'benchmark_train.csv'),
            "benchmark_val": str(tmp_processed / 'benchmark_val.csv'),
            "benchmark_test": str(tmp_processed / 'benchmark_test.csv'),
            "ba_val": str(tmp_processed / 'ba_val.csv'),
            "ba_test": str(tmp_processed / 'ba_test.csv')
        }
    }

    features_config = {'common_features': ['sprtrn'], 'sp500_returns': 'sprtrn'}
    hparams_config = {'rolling_windows': {'out_size': 60}}

    # run pipeline (integration smoke test)
    run_processing_pipeline(paths_config, hparams_config, features_config)

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

        null_count = processed_df.isna().sum().sum()
        assert null_count == 0, f'Found {null_count} NaNs in {file_name} after processing'

    # Loop over raw files
    for split, raw_df in raw_files.items():
        # Build expected feature columns and check processed files
        expected_all_feats_cols, expected_tickers, expected_ba_cols = _build_expected_feats(
            raw_df,
            features_config['common_features']
        )

        # Check if all features match
        assert len(list(processed_files[f'processed_{split}'].columns)) \
            == len(expected_all_feats_cols), f'Lengths of feature columns in processed {split} do not match'
        
        assert list(processed_files[f'processed_{split}'].columns) \
            == expected_all_feats_cols, f'Feature columns are different {split}'

        # Check if returns features match (will be only ticker symbol)
        assert list(processed_files[f'returns_{split}'].columns) \
            == expected_tickers, f'Returns columns in {split} do not match'
        
        # Check if BA spreads match. <Ticker>_BA_SPREAD
        if split == 'train':
            pass #### SINCE WE DONT HAVE BA SPREAD FOR TRAIN ####
        else:
            assert list(processed_files[f'ba_{split}'].columns) \
                == expected_ba_cols, f'BA Spread columns in {split} do not match'

        # Check if s&p500 benchmarks returns match
        assert len(processed_files[f'benchmark_{split}'].columns) == 1, 'Only 1 column of S&P500 returns must be present'
        assert list(processed_files[f'benchmark_{split}'].columns)[0] == \
            features_config['sp500_returns']
        
        # Check that feature indices match target (returns) indices exactly
        proc = processed_files[f'processed_{split}']
        rets = processed_files[f'returns_{split}']
        pd.testing.assert_index_equal(
            proc.index, 
            rets.index,
            obj=f'Index mismatch between features and returns in {split}'
        )

        # Check that BA Spread indices match target (returns) indices exactly
        if split == 'train': # We dont have ba spreads for train
            pass
        else:
            ba_spreads = processed_files[f'ba_{split}']
            pd.testing.assert_index_equal(
                ba_spreads.index,
                rets.index,
                obj=f'Index mismatch between ba spreads and returns in {split}'
            )
        
        # Check that Benchmark (S&P500) indices match target (returns) indices exactly
        bench = processed_files[f'benchmark_{split}']
        pd.testing.assert_index_equal(
            bench.index, 
            rets.index,
            obj=f'Index mismatch between benchmark returns and all stock returns in {split}'
        )
    
    # Verify all processed files have identical schemas
    train_cols = processed_files['processed_train'].columns
    for split in ['val', 'test']:
        pd.testing.assert_index_equal(
            processed_files[f'processed_{split}'].columns, 
            train_cols,
            obj=f'Columns in {split} do not match train set'
        )
