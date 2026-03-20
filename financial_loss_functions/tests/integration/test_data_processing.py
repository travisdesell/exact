import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.utils import load_path_config, load_config
from src.data_processing.pipeline import run_processing_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _create_synthetic_macro(tmp_path: Path, n_months: int = 120) -> Path:
    """Create minimal synthetic macro CSVs in a temp directory."""
    macro_dir = tmp_path / 'macro'
    macro_dir.mkdir()
    rng = np.random.default_rng(42)
    idx = pd.date_range('2016-01-01', periods=n_months, freq='MS')

    rates = pd.DataFrame({
        'FEDFUNDS': np.cumsum(rng.normal(0, 0.05, n_months)) + 2.0,
        'TB3MS': np.cumsum(rng.normal(0, 0.03, n_months)) + 1.5,
    }, index=idx)
    rates.to_csv(macro_dir / 'Rates_FX.csv')

    prices = pd.DataFrame({
        'CPIAUCSL': np.linspace(240, 280, n_months) + rng.normal(0, 0.5, n_months),
    }, index=idx)
    prices.to_csv(macro_dir / 'Prices.csv')

    return macro_dir


@pytest.mark.integration
def test_pipeline_with_macro_produces_processed_files(tmp_path):
    """Pipeline should produce processed CSVs with macro columns when macro data is present."""
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()

    macro_dir = _create_synthetic_macro(tmp_path)

    paths_config = load_path_config(
        str(PROJECT_ROOT / 'config' / 'paths.json'),
        'sample',
    )

    paths_config['data']['raw_macro_dir'] = str(macro_dir)
    paths_config['data']['processed_dir'] = str(processed_dir)

    for key in ['returns_train', 'returns_val', 'returns_test',
                'processed_train', 'processed_val', 'processed_test']:
        paths_config['processed_paths'][key] = str(processed_dir / f'{key}.csv')

    paths_config['processed_paths']['updated_common_features'] = str(
        processed_dir / 'updated_common_features.json'
    )

    features_config = {
        'common_features': ['sprtrn'],
        'macro_per_stock': 1,
        'feature_selection': {
            'lags': [5, 10],
            'low_corr_threshold': 0.1,
        },
    }

    run_processing_pipeline(paths_config, features_config)

    assert (processed_dir / 'processed_train.csv').exists()
    assert (processed_dir / 'processed_val.csv').exists()
    assert (processed_dir / 'processed_test.csv').exists()
    assert (processed_dir / 'returns_train.csv').exists()

    train_df = pd.read_csv(processed_dir / 'processed_train.csv', index_col=0, nrows=3)
    assert train_df.shape[1] > 0, 'Processed train should have columns'


@pytest.mark.integration
def test_pipeline_with_macro_updates_common_features(tmp_path):
    """Updated common features should include macro columns after pipeline runs."""
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    macro_dir = _create_synthetic_macro(tmp_path)

    paths_config = load_path_config(
        str(PROJECT_ROOT / 'config' / 'paths.json'),
        'sample',
    )
    paths_config['data']['raw_macro_dir'] = str(macro_dir)
    paths_config['data']['processed_dir'] = str(processed_dir)

    for key in ['returns_train', 'returns_val', 'returns_test',
                'processed_train', 'processed_val', 'processed_test']:
        paths_config['processed_paths'][key] = str(processed_dir / f'{key}.csv')

    updated_path = processed_dir / 'updated_common_features.json'
    paths_config['processed_paths']['updated_common_features'] = str(updated_path)

    features_config = {
        'common_features': ['sprtrn'],
        'macro_per_stock': 1,
        'feature_selection': {
            'lags': [5],
            'low_corr_threshold': 0.1,
        },
    }

    run_processing_pipeline(paths_config, features_config)

    assert updated_path.exists()
    with open(updated_path) as f:
        updated_features = json.load(f)

    assert 'sprtrn' in updated_features
    assert len(updated_features) > 1, 'Should include at least one macro feature beyond sprtrn'


@pytest.mark.integration
def test_pipeline_without_macro_still_works(tmp_path):
    """Pipeline should work gracefully when no macro data directory exists."""
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()

    paths_config = load_path_config(
        str(PROJECT_ROOT / 'config' / 'paths.json'),
        'sample',
    )
    paths_config['data']['raw_macro_dir'] = str(tmp_path / 'nonexistent_macro')
    paths_config['data']['processed_dir'] = str(processed_dir)

    for key in ['returns_train', 'returns_val', 'returns_test',
                'processed_train', 'processed_val', 'processed_test']:
        paths_config['processed_paths'][key] = str(processed_dir / f'{key}.csv')
    paths_config['processed_paths']['updated_common_features'] = str(
        processed_dir / 'updated_common_features.json'
    )

    features_config = {
        'common_features': ['sprtrn'],
        'macro_per_stock': 2,
        'feature_selection': {
            'lags': [10],
            'low_corr_threshold': 0.1,
        },
    }

    run_processing_pipeline(paths_config, features_config)

    assert (processed_dir / 'processed_train.csv').exists()
    assert (processed_dir / 'processed_val.csv').exists()
