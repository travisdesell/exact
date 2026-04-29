import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_json
from src.evaluation.metrics import MetricLibrary
from src.models.registry import TradModelLibrary

# Helper to create processed CSV files (train, val, test)
def _create_minimal_processed_data(
        processed_dir: Path, num_train=360, num_val=120, num_test=120
):
    num_stocks = 50
    num_features_per_stock = 5
    tickers = [f'stock{i}' for i in range(num_stocks)]
    feature_suffixes = [f'feat{j}' for j in range(num_features_per_stock)]
    feature_cols = [f'{t}_{s}' for t in tickers for s in feature_suffixes]
    common_col = 'sprtrn'

    # Training data
    train_dates = pd.date_range('2000-01-01', periods=num_train, freq='D')
    train_features = pd.DataFrame(np.random.randn(num_train, len(feature_cols)), columns=feature_cols, index=train_dates)
    train_features[common_col] = np.random.randn(num_train)
    train_features.to_csv(processed_dir / 'processed_train.csv')

    train_returns = pd.DataFrame(np.random.randn(num_train, num_stocks), columns=tickers, index=train_dates)
    train_returns.to_csv(processed_dir / 'returns_train.csv')

    # Validation data
    val_dates = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=num_val, freq='D')
    val_features = pd.DataFrame(np.random.randn(num_val, len(feature_cols)), columns=feature_cols, index=val_dates)
    val_features[common_col] = np.random.randn(num_val)
    val_features.to_csv(processed_dir / 'processed_val.csv')

    val_returns = pd.DataFrame(np.random.randn(num_val, num_stocks), columns=tickers, index=val_dates)
    val_returns.to_csv(processed_dir / 'returns_val.csv')

    ba_cols = [f'{t}_BA_SPREAD' for t in tickers]
    ba_val = pd.DataFrame(np.random.randn(num_val, num_stocks) * 0.01, columns=ba_cols, index=val_dates)
    ba_val.to_csv(processed_dir / 'ba_val.csv')

    benchmark_val = pd.DataFrame({'sprtrn': np.random.randn(num_val)}, index=val_dates)
    benchmark_val.to_csv(processed_dir / 'benchmark_val.csv')

    # Test data
    test_dates = pd.date_range(val_dates[-1] + pd.Timedelta(days=1), periods=num_test, freq='D')
    test_features = pd.DataFrame(np.random.randn(num_test, len(feature_cols)), columns=feature_cols, index=test_dates)
    test_features[common_col] = np.random.randn(num_test)
    test_features.to_csv(processed_dir / 'processed_test.csv')

    test_returns = pd.DataFrame(np.random.randn(num_test, num_stocks), columns=tickers, index=test_dates)
    test_returns.to_csv(processed_dir / 'returns_test.csv')

    ba_test = pd.DataFrame(np.random.randn(num_test, num_stocks) * 0.01, columns=ba_cols, index=test_dates)
    ba_test.to_csv(processed_dir / 'ba_test.csv')

    benchmark_test = pd.DataFrame({'sprtrn': np.random.randn(num_test)}, index=test_dates)
    benchmark_test.to_csv(processed_dir / 'benchmark_test.csv')

    benchmark_train = pd.DataFrame({'sprtrn': np.random.randn(num_train)}, index=train_dates)
    benchmark_train.to_csv(processed_dir / 'benchmark_train.csv')


# Helper to create optimized hyperparameters JSON artifact
def _create_optimized_hparams(artifacts_dir: Path, model_loss: str, model_name: str):
    """Create dummy optimized hyperparameters file named by model name."""
    hparams_dir = artifacts_dir / 'optimized_hparams/'
    hparams_dir.mkdir(parents=True, exist_ok=True)
    hparams_data = {
        model_loss: {
            'model': {'hidden_size': 8, 'num_layers': 1, 'dropout': 0.2},
            'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4},
            'train': {
                'train_batch_size': 4, 'val_batch_size': 4,
                'clip_grad_norm': 0.5, 'epochs': 5,
                'early_stopping': False
            },
            'loss': {'cvar_lambda': 0.1, 'risk_p_lambda': 0.1}
        }
    }
    with open(hparams_dir / f'optimized_{model_name}.json', 'w') as f:
        json.dump(hparams_data, f, indent=4)


# ---------- Integration test for test‑set evaluation pipeline ---------- #
@pytest.mark.integration
def test_evaluation_pipeline_smoke(tmp_path):
    # 1. Create directories
    processed_dir = tmp_path / 'processed'
    artifacts_dir = tmp_path / 'artifacts'
    processed_dir.mkdir()
    artifacts_dir.mkdir()

    # Generate synthetic processed data (train, val, test)
    _create_minimal_processed_data(processed_dir, num_train=360, num_val=120, num_test=120)

    # Create artifacts (optimized hyperparameters) for the model+loss we will test
    model_loss = 'BaseLSTM-custom_loss_10'          # must match a loss in hparams_config
    model_name = 'BaseLSTM'                         # file name uses model name only
    _create_optimized_hparams(artifacts_dir, model_loss, model_name)

    # Build paths_config, hparams_config, features_config
    paths_config = {
        'processed_paths': {
            'returns_train': processed_dir / 'returns_train.csv',
            'returns_val': processed_dir / 'returns_val.csv',
            'returns_test': processed_dir / 'returns_test.csv',
            'processed_train': processed_dir / 'processed_train.csv',
            'processed_val': processed_dir / 'processed_val.csv',
            'processed_test': processed_dir / 'processed_test.csv',
            'ba_test': processed_dir / 'ba_test.csv',
            'benchmark_test': processed_dir / 'benchmark_test.csv'
        },
        'artifacts': {
            'avg_perf_dir': artifacts_dir / 'avg_perf/',
            'hparams_dir': artifacts_dir / 'optimized_hparams/',
            'temp_dir': artifacts_dir / 'temp/',
            'tuned_plots_dir': artifacts_dir / 'plots/',
            'wfv_rets_dir': artifacts_dir / 'daily_rets/',
            'test_perf_dir': artifacts_dir / 'test_perf/'
        },
        'models_module': 'src.models'
    }
    # Create artifact subdirectories
    for sub in paths_config['artifacts'].values():
        sub.mkdir(parents=True, exist_ok=True)

    # Verify the optimized file was created correctly
    assert (artifacts_dir / 'optimized_hparams' / f'optimized_{model_name}.json').exists()

    hparams_config = {
        'rolling_windows': {'in_size': 180, 'out_size': 60, 'stride': 1},
        'tuner': {},
        'seed_list': [42, 123],
        'nn_models': {'BaseLSTM': {}},
        'losses': {},
        "trad_models": {
            "GlobalMinimumVariance": {
                "allow_short": False
            },
            "MeanVariancePortfolio": {
                "expected_returns_method": "arithmetic",
                "risk_aversion": 1.0,
                "allow_short": False
            },
            "NestedClusteredOptimization": {
                "de_noise": True
            }
        }
    }

    features_config = {
        'common_features': ['sprtrn'],
        'sp500_returns': 'sprtrn'
    }

    # Run the evaluation pipeline
    from scripts.run_evaluation import run_evaluation_pipeline
    run_evaluation_pipeline(
        paths_config=paths_config,
        hparams_config=hparams_config,
        features_config=features_config,
        prev_grid_mode='one_model',
        model_losses=[model_loss],
        mpi=False
    )

    # Check if output files exist
    test_perf_dir = paths_config['artifacts']['test_perf_dir']
    test_avg_perf_path = test_perf_dir / \
        f'avg_test_perf_all_{len(hparams_config['seed_list'])}_seeds.csv'
    test_rets_path = test_perf_dir / 'test_returns_all.json'
    test_loss_curves_path = test_perf_dir / 'test_loss_curves_all.json'
    assert (test_avg_perf_path).exists()
    assert (test_rets_path).exists()
    assert (test_loss_curves_path).exists()

    # Check contents of average performance file
    avg_perf_df = pd.read_csv(test_avg_perf_path, index_col=0)

    benchmarks = list(TradModelLibrary.list_models())
    benchmarks.extend(['Equal_Weight', 'S&P500'])
    pf_metrics = MetricLibrary.items().keys()
    assert avg_perf_df.shape == (1 + len(benchmarks), len(pf_metrics)) # 1 model-loss and 2 is for benchmarks
    assert model_loss in avg_perf_df.index
    assert list(avg_perf_df.columns) == list(pf_metrics)

    # Check contents of daily returns file
    all_seed_daily_rets = load_json(test_rets_path)
    assert ['nn_models', 'benchmarks'] == list(all_seed_daily_rets.keys())

    for seed, models_dict in all_seed_daily_rets['nn_models'].items():
        assert int(seed) in hparams_config['seed_list'], f'Seed {seed} not saved in returns data.'
        assert len(models_dict) == 1, f'More than 1 model-loss was saved in returns for seed = {seed}.' # Since this test uses 1
        assert model_loss == list(models_dict.keys())[0]

        assert len(models_dict[model_loss]) == 2, 'There must be only 2 output windows in the test set for this test'
    
    for trad_mod, rets_dict in all_seed_daily_rets['benchmarks'].items():
        assert trad_mod in benchmarks

        assert len(rets_dict) == 2, 'There must be only 2 output windows in the test set for this test'