import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Syntheitc processed CSV files
def _create_minimal_processed_data(processed_dir: Path, num_train=360, num_val=60):
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

    # BA spreads (ticker_BA_SPREAD)
    ba_cols = [f'{t}_BA_SPREAD' for t in tickers]
    ba_val = pd.DataFrame(np.random.randn(num_val, num_stocks) * 0.01, columns=ba_cols, index=val_dates)
    ba_val.to_csv(processed_dir / 'ba_val.csv')

    # Benchmark (S&P500)
    benchmark_val = pd.DataFrame({'sprtrn': np.random.randn(num_val)}, index=val_dates)
    benchmark_val.to_csv(processed_dir / 'benchmark_val.csv')
    benchmark_train = pd.DataFrame({'sprtrn': np.random.randn(num_train)}, index=train_dates)
    benchmark_train.to_csv(processed_dir / 'benchmark_train.csv')


# ---------- Integration Test for Single model training pipeline ---------- #
@pytest.mark.integration
def test_training_pipeline_smoke(tmp_path):
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    artifacts_dir = tmp_path / 'artifacts'
    artifacts_dir.mkdir()

    _create_minimal_processed_data(processed_dir)

    paths_config = {
        'processed_paths': {
            'returns_train': processed_dir / 'returns_train.csv',
            'returns_val': processed_dir / 'returns_val.csv',
            'processed_train': processed_dir / 'processed_train.csv',
            'processed_val': processed_dir / 'processed_val.csv',
            'benchmark_val': processed_dir / 'benchmark_val.csv',
            'ba_val': processed_dir / 'ba_val.csv'
        },
        'artifacts': {
            'avg_perf_dir': artifacts_dir / 'avg_perf',
            'hparams_dir': artifacts_dir / 'hparams',
            'temp_dir': artifacts_dir / 'temp',
            'tuned_plots_dir': artifacts_dir / 'plots',
            'wfv_rets_dir': artifacts_dir / 'daily_rets'
        },
        'models_module': 'src.models'
    }
    for sub in paths_config['artifacts'].values():
        sub.mkdir(parents=True, exist_ok=True)

    hparams_config = {
        'rolling_windows': {'in_size': 180, 'out_size': 60, 'stride': 1},
        'tuner': {'n_tuning_trials': 2},
        'nn_models': {
            'BaseLSTM': {
                'model': {'hidden_size': 8, 'num_layers': 1, 'dropout': 0.2},
                'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4},
                'train': {
                    'train_batch_size': 4, 'val_batch_size': 4,
                    'clip_grad_norm': 0.5, 'epochs': 5,
                    'early_stopping': False
                },
                'tuning': {
                    'hidden_size': {'type': 'categorical', 'choices': [8]},
                    'lr': {'type': 'float', 'low': 1e-3, 'high': 1e-3, 'log': False}
                }
            }
        },
        'losses': {
            'custom_loss_10': {
                'lambdas': {'cvar_lambda': 0.1, 'risk_p_lambda': 0.1},
                'tuning': {
                    'cvar_lambda': {'type': 'categorical', 'choices': [0.1]},
                    'risk_p_lambda': {'type': 'categorical', 'choices': [0.1]}
                }
            },
            'custom_loss_11': {
                'lambdas': {'cvar_lambda': 0.1, 'risk_p_lambda': 0.1},
                'tuning': {
                    'cvar_lambda': {'type': 'categorical', 'choices': [0.1]},
                    'risk_p_lambda': {'type': 'categorical', 'choices': [0.1]}
                }
            }
        }
        
    }

    features_config = {'common_features': ['sprtrn'], 'sp500_returns': 'sprtrn'}

    from scripts.run_training import run_tuning_pipeline
    run_tuning_pipeline(
        paths_config=paths_config,
        hparams_config=hparams_config,
        features_config=features_config,
        grid_mode='one_model',
        loss_mode='custom',
        model_name='BaseLSTM',
        tune=True,
        mpi=False
    )

    # Checks if output files exist
    assert (artifacts_dir / 'avg_perf' / 'avg_perf_BaseLSTM.csv').exists()
    assert (artifacts_dir / 'daily_rets' / 'daily_rets_BaseLSTM.json').exists()
    assert (artifacts_dir / 'hparams' / 'optimized_BaseLSTM.json').exists()