import numpy as np
import pandas as pd
from torch import optim
from typing import Dict, Optional
from pathlib import Path
from src.utils import create_directory
from src.data_processing.dataset import Reshaper
from src.data_processing.dataset import WindowDataset
from src.data_processing.loading import load_csv_files
from src.models.lstm import BaseLSTM, AttentionLSTM
from src.models.cvar_benchmark import CVaRBenchmark, CVaRParams
from src.training.train import (
    Trainer,
    train_val_losses_plot,
    Evaluator
)
from src.training.loss_functions import (
    raw_sharpe_loss,
    raw_sortino_loss,
    differentiable_sharpe_loss,
    CompositeSRLoss,
)
from src.evaluation.pyfolio_viz import (
    weights_to_pyfolio,
    build_window_dates,
    generate_comparison_tearsheets,
    comparison_summary,
)


def _build_composite_loss(reshaper: Reshaper, hparams_config: Dict) -> CompositeSRLoss:
    """Instantiate CompositeSRLoss using feature indices from the Reshaper."""
    features = reshaper.get_features()
    tickers = reshaper.get_tickers()

    feat_index = {f: i for i, f in enumerate(features)}
    ret_idx = feat_index.get("RET", 0)
    turn_idx = feat_index.get("TURNOVER", 1)
    illiq_idx = feat_index.get("ILLIQUIDITY", 2)
    spread_idx = feat_index.get("BA_SPREAD", 3)

    known_stock_features = {"RET", "TURNOVER", "ILLIQUIDITY", "BA_SPREAD", "VOL_CHANGE", "sprtrn"}
    macro_indices = [
        feat_index[f] for f in features
        if f not in known_stock_features
    ]

    loss_cfg = hparams_config.get("CompositeSRLoss", {})

    return CompositeSRLoss(
        num_tickers=len(tickers),
        num_features_per_ticker=len(features),
        ret_feature_idx=ret_idx,
        turnover_feature_idx=turn_idx,
        illiq_feature_idx=illiq_idx,
        ba_spread_feature_idx=spread_idx,
        macro_col_indices=macro_indices if macro_indices else None,
        alpha=loss_cfg.get("alpha", 0.10),
        beta=loss_cfg.get("beta", 0.05),
        gamma=loss_cfg.get("gamma", 0.10),
        delta=loss_cfg.get("delta", 0.10),
        psych_thresholds=loss_cfg.get("psych_thresholds"),
        psych_sigma=loss_cfg.get("psych_sigma", 0.01),
        ema_span=loss_cfg.get("ema_span", 10),
    )


def _extract_returns_from_X(
    X: np.ndarray, reshaper: Reshaper,
) -> np.ndarray:
    """
    Extract per-ticker returns from the flat (W, T, N*F) input array.
    Returns (W, T, N).
    """
    features = reshaper.get_features()
    tickers = reshaper.get_tickers()
    F = len(features)
    ret_idx = sorted(features).index("RET") if "RET" in features else 0

    W, T, _ = X.shape
    N = len(tickers)
    out = np.zeros((W, T, N), dtype=X.dtype)
    for j in range(N):
        out[:, :, j] = X[:, :, j * F + ret_idx]
    return out


def _load_sec_fundamentals(
    paths_config: Dict,
    tickers: list,
    good_starts_train: np.ndarray,
    good_starts_val: np.ndarray,
    in_size: int,
    date_index_train: Optional[pd.DatetimeIndex] = None,
    date_index_val: Optional[pd.DatetimeIndex] = None,
):
    """
    Load pre-computed composite fundamental scores and window them.

    Returns (fund_train, fund_val) as numpy arrays of shape (W, N),
    or (None, None) if the file doesn't exist.
    """
    sec_dir = Path(paths_config.get("data", {}).get("sec_filings_dir", "data/raw/sec_filings"))
    score_path = sec_dir / "composite_fundamental_scores.csv"
    if not score_path.exists():
        return None, None

    scores = pd.read_csv(score_path, index_col=0, parse_dates=True)
    available_tickers = [t for t in tickers if t in scores.columns]
    if not available_tickers:
        return None, None

    scores = scores[available_tickers]

    def _window_scores(starts, date_index):
        if date_index is None:
            return None
        aligned = scores.reindex(date_index).ffill().bfill().fillna(0.0)
        windows = []
        for s in starts:
            idx = s + in_size - 1
            if idx < len(aligned):
                windows.append(aligned.iloc[idx].values)
            else:
                windows.append(aligned.iloc[-1].values)
        return np.array(windows) if windows else None

    fund_train = _window_scores(good_starts_train, date_index_train)
    fund_val = _window_scores(good_starts_val, date_index_val)
    return fund_train, fund_val


def run_training_pipeline(paths_config: Dict, hparams_config: Dict):
    """
    All models training pipeline entry point.

    Trains BaseLSTM and AttentionLSTM with both the legacy Sharpe loss
    and the composite S/R loss, runs a CVaR benchmark, and generates
    pyfolio tearsheets comparing all strategies.

    @param paths_config Dict Dictionary containing paths
    @param hparams_config Dict Dictionary containing hyperparameter information
    """
    print('\n', '=' * 20, ' Training Pipeline ', '=' * 20)
    
    # Create plots directory if it doesnt exist
    plots_dir = Path(paths_config['artifacts']['plots'])
    create_directory(plots_dir)
    
    # -------------------- Loading Processed Data -------------------- #
    processed_files = {
        'processed_train': Path(paths_config['processed_paths']['processed_train']),
        'processed_val': Path(paths_config['processed_paths']['processed_val']),
        'returns_train': Path(paths_config['processed_paths']['returns_train']),
        'returns_val': Path(paths_config['processed_paths']['returns_val'])
    }

    processed_dfs = load_csv_files(processed_files)
    train_data = processed_dfs['processed_train']
    returns_train = processed_dfs['returns_train']

    val_data = processed_dfs['processed_val']
    returns_val = processed_dfs['returns_val']

    print('Train shape:', train_data.shape)
    print('Val shape:', val_data.shape)

    # -------------------- Preprocessing (Reshaping) -------------------- #
    in_size = hparams_config['rolling_windows']['in_size']
    out_size = hparams_config['rolling_windows']['out_size']

    reshaper = Reshaper(
        in_size,
        out_size,
        hparams_config['rolling_windows']['stride']
    )
    reshaper.extract_features(train_data)
    
    X_train, y_train, good_starts_train = reshaper.reshape(train_data, returns_train)
    print('-'*10, ' train shapes ', '-'*10)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    X_val, y_val, good_starts_val = reshaper.reshape(val_data, returns_val)
    print('-'*10, ' val shapes ', '-'*10)
    print('X_val shape:', X_val.shape)
    print('y_val shape:', y_val.shape)

    # -------------------- SEC Fundamentals (optional) -------------------- #
    tickers = reshaper.get_tickers()
    fund_train, fund_val = _load_sec_fundamentals(
        paths_config, tickers,
        good_starts_train, good_starts_val, in_size,
        date_index_train=train_data.index if hasattr(train_data, 'index') else None,
        date_index_val=val_data.index if hasattr(val_data, 'index') else None,
    )
    if fund_train is not None:
        print(f'SEC fundamentals loaded: train={fund_train.shape}, val={fund_val.shape}')
    else:
        print('SEC fundamentals not available -- skipping L_fundamental')

    # -------------------- Build Composite Loss -------------------- #
    composite_loss = _build_composite_loss(reshaper, hparams_config)

    # -------------------- Training Models -------------------- #
    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    evaluator = Evaluator(y_val)
    all_strategy_weights: Dict[str, np.ndarray] = {}

    # ---- BaseLSTM (Sharpe-only, baseline) ----
    model1_name = 'BaseLSTM'
    print('\n', '-'*10, f' Training {model1_name} (Sharpe) ', '-'*10)
    try:
        trainer = Trainer(
            model=BaseLSTM,
            optimizer=optim.AdamW,
            loss=differentiable_sharpe_loss,
            model_hparams=hparams_config[model1_name]['model'],
            optimizer_hparams=hparams_config[model1_name]['optimizer'],
            train_hparams=hparams_config[model1_name]['train'],
            in_size=X_train.shape[2],
            num_stocks=y_train.shape[2]
        )
        trainer.train(train_ds)
        trainer.evaluate(val_ds)

        train_val_losses_plot(
            trainer.train_losses, trainer.val_losses,
            model1_name + ' (Sharpe) Loss Curves',
            plots_dir / (model1_name + '_Sharpe_Loss.png'),
        )
        alloc_weights = trainer.get_val_alloc_weights()
        evaluator.calc_pf_daily_rets(alloc_weights, model1_name + ' (Sharpe)')
        all_strategy_weights[model1_name + ' (Sharpe)'] = alloc_weights
    except Exception as error:
        print(f'DEBUG: Error training {model1_name} (Sharpe). Skipping.', error)

    # ---- BaseLSTM (Composite S/R Loss) ----
    model1c_name = 'BaseLSTM (Composite)'
    print('\n', '-'*10, f' Training {model1c_name} ', '-'*10)
    try:
        trainer = Trainer(
            model=BaseLSTM,
            optimizer=optim.AdamW,
            loss=composite_loss,
            model_hparams=hparams_config['BaseLSTM']['model'],
            optimizer_hparams=hparams_config['BaseLSTM']['optimizer'],
            train_hparams=hparams_config['BaseLSTM']['train'],
            in_size=X_train.shape[2],
            num_stocks=y_train.shape[2],
            fundamentals_train=fund_train,
            fundamentals_val=fund_val,
        )
        trainer.train(train_ds)
        trainer.evaluate(val_ds)

        train_val_losses_plot(
            trainer.train_losses, trainer.val_losses,
            model1c_name + ' Loss Curves',
            plots_dir / 'BaseLSTM_Composite_Loss.png',
        )
        alloc_weights = trainer.get_val_alloc_weights()
        evaluator.calc_pf_daily_rets(alloc_weights, model1c_name)
        all_strategy_weights[model1c_name] = alloc_weights
    except Exception as error:
        print(f'DEBUG: Error training {model1c_name}. Skipping.', error)

    # ---- AttentionLSTM (Composite S/R Loss) ----
    model2_name = 'AttentionLSTM (Composite)'
    print('\n', '-'*10, f' Training {model2_name} ', '-'*10)
    try:
        trainer = Trainer(
            model=AttentionLSTM,
            optimizer=optim.AdamW,
            loss=composite_loss,
            model_hparams=hparams_config['AttentionLSTM']['model'],
            optimizer_hparams=hparams_config['AttentionLSTM']['optimizer'],
            train_hparams=hparams_config['AttentionLSTM']['train'],
            in_size=X_train.shape[2],
            num_stocks=y_train.shape[2],
            fundamentals_train=fund_train,
            fundamentals_val=fund_val,
        )
        trainer.train(train_ds)
        trainer.evaluate(val_ds)

        train_val_losses_plot(
            trainer.train_losses, trainer.val_losses,
            model2_name + ' Loss Curves',
            plots_dir / 'AttentionLSTM_Composite_Loss.png',
        )
        alloc_weights = trainer.get_val_alloc_weights()
        evaluator.calc_pf_daily_rets(alloc_weights, model2_name)
        all_strategy_weights[model2_name] = alloc_weights
    except Exception as error:
        print(f'DEBUG: Error training {model2_name}. Skipping.', error)

    # -------------------- CVaR Benchmark -------------------- #
    print('\n', '-'*10, ' CVaR Benchmark ', '-'*10)
    try:
        cvar_cfg = hparams_config.get('CVaRBenchmark', {})
        cvar_params = CVaRParams(
            confidence=cvar_cfg.get('confidence', 0.95),
            risk_aversion=cvar_cfg.get('risk_aversion', 1.0),
            w_min=cvar_cfg.get('w_min', 0.0),
            w_max=cvar_cfg.get('w_max', 0.30),
            L_tar=cvar_cfg.get('L_tar', 1.6),
        )
        cvar_bench = CVaRBenchmark(params=cvar_params)

        X_val_rets = _extract_returns_from_X(X_val, reshaper)
        cvar_weights = cvar_bench.rolling_optimize(X_val_rets)

        evaluator.calc_pf_daily_rets(cvar_weights, 'CVaR Benchmark')
        all_strategy_weights['CVaR Benchmark'] = cvar_weights
        print(f'CVaR Benchmark weights shape: {cvar_weights.shape}')
    except Exception as error:
        print(f'DEBUG: CVaR Benchmark failed. Skipping.', error)

    # -------------------- Equal Weight -------------------- #
    evaluator.calc_eq_wt_daily_rets()
    eq_w = np.full((y_val.shape[0], y_val.shape[2]), 1.0 / y_val.shape[2])
    all_strategy_weights['Equal Weight'] = eq_w

    # -------------------- Windowed Comparison Plot -------------------- #
    evaluator.plot_windowed_comparison(plots_dir / 'Daily_Returns.png')

    total_returns = evaluator.calc_total_performance('returns')
    total_sharpes = evaluator.calc_total_performance('sharpe')

    print('\n', '-'*10, ' Portfolio Performance Metrics ', '-'*10)
    print('\n', 'Compounded returns for each window:\n', total_returns)
    print('\n', 'Basic sharpe ratios for each window:\n', total_sharpes)

    # -------------------- pyfolio Tearsheets -------------------- #
    print('\n', '-'*10, ' Generating pyfolio Tearsheets ', '-'*10)
    try:
        val_dates = val_data.index if hasattr(val_data, 'index') else pd.RangeIndex(len(val_data))
        window_dates = build_window_dates(val_dates, good_starts_val, in_size, out_size)

        benchmark_col = 'sprtrn'
        if benchmark_col in val_data.columns:
            bench_rets = val_data[benchmark_col]
        elif hasattr(val_data, 'index'):
            bench_rets = pd.Series(0.0, index=val_data.index)
        else:
            bench_rets = pd.Series(0.0, index=pd.RangeIndex(len(val_data)))

        strategies: Dict[str, dict] = {}
        for name, w_arr in all_strategy_weights.items():
            try:
                pf_data = weights_to_pyfolio(
                    weights=w_arr,
                    returns=y_val,
                    tickers=tickers,
                    window_dates=window_dates,
                    benchmark_returns=bench_rets,
                )
                strategies[name] = pf_data
            except Exception as exc:
                print(f'DEBUG: pyfolio conversion failed for {name}: {exc}')

        if strategies:
            pyfolio_dir = plots_dir / 'pyfolio'
            generate_comparison_tearsheets(strategies, pyfolio_dir)
            summary = comparison_summary(strategies)
            print('\n', '-'*10, ' Strategy Comparison Summary ', '-'*10)
            print(summary.to_string())
            summary.to_csv(plots_dir / 'strategy_comparison.csv')
    except Exception as error:
        print(f'DEBUG: pyfolio tearsheets failed. Skipping.', error)