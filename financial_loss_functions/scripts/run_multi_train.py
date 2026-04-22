"""
Multi-Run Training Script
=========================
Trains a single model+loss combination N times with different seeds,
tracks all results, and saves the best-performing run.

Usage (from financial_loss_functions/ directory):
    python -m scripts.run_multi_train -m BaseLSTM -l custom_loss_1 --n_runs 5
    python -m scripts.run_multi_train -m TFT -l custom_loss_3 --seeds 42,50,67,313,3
"""

import os
import sys
import copy
import signal
import argparse
import time
import json

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_path_config, load_json, create_directory, save_to_csv, save_to_json
from src.utils.device import set_seed, get_best_device
from src.utils.formatting import reformat_hparams
from src.training.train_nn import Trainer
from src.training.loss_functions import LossLibrary
from src.models.registry import NNModelLibrary, TradModelLibrary
from src.evaluation.evaluator import Evaluator, EqualWeightCalculator
from src.evaluation.metrics import MetricLibrary
from src.data_processing.loading import load_csv_files
from src.data_processing.dataset import Reshaper, calc_in_out_idx

# ────────────────────────────────────────────────────────────────
# Signal handling (graceful Ctrl+C)
# ────────────────────────────────────────────────────────────────
_interrupted = False


def signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        print("\nForce quitting immediately...")
        sys.exit(1)
    _interrupted = True
    print(f"\n{'='*60}")
    print("INTERRUPTED - Cleaning up before exit...")
    print("Press Ctrl+C again to force quit immediately.")
    print(f"{'='*60}")
    _cleanup()
    sys.exit(130 if signum == signal.SIGINT else 1)


def _cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Cleared GPU/MPS memory.")
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    print("Cleanup complete.")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ────────────────────────────────────────────────────────────────
# Helper: search registries (mirrors CandidatesGrid internals)
# ────────────────────────────────────────────────────────────────

def _search_model(model_lib, model_name):
    """Return model class or None."""
    for _cat, models_dict in model_lib.items():
        if model_name in models_dict:
            return models_dict[model_name]
    return None


def _search_loss_func(loss_lib, loss_name):
    """Return loss callable or None."""
    for _cat, cat_dict in loss_lib.items():
        for _sub, sub_dict in cat_dict.items():
            if loss_name in sub_dict:
                return sub_dict[loss_name]
    return None

# ────────────────────────────────────────────────────────────────
# Core multi-run logic
# ────────────────────────────────────────────────────────────────

def run_multi_train(
    model_name: str,
    loss_name: str,
    n_runs: int = 5,
    seeds: list[int] | None = None,
    rank_metric: str = "sharpe",
    resume: bool = False,
):
    """
    Train *model_name* with *loss_name* for *n_runs* seeds.
    After all runs, pick the best one by *rank_metric* and persist:
      - best model weights  (.pt)
      - all-runs summary    (.csv)
      - best run config     (.json)
    """
    start_time = time.time()

    # ── Seeds ─────────────────────────────────────────────────
    if seeds is None:
        rng = np.random.default_rng(42)
        seeds = rng.integers(1, 10_000, size=n_runs).tolist()
    else:
        n_runs = len(seeds)

    print(f"\n{'='*60}")
    print(f"  Multi-Run Training: {model_name} + {loss_name}")
    print(f"  Runs: {n_runs} | Seeds: {seeds}")
    print(f"  Ranking metric: {rank_metric}")
    print(f"{'='*60}\n")

    # ── Load configs (relative paths) ─────────────────────────
    paths_config = load_path_config(os.path.join("config", "paths.json"))
    hparams_config = load_json(os.path.join("config", "hparams.json"))
    features_config = load_json(os.path.join("config", "features.json"))

    # ── Create artifact directories ───────────────────────────
    artifacts_paths = {}
    for name, path in paths_config["artifacts"].items():
        dir_path = Path(path)
        create_directory(dir_path)
        artifacts_paths[name] = dir_path

    # Multi-run specific dirs
    best_models_dir = artifacts_paths["results_dir"] / "best_models"
    multi_run_dir = artifacts_paths["results_dir"] / "multi_run"
    checkpoints_root = artifacts_paths["results_dir"] / "checkpoints"
    create_directory(best_models_dir)
    create_directory(multi_run_dir)
    create_directory(checkpoints_root)

    # ── Register models & losses ──────────────────────────────
    models_module = paths_config["models_module"]
    TradModelLibrary.autodiscover(models_module)
    NNModelLibrary.autodiscover(models_module)

    model_lib = NNModelLibrary.items()
    loss_lib = LossLibrary.items()

    # ── Resolve model & loss ──────────────────────────────────
    model_class = _search_model(model_lib, model_name)
    if model_class is None:
        print(f"ERROR: Model '{model_name}' not found in the registry.")
        print("Available models:")
        for cat in NNModelLibrary.list_categories():
            print(f"  [{cat}]: {NNModelLibrary.list_models(cat)}")
        sys.exit(1)

    loss_func = _search_loss_func(loss_lib, loss_name)
    if loss_func is None:
        print(f"ERROR: Loss function '{loss_name}' not found in the registry.")
        print("Available loss functions:")
        for cat in LossLibrary.list_categories():
            for sub in LossLibrary.list_subcategories(cat):
                print(f"  [{cat}/{sub}]: {LossLibrary.list_functions(cat, sub)}")
        sys.exit(1)

    # ── Load & preprocess data ────────────────────────────────
    processed_files = {
        "processed_train": Path(paths_config["processed_paths"]["processed_train"]),
        "processed_val": Path(paths_config["processed_paths"]["processed_val"]),
        "returns_train": Path(paths_config["processed_paths"]["returns_train"]),
        "returns_val": Path(paths_config["processed_paths"]["returns_val"]),
    }
    processed_dfs = load_csv_files(processed_files)
    train_data = processed_dfs["processed_train"]
    returns_train = processed_dfs["returns_train"]
    val_data = processed_dfs["processed_val"]
    returns_val = processed_dfs["returns_val"]

    print(f"Train shape: {train_data.shape}")
    print(f"Val shape:   {val_data.shape}")

    windows_cfg = hparams_config["rolling_windows"]
    common_features = features_config["common_features"]
    reshaper = Reshaper(
        windows_cfg["in_size"],
        windows_cfg["out_size"],
        windows_cfg["stride"],
        common_features,
    )
    reshaper.extract_features(train_data.columns)

    X_train, y_train, _ = reshaper.reshape(train_data, returns_train)
    X_val, y_val, _ = reshaper.reshape(val_data, returns_val)
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")

    # ── Resolve hparam configs for this combo ─────────────────
    model_cfg = hparams_config["nn_models"][model_name]
    loss_cfg = hparams_config.get("losses", {}).get(loss_name, {})
    best_config = reformat_hparams(model_cfg, loss_cfg)

    # ── Device ────────────────────────────────────────────────
    torch_device = get_best_device()
    # DeformTime fallback to CPU
    if model_name == "DeformTime":
        torch_device = torch.device("cpu")

    # ── Convert to dataset tensors ────────────────────────────
    from src.data_processing.dataset import WindowDataset

    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val)
    X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

    # ── Metric functions to evaluate each run ─────────────────
    metric_funcs = {
        "sharpe": MetricLibrary.get("sharpe"),
        "sortino": MetricLibrary.get("sortino"),
        "max_drawdown": MetricLibrary.get("max_drawdown"),
        "cvar": MetricLibrary.get("cvar"),
        "omega": MetricLibrary.get("omega"),
        "calmar": MetricLibrary.get("calmar"),
    }

    # ── Run loop ──────────────────────────────────────────────
    run_records = []       # list of dicts for the summary CSV
    best_metric_val = -np.inf
    best_run_idx = -1
    best_model_state = None

    for run_idx, seed in enumerate(seeds):
        run_start = time.time()
        print(f"\n{'─'*50}")
        print(f"  Run {run_idx + 1}/{n_runs}  |  Seed: {seed}")
        print(f"{'─'*50}")

        set_seed(seed)

        try:
            trainer = Trainer(
                model=model_class,
                loss=loss_func,
                model_hparams=copy.deepcopy(best_config["model"]),
                optimizer_hparams=copy.deepcopy(best_config["optimizer"]),
                train_hparams=best_config["train"],
                in_size=X_train_shape[2],
                num_stocks=y_train_shape[2],
                max_seq_len=X_train_shape[1],
                scheduler_hparams=best_config.get("scheduler"),
                loss_hparams=copy.deepcopy(best_config.get("loss")) if best_config.get("loss") else None,
                device=torch_device,
            )

            # Per-seed checkpoint directory.
            run_ckpt_dir = checkpoints_root / f"{model_name}-{loss_name}-seed{seed}"
            create_directory(run_ckpt_dir)

            # If resuming, pick up from the most recent checkpoint for this seed.
            if resume:
                latest = Trainer.find_latest_checkpoint(str(run_ckpt_dir))
                if latest is not None:
                    trainer.resume_from(latest)

            # Train with validation, writing checkpoints along the way.
            trainer.train(train_ds, val_ds, checkpoint_dir=str(run_ckpt_dir))
            # Evaluate to get allocation weights
            trainer.evaluate(val_ds)
            alloc_weights = trainer.get_eval_alloc_weights()

            # ── Compute portfolio metrics ─────────────────────
            combo_label = f"{model_name}-{loss_name}-seed{seed}"
            evaluator = Evaluator(y_val, None)
            evaluator.calc_pf_daily_rets(alloc_weights, combo_label)

            run_metrics = {"run": run_idx + 1, "seed": seed}
            for met_name, met_func in metric_funcs.items():
                met_series = evaluator.calc_metric_performance(met_func, mean=True)
                run_metrics[met_name] = round(float(met_series.item()), 6)

            run_metrics["best_val_loss"] = round(float(trainer.best_val_loss), 6)
            run_metrics["best_train_loss"] = round(float(trainer.best_train_loss), 6)
            run_records.append(run_metrics)

            run_time = round(time.time() - run_start, 1)
            print(f"\n  Run {run_idx + 1} Results ({run_time}s):")
            for k, v in run_metrics.items():
                if k not in ("run", "seed"):
                    print(f"    {k:>16s}: {v}")

            # ── Track best ────────────────────────────────────
            current_metric = run_metrics.get(rank_metric, -np.inf)
            if current_metric > best_metric_val:
                best_metric_val = current_metric
                best_run_idx = run_idx
                best_model_state = copy.deepcopy(trainer.best_model_state or trainer.model.state_dict())
                print(f"  ★ New best! {rank_metric}={current_metric:.6f}")

        except Exception as e:
            print(f"  ✗ Run {run_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            run_records.append({
                "run": run_idx + 1,
                "seed": seed,
                "error": str(e),
            })

        finally:
            # Free memory between runs
            try:
                del trainer
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # ── Save results ──────────────────────────────────────────
    combo_tag = f"{model_name}-{loss_name}"

    #  1) All-runs summary CSV
    summary_df = pd.DataFrame(run_records)
    summary_path = multi_run_dir / f"{combo_tag}_runs.csv"
    save_to_csv(summary_df, summary_path)
    print(f"\n✓ Runs summary saved to: {summary_path}")

    #  2) Best model weights
    if best_model_state is not None:
        weights_path = best_models_dir / f"{combo_tag}_best.pt"
        torch.save(best_model_state, weights_path)
        print(f"✓ Best model weights saved to: {weights_path}")

        #  3) Best run config JSON
        best_record = run_records[best_run_idx]
        best_info = {
            "model": model_name,
            "loss": loss_name,
            "best_seed": best_record.get("seed"),
            "best_run": best_record.get("run"),
            "rank_metric": rank_metric,
            "rank_metric_value": best_record.get(rank_metric),
            "metrics": {
                k: v for k, v in best_record.items()
                if k not in ("run", "seed", "error")
            },
            "hparams": best_config,
            "weights_file": str(weights_path),
            "n_runs": n_runs,
            "all_seeds": seeds,
        }
        config_path = multi_run_dir / f"{combo_tag}_best_config.json"
        save_to_json(best_info, str(config_path))
        print(f"✓ Best config saved to: {config_path}")
    else:
        print("\n⚠ No successful runs — nothing to save.")

    # ── Final summary ─────────────────────────────────────────
    total_time = round((time.time() - start_time) / 60, 2)
    print(f"\n{'='*60}")
    print(f"  Multi-Run Complete — {n_runs} runs in {total_time} min")
    if best_run_idx >= 0 and best_run_idx < len(run_records):
        br = run_records[best_run_idx]
        print(f"  Best Run: #{br.get('run')} (seed={br.get('seed')})")
        print(f"  Best {rank_metric}: {best_metric_val:.6f}")
    print(f"{'='*60}\n")

    # Print full results table
    if run_records:
        print("\nAll Runs Summary:")
        print(summary_df.to_string(index=False))

    return summary_df


# ────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Multi-Run Training: train a model+loss combo N times and save the best."
        )
        parser.add_argument(
            "-m", "--model", required=True,
            help="Model name (e.g. BaseLSTM, TFT, AttentionLSTM)",
        )
        parser.add_argument(
            "-l", "--loss", required=True,
            help="Loss function name (e.g. custom_loss_1, raw_sharpe_objective)",
        )
        parser.add_argument(
            "--n_runs", type=int, default=5,
            help="Number of training runs (default: 5). Ignored if --seeds is provided.",
        )
        parser.add_argument(
            "--seeds", type=str, default=None,
            help="Comma-separated list of seeds (e.g. 42,50,67). Overrides --n_runs.",
        )
        parser.add_argument(
            "--rank_metric", type=str, default="sharpe",
            choices=["sharpe", "sortino", "max_drawdown", "cvar", "omega", "calmar"],
            help="Metric used to rank runs and pick the best (default: sharpe).",
        )
        parser.add_argument(
            "--resume", action="store_true",
            help="If set, resume each per-seed run from the latest checkpoint on disk "
                 "(if any). Checkpoint frequency is controlled by "
                 "hparams.json > nn_models[model].train.checkpoint_every.",
        )

        args = parser.parse_args()

        seed_list = None
        if args.seeds:
            seed_list = [int(s.strip()) for s in args.seeds.split(",")]

        run_multi_train(
            model_name=args.model,
            loss_name=args.loss,
            n_runs=args.n_runs,
            seeds=seed_list,
            rank_metric=args.rank_metric,
            resume=args.resume,
        )

    except SystemExit:
        pass
    except Exception as e:
        print(f"\nMulti-run training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
