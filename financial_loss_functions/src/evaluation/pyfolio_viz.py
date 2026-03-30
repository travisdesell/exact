"""
pyfolio integration for portfolio strategy visualization.

Converts model allocation weights and return arrays into the
pandas-based formats that pyfolio expects, then generates
comparison tearsheets across strategies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def weights_to_pyfolio(
    weights: np.ndarray,
    returns: np.ndarray,
    tickers: List[str],
    window_dates: List[pd.DatetimeIndex],
    benchmark_returns: pd.Series,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Convert rolling-window model outputs into pyfolio-compatible data.

    @param weights np.ndarray (W, N) portfolio weights per window.
    @param returns np.ndarray (W, T_out, N) future returns per window.
    @param tickers list[str] ordered ticker symbols.
    @param window_dates list[pd.DatetimeIndex] dates for each window's
        T_out period.  ``len(window_dates)`` must equal W, and each
        element has length T_out.
    @param benchmark_returns pd.Series daily benchmark returns indexed
        by date (e.g. sprtrn).
    @param initial_capital float notional starting capital for the
        positions table.

    @return dict with keys:
        ``returns``  – pd.Series (daily non-cumulative portfolio returns)
        ``positions`` – pd.DataFrame (daily dollar notionals per ticker + cash)
        ``benchmark_rets`` – pd.Series (aligned benchmark returns)
    """
    W, T_out, N = returns.shape

    all_port_rets: Dict[pd.Timestamp, float] = {}
    all_positions: Dict[pd.Timestamp, Dict[str, float]] = {}

    portfolio_value = initial_capital

    for w_idx in range(W):
        w = weights[w_idx]  # (N,)
        for t in range(T_out):
            date = window_dates[w_idx][t]
            if date in all_port_rets:
                continue

            daily_ret = float((w * returns[w_idx, t, :]).sum())
            all_port_rets[date] = daily_ret

            pos = {}
            for k, ticker in enumerate(tickers):
                pos[ticker] = portfolio_value * w[k]
            pos["cash"] = 0.0
            all_positions[date] = pos

            portfolio_value *= (1 + daily_ret)

    ret_series = pd.Series(all_port_rets).sort_index()
    ret_series.index = pd.to_datetime(ret_series.index)
    ret_series.index.name = None

    pos_df = pd.DataFrame.from_dict(all_positions, orient="index").sort_index()
    pos_df.index = pd.to_datetime(pos_df.index)
    pos_df.index.name = None

    aligned_bench = benchmark_returns.reindex(ret_series.index).fillna(0.0)

    return {
        "returns": ret_series,
        "positions": pos_df,
        "benchmark_rets": aligned_bench,
    }


def build_window_dates(
    full_date_index: pd.DatetimeIndex,
    good_starts: np.ndarray,
    in_size: int,
    out_size: int,
) -> List[pd.DatetimeIndex]:
    """
    Compute the T_out date ranges for each rolling window.

    @param full_date_index pd.DatetimeIndex  All dates in the split.
    @param good_starts np.ndarray  Starting indices of valid windows.
    @param in_size int  Input window length.
    @param out_size int  Output window length.

    @return list[pd.DatetimeIndex] one DatetimeIndex per window.
    """
    window_dates = []
    for s in good_starts:
        start = s + in_size
        end = start + out_size
        window_dates.append(full_date_index[start:end])
    return window_dates


# ---------------------------------------------------------------------------
# Tearsheet generation
# ---------------------------------------------------------------------------

def _try_import_pyfolio():
    """Lazy-import pyfolio (the reloaded fork)."""
    try:
        import pyfolio as pf
        return pf
    except ImportError as exc:
        raise ImportError(
            "pyfolio-reloaded is required for tearsheet generation. "
            "Install with: pip install pyfolio-reloaded"
        ) from exc


def generate_returns_tearsheet(
    strategy_data: Dict,
    title: str = "",
    output_path: Optional[Path] = None,
):
    """
    Generate a pyfolio returns tearsheet for a single strategy.

    @param strategy_data dict  Output of ``weights_to_pyfolio``.
    @param title str  Optional plot title prefix.
    @param output_path Path  If provided, save the figure to this path.
    """
    pf = _try_import_pyfolio()

    fig = pf.create_returns_tear_sheet(
        strategy_data["returns"],
        benchmark_rets=strategy_data["benchmark_rets"],
        return_fig=True,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Tearsheet saved to %s", output_path)

    plt.close("all")


def generate_comparison_tearsheets(
    strategies: Dict[str, Dict],
    output_dir: Path,
):
    """
    Generate and save a pyfolio returns tearsheet for each strategy.

    @param strategies dict  Mapping strategy name -> pyfolio data dict.
    @param output_dir Path  Directory to save tearsheet PNGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in strategies.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        out_path = output_dir / f"tearsheet_{safe_name}.png"
        try:
            generate_returns_tearsheet(data, title=name, output_path=out_path)
            logger.info("Generated tearsheet for %s", name)
        except Exception as exc:
            logger.warning("Tearsheet generation failed for %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Comparison summary table
# ---------------------------------------------------------------------------

def comparison_summary(
    strategies: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Compute summary performance metrics for all strategies.

    @param strategies dict  strategy name -> pyfolio data dict.
    @return pd.DataFrame  Rows = strategies, columns = metrics.
    """
    rows = []
    for name, data in strategies.items():
        rets = data["returns"]
        ann_factor = 252
        total_ret = float((1 + rets).prod() - 1)
        ann_ret = float((1 + total_ret) ** (ann_factor / max(len(rets), 1)) - 1)
        ann_vol = float(rets.std() * np.sqrt(ann_factor))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = float(dd.min())

        rows.append({
            "strategy": name,
            "total_return": round(total_ret, 4),
            "ann_return": round(ann_ret, 4),
            "ann_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
        })
    return pd.DataFrame(rows).set_index("strategy")
