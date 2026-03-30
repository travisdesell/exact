"""
SEC filing data pipeline using edgartools.

Fetches 10-K, 10-Q, and 8-K filings for a set of tickers, extracts
XBRL financial statements, computes fundamental features, aligns them
to a daily business-day grid, and caches results as Parquet.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNDAMENTAL_WEIGHTS = {
    "revenue_growth": 0.30,
    "operating_margin": 0.25,
    "debt_to_equity": -0.20,
    "fcf_yield": 0.15,
    "event_signal": 0.10,
}

_EVENT_WINDOW_DAYS = 2


# ---------------------------------------------------------------------------
# Filing retrieval helpers
# ---------------------------------------------------------------------------

def _init_edgar(identity: str = "FinLossFunctions research@example.com"):
    """Lazy-import edgartools and set SEC identity."""
    try:
        import edgar
    except ImportError as exc:
        raise ImportError(
            "edgartools is required for the SEC filing pipeline. "
            "Install with: pip install edgartools"
        ) from exc
    edgar.set_identity(identity)
    return edgar


def fetch_filings_for_ticker(
    ticker: str,
    forms: Tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    *,
    edgar_module=None,
) -> Dict[str, list]:
    """
    Retrieve filings for *ticker* grouped by form type.

    Returns dict mapping form string to a list of Filing objects.
    """
    if edgar_module is None:
        edgar_module = _init_edgar()

    company = edgar_module.Company(ticker)
    result: Dict[str, list] = {}
    for form in forms:
        try:
            filings = company.get_filings(form=form)
            result[form] = list(filings) if filings is not None else []
        except Exception as exc:
            logger.warning("Could not fetch %s for %s: %s", form, ticker, exc)
            result[form] = []
    return result


# ---------------------------------------------------------------------------
# XBRL extraction helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_statement_value(statement, *concept_labels) -> Optional[float]:
    """
    Try to pull a numeric value from an XBRL statement by trying
    multiple concept label variants.
    """
    if statement is None:
        return None
    try:
        df = statement.to_dataframe() if hasattr(statement, "to_dataframe") else None
    except Exception:
        return None
    if df is None or df.empty:
        return None

    for label in concept_labels:
        for col in df.columns:
            if label.lower() in str(col).lower():
                vals = df[col].dropna()
                if not vals.empty:
                    return _safe_float(vals.iloc[0])
    return None


def extract_financials_from_filing(filing) -> Dict[str, Optional[float]]:
    """
    Extract key financial metrics from a single 10-K or 10-Q filing
    via its XBRL data.

    Returns a dict with keys:
        revenue, operating_income, net_income,
        total_assets, total_liabilities, stockholders_equity,
        current_assets, current_liabilities,
        operating_cashflow, capex
    """
    metrics: Dict[str, Optional[float]] = {
        "revenue": None,
        "operating_income": None,
        "net_income": None,
        "total_assets": None,
        "total_liabilities": None,
        "stockholders_equity": None,
        "current_assets": None,
        "current_liabilities": None,
        "operating_cashflow": None,
        "capex": None,
    }

    try:
        obj = filing.obj()
    except Exception:
        return metrics

    financials = getattr(obj, "financials", None)
    if financials is None:
        try:
            from edgar.financials import Financials
            financials = Financials.extract(filing)
        except Exception:
            return metrics

    if financials is None:
        return metrics

    # Income statement
    try:
        inc = financials.income_statement()
        metrics["revenue"] = _extract_statement_value(
            inc, "Revenue", "Revenues", "Net Revenue", "SalesRevenueNet",
            "RevenueFromContractWithCustomer",
        )
        metrics["operating_income"] = _extract_statement_value(
            inc, "OperatingIncome", "Operating Income",
        )
        metrics["net_income"] = _extract_statement_value(
            inc, "NetIncome", "Net Income",
        )
    except Exception:
        pass

    # Balance sheet
    try:
        bs = financials.balance_sheet()
        metrics["total_assets"] = _extract_statement_value(
            bs, "Assets", "TotalAssets",
        )
        metrics["total_liabilities"] = _extract_statement_value(
            bs, "Liabilities", "TotalLiabilities",
        )
        metrics["stockholders_equity"] = _extract_statement_value(
            bs, "StockholdersEquity", "Equity",
        )
        metrics["current_assets"] = _extract_statement_value(
            bs, "CurrentAssets", "AssetsCurrent",
        )
        metrics["current_liabilities"] = _extract_statement_value(
            bs, "CurrentLiabilities", "LiabilitiesCurrent",
        )
    except Exception:
        pass

    # Cash flow statement
    try:
        cf = financials.cashflow_statement()
        metrics["operating_cashflow"] = _extract_statement_value(
            cf, "OperatingCashFlow", "NetCashFromOperating",
            "CashFromOperatingActivities",
        )
        metrics["capex"] = _extract_statement_value(
            cf, "CapitalExpenditure", "Capex",
            "PaymentsToAcquirePropertyPlantAndEquipment",
        )
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_ticker_fundamentals(
    filings_10k: list,
    filings_10q: list,
) -> pd.DataFrame:
    """
    Build a time-indexed DataFrame of fundamental features from 10-K
    and 10-Q filings for a single ticker.

    Columns: revenue_growth, operating_margin, debt_to_equity,
             current_ratio, fcf_yield
    """
    records = []
    all_filings = []
    for f in filings_10k:
        all_filings.append(("10-K", f))
    for f in filings_10q:
        all_filings.append(("10-Q", f))

    for form_type, filing in all_filings:
        try:
            filing_date = pd.Timestamp(filing.filing_date)
        except Exception:
            continue

        data = extract_financials_from_filing(filing)
        records.append({"date": filing_date, "form": form_type, **data})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").drop_duplicates(
        subset=["date"], keep="last"
    )
    df = df.set_index("date")

    # Revenue growth (QoQ)
    if "revenue" in df.columns:
        df["revenue_growth"] = df["revenue"].pct_change()
    else:
        df["revenue_growth"] = np.nan

    # Operating margin
    rev = df.get("revenue")
    op_inc = df.get("operating_income")
    if rev is not None and op_inc is not None:
        df["operating_margin"] = np.where(
            (rev != 0) & rev.notna() & op_inc.notna(),
            op_inc / rev,
            np.nan,
        )
    else:
        df["operating_margin"] = np.nan

    # Debt-to-equity
    liab = df.get("total_liabilities")
    equity = df.get("stockholders_equity")
    if liab is not None and equity is not None:
        df["debt_to_equity"] = np.where(
            (equity != 0) & equity.notna() & liab.notna(),
            liab / equity,
            np.nan,
        )
    else:
        df["debt_to_equity"] = np.nan

    # Current ratio
    ca = df.get("current_assets")
    cl = df.get("current_liabilities")
    if ca is not None and cl is not None:
        df["current_ratio"] = np.where(
            (cl != 0) & cl.notna() & ca.notna(),
            ca / cl,
            np.nan,
        )
    else:
        df["current_ratio"] = np.nan

    # FCF yield (proxy: (operating CF - capex) / total_assets as stand-in)
    ocf = df.get("operating_cashflow")
    capex = df.get("capex")
    ta = df.get("total_assets")
    if ocf is not None and ta is not None:
        capex_vals = capex.fillna(0) if capex is not None else 0
        df["fcf_yield"] = np.where(
            (ta != 0) & ta.notna() & ocf.notna(),
            (ocf - capex_vals.abs()) / ta,
            np.nan,
        )
    else:
        df["fcf_yield"] = np.nan

    feature_cols = [
        "revenue_growth",
        "operating_margin",
        "debt_to_equity",
        "current_ratio",
        "fcf_yield",
    ]
    return df[feature_cols]


def _compute_event_signal(
    filings_8k: list,
    target_index: pd.DatetimeIndex,
    window_days: int = _EVENT_WINDOW_DAYS,
) -> pd.Series:
    """
    Create a binary event indicator that is 1 within +/- *window_days*
    of any 8-K filing date.
    """
    event_dates = set()
    for f in filings_8k:
        try:
            event_dates.add(pd.Timestamp(f.filing_date))
        except Exception:
            continue

    signal = pd.Series(0.0, index=target_index, name="event_signal")
    for ed in event_dates:
        mask = (target_index >= ed - pd.Timedelta(days=window_days)) & (
            target_index <= ed + pd.Timedelta(days=window_days)
        )
        signal.loc[mask] = 1.0
    return signal


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_sec_pipeline_for_ticker(
    ticker: str,
    target_index: pd.DatetimeIndex,
    cache_dir: Optional[Path] = None,
    edgar_module=None,
) -> pd.DataFrame:
    """
    End-to-end: fetch filings, extract features, align to daily grid.

    Returns DataFrame indexed by *target_index* with columns:
        revenue_growth, operating_margin, debt_to_equity,
        current_ratio, fcf_yield, event_signal
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{ticker}_fundamentals.parquet"
        if cache_path.exists():
            logger.info("Loading cached SEC data for %s", ticker)
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            return cached.reindex(target_index).ffill().bfill()

    if edgar_module is None:
        edgar_module = _init_edgar()

    filings_by_form = fetch_filings_for_ticker(
        ticker, ("10-K", "10-Q", "8-K"), edgar_module=edgar_module
    )

    fund_df = _compute_ticker_fundamentals(
        filings_by_form.get("10-K", []),
        filings_by_form.get("10-Q", []),
    )

    event_signal = _compute_event_signal(
        filings_by_form.get("8-K", []),
        target_index,
    )

    if fund_df.empty:
        result = pd.DataFrame(
            0.0,
            index=target_index,
            columns=[
                "revenue_growth", "operating_margin", "debt_to_equity",
                "current_ratio", "fcf_yield", "event_signal",
            ],
        )
    else:
        aligned = fund_df.reindex(target_index).ffill().bfill()
        aligned["event_signal"] = event_signal.values
        result = aligned

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)

    return result


def compute_composite_fundamental_scores(
    all_ticker_fundamentals: Dict[str, pd.DataFrame],
    tickers: List[str],
    target_index: pd.DatetimeIndex,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Combine per-ticker fundamental DataFrames into a single composite
    score DataFrame with columns = tickers, index = target_index.

    Each ticker's features are z-scored cross-sectionally, then
    combined with *weights* (defaults to FUNDAMENTAL_WEIGHTS).
    """
    if weights is None:
        weights = FUNDAMENTAL_WEIGHTS

    feature_names = [
        "revenue_growth", "operating_margin", "debt_to_equity",
        "fcf_yield", "event_signal",
    ]

    panels: Dict[str, pd.DataFrame] = {}
    for feat in feature_names:
        feat_df = pd.DataFrame(index=target_index)
        for t in tickers:
            if t in all_ticker_fundamentals and feat in all_ticker_fundamentals[t].columns:
                feat_df[t] = all_ticker_fundamentals[t][feat].reindex(target_index)
            else:
                feat_df[t] = 0.0
        feat_df = feat_df.ffill().bfill().fillna(0.0)
        panels[feat] = feat_df

    # Cross-sectional z-score per date and feature
    z_panels: Dict[str, pd.DataFrame] = {}
    for feat, df in panels.items():
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1).replace(0, 1)
        z_panels[feat] = df.sub(row_mean, axis=0).div(row_std, axis=0)

    composite = pd.DataFrame(0.0, index=target_index, columns=tickers)
    for feat, w in weights.items():
        if feat in z_panels:
            composite += w * z_panels[feat]

    return composite


def run_sec_filing_pipeline(
    tickers: List[str],
    target_index: pd.DatetimeIndex,
    cache_dir: Optional[Path] = None,
    identity: str = "FinLossFunctions research@example.com",
) -> pd.DataFrame:
    """
    Top-level entry: fetch SEC data for all *tickers*, compute composite
    fundamental scores, return DataFrame (dates x tickers).
    """
    edgar_module = _init_edgar(identity)

    all_funds: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        logger.info("Processing SEC filings for %s", ticker)
        try:
            df = run_sec_pipeline_for_ticker(
                ticker, target_index, cache_dir=cache_dir,
                edgar_module=edgar_module,
            )
            all_funds[ticker] = df
        except Exception as exc:
            logger.warning("SEC pipeline failed for %s: %s", ticker, exc)

    return compute_composite_fundamental_scores(
        all_funds, tickers, target_index
    )
