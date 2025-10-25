import argparse
import sys
import math
import json
from typing import Dict, List, Tuple
from datetime import datetime

import requests
import pandas as pd
import numpy as np

from fred_config import (
    FRED_API_KEY,
    FRED_START_DATE,
    FRED_END_DATE,
    FRED_FREQUENCY,
)

FRED_API_BASE = "https://api.stlouisfed.org/fred"
SESSION = requests.Session()


# ----------------------------
# Helpers: API + transforms
# ----------------------------

def fred_series_observations(series_id: str,
                             api_key: str,
                             start: str,
                             end: str,
                             frequency: str = "m",
                             aggregation_method: str = "avg") -> pd.Series:
    """
    Fetch a FRED series as a monthly pandas Series of floats.
    Uses FRED's internal aggregation when original frequency != monthly.

    frequency: 'm'  (monthly)
    aggregation_method: 'avg' for daily/weekly series; 'eop' for end-of-period if needed.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "frequency": frequency,
        "aggregation_method": aggregation_method,
        "units": "lin"
    }
    url = f"{FRED_API_BASE}/series/observations"
    r = SESSION.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"FRED API error {r.status_code} for {series_id}: {r.text[:200]}")
    data = r.json()
    if "observations" not in data:
        return pd.Series(dtype=float)
    obs = data["observations"]
    if not obs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(obs)
    # Some series return "." for missing values
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=series_id).astype(float)
    # Ensure monthly index
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp("M")
    return s


def tcode_transform(x: pd.Series, tcode: int) -> pd.Series:
    """
    Apply FRED-MD TCODE transformations (Appendix TCODE 1–7):contentReference[oaicite:10]{index=10}.
    Returns a pandas Series aligned with x.
    """
    x = x.copy()
    if tcode == 1:  # level
        return x
    elif tcode == 2:  # Δx
        return x.diff()
    elif tcode == 3:  # Δ²x
        return x.diff().diff()
    elif tcode == 4:  # log x
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(x.replace(0, np.nan))
        return y
    elif tcode == 5:  # Δ log x
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(x.replace(0, np.nan)).diff()
        return y
    elif tcode == 6:  # Δ² log x
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(x.replace(0, np.nan)).diff().diff()
        return y
    elif tcode == 7:  # Δ (x_t/x_{t-1} - 1)
        pct = x.pct_change()
        return pct.diff()
    else:
        raise ValueError(f"Unknown TCODE {tcode}")


# ----------------------------
# Series configuration
# ----------------------------
# Each entry: ("GroupName", "SeriesID", TCODE, "Description")
# The list below covers a broad, representative subset across all eight sectors,
# including all canonical FRED-MD workhorses and the derived series we build below.
# You can extend this list safely; the pipeline is generic.

BASE_SERIES: List[Tuple[str, str, int, str]] = [
    # Group 1: Output & income (mostly Δlog):contentReference[oaicite:11]{index=11}
    ("Output & Income", "RPI", 5, "Real Personal Income"),
    ("Output & Income", "W875RX1", 5, "Real Personal Income Excluding Transfers"),
    ("Output & Income", "INDPRO", 5, "Industrial Production Index"),
    ("Output & Income", "IPFPNSS", 5, "IP: Final Products & Nonindustrial Supplies"),
    ("Output & Income", "IPFINAL", 5, "IP: Final Products (Market Group)"),
    ("Output & Income", "IPCONGD", 5, "IP: Consumer Goods"),
    ("Output & Income", "IPDCONGD", 5, "IP: Durable Consumer Goods"),
    ("Output & Income", "IPNCONGD", 5, "IP: Nondurable Consumer Goods"),
    ("Output & Income", "IPBUSEQ", 5, "IP: Business Equipment"),
    ("Output & Income", "IPMAT", 5, "IP: Materials"),
    ("Output & Income", "IPDMAT", 5, "IP: Durable Materials"),
    ("Output & Income", "IPNMAT", 5, "IP: Nondurable Materials"),
    ("Output & Income", "IPMANSICS", 5, "IP: Manufacturing (SIC)"),
    ("Output & Income", "IPB51222S", 5, "IP: Residential Utilities"),
    ("Output & Income", "IPFUELS", 5, "IP: Fuels"),
    ("Output & Income", "NAPMPI", 1, "ISM Manufacturing: Production Index"),
    ("Output & Income", "CUMFNS", 2, "Capacity Utilization: Manufacturing"),

    # Group 2: Labor market:contentReference[oaicite:12]{index=12}
    ("Labor Market", "HWI", 2, "Help-Wanted Index (best-effort)"),
    ("Labor Market", "HWIURATIO", 2, "Help-Wanted to Unemployed Ratio"),
    ("Labor Market", "CLF16OV", 5, "Civilian Labor Force"),
    ("Labor Market", "CE16OV", 5, "Civilian Employment"),
    ("Labor Market", "UNRATE", 2, "Unemployment Rate"),
    ("Labor Market", "UEMPMEAN", 2, "Average Duration of Unemployment"),
    ("Labor Market", "PAYEMS", 5, "All Employees: Total Nonfarm"),
    ("Labor Market", "USGOOD", 5, "All Employees: Goods-Producing"),
    ("Labor Market", "USCONS", 5, "All Employees: Construction"),
    ("Labor Market", "MANEMP", 5, "All Employees: Manufacturing"),
    ("Labor Market", "DMANEMP", 5, "All Employees: Durable Goods"),
    ("Labor Market", "NDMANEMP", 5, "All Employees: Nondurable Goods"),
    ("Labor Market", "SRVPRD", 5, "All Employees: Service-Providing"),
    ("Labor Market", "USTPU", 5, "All Employees: Trade, Transportation & Utilities"),
    ("Labor Market", "USTRADE", 5, "All Employees: Retail Trade"),
    ("Labor Market", "USFIRE", 5, "All Employees: Financial Activities"),
    ("Labor Market", "USGOVT", 5, "All Employees: Government"),
    ("Labor Market", "CES0600000007", 1, "Avg Weekly Hours: Goods-Producing"),
    ("Labor Market", "AWOTMAN", 1, "Avg Weekly Overtime Hours: Manufacturing"),
    ("Labor Market", "AWHMAN", 1, "Avg Weekly Hours: Manufacturing"),
    ("Labor Market", "NAPMEI", 1, "ISM Manufacturing: Employment Index"),
    ("Labor Market", "CES0600000008", 6, "Avg Hourly Earnings: Goods-Producing"),
    ("Labor Market", "CES2000000008", 6, "Avg Hourly Earnings: Construction"),
    ("Labor Market", "CES3000000008", 6, "Avg Hourly Earnings: Manufacturing"),

    # Group 3: Housing (counts; use Δlog):contentReference[oaicite:13]{index=13}
    ("Housing", "HOUST", 5, "Housing Starts: Total"),
    ("Housing", "HOUSTNE", 5, "Housing Starts: Northeast"),
    ("Housing", "HOUSTMW", 5, "Housing Starts: Midwest"),
    ("Housing", "HOUSTS", 5, "Housing Starts: South"),
    ("Housing", "HOUSTW", 5, "Housing Starts: West"),
    ("Housing", "PERMIT", 5, "New Private Housing Permits (SAAR)"),
    ("Housing", "PERMITNE", 5, "Permits: Northeast"),
    ("Housing", "PERMITMW", 5, "Permits: Midwest"),
    ("Housing", "PERMITS", 5, "Permits: South"),
    ("Housing", "PERMITW", 5, "Permits: West"),

    # Group 4: Consumption, orders, inventories:contentReference[oaicite:14]{index=14}
    ("Consumption/Orders/Inventories", "DPCERA3M086SBEA", 5, "Real PCE"),
    ("Consumption/Orders/Inventories", "CMRMTSPL", 5, "Real Manufacturing & Trade Sales"),
    ("Consumption/Orders/Inventories", "RSAFS", 5, "Retail and Food Services Sales"),
    ("Consumption/Orders/Inventories", "NAPM", 1, "ISM PMI Composite Index"),
    ("Consumption/Orders/Inventories", "NAPMNOI", 1, "ISM New Orders Index"),
    ("Consumption/Orders/Inventories", "NAPMSDI", 1, "ISM Supplier Deliveries Index"),
    ("Consumption/Orders/Inventories", "NAPMII", 1, "ISM Inventories Index"),
    ("Consumption/Orders/Inventories", "ACOGNO", 5, "New Orders: Consumer Goods"),
    ("Consumption/Orders/Inventories", "AMDMNO", 5, "New Orders: Durable Goods"),
    ("Consumption/Orders/Inventories", "ANDENO", 5, "New Orders: Nondefense Capital Goods"),
    ("Consumption/Orders/Inventories", "AMDMUO", 5, "Unfilled Orders: Durable Goods"),
    ("Consumption/Orders/Inventories", "BUSINV", 5, "Total Business Inventories"),
    ("Consumption/Orders/Inventories", "ISRATIO", 2, "Inventories-to-Sales Ratio"),
    ("Consumption/Orders/Inventories", "UMCSENT", 2, "Consumer Sentiment Index"),

    # Group 5: Money & credit:contentReference[oaicite:15]{index=15}
    ("Money & Credit", "M1SL", 6, "M1 Money Stock"),
    ("Money & Credit", "M2SL", 6, "M2 Money Stock"),
    ("Money & Credit", "M2REAL", 5, "Real M2"),
    ("Money & Credit", "AMBSL", 6, "Adjusted Monetary Base"),
    ("Money & Credit", "TOTRESNS", 6, "Total Reserves"),
    ("Money & Credit", "NONBORRES", 6, "Nonborrowed Reserves"),
    ("Money & Credit", "BUSLOANS", 6, "Commercial & Industrial Loans"),
    ("Money & Credit", "REALLN", 6, "Real Estate Loans (All Commercial Banks)"),
    ("Money & Credit", "NONREVSL", 6, "Total Nonrevolving Credit"),
    ("Money & Credit", "MZMSL", 6, "MZM Money Stock"),
    ("Money & Credit", "DTCOLNVHFNM", 6, "Consumer Motor Vehicle Loans"),
    ("Money & Credit", "DTCTHFNM", 6, "Total Consumer Loans/Leases"),
    ("Money & Credit", "INVEST", 6, "Securities in Bank Credit"),

    # Group 6: Interest & exchange rates:contentReference[oaicite:16]{index=16}
    ("Rates/FX", "FEDFUNDS", 2, "Effective Federal Funds Rate"),
    ("Rates/FX", "CPF3M", 2, "3-Month AA Financial Commercial Paper (post-1997)"),
    ("Rates/FX", "M13002US35620M156NNBR", 2, "3-Month Commercial Paper (pre-1997, for splice)"),
    ("Rates/FX", "TB3MS", 2, "3-Month Treasury Bill"),
    ("Rates/FX", "TB6MS", 2, "6-Month Treasury Bill"),
    ("Rates/FX", "GS1", 2, "1-Year Treasury"),
    ("Rates/FX", "GS5", 2, "5-Year Treasury"),
    ("Rates/FX", "GS10", 2, "10-Year Treasury"),
    ("Rates/FX", "AAA", 2, "Moody's Aaa"),
    ("Rates/FX", "BAA", 2, "Moody's Baa"),
    ("Rates/FX", "TWEXMMTH", 5, "Trade-Weighted Dollar Index (Major Currencies)"),
    ("Rates/FX", "EXSZUS", 5, "Switzerland/US FX"),
    ("Rates/FX", "EXJPUS", 5, "Japan/US FX"),
    ("Rates/FX", "EXUSUK", 5, "US/UK FX"),
    ("Rates/FX", "EXCAUS", 5, "Canada/US FX"),

    # Group 7: Prices:contentReference[oaicite:17]{index=17}
    ("Prices", "PPIFGS", 6, "PPI: Finished Goods"),
    ("Prices", "PPIFCG", 6, "PPI: Finished Consumer Goods"),
    ("Prices", "PPIITM", 6, "PPI: Intermediate Materials"),
    ("Prices", "PPICRM", 6, "PPI: Crude Materials"),
    ("Prices", "MCOILWTICO", 6, "Crude Oil (WTI), monthly"),
    ("Prices", "PPICMM", 6, "PPI: Metals & Metal Products"),
    ("Prices", "NAPMPRI", 1, "ISM Manufacturing: Prices Index"),
    ("Prices", "CPIAUCSL", 6, "CPI: All Items"),
    ("Prices", "CPIAPPSL", 6, "CPI: Apparel"),
    ("Prices", "CPITRNSL", 6, "CPI: Transportation"),
    ("Prices", "CPIMEDSL", 6, "CPI: Medical Care"),
    ("Prices", "CUSR0000SAC", 6, "CPI: Commodities"),
    ("Prices", "CUUR0000SAD", 6, "CPI: Durables"),
    ("Prices", "CUSR0000SAS", 6, "CPI: Services"),
    ("Prices", "CPIULFSL", 6, "CPI: Less Food"),
    ("Prices", "CUUR0000SA0L2", 6, "CPI: Less Shelter"),
    ("Prices", "CUSR0000SA0L5", 6, "CPI: Less Medical Care"),
    ("Prices", "PCEPI", 6, "PCE Price Index"),
    ("Prices", "DDURRG3M086SBEA", 6, "PCE Deflator: Durables"),
    ("Prices", "DNDGRG3M086SBEA", 6, "PCE Deflator: Nondurables"),
    ("Prices", "DSERRG3M086SBEA", 6, "PCE Deflator: Services"),

    # Group 8: Stock market (use Δlog for price; levels for yields/ratios):contentReference[oaicite:18]{index=18}
    ("Stock Market", "SP500", 5, "S&P 500 (daily -> monthly avg)"),
    ("Stock Market", "SPDIVY", 1, "S&P 500 Dividend Yield"),
    ("Stock Market", "CAPE", 1, "Shiller CAPE P/E Ratio"),
]

# Derived series recipes (id -> function)
def build_cp3m_spliced(api_key: str, start: str, end: str) -> pd.Series:
    """CP3Mx: splice historical commercial paper (NBER code) with CPF3M."""
    try:
        new = fred_series_observations("CPF3M", api_key, start, end, frequency="m", aggregation_method="avg")
    except Exception:
        new = pd.Series(dtype=float)
    try:
        old = fred_series_observations("M13002US35620M156NNBR", api_key, start, end, frequency="m", aggregation_method="avg")
    except Exception:
        old = pd.Series(dtype=float)
    if new.empty and old.empty:
        return pd.Series(dtype=float)
    cp3m = new.copy()
    cp3m = cp3m.combine_first(old)
    cp3m.name = "CP3Mx"
    return cp3m

def build_claimsx(api_key: str, start: str, end: str) -> pd.Series:
    """
    CLAIMSx: splice monthly historical with weekly ICSA averaged monthly.
    Best-effort approximation; uses FRED aggregation.
    """
    try:
        weekly = fred_series_observations("ICSA", api_key, start, end, frequency="m", aggregation_method="avg")
    except Exception:
        weekly = pd.Series(dtype=float)
    try:
        old_m = fred_series_observations("M08297USM548NNBR", api_key, start, end, frequency="m", aggregation_method="avg")
    except Exception:
        old_m = pd.Series(dtype=float)
    s = weekly.combine_first(old_m)
    s.name = "CLAIMSx"
    return s

def build_spread(api_key: str, a: str, b: str, start: str, end: str, name: str) -> pd.Series:
    sa = fred_series_observations(a, api_key, start, end, frequency="m", aggregation_method="avg")
    sb = fred_series_observations(b, api_key, start, end, frequency="m", aggregation_method="avg")
    s = sa - sb
    s.name = name
    return s

def build_conspi(api_key: str, start: str, end: str) -> pd.Series:
    num = fred_series_observations("NONREVSL", api_key, start, end, frequency="m", aggregation_method="avg")
    den = fred_series_observations("PI", api_key, start, end, frequency="m", aggregation_method="avg")
    s = num / den
    s.name = "CONSPI"
    return s

def build_oil_spliced(api_key: str, start: str, end: str) -> pd.Series:
    """
    OILPRICEx (best-effort): monthly WTI (MCOILWTICO). If legacy OILPRICE exists, combine.
    """
    wti = fred_series_observations("MCOILWTICO", api_key, start, end, frequency="m", aggregation_method="avg")
    try:
        legacy = fred_series_observations("OILPRICE", api_key, start, end, frequency="m", aggregation_method="avg")
    except Exception:
        legacy = pd.Series(dtype=float)
    s = wti.combine_first(legacy)
    s.name = "OILPRICEx"
    return s

DERIVED_BUILDERS = {
    "CP3Mx": lambda k, st, en: build_cp3m_spliced(k, st, en),
    "COMPAPFFx": lambda k, st, en: build_spread(k, "CP3Mx", "FEDFUNDS", st, en, "COMPAPFFx"),  # computed after CP3Mx is present
    "CLAIMSx": lambda k, st, en: build_claimsx(k, st, en),
    "CONSPI": lambda k, st, en: build_conspi(k, st, en),
    "T1YFFM": lambda k, st, en: build_spread(k, "GS1", "FEDFUNDS", st, en, "T1YFFM"),
    "T5YFFM": lambda k, st, en: build_spread(k, "GS5", "FEDFUNDS", st, en, "T5YFFM"),
    "T10YFFM": lambda k, st, en: build_spread(k, "GS10", "FEDFUNDS", st, en, "T10YFFM"),
    "TB3SMFFM": lambda k, st, en: build_spread(k, "TB3MS", "FEDFUNDS", st, en, "TB3SMFFM"),
    "TB6SMFFM": lambda k, st, en: build_spread(k, "TB6MS", "FEDFUNDS", st, en, "TB6SMFFM"),
    "AAAFFM": lambda k, st, en: build_spread(k, "AAA", "FEDFUNDS", st, en, "AAAFFM"),
    "BAAFFM": lambda k, st, en: build_spread(k, "BAA", "FEDFUNDS", st, en, "BAAFFM"),
    "OILPRICEx": lambda k, st, en: build_oil_spliced(k, st, en),
}

DERIVED_TCODE = {
    # Spreads are in levels (no time differencing)
    "T1YFFM": 1, "T5YFFM": 1, "T10YFFM": 1, "TB3SMFFM": 1, "TB6SMFFM": 1,
    "AAAFFM": 1, "BAAFFM": 1, "COMPAPFFx": 1,
    # CP3Mx is a rate level (matches CP series transform)
    "CP3Mx": 2,
    # Ratios:
    "CONSPI": 2,
    # Claims (counts) — Δlog often used; here we keep Δlog for stationarity
    "CLAIMSx": 5,
    # Oil splice inherits price transform (Δ² log commonly used in FRED-MD)
    "OILPRICEx": 6,
}


# ----------------------------
# Factor extraction (PCA)
# ----------------------------

def pca_factors(df_trans: pd.DataFrame, n_factors: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute first n_factors principal components on standardized data.
    Simple, transparent PCA with NaN handling by interpolation/ffill/bfill per series.

    Returns:
      factors_df: T x n_factors
      var_explained: array of variance explained ratios
    """
    Z = df_trans.copy()

    # Fill remaining gaps: interpolate (time), then forward/back fill, then drop rows still empty
    Z = Z.sort_index()
    Z = Z.interpolate(method="time", limit_direction="both")
    Z = Z.fillna(method="ffill").fillna(method="bfill")
    Z = Z.dropna(axis=0, how="any")

    # Standardize each column
    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=1).replace(0, np.nan)
    Zs = (Z - means) / stds

    # Covariance PCA
    cov = np.cov(Zs.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Scores (factors): Z * eigenvectors
    F = Zs.values @ eigvecs[:, :n_factors]
    factors_df = pd.DataFrame(F, index=Zs.index, columns=[f"F{i+1}" for i in range(n_factors)])

    var_explained = eigvals[:n_factors] / np.sum(eigvals)
    return factors_df, var_explained


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch FRED series data and export the raw values.")
    ap.add_argument("--out", default="fred_series_data.csv", help="Output CSV for retrieved data")
    args = ap.parse_args()

    api_key = FRED_API_KEY.strip()
    start = FRED_START_DATE
    end = FRED_END_DATE
    frequency = (FRED_FREQUENCY or "m").strip().lower()

    if not api_key:
        print("FRED_API_KEY is not set in fred_config.py.")
        sys.exit(1)

    # 1) Fetch all base series
    print(f"Fetching base series from FRED (frequency='{frequency}')...")
    series_frames = {}

    for group, sid, tcode, desc in BASE_SERIES:
        try:
            s = fred_series_observations(
                sid,
                api_key,
                start,
                end,
                frequency=frequency,
                aggregation_method="avg",
            )
            if s.empty:
                print(f"  [warn] No data for {sid}")
                continue
            series_frames[sid] = s.rename(sid)
        except Exception as e:
            print(f"  [warn] Failed {sid}: {e}")

    if not series_frames:
        print("No series retrieved. Check your API key and network.")
        sys.exit(1)

    # 2) Assemble raw panel (align on index and trim to requested date range)
    print("Assembling dataset...")
    all_df = pd.concat(series_frames.values(), axis=1).sort_index()
    if start:
        all_df = all_df[all_df.index >= pd.to_datetime(start)]
    if end:
        all_df = all_df[all_df.index <= pd.to_datetime(end)]

    # 3) Save raw data to CSV
    output_path = args.out
    print(f"Writing dataset to {output_path} ...")
    all_df.to_csv(output_path, index_label="date")
    print("Done.")


if __name__ == "__main__":
    main()
