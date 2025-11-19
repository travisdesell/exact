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
# Helper: API
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


# ----------------------------
# Series configuration
# ----------------------------
# Each entry: ("GroupName", "SeriesID", "Description")
# The list below covers a broad, representative subset across all eight sectors.

BASE_SERIES: List[Tuple[str, str, str]] = [
    # Group 1: Output & income
    ("Output & Income", "RPI", "Real Personal Income"),
    ("Output & Income", "W875RX1", "Real Personal Income Excluding Transfers"),
    ("Output & Income", "INDPRO", "Industrial Production Index"),
    ("Output & Income", "IPFPNSS", "IP: Final Products & Nonindustrial Supplies"),
    ("Output & Income", "IPFINAL", "IP: Final Products (Market Group)"),
    ("Output & Income", "IPCONGD", "IP: Consumer Goods"),
    ("Output & Income", "IPDCONGD", "IP: Durable Consumer Goods"),
    ("Output & Income", "IPNCONGD", "IP: Nondurable Consumer Goods"),
    ("Output & Income", "IPBUSEQ", "IP: Business Equipment"),
    ("Output & Income", "IPMAT", "IP: Materials"),
    ("Output & Income", "IPDMAT", "IP: Durable Materials"),
    ("Output & Income", "IPNMAT", "IP: Nondurable Materials"),
    ("Output & Income", "IPMANSICS", "IP: Manufacturing (SIC)"),
    ("Output & Income", "IPB51222S", "IP: Residential Utilities"),
    ("Output & Income", "IPFUELS", "IP: Fuels"),
    ("Output & Income", "CUMFNS", "Capacity Utilization: Manufacturing"),

    # Group 2: Labor market
    ("Labor Market", "CLF16OV", "Civilian Labor Force"),
    ("Labor Market", "CE16OV", "Civilian Employment"),
    ("Labor Market", "UNRATE", "Unemployment Rate"),
    ("Labor Market", "UEMPMEAN", "Average Duration of Unemployment"),
    ("Labor Market", "PAYEMS", "All Employees: Total Nonfarm"),
    ("Labor Market", "USGOOD", "All Employees: Goods-Producing"),
    ("Labor Market", "USCONS", "All Employees: Construction"),
    ("Labor Market", "MANEMP", "All Employees: Manufacturing"),
    ("Labor Market", "DMANEMP", "All Employees: Durable Goods"),
    ("Labor Market", "NDMANEMP", "All Employees: Nondurable Goods"),
    ("Labor Market", "SRVPRD", "All Employees: Service-Providing"),
    ("Labor Market", "USTPU", "All Employees: Trade, Transportation & Utilities"),
    ("Labor Market", "USTRADE", "All Employees: Retail Trade"),
    ("Labor Market", "USFIRE", "All Employees: Financial Activities"),
    ("Labor Market", "USGOVT", "All Employees: Government"),
    ("Labor Market", "CES0600000007", "Avg Weekly Hours: Goods-Producing"),
    ("Labor Market", "AWOTMAN", "Avg Weekly Overtime Hours: Manufacturing"),
    ("Labor Market", "AWHMAN", "Avg Weekly Hours: Manufacturing"),
    ("Labor Market", "CES0600000008", "Avg Hourly Earnings: Goods-Producing"),
    ("Labor Market", "CES2000000008", "Avg Hourly Earnings: Construction"),
    ("Labor Market", "CES3000000008", "Avg Hourly Earnings: Manufacturing"),

    # Group 3: Housing
    ("Housing", "HOUST", "Housing Starts: Total"),
    ("Housing", "HOUSTNE", "Housing Starts: Northeast"),
    ("Housing", "HOUSTMW", "Housing Starts: Midwest"),
    ("Housing", "HOUSTS", "Housing Starts: South"),
    ("Housing", "HOUSTW", "Housing Starts: West"),
    ("Housing", "PERMIT", "New Private Housing Permits (SAAR)"),
    ("Housing", "PERMITNE", "Permits: Northeast"),
    ("Housing", "PERMITMW", "Permits: Midwest"),
    ("Housing", "PERMITS", "Permits: South"),
    ("Housing", "PERMITW", "Permits: West"),

    # Group 4: Consumption, orders, inventories
    ("Consumption/Orders/Inventories", "DPCERA3M086SBEA", "Real PCE"),
    ("Consumption/Orders/Inventories", "CMRMTSPL", "Real Manufacturing & Trade Sales"),
    ("Consumption/Orders/Inventories", "RSAFS", "Retail and Food Services Sales"),
    ("Consumption/Orders/Inventories", "ACOGNO", "New Orders: Consumer Goods"),
    ("Consumption/Orders/Inventories", "ANDENO", "New Orders: Nondefense Capital Goods"),
    ("Consumption/Orders/Inventories", "AMDMUO", "Unfilled Orders: Durable Goods"),
    ("Consumption/Orders/Inventories", "BUSINV", "Total Business Inventories"),
    ("Consumption/Orders/Inventories", "ISRATIO", "Inventories-to-Sales Ratio"),
    ("Consumption/Orders/Inventories", "UMCSENT", "Consumer Sentiment Index"),

    # Group 5: Money & credit
    ("Money & Credit", "M1SL", "M1 Money Stock"),
    ("Money & Credit", "M2SL", "M2 Money Stock"),
    ("Money & Credit", "M2REAL", "Real M2"),
    ("Money & Credit", "AMBSL", "Adjusted Monetary Base"),
    ("Money & Credit", "TOTRESNS", "Total Reserves"),
    ("Money & Credit", "NONBORRES", "Nonborrowed Reserves"),
    ("Money & Credit", "BUSLOANS", "Commercial & Industrial Loans"),
    ("Money & Credit", "REALLN", "Real Estate Loans (All Commercial Banks)"),
    ("Money & Credit", "NONREVSL", "Total Nonrevolving Credit"),
    ("Money & Credit", "MZMSL", "MZM Money Stock"),
    ("Money & Credit", "DTCOLNVHFNM", "Consumer Motor Vehicle Loans"),
    ("Money & Credit", "DTCTHFNM", "Total Consumer Loans/Leases"),
    ("Money & Credit", "INVEST", "Securities in Bank Credit"),

    # Group 6: Interest & exchange rates
    ("Rates/FX", "FEDFUNDS", "Effective Federal Funds Rate"),
    ("Rates/FX", "CPF3M", "3-Month AA Financial Commercial Paper (post-1997)"),
    ("Rates/FX", "M13002US35620M156NNBR", "3-Month Commercial Paper (pre-1997, for splice)"),
    ("Rates/FX", "TB3MS", "3-Month Treasury Bill"),
    ("Rates/FX", "TB6MS", "6-Month Treasury Bill"),
    ("Rates/FX", "GS1", "1-Year Treasury"),
    ("Rates/FX", "GS5", "5-Year Treasury"),
    ("Rates/FX", "GS10", "10-Year Treasury"),
    ("Rates/FX", "AAA", "Moody's Aaa"),
    ("Rates/FX", "BAA", "Moody's Baa"),
    ("Rates/FX", "EXSZUS", "Switzerland/US FX"),
    ("Rates/FX", "EXJPUS", "Japan/US FX"),
    ("Rates/FX", "EXUSUK", "US/UK FX"),
    ("Rates/FX", "EXCAUS", "Canada/US FX"),

    # Group 7: Prices
    ("Prices", "PPIFGS", "PPI: Finished Goods"),
    ("Prices", "PPIFCG", "PPI: Finished Consumer Goods"),
    ("Prices", "PPIITM", "PPI: Intermediate Materials"),
    ("Prices", "PPICRM", "PPI: Crude Materials"),
    ("Prices", "MCOILWTICO", "Crude Oil (WTI), monthly"),
    ("Prices", "PPICMM", "PPI: Metals & Metal Products"),
    ("Prices", "CPIAUCSL", "CPI: All Items"),
    ("Prices", "CPIAPPSL", "CPI: Apparel"),
    ("Prices", "CPITRNSL", "CPI: Transportation"),
    ("Prices", "CPIMEDSL", "CPI: Medical Care"),
    ("Prices", "CUSR0000SAC", "CPI: Commodities"),
    ("Prices", "CUUR0000SAD", "CPI: Durables"),
    ("Prices", "CUSR0000SAS", "CPI: Services"),
    ("Prices", "CPIULFSL", "CPI: Less Food"),
    ("Prices", "CUUR0000SA0L2", "CPI: Less Shelter"),
    ("Prices", "CUSR0000SA0L5", "CPI: Less Medical Care"),
    ("Prices", "PCEPI", "PCE Price Index"),
    ("Prices", "DDURRG3M086SBEA", "PCE Deflator: Durables"),
    ("Prices", "DNDGRG3M086SBEA", "PCE Deflator: Nondurables"),
    ("Prices", "DSERRG3M086SBEA", "PCE Deflator: Services"),

    # Group 8: Stock market
    ("Stock Market", "SP500", "S&P 500 (daily -> monthly avg)"),
]

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

    for group, sid, desc in BASE_SERIES:
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
