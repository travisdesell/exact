import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd

MACRO_GROUPS = (
    'Consumption_Orders_Inventories',
    'Housing',
    'Labor_Market',
    'Money_Credit',
    'Output_Income',
    'Prices',
    'Rates_FX',
    'Stock_Market',
)

# Small baseline keeps all groups eligible before sector priorities are applied.
BASE_GROUP_WEIGHT = 0.05
PRIMARY_GROUP_WEIGHT = 1.0
# Secondary-sector overlap nudges, but does not dominate, the primary-sector signal.
SECONDARY_SECTOR_BONUS = 0.2

# Mapping from sector -> macro groups expected to have higher causal relevance.
SECTOR_PRIOR_GROUPS = {
    'Financials': ('Rates_FX', 'Money_Credit'),
    'Information Technology': ('Labor_Market', 'Prices', 'Stock_Market'),
    'Consumer Discretionary': (
        'Consumption_Orders_Inventories',
        'Labor_Market',
        'Prices',
    ),
    'Consumer Staples': (
        'Consumption_Orders_Inventories',
        'Labor_Market',
        'Prices',
    ),
    'Utilities': ('Rates_FX', 'Housing', 'Money_Credit'),
    'Real Estate': ('Rates_FX', 'Housing', 'Money_Credit'),
    'Industrials': ('Output_Income', 'Prices', 'Labor_Market'),
    'Materials': ('Output_Income', 'Prices', 'Labor_Market'),
    'Health Care': ('Output_Income', 'Prices', 'Labor_Market'),
}


def _default_sector_mapping_path() -> Path:
    """Default location of ticker-sector classification file."""

    return Path(__file__).resolve().parents[2] / 'config' / 'sector_classification.json'


@lru_cache(maxsize=1)
def _load_default_sector_mapping() -> Dict[str, Dict[str, str | None]]:
    """Cached mapping load to avoid repeated JSON reads in ticker loops."""

    return load_sector_mapping()


def load_sector_mapping(
    mapping_path: Path | None = None,
) -> Dict[str, Dict[str, str | None]]:
    """Load ticker->sector metadata from JSON config."""

    path = mapping_path or _default_sector_mapping_path()
    with path.open('r', encoding='utf-8') as f:
        mapping_json = json.load(f)

    return mapping_json


def build_sector_mapping_df(mapping_json: Dict[str, Dict[str, str | None]]) -> pd.DataFrame:
    """Convert mapping JSON into a flat dataframe for reporting/export."""

    rows = []
    for ticker, info in mapping_json.items():
        rows.append(
            {
                'ticker': ticker,
                'company': info.get('company'),
                'sector': info.get('sector'),
                'secondary_sector': info.get('secondary_sector'),
                'key_business': info.get('key_business'),
            }
        )

    sector_df = pd.DataFrame(rows)
    sector_df = sector_df.sort_values('ticker').reset_index(drop=True)
    return sector_df


def get_sector_prior_weights(ticker: str) -> Dict[str, float]:
    """
    Build macro-group prior weights for a ticker.

    Strategy:
    - initialize all groups with a low baseline,
    - assign full weight to primary-sector groups,
    - apply a capped bonus for secondary-sector groups when present.
    """

    mapping_json = _load_default_sector_mapping()
    if ticker not in mapping_json:
        raise KeyError(f'Sector mapping not found for ticker: {ticker}')

    ticker_info = mapping_json[ticker]
    primary_sector = ticker_info.get('sector')
    secondary_sector = ticker_info.get('secondary_sector')

    if primary_sector not in SECTOR_PRIOR_GROUPS:
        raise KeyError(f'Unsupported sector in mapping: {primary_sector}')

    group_weights = {group: BASE_GROUP_WEIGHT for group in MACRO_GROUPS}

    for macro_group in SECTOR_PRIOR_GROUPS[primary_sector]:
        group_weights[macro_group] = PRIMARY_GROUP_WEIGHT

    if secondary_sector:
        for macro_group in SECTOR_PRIOR_GROUPS.get(secondary_sector, ()):
            boosted = group_weights[macro_group] + SECONDARY_SECTOR_BONUS
            group_weights[macro_group] = min(PRIMARY_GROUP_WEIGHT, boosted)

    return group_weights
