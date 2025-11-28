# scripts/generate_synthetic_crsp_csv_splits_with_sprtrn.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

def generate_synthetic_crsp(
    tickers: Optional[List[str]] = None,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    freq: str = "B",          # business days
    seed: int = 42,
    market_rets: Optional[np.ndarray] = None,
    market_noise_scale: float = 0.0005,
    market_weighted: bool = False,
) -> pd.DataFrame:
    """Return a wide dataframe with date column, per-ticker features, and one column 'sprtrn'.

    Parameters
    ----------
    tickers: list[str] | None
        List of tickers. If None, default 50 synthetic tickers are created.
    market_rets: np.ndarray | None
        Optional externally supplied market return series (length must match dates).
    market_noise_scale: float
        Additional noise to add to the generated market return so it's not a perfect average.
    market_weighted: bool
        If True, create a random-weighted market (cap-like). Else equal-weighted.

    Returns
    -------
    pd.DataFrame
        Columns: date, <TICKER>_RET, <TICKER>_ILLIQUIDITY, <TICKER>_VOL_CHANGE,
                 <TICKER>_TURNOVER, <TICKER>_BA_SPREAD, sprtrn
    """
    rng = np.random.default_rng(seed)

    if tickers is None:
        tickers = [f"TICK{str(i).zfill(3)}" for i in range(1, 51)]

    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)

    data = {"date": dates}
    all_rets = []

    for tk in tickers:
        # simulate returns
        mu = rng.normal(0.0002, 0.0003)
        sigma = rng.uniform(0.01, 0.04)
        rets = rng.normal(loc=mu, scale=sigma, size=n)

        # simulate price series (not used directly but could be helpful)
        price0 = rng.uniform(10, 500)
        log_prices = np.log(price0) + np.cumsum(rets)
        # prices = np.exp(log_prices)

        # volume-ish process and derived features
        base_vol = rng.lognormal(mean=12.5, sigma=0.7)
        vols = (base_vol * np.exp(rng.normal(scale=0.1, size=n))).astype(float)
        shares_outstanding = rng.uniform(1e6, 5e9)
        turnover = vols / shares_outstanding

        # illiquidity proxy
        illiq = np.abs(rets) / (vols + 1e-9) * 1e6

        # rolling volatility change
        vol_roll = pd.Series(rets).rolling(window=21, min_periods=1).std().values
        vol_change = np.zeros_like(vol_roll)
        prev = vol_roll[:-1]
        curr = vol_roll[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(prev != 0, (curr - prev) / (prev + 1e-9), 0.0)
        vol_change[1:] = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)

        # bid-ask spread proxy (positive)
        ba_base = rng.normal(loc=0.0015, scale=0.0007)
        ba_spread = np.abs(ba_base + 0.5 * vol_roll * rng.normal(scale=1, size=n))
        ba_spread = np.clip(ba_spread, 1e-6, None)

        # assign per-ticker columns
        data[f"{tk}_RET"] = rets
        data[f"{tk}_ILLIQUIDITY"] = illiq
        data[f"{tk}_VOL_CHANGE"] = vol_change
        data[f"{tk}_TURNOVER"] = turnover
        data[f"{tk}_BA_SPREAD"] = ba_spread

        all_rets.append(rets)

    all_rets = np.vstack(all_rets)  # shape (n_tickers, n)

    # create sprtrn
    if market_rets is not None:
        market_rets = np.asarray(market_rets)
        if market_rets.shape[0] != n:
            raise ValueError(f"market_rets length {market_rets.shape[0]} != n ({n})")
        sprtrn = market_rets
    else:
        if market_weighted:
            # random positive weights that sum to 1
            w = rng.random(all_rets.shape[0])
            w = w / w.sum()
            market_base = (w[:, None] * all_rets).sum(axis=0)
        else:
            market_base = all_rets.mean(axis=0)

        sprtrn = market_base + rng.normal(loc=0.0, scale=market_noise_scale, size=n)

    data["sprtrn"] = sprtrn

    df = pd.DataFrame(data)
    df = df[["date"] + [c for c in df.columns if c != "date"]]
    return df


def time_split_df(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split of dataframe rows (preserves ordering).
    Returns (train_df, val_df, test_df)
    """
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1
    assert train_frac + val_frac < 1.0

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def save_csv_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str = "data/processed/synthetic_crsp_v1",
    base_name: str = "synthetic_crsp",
) -> Dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = out_path / f"{base_name}_train.csv"
    val_csv = out_path / f"{base_name}_validation.csv"
    test_csv = out_path / f"{base_name}_test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
    }


if __name__ == "__main__":
    # Parameters you can change
    tickers = None  # default 50 tickers; or supply: ["AAPL","MSFT"]
    start = "2020-01-01"
    end = "2023-12-31"
    freq = "B"
    seed = 42
    train_frac = 0.7
    val_frac = 0.15
    out_dir = "data/raw/sample"
    base_name = "combined_predictors"

    print("Generating synthetic CRSP-like dataset with single 'sprtrn' column...")
    df = generate_synthetic_crsp(tickers=tickers, start=start, end=end, freq=freq, seed=seed)
    print("Dataset shape:", df.shape)

    print("Splitting into train/val/test...")
    train_df, val_df, test_df = time_split_df(df, train_frac=train_frac, val_frac=val_frac)
    print("Sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    print("Saving CSV splits to:", out_dir)
    saved = save_csv_splits(train_df, val_df, test_df, out_dir=out_dir, base_name=base_name)
    for k, v in saved.items():
        print(f"  {k}: {v}")
    print("Done.")