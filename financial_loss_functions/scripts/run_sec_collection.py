"""
CLI entry point for SEC filing data collection.

Usage:
    python -m scripts.run_sec_collection [--identity "Name email@domain.com"]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from scripts.utils import load_config
from src.data_collection.sec_filings import run_sec_filing_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch SEC filings and compute fundamental features")
    parser.add_argument(
        "--identity",
        default="FinLossFunctions research@example.com",
        help="SEC EDGAR identity string (Name + email)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for cached Parquet files",
    )
    args = parser.parse_args()

    paths_config = load_config("paths")

    sector_path = Path("config/sector_classification.json")
    with open(sector_path, "r") as f:
        sector_map = json.load(f)
    tickers = sorted(sector_map.keys())
    logger.info("Found %d tickers from sector classification", len(tickers))

    crsp_dir = Path(paths_config["data"]["crsp_dir"])
    if not crsp_dir.exists():
        crsp_dir = Path(paths_config["data"]["raw_dir"]) / "sample"

    train_file = crsp_dir / paths_config["raw_files"]["train"]
    if not train_file.exists():
        logger.error("Training data not found at %s", train_file)
        return

    train_df = pd.read_csv(train_file)
    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.set_index("date")

    cache_dir = Path(args.output_dir) if args.output_dir else Path(
        paths_config.get("data", {}).get("sec_filings_dir", "data/raw/sec_filings")
    )

    scores = run_sec_filing_pipeline(
        tickers=tickers,
        target_index=train_df.index,
        cache_dir=cache_dir,
        identity=args.identity,
    )

    output_path = cache_dir / "composite_fundamental_scores.csv"
    scores.to_csv(output_path)
    logger.info("Composite scores saved to %s", output_path)


if __name__ == "__main__":
    main()
