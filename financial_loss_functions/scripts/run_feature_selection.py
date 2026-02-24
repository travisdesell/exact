import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.utils import load_path_config
from src.feature_selection.analysis import run_feature_selection_pipeline


def _parse_lags(lags_arg: str) -> list[int]:
    """Parse comma-separated lag list from CLI into integer business-day lags."""

    return [int(item.strip()) for item in lags_arg.split(',') if item.strip()]


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Run CRSP + macro feature engineering and feature selection analyses.'
    )
    parser.add_argument(
        '--crsp-dir',
        default=os.getenv('CRSP_DIR'),
        help='CRSP directory name under data/raw or absolute path. Defaults to CRSP_DIR env var.',
    )
    parser.add_argument(
        '--lags',
        default='10,30,50,60',
        help='Comma-separated lags in business days. Default: 10,30,50,60',
    )
    parser.add_argument(
        '--low-corr-threshold',
        type=float,
        default=0.1,
        help='Absolute Spearman threshold used to classify low correlations.',
    )
    parser.add_argument(
        '--output-dir',
        default='data/feature_selection',
        help='Output directory path (relative to project root or absolute).',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of selected macro features per ticker. Default: 10',
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths_config = load_path_config(
        os.path.join('config', 'paths.json'),
        args.crsp_dir,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    artifacts = run_feature_selection_pipeline(
        paths_config=paths_config,
        output_dir=output_dir,
        lags=_parse_lags(args.lags),
        low_corr_threshold=args.low_corr_threshold,
        top_k=args.top_k,
    )

    print('\n', '=' * 20, ' Feature Selection Pipeline ', '=' * 20)
    print(f'Output directory: {output_dir}')
    print(f'Sector assignments: {artifacts.sector_map.shape}')
    print(f'Macro aligned features: {artifacts.macro_aligned.shape}')
    print(f'Ticker-macro rankings: {artifacts.ticker_rankings.shape}')
    print(f'Ticker selected features: {artifacts.ticker_selected.shape}')
    print(f'Rankings file: {output_dir / "ticker_macro_rankings.csv"}')
    print(f'Selected file: {output_dir / "ticker_selected_features.csv"}')
    print(f'Sector map file: {output_dir / "sector_assignment_50.csv"}')
