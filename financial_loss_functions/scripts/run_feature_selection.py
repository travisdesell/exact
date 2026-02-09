import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.utils import load_path_config
from src.feature_selection.analysis import run_feature_selection_pipeline


def _parse_lags(lags_arg: str) -> list[int]:
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
    )

    print('\n', '=' * 20, ' Feature Selection Pipeline ', '=' * 20)
    print(f'Output directory: {output_dir}')
    print(f'Non-lagged CRSP features: {artifacts.crsp_non_lagged.shape}')
    print(f'Lagged CRSP features: {artifacts.crsp_lagged.shape}')
    print(f'Macro aligned features: {artifacts.macro_aligned.shape}')
    print(f'Model dataset shape: {artifacts.model_data.shape}')
    print(f'Top ranked features file: {output_dir / "top_50_features_comparison.csv"}')
    print(f'Summary report: {output_dir / "feature_selection_summary.md"}')
