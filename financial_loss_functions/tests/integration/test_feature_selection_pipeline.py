import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.io import load_path_config
from src.feature_selection.analysis import run_feature_selection_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_feature_selection_cli_creates_outputs(tmp_path):
    output_dir = tmp_path / 'feature_selection_cli'
    cmd = [
        sys.executable,
        '-m',
        'scripts.run_feature_selection',
        '--crsp-dir',
        '2023_sp_500_select_50',
        '--lags',
        '10,30',
        '--top-k',
        '10',
        '--output-dir',
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    assert (output_dir / 'ticker_macro_rankings.csv').exists()
    assert (output_dir / 'ticker_selected_features.csv').exists()
    assert (output_dir / 'sector_assignment_50.csv').exists()


@pytest.mark.integration
def test_feature_selection_sector_priors_reflected_in_output(tmp_path):
    paths_config = load_path_config(
        str(PROJECT_ROOT / 'config' / 'paths.json'),
        '2023_sp_500_select_50',
    )
    output_dir = tmp_path / 'feature_selection_api'

    artifacts = run_feature_selection_pipeline(
        paths_config=paths_config,
        output_dir=output_dir,
        lags=[10, 30, 50, 60],
        low_corr_threshold=0.1,
        top_k=10,
    )

    selected = artifacts.ticker_selected
    assert selected['ticker'].nunique() == 50
    assert (selected.groupby('ticker').size() == 10).all()

    finance_groups = {'Rates_FX', 'Money_Credit'}
    tech_groups = {'Labor_Market', 'Prices', 'Stock_Market'}

    for ticker in ('HBAN', 'NTRS'):
        ticker_groups = set(selected[selected['ticker'] == ticker]['macro_group'])
        assert ticker_groups & finance_groups

    for ticker in ('AKAM', 'FFIV'):
        ticker_groups = set(selected[selected['ticker'] == ticker]['macro_group'])
        assert ticker_groups & tech_groups
