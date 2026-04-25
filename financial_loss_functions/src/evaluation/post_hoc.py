import pandas as pd
import numpy as np
from scipy.stats import shapiro

def test_normality_by_model(
        all_seed_perf: pd.DataFrame,
        nn_model_losses: list[str],
        metrics:list[str]|None=None,
        alpha=0.05
    ):
    """
    Perform Shapiro-Wilk normality test for each metric and each model-loss combination,
    with Bonferroni correction for multiple comparisons.
    
    Args:
        all_seed_perf : pd.DataFrame
            Must contain a column 'model_loss' and metric columns (numerical).
            Each model-loss should have exactly 30 rows (one per seed).
        nn_model_losses : list
            List of model-loss strings to include in the test.
        metrics : list, optional
            List of metric column names to test. If None, all numeric columns except 'model_loss' are used.
        alpha : float, default 0.05
            Significance level before correction.
    
    Returns:
        pd.DataFrame: Columns = 'model_loss', 'metric', 'statistic', 'p_value_raw', 'p_value_corrected',
            'normality_assumed' (True if p_corrected > alpha).
    """
    if metrics is None:
        metrics = all_seed_perf.select_dtypes(include=[np.number]).columns.tolist()
        # remove any non‑metric identifier column named 'seed' if present
        if 'seed' in metrics:
            metrics.remove('seed')
    
    results = []
    total_tests = len(nn_model_losses) * len(metrics)
    
    for model in nn_model_losses:
        sub = all_seed_perf[all_seed_perf.index == model]
        for metric in metrics:
            values = sub[metric].dropna().values
            if len(values) < 3:
                # Not enough samples for Shapiro‑Wilk
                stat, p_raw = np.nan, np.nan
                normal = False
            else:
                stat, p_raw = shapiro(values)
            results.append({
                'model_loss': model,
                'metric': metric,
                'statistic': stat,
                'p_value_raw': p_raw,
                'normality_assumed': False  # placeholder
            })
    
    results_df = pd.DataFrame(results)
    # Apply Bonferroni correction
    results_df['p_value_corrected'] = results_df['p_value_raw'] * total_tests
    results_df['normality_assumed'] = results_df['p_value_corrected'] > alpha
    # Clip corrected p‑values to 1.0
    results_df['p_value_corrected'] = results_df['p_value_corrected'].clip(upper=1.0)
    
    return results_df