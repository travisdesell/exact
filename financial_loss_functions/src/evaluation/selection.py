import numpy as np
import pandas as pd

def filter_models(
        avg_perf: pd.DataFrame, bench_name: str, bench_met: str, keep: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter out models that do not beat the benchmark (eg. Equal_Weight) and keep ones that do.

    Args:
        avg_perf (pd.DataFrame): Average Performance of all models across all metrics.
        bench_name (str): String name of the benchmark that exists in the avg_per dataframe.
        bench_met (str): String name of the metric that should be used to compare the models.
        keep (list[str]): List of all benchmarks or indexes to keep.
    
    Returns:
        tuple: A tuple containing,

            - filtered_avg_perf (pd.DataFrame): Dataframe containing only models that 
            outperformed the benchmark on the specified metric.
            - filtered_models (list[str]): List of names of the models that beat the benchmark.
    """

    # Get the equal‑weight Metric (Sharpe) value
    ew_sharpe = avg_perf.loc[bench_name, bench_met]

    # Create mask: keep if (1) it's a benchmark OR (2) its Sharpe > ew_sharpe
    mask = avg_perf.index.isin(keep) | (avg_perf[bench_met] > ew_sharpe)

    filtered_df = avg_perf[mask]

    filtered_models = filtered_df.index[
        ~filtered_df.index.isin(keep)
    ].to_list()

    return filtered_df, filtered_models

def high_corr_with_each_metric(corr: pd.DataFrame, threshold: float=0.8):
    """
    Print the correlation of each portfolio performance metric with all others,
    with a threshold for correlation,

    Args:
        corr (pd.DataFrame): Correlation matrix of the performance metrics.
        threshold (float): Threshold to print the correlations. Default = 0.8.
    """
    for metric in corr.columns:
        # Exclude self (corr=1.0)
        others = corr[metric][corr[metric].index != metric]
        high = others[abs(others) > threshold]
        if not high.empty:
            print(f'\n{metric} is highly correlated with:')
            for name, val in high.items():
                print(f'  {name}: {val:.3f}')

def pareto_dominance(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Returns a boolean Series indicating whether each row is dominated.
    Assumes all columns are "higher is better".

    Args:
        df (pd.DataFrame): Dataframe from which we select dominating models/methods
            in the index based on the multi-metric columns. Models/methods must be in 
            the index and metrics must be in the columns
        columns (lis[str]): List of column names from the dataframe, that must be used 
            for the pareto dominance.
    
    Returns:
        pd.Series: Series containing the dominated status and the names of the model/methods.
    """
    n_rows = len(df)
    dominated = np.zeros(n_rows, dtype=bool)
    # Convert to numpy array for faster comparisons
    values = df[columns].values
    for i in range(n_rows):
        if dominated[i]:
            continue
        for j in range(n_rows):
            if i == j:
                continue
            # Check if row j dominates row i
            if (values[j] >= values[i]).all() and (values[j] > values[i]).any():
                dominated[i] = True
                break
    return pd.Series(dominated, index=df.index)