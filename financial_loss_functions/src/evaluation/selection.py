import pandas as pd
import numpy as np

def high_corr_with_each_metric(corr, threshold=0.8):
    for metric in corr.columns:
        # Exclude self (corr=1.0)
        others = corr[metric][corr[metric].index != metric]
        high = others[abs(others) > threshold]
        if not high.empty:
            print(f"\n{metric} is highly correlated with:")
            for name, val in high.items():
                print(f"  {name}: {val:.3f}")

def pareto_dominance(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Returns a boolean Series indicating whether each row is dominated.
    Assumes all columns are "higher is better".
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