import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from src.models import cov_models

def gvmp(cov):
    # Global Minimum Variance - long-only
    gm = cov_models.GlobalMinimumVariance(allow_short=False)
    gm_weights = gm.calculate_weights(cov)
    print("GMVP (long-only):", gm_weights)

    # Global Minimum Variance - unconstrained analytic
    gm2 = cov_models.GlobalMinimumVariance(allow_short=True)
    gm2_weights = gm2.calculate_weights(cov)
    print("GMVP (unconstrained):", gm2_weights)

def naive_MVP(cov):
    mvp_weights = cov_models.naive_mvp(cov)
    print('Naive MVP:', mvp_weights)

def hrp(cov, corr):
    hrp = cov_models.HierarchialRiskParity()
    hrp_weights = hrp.calculate_weights(cov, corr)
    print('HRP (single):\n', hrp_weights)

def mean_var(cov, returns):
    mean_var = cov_models.MeanVariancePortfolio(
        expected_returns_method='arithmetic'
    )

    arth_weights = mean_var.calculate_weights(cov, returns)
    print('Mean-Variance Using Arithmetic Mean Expected Eeturns:', arth_weights)

    mean_var = cov_models.MeanVariancePortfolio(
        expected_returns_method='geometric'
    )

    arth_weights = mean_var.calculate_weights(cov, returns)
    print('Mean-Variance Using Geometric Mean Expected Eeturns:', arth_weights)


if __name__ == '__main__':
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(500, 4) * 0.01, columns=['A', 'B', 'C', 'D'])
    cov = returns.cov()
    corr = returns.corr()
    
    gvmp(cov)

    naive_MVP(cov)

    hrp(cov, corr)

    mean_var(cov, returns)