import sys
import numpy as np
import pandas as pd

sys.path.append("..")
import cov_models

def GMVP(cov):
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


if __name__ == '__main__':
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(500, 4) * 0.01, columns=list("ABCD"))
    cov = returns.cov()
    
    GMVP(cov)

    naive_MVP(cov)