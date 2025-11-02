import pytest
import numpy as np
import pandas as pd
from cov_models import (
    HierarchialRiskParity,
    naive_mvp,
    BaseQuadraticOptimizer
)

# -------------------- Fixtures -------------------- #
@pytest.fixture
def hrp():
    """Create HRP instance"""
    return HierarchialRiskParity()

@pytest.fixture
def sample_data():
    # Deterministic synthetic returns so covariance ordering is predictable:
    # A = low variance, C = medium variance, B = high variance
    returns = pd.DataFrame({
        'A': [0.001, -0.001, 0.002, -0.0015, 0.001],   # tiny moves -> low var
        'B': [0.05, -0.04, 0.06, -0.05, 0.045],       # large moves -> high var
        'C': [0.02, -0.015, 0.025, -0.02, 0.03],      # medium moves -> middle var
    })

    cov = returns.cov()
    corr = returns.corr()

    # Sanity assert inside fixture to ensure ordering we expect
    vars_ = cov.values.diagonal()
    # should be A < C < B
    assert vars_[0] < vars_[2] < vars_[1], f'Unexpected variances: {vars_}'

    return returns, cov, corr

@pytest.fixture
def sample_linkage():
    link = np.array([
            [0, 1, 0.1, 2],  # cluster (0,1)
            [2, 3, 0.2, 2],  # cluster (2,3)
            [4, 5, 0.3, 4]   # final cluster ((0,1),(2,3))
        ], dtype=float)
    return link

# -------------------- HRP Tests -------------------- #
def test_correlDist(hrp, sample_data):
    _, _, corr = sample_data

    dist = hrp._correlDist(corr)

    assert dist.shape == corr.shape
    assert np.allclose(dist.values, dist.values.T), 'Matrix should symmetric'
    assert np.allclose(np.diag(dist), 0), 'Diagonal should be 0'
    assert (dist >= 0).all().all()
    assert (dist <= 1).all().all()

def test_getQuasiDiag_vaild_output(hrp, sample_linkage):    
    order = hrp._getQuasiDiag(sample_linkage)

    # Should produce ordering of all 4 original indices
    assert isinstance(order, list)
    assert len(order) == 4
    assert set(order) == {0, 1, 2, 3}, 'Should contain all asset indices'
    assert all(isinstance(x, int) for x in order), 'All indices should be int'

def test_getQuasiDiag_deterministic(hrp, sample_linkage):
    result1 = hrp._getQuasiDiag(sample_linkage)
    result2 = hrp._getQuasiDiag(sample_linkage)

    assert result1 == result2, 'Output should be deterministic for same linkage'

def test_getQuasiDiag_single_item(hrp):
    link = np.array([[0, 1, 0.1, 2]], dtype=float)
    order = hrp._getQuasiDiag(link)

    assert isinstance(order, list)
    assert set(order) == {0, 1}
    assert len(order) == 2

def test_getIVP(hrp, sample_data):
    _, cov, _ = sample_data
    ivp = hrp._getIVP(cov.values)

    assert ivp.shape[0] == cov.shape[0]
    assert np.isclose(ivp.sum(), 1.0)  # weights sum to 1
    assert np.all(ivp >= 0)  # no negative weights

def test_getClusterVar(hrp, sample_data):
    _, cov, _ = sample_data
    assets = list(cov.columns)
    var = hrp._getClusterVar(cov, assets)

    assert var > 0  # variance must be positive
    assert isinstance(var, float)

def test_getRecBipart_unit(hrp, sample_data):
    _, cov, _ = sample_data

    # Provide a fixed, deterministic ordering of labels (do not call clustering)
    sortIx = ['A', 'B', 'C']

    weights = hrp._getRecBipart(cov, sortIx)

    # Basic structural checks
    assert isinstance(weights, pd.Series)
    assert list(weights.index) == sortIx
    assert len(weights) == 3

    # Numeric checks: positive and normalized
    assert (weights >= 0).all(), 'All weights must be non-negative'
    assert np.isclose(weights.sum(), 1.0), f'Weights do not sum to 1: sum={weights.sum()}'

    # asset with lowest variance should get the largest weight
    variances = cov.values.diagonal()
    idx_low_var = cov.index[np.argmin(variances)]   # expected 'A'
    idx_high_var = cov.index[np.argmax(variances)]  # expected 'B'

    assert weights[idx_low_var] > weights[idx_high_var], (
        f'Expected low-variance asset {idx_low_var} to have higher weight than '
        f'high-variance asset {idx_high_var}: {weights.to_dict()}'
    )

    # Middle variance asset should have weight between low and high
    mid_idx = [i for i in cov.index if i not in {idx_low_var, idx_high_var}][0]
    assert weights[idx_low_var] >= weights[mid_idx] >= weights[idx_high_var], (
        f'Expected ordering low >= mid >= high but got {weights.to_dict()}'
    )

def test_calculate_weights(hrp, sample_data):
    _, cov, corr = sample_data

    weights = hrp.calculate_weights(cov, corr)

    # Structural checks
    assert isinstance(weights, pd.Series)
    assert list(weights.index) == list(cov.index)
    assert np.isclose(weights.sum(), 1.0)

    # Behavioral checks
    # Lowest variance asset should have largest weight
    idx_low_var = cov.index[np.argmin(np.diag(cov.values))]
    idx_high_var = cov.index[np.argmax(np.diag(cov.values))]

    assert weights[idx_low_var] > weights[idx_high_var]

    # Check getter
    retrieved_weights = hrp.get_weights()
    pd.testing.assert_series_equal(retrieved_weights, weights)

# -------------------- Naive MVP Tests -------------------- #
def test_naive_mvp_simple_case():
    # Covariance matrix for 3 assets
    cov = np.array([
        [0.1, 0.02, 0.01],
        [0.02, 0.2, 0.03],
        [0.01, 0.03, 0.15]
    ])

    weights = naive_mvp(cov)

    # Check type
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == cov.shape[0]

    # Sum of weights = 1
    np.testing.assert_almost_equal(weights.sum(), 1.0)

    # All weights >= 0
    assert (weights >= 0).all()

def test_naive_mvp_diagonal_cov():
    # Diagonal covariance matrix: all variances equal
    cov = np.diag([0.1, 0.1, 0.1])
    weights = naive_mvp(cov)

    # All weights should be equal for identical variances
    np.testing.assert_allclose(weights, np.array([1/3, 1/3, 1/3]))

# -------------------- BaseQuadraticOptimizer Tests -------------------- #
@pytest.fixture
def basequad():
    """Create Base Quadratic Optimizer Instance"""
    return BaseQuadraticOptimizer(solver='auto')

def test_set_ridge(basequad):
    # Using string as input
    basequad.set_ridge('1e-6')

    assert basequad.reg == float('1e-6'), 'String input should convert to float'

    # Using float as input
    basequad.set_ridge(float('1e-9'))

    assert basequad.reg == float('1e-9'), 'Float input should set the ridge value'