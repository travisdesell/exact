import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.models.cov_models import (
    HierarchialRiskParity,
    NaiveMVP,
    BaseQuadraticOptimizer,
    GlobalMinimumVariance,
    MeanVariancePortfolio,
    NestedClusteredOptimization
)

# -------------------- Common Fixtures -------------------- #
@pytest.fixture
def sample_data():
    # IMPORTANT: Do not modify values without updating expected HRP test behavior.
    # This dataset enforces A < C < B variance for deterministic clustering.

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

# -------------------- Naive MVP Tests -------------------- #
def test_naive_mvp_simple_case():
    # Covariance matrix for 3 assets
    cov = np.array([
        [0.1, 0.02, 0.01],
        [0.02, 0.2, 0.03],
        [0.01, 0.03, 0.15]
    ])

    naive_mvp = NaiveMVP()
    weights = naive_mvp.calculate_weights(cov)

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
    naive_mvp = NaiveMVP()
    weights = naive_mvp.calculate_weights(cov)

    # All weights should be equal for identical variances
    np.testing.assert_allclose(weights, np.array([1/3, 1/3, 1/3]))

# -------------------- BaseQuadraticOptimizer Tests -------------------- #
@pytest.fixture
def base_quad():
    """Create Base Quadratic Optimizer Instance"""
    return BaseQuadraticOptimizer(solver='auto', reg='auto')

@pytest.fixture
def simple_cov():
    # Positive definite covariance matrix
    return np.array([
        [0.1, 0.02],
        [0.02, 0.1]
    ])

@pytest.fixture
def simple_q():
    return np.array([0.0, 0.0])

def test_ensure_symmetry(base_quad):
    # Testing function that ensures symmetry of matrix
    mat = np.array([[1., 2.],
                    [3., 4.]])
    sym = base_quad._ensure_symmetry(mat)

    assert np.allclose(sym, np.array([[1., 2.5],
                                      [2.5, 4.]]))
    # Should be symmetric
    assert np.allclose(sym, sym.T), 'Matrix Should be symmetric'

def test_compute_ridge_auto(base_quad, simple_cov):
    # Testing computation of ridge value. 
    # A small value used to safely invert matrices to avoid numerical errors.
    ridge = base_quad._compute_ridge(simple_cov)

    trace = np.trace(simple_cov)
    expected = 1e-8 * (trace / simple_cov.shape[0])

    assert np.isclose(ridge, expected)
    assert ridge > 0

def test_compute_ridge_numeric(simple_cov):
    # Testing ridge value is same as input ridge value
    base_quad = BaseQuadraticOptimizer(reg='1e-9')
    ridge = base_quad._compute_ridge(simple_cov)
    assert np.isclose(ridge, 1e-9)

def test_safe_inv(base_quad, simple_cov):
    # Testing function to safely invert matrices
    inv = base_quad._safe_inv(simple_cov)

    # inverse should be symmetric
    assert np.allclose(inv, inv.T), 'Inverse should be symetric'

    # Validate inverse property approximately: A * A^-1 = I
    approx_identity = simple_cov @ inv
    assert np.allclose(approx_identity, np.eye(2), atol=1e-6)

def test_qp_solve_sum_to_one(base_quad, simple_cov, simple_q):
    # Testing solving of quadratic equation, should give weights that sum to 1
    n = simple_cov.shape[0]
    A = np.ones((1, n))
    b = np.array([1.0])

    x, success = base_quad._qp_solve(
        P=simple_cov, 
        q=simple_q,
        A=A, 
        b=b
    )

    assert success
    assert np.isclose(np.sum(x), 1.0, atol=1e-6)

def test_qp_solve_nonnegative(base_quad, simple_cov, simple_q):
    # Weights should be non-negative after solving quadrativ problem
    n = simple_cov.shape[0]
    A = np.ones((1, n))
    b = np.array([1.0])

    # x >= 0  ->  -I x <= 0
    G = -np.eye(n)
    h = np.zeros(n)

    x, success = base_quad._qp_solve(
        P=simple_cov,
        q=simple_q,
        A=A, b=b,
        G=G, h=h
    )

    assert success
    assert np.all(x >= -1e-8)  # numerical tolerance
    assert np.isclose(np.sum(x), 1.0, atol=1e-6)

def test_set_ridge(base_quad):
    # Testing Setter method
    # Using string as input
    base_quad.set_ridge('1e-6')
    assert base_quad.reg == float('1e-6'), 'String input should convert to float'

    # Using float as input
    base_quad.set_ridge(0.001)
    assert base_quad.reg == 0.001, 'Float input should set the ridge value'

def test_qp_solve_scipy(simple_cov, simple_q):
    # Testing if Scipy implementation of quadratic problem solving works
    # Scipy is a backup for when cvxopt doesnt work
    base_quad = BaseQuadraticOptimizer(solver='scipy')
    n = simple_cov.shape[0]
    A = np.ones((1, n))
    b = np.array([1.0])

    x, success = base_quad._qp_solve(
        P=simple_cov, 
        q=simple_q,
        A=A, 
        b=b
    )

    assert success
    assert np.isclose(np.sum(x), 1.0, atol=1e-6)

# -------------------- Global Minimum Variance Tests -------------------- #
@pytest.fixture
def gmvp():
    """Create Gloab Minimum Variance Instance"""
    return GlobalMinimumVariance()

def test_calculate_weights_long_only(gmvp, sample_data):
    """Long-only QP: weights must be >= 0 and sum to 1."""
    _, cov, _ = sample_data
    weights = gmvp.calculate_weights(cov)

    assert (weights >= -1e-12).all()  # no negatives (allow small numerics)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert gmvp.success_ is True

    # Check getter
    retrieved_weights = gmvp.get_weights()
    assert np.array_equal(retrieved_weights, weights)

def analytic_gmvp_weights(cov):
    ones = np.ones(cov.shape[0])
    
    # Ensuring symmetry
    cov = 0.5 * (cov + cov.T)

    # Adding ridge
    cov_r = cov + 1e-8 * np.eye(cov.shape[0])
    
    # Safe Inversion
    inv = np.linalg.inv(cov_r)
    raw = inv @ ones
    
    return raw / (ones @ raw)

def test_calculate_weights_allow_short(sample_data):
    """When allow_short=True, closed form solution must match numpy inverse formula."""
    _, cov, _ = sample_data
    gmvp = GlobalMinimumVariance(allow_short=True)
    weights = gmvp.calculate_weights(cov)
    expected = analytic_gmvp_weights(cov)

    assert np.allclose(weights, expected, atol=1e-6)
    assert gmvp.success_ is True
    assert np.isclose(weights.sum(), 1.0)

    # Check getter
    retrieved_weights = gmvp.get_weights()
    assert np.array_equal(retrieved_weights, weights)

def test_get_weights_raises_if_not_fit(gmvp):
    with pytest.raises(ValueError):
        gmvp.get_weights()

def test_input_not_modified(sample_data):
    """Ensure we don't mutate original covariance input."""
    _, cov, _ = sample_data
    in_cov = cov.copy()
    gmvp = GlobalMinimumVariance(allow_short=True)
    _ = gmvp.calculate_weights(in_cov)
    assert np.allclose(in_cov, cov)

    assert np.allclose(gmvp.cov, cov)

def test_repeatability_short(sample_data):
    """Deterministic output: calling twice gives same result."""
    _, cov, _ = sample_data
    model = GlobalMinimumVariance(allow_short=True)
    w1 = model.calculate_weights(cov)
    w2 = model.calculate_weights(cov)
    assert np.allclose(w1, w2)

def test_repeatability_no_short(gmvp, sample_data):
    """Deterministic output: calling twice gives same result."""
    _, cov, _ = sample_data
    w1 = gmvp.calculate_weights(cov)
    w2 = gmvp.calculate_weights(cov)
    
    assert np.allclose(w1, w2)

# -------------------- Mean Variance Tests -------------------- #
def test_arith_mean_from_returns():
    mvp = MeanVariancePortfolio()
    # 2D returns array
    returns = np.array([[0.01, 0.02], [0.03, -0.01]])
    mu = mvp._arith_mean_from_returns(returns)
    expected = np.nanmean(returns, axis=0)
    np.testing.assert_allclose(mu, expected)

def test_arith_mean_raises_on_1d_input():
    mvp = MeanVariancePortfolio()
    bad = np.array([0.01, 0.02])  # 1-D
    with pytest.raises(ValueError):
        mvp._arith_mean_from_returns(bad)

def test_geom_mean_from_returns():
    mvp = MeanVariancePortfolio()
    returns = np.array([[0.01, 0.02], [0.03, -0.01]])
    gm = mvp._geom_mean_from_returns(returns)
    # compute expected: exp(mean(log1p(returns))) - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log = np.nanmean(np.log1p(returns), axis=0)
        expected = np.expm1(mean_log)
    np.testing.assert_allclose(gm, expected)

def test_geom_mean_handles_nans():
    mvp = MeanVariancePortfolio()
    returns = np.array([[0.01, np.nan], [np.nan, 0.02], [0.03, 0.01]])
    gm = mvp._geom_mean_from_returns(returns)
    # compute expected robustly
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log = np.nanmean(np.log1p(returns), axis=0)
        expected = np.expm1(mean_log)
    np.testing.assert_allclose(gm, expected, atol=1e-12)

def test_geom_mean_raises_on_1d_input():
    mvp = MeanVariancePortfolio()
    bad = np.array([0.01, 0.02])  # 1-D
    with pytest.raises(ValueError):
        mvp._geom_mean_from_returns(bad)

def test_calculate_weights_uses_arithmetic_mean(sample_data):
    returns, cov, _ = sample_data
    # Use DataFrame input for returns (class should accept DataFrame-like)
    mvp = MeanVariancePortfolio(expected_returns_method='arithmetic')

    # Run calculation (long-only QP path)
    w = mvp.calculate_weights(cov=cov.values, returns=returns)  # returns may be DataFrame or ndarray

    # expected mu computed via helper (call on numpy array to mirror internal behavior)
    expected_mu = np.nanmean(returns.values, axis=0)
    # stored expected_returns_ should match
    np.testing.assert_allclose(mvp.get_expected_returns(), expected_mu, atol=1e-12)

    # weight invariants
    assert isinstance(w, np.ndarray)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-12)  # long-only non-negativity
    assert mvp.success_ is True

def test_calculate_weights_uses_geometric_mean(sample_data):
    returns, cov, _ = sample_data
    mvp = MeanVariancePortfolio(expected_returns_method='geometric')

    w = mvp.calculate_weights(cov=cov.values, returns=returns)

    # compute geometric mean same way as class: exp(mean(log1p(returns))) - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_log = np.nanmean(np.log1p(returns.values), axis=0)
        expected_gm = np.expm1(mean_log)
    
    np.testing.assert_allclose(mvp.get_expected_returns(), expected_gm, atol=1e-12)

    # weight invariants
    assert isinstance(w, np.ndarray)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-12)
    assert mvp.success_ is True

def test_calculate_weights_raises_if_returns_missing_when_method_set(
        sample_data
    ):
    _, cov, _ = sample_data
    mvp = MeanVariancePortfolio(expected_returns_method='arithmetic')
    with pytest.raises(ValueError):
        mvp.calculate_weights(cov=cov, returns=None)  # returns required for arithmetic/geometric

def test_error_when_no_expected_returns_and_method_none(sample_data):
    _, cov, _ = sample_data
    mvp = MeanVariancePortfolio(expected_returns_method=None)

    # calling without expected_returns or returns should raise
    with pytest.raises(ValueError):
        mvp.calculate_weights(cov=cov)

def test_expected_returns_length_mismatch(sample_data):
    _, cov, _ = sample_data
    mvp = MeanVariancePortfolio(expected_returns_method=None)

    # pass expected_returns with wrong length
    wrong_mu = np.array([0.1, 0.2])  # length 2 but cov is 3x3
    with pytest.raises(ValueError):
        mvp.calculate_weights(cov=cov, expected_returns=wrong_mu)

def test_allow_short_analytic_solution_matches_properties(sample_data):
    returns, cov, _ = sample_data
    # provide expected returns explicitly so method doesn't need to compute
    mu = np.nanmean(returns.values, axis=0)

    mvp = MeanVariancePortfolio(expected_returns_method=None)

    w = mvp.calculate_weights(cov=cov.values, expected_returns=mu)

    # analytic solution: weights sum to 1, can be negative (shorting allowed), success True
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert isinstance(w, np.ndarray)
    assert mvp.success_ is True

def test_get_weights_and_expected_returns_before_fit_raise():
    mvp = MeanVariancePortfolio()
    with pytest.raises(ValueError):
        mvp.get_weights()
    with pytest.raises(ValueError):
        mvp.get_expected_returns()

def test_get_expected_returns_returns_copy(sample_data):
    returns, cov, _ = sample_data
    mvp = MeanVariancePortfolio(expected_returns_method='arithmetic')
    mvp.calculate_weights(cov=cov, returns=returns)

    # get_expected_returns should return a copy, modifying it shouldn't change internal state
    mu_copy = mvp.get_expected_returns()
    mu_copy[0] = 999.0
    # internal expected_returns_ should remain original
    assert mvp.expected_returns_[0] != 999.0

# -------------------- HRP Tests -------------------- #
@pytest.fixture
def hrp():
    """Create HRP instance"""
    return HierarchialRiskParity()

@pytest.fixture
def sample_linkage():
    link = np.array([
            [0, 1, 0.1, 2],  # cluster (0,1)
            [2, 3, 0.2, 2],  # cluster (2,3)
            [4, 5, 0.3, 4]   # final cluster ((0,1),(2,3))
        ], dtype=float)
    return link

def test_correlDist(hrp, sample_data):
    # Testing correlation distance matrix for symmetry and digonal
    _, _, corr = sample_data

    dist = hrp._correlDist(corr)

    assert dist.shape == corr.shape
    assert np.allclose(dist.values, dist.values.T), 'Matrix should symmetric'
    assert np.allclose(np.diag(dist), 0), 'Diagonal should be 0'
    assert (dist >= 0).all().all()
    assert (dist <= 1).all().all()

def test_getQuasiDiag_vaild_output(hrp, sample_linkage):
    # Testing if function outputs correct indices representing assets    
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

def test_get_weights_raises_if_not_fit(hrp):
    with pytest.raises(ValueError):
        hrp.get_weights()

# -------------------- NCO Tests -------------------- #
@pytest.fixture
def nco():
    """Create NCO instance"""
    return NestedClusteredOptimization(de_noise=True)

def test_cov2corr(nco, sample_data):
    returns, cov, corr = sample_data
    computed_corr = nco._cov2corr(cov.values)
    np.testing.assert_almost_equal(computed_corr, corr.values, decimal=6)
    # Diagonal should be 1
    assert np.allclose(np.diag(computed_corr), 1.0)

def test_getPCA(nco, sample_data):
    returns, cov, corr = sample_data
    eVal, eVec = nco._getPCA(corr.values)
    assert eVal.shape == (3,3)
    assert np.all(np.diag(eVal) >= 0)
    # Orthonormality
    np.testing.assert_almost_equal(eVec.T @ eVec, np.eye(3), decimal=6)

def test_mpPDF(nco):
    pdf = nco._mpPDF(var=0.5, q=2.0, pts=100)
    assert isinstance(pdf, pd.Series)
    assert len(pdf) == 100
    assert pdf.index[0] < pdf.index[-1]

def test_fitKDE(nco):
    obs = np.random.normal(0, 1, 100)
    pdf = nco._fitKDE(obs, bWidth=0.25, kernel='gaussian')
    assert isinstance(pdf, pd.Series)
    assert len(pdf) > 0

def test_errPDFs(nco):
    var = 0.5
    eVal = np.array([0.2, 0.5, 1.2])
    q = 2.0
    bWidth = 0.1
    sse = nco._errPDFs(var, eVal, q, bWidth)
    assert isinstance(sse, float)
    assert sse >= 0

def test_findMaxEval(nco, sample_data):
    returns, cov, corr = sample_data
    eVal, _ = nco._getPCA(corr.values)
    q = len(returns) / returns.shape[1]
    eMax, var = nco._findMaxEval(np.diag(eVal), q, bWidth=0.1)
    assert eMax > 0
    assert 0 < var < 1

def test_denoisedCorr(nco, sample_data):
    returns, cov, corr = sample_data
    eVal, eVec = nco._getPCA(corr.values)
    nFacts = 2
    corr_denoised = nco._denoisedCorr(eVal, eVec, nFacts)
    assert corr_denoised.shape == (3,3)
    np.testing.assert_almost_equal(np.diag(corr_denoised), 1.0, decimal=6)

def test_corr2cov(nco, sample_data):
    returns, cov, corr = sample_data
    std = np.sqrt(np.diag(cov.values))
    cov_reconstructed = nco._corr2cov(corr.values, std)
    np.testing.assert_almost_equal(cov_reconstructed, cov.values, decimal=6)

def test_deNoiseCov(nco, sample_data):
    returns, cov, corr = sample_data
    T, N = returns.shape
    q = T / N
    cov_denoised = nco._deNoiseCov(cov.values, q, bWidth=0.1)
    assert cov_denoised.shape == (3,3)
    np.testing.assert_almost_equal(cov_denoised, cov_denoised.T, decimal=6)

def test_de_noise(nco, sample_data):
    returns, cov, corr = sample_data
    T, N = returns.shape
    cov_denoised = nco._de_noise(cov, T, N)
    assert isinstance(cov_denoised, pd.DataFrame)
    assert cov_denoised.shape == cov.shape
    assert cov_denoised.index.equals(cov.index)
    assert cov_denoised.columns.equals(cov.columns)

def test_clusterKMeansBase(nco, sample_data):
    returns, cov, corr = sample_data
    corr1, clstrs, silh = nco._clusterKMeansBase(corr, maxNumClusters=2, n_init=2)
    assert isinstance(corr1, pd.DataFrame)
    assert corr1.shape == corr.shape
    assert isinstance(clstrs, dict)
    assert len(clstrs) <= 2
    assert isinstance(silh, pd.Series)
    assert len(silh) == len(corr.columns)

def test_calc_nco_no_mu(nco, sample_data):
    returns, cov, corr = sample_data
    weights = nco._calc_nco(cov, mu=None, maxNumClusters=int(cov.shape[0]/2))
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(cov.columns)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

def test_calc_nco_with_mu(nco, sample_data):
    returns, cov, corr = sample_data
    mu = np.array([0.01, 0.02, 0.015]).reshape(-1,1)
    weights = nco._calc_nco(cov, mu=mu, maxNumClusters=int(cov.shape[0]/2))
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(cov.columns)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

def test_calculate_weights_with_denoise(nco, sample_data):
    returns, cov, corr = sample_data
    weights = nco.calculate_weights(cov, returns)
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(cov.columns)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

def test_calculate_weights_without_denoise(nco, sample_data):
    returns, cov, corr = sample_data
    weights = nco.calculate_weights(cov, returns)
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(cov.columns)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

def test_single_asset():
    returns_single = pd.DataFrame({'A': [0.01, -0.01, 0.02]})
    cov_single = returns_single.cov()
    nco = NestedClusteredOptimization(de_noise=False)
    weights = nco.calculate_weights(cov_single, returns_single)
    assert len(weights) == 1
    assert weights.iloc[0] == 1.0