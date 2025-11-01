import pytest
import numpy as np
import pandas as pd
from cov_models import HierarchialRiskParity

@pytest.fixture
def hrp():
    """Create HRP instance"""
    return HierarchialRiskParity()

@pytest.fixture
def sample_data():
    # synthetic returns for 3 assets over 100 periods
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 3) * [0.1, 0.2, 0.15],  # asset vol differences
        columns=['A', 'B', 'C']
    )

    cov = returns.cov()
    corr = returns.corr()

    return returns, cov, corr

@pytest.fixture
def sample_linkage():
    link = np.array([
            [0, 1, 0.1, 2],  # cluster (0,1)
            [2, 3, 0.2, 2],  # cluster (2,3)
            [4, 5, 0.3, 4]   # final cluster ((0,1),(2,3))
        ], dtype=float)
    return link

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

# TODO: Continue unit tests for hrp, from getRecBipart to complete HRP model