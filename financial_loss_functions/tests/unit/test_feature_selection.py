import numpy as np
import pandas as pd

from src.feature_selection.analysis import aggregate_ticker_features, create_lagged_features


def test_aggregate_ticker_features_means_by_suffix():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    raw = pd.DataFrame({
        'AAA_RET': [0.01, 0.02, 0.03],
        'BBB_RET': [0.03, 0.02, 0.01],
        'AAA_VOL_CHANGE': [1.0, 2.0, 3.0],
        'BBB_VOL_CHANGE': [3.0, 2.0, 1.0],
        'AAA_BA_SPREAD': [10.0, 20.0, 30.0],
        'BBB_BA_SPREAD': [30.0, 20.0, 10.0],
    }, index=dates)

    engineered = aggregate_ticker_features(raw)

    assert list(sorted(engineered.columns)) == sorted(
        ['RET', 'VOL_CHANGE', 'BA_SPREAD', 'CUM_RET', 'LOG_CUM_RET']
    )
    np.testing.assert_allclose(engineered['RET'].values, np.array([0.02, 0.02, 0.02]))
    np.testing.assert_allclose(engineered['VOL_CHANGE'].values, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(engineered['BA_SPREAD'].values, np.array([20.0, 20.0, 20.0]))
    np.testing.assert_allclose(engineered['CUM_RET'].values, np.array([0.02, 0.04, 0.06]))
    np.testing.assert_allclose(engineered['LOG_CUM_RET'].values, np.log1p(np.array([0.02, 0.04, 0.06])))


def test_create_lagged_features_multiple_lags():
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'RET': [0.01, 0.02, 0.03, 0.04, 0.05],
        'VOL_CHANGE': [1, 2, 3, 4, 5],
    }, index=dates)

    lagged = create_lagged_features(data, lags=[1, 3])

    assert 'RET_LAG_1' in lagged.columns
    assert 'RET_LAG_3' in lagged.columns
    assert 'VOL_CHANGE_LAG_1' in lagged.columns
    assert 'VOL_CHANGE_LAG_3' in lagged.columns
    assert pd.isna(lagged.loc[dates[0], 'RET_LAG_1'])
    assert lagged.loc[dates[4], 'RET_LAG_1'] == data.loc[dates[3], 'RET']
    assert lagged.loc[dates[4], 'RET_LAG_3'] == data.loc[dates[1], 'RET']
