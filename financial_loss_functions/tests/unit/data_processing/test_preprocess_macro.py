import pytest
import numpy as np
import pandas as pd
from src.data_processing.preprocess_macro import MacroCombiner

# --------- MacroCombiner tests ----------#
@pytest.fixture
def macro_combiner():
    return MacroCombiner(resample_freq='B') # B for business days

def test_combine_macro_data_parses_datetimes_and_sorts(macro_combiner):
    """ Test for combining macro data csvs, and sorting by date"""
    # df1 uses string dates (unsorted); df2 uses strings too but covers other dates
    df1 = pd.DataFrame(
        {'A': [1, 2, 3]}, index=['2020-01-03', '2020-01-01', '2020-01-02']
    )
    df2 = pd.DataFrame(
        {'B': [10, 20, np.nan]}, index=['2020-01-01', '2020-01-02', '2020-01-04']
    )

    raw_macro = {'one': df1, 'two': df2}

    combined = macro_combiner.combine_macro_data(raw_macro)

    # Index must be datetime and sorted ascending
    assert isinstance(combined.index, pd.DatetimeIndex)
    assert list(combined.index) == sorted(list(combined.index))

    # Both columns should be present and aligned by date
    assert 'A' in combined.columns and 'B' in combined.columns
    # Check that a known value is present (A at 2020-01-01)
    assert combined.loc[pd.Timestamp('2020-01-01'), 'A'] == 2

def test_combine_macro_data_drops_all_nan_columns(macro_combiner):
    df1 = pd.DataFrame(
        {'A': [1, 2]}, index=['2020-01-01', '2020-01-02']
    )
    df2 = pd.DataFrame(
        {'ALLNAN': [np.nan, np.nan]}, index=['2020-01-01', '2020-01-02']
    )
    raw_macro = {'d1': df1, 'd2': df2}

    combined = macro_combiner.combine_macro_data(raw_macro)

    # Column that is entirely NaN should be dropped
    assert 'ALLNAN' not in combined.columns
    assert 'A' in combined.columns

def test_to_daily_resample_ffill_then_bfill_and_drop_all_nan():
    # Use daily resampling frequency in test to make expected indexes deterministic
    mc = MacroCombiner(resample_freq='D')

    # initial DF: first two days are NaN, the third day has 1.0
    df = pd.DataFrame(
        {'M': [np.nan, np.nan, 1.0]},
        index=pd.to_datetime(['2020-01-03', '2020-01-04', '2020-01-05'])
    )

    daily = mc.to_daily(df)

    expected_idx = pd.date_range('2020-01-03', '2020-01-05', freq='D')
    assert list(daily.index) == list(expected_idx)

    # bfill should have filled the leading NaNs with 1.0
    assert daily.loc[pd.Timestamp('2020-01-03'), 'M'] == 1.0
    assert daily.loc[pd.Timestamp('2020-01-04'), 'M'] == 1.0

    # Append a trailing all-NaN date and make sure it is dropped after to_daily
    trailing = pd.DataFrame({'M': [np.nan]}, index=[pd.Timestamp('2020-01-10')])
    df2 = pd.concat([df, trailing])
    daily2 = mc.to_daily(df2)
    assert pd.Timestamp('2020-01-10') not in daily2.index