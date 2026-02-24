import pytest
from src.utils.formatting import split_col, extract_req_cols

def test_split_col_valid_and_invalid():    # valid split
    t, f = split_col(col_sep='_', col='ABC_feature_name')
    assert t == 'ABC' and f == 'feature_name'

    # invalid format (no separator) should raise
    with pytest.raises(ValueError):
        split_col('_', 'noseparator')

def test_split_col_hyphen():    # valid split
    t, f = split_col(col_sep='-', col='ABC-feature-name')
    assert t == 'ABC' and f == 'feature-name'

    # invalid format (no separator) should raise
    with pytest.raises(ValueError):
        split_col('-', 'no_separator')

def test_extract_req_cols():
    my_cols = ['AAPL_RET', 'MSFT_VOL_CHANGE', 'GOOG_RET', 'AAPL_VOL_CHANGE']
    
    # 2. Run the function
    result = extract_req_cols(my_cols, '_VOL_CHANGE')
    
    # 3. Check the answer
    assert result == ['MSFT_VOL_CHANGE', 'AAPL_VOL_CHANGE']
    
    # 4. Check a case with no matches
    assert extract_req_cols(my_cols, 'PRICE') == []