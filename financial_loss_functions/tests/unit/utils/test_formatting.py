import sys
import pytest
import numpy as np
import pandas as pd
from io import StringIO
from src.utils.formatting import (
    split_col,
    extract_req_cols,
    reformat_hparams,
    split_combo_names,
    serialize_np_dict,
    deserialize_np_dict,
    print_evaluation_info,
    reform_returns_w_dates,
    reformat_model_perfs
)

# -------------------- Tests for extract_req_cols -------------------- #
def test_extract_req_cols():
    my_cols = ['AAPL_RET', 'MSFT_VOL_CHANGE', 'GOOG_RET', 'AAPL_VOL_CHANGE']
    
    # 2. Run the function
    result = extract_req_cols(my_cols, '_VOL_CHANGE')
    
    # 3. Check the answer
    assert result == ['MSFT_VOL_CHANGE', 'AAPL_VOL_CHANGE']
    
    # 4. Check a case with no matches
    assert extract_req_cols(my_cols, 'PRICE') == []

# -------------------- Tests for split_col -------------------- #
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

# -------------------- Tests for reformat_hparams --------------------- #
def test_reformat_hparams_full():
    model_cfg = {
        'model': {'hidden_size': 32, 'dropout': 0.2},
        'optimizer': {'lr': 0.001, 'weight_decay': 0.0001},
        'train': {'epochs': 100, 'batch_size': 64},
        'scheduler': {'patience': 10}
    }
    loss_cfg = {'lambdas': {'lambda1': 0.5, 'lambda2': 1.0}}
    result = reformat_hparams(model_cfg, loss_cfg)
    expected = {
        'model': {'hidden_size': 32, 'dropout': 0.2},
        'optimizer': {'lr': 0.001, 'weight_decay': 0.0001},
        'train': {'epochs': 100, 'batch_size': 64},
        'scheduler': {'patience': 10},
        'loss': {'lambda1': 0.5, 'lambda2': 1.0}
    }
    assert result == expected

def test_reformat_hparams_no_scheduler():
    model_cfg = {
        'model': {'a': 1},
        'optimizer': {'b': 2},
        'train': {'c': 3}
    }
    loss_cfg = {'lambdas': {}}
    result = reformat_hparams(model_cfg, loss_cfg)
    assert result['scheduler'] == {}
    assert result['loss'] == {}

def test_reformat_hparams_no_loss_lambdas():
    model_cfg = {
        'model': {'x': 10},
        'optimizer': {'y': 20},
        'train': {'z': 30}
    }
    loss_cfg = {}   # missing 'lambdas'
    result = reformat_hparams(model_cfg, loss_cfg)
    assert result['loss'] == {}   # empty dict default

def test_reformat_hparams_deepcopy_independence():
    model_cfg = {'model': {'val': 5}, 'optimizer': {'val': 1}, 'train': {'val': 2}}
    loss_cfg = {'lambdas': {'lam': 0.1}}
    result = reformat_hparams(model_cfg, loss_cfg)
    # modify original should not affect result
    model_cfg['model']['val'] = 99
    assert result['model']['val'] == 5
    loss_cfg['lambdas']['lam'] = 0.9
    assert result['loss']['lam'] == 0.1
    model_cfg['optimizer']['val'] = 10
    assert result['optimizer']['val'] == 1
    model_cfg['train']['val'] == 7
    assert result['train']['val'] == 2

# -------------------- Tests for split_combo_names -------------------- #
def test_split_combo_names_valid():
    names = ['modelA-loss1', 'modelB-loss2', 'modelC-loss3']
    result = split_combo_names(names, '-')
    expected = [('modelA', 'loss1'), ('modelB', 'loss2'), ('modelC', 'loss3')]
    assert result == expected

def test_split_combo_names_sep_not_found():
    names = ['modelA_loss1', 'modelB_loss2']
    with pytest.raises(ValueError, match="Model \\+ Loss combo name string is incorrect"):
        split_combo_names(names, '-')

def test_split_combo_names_empty_list():
    result = split_combo_names([], '-')
    assert result == []

def test_split_combo_names_multiple_sep_uses_first_only():
    names = ['modelA-loss1-extra']
    # split with maxsplit=1, so only first separator matters
    result = split_combo_names(names, '-')
    assert result == [('modelA', 'loss1-extra')]

# -------------------- Tests for serialize_np_dict -------------------- #
def test_serialize_np_dict_simple():
    obj = {'a': np.array([1, 2, 3]), 'b': np.array([[4,5],[6,7]])}
    result = serialize_np_dict(obj)
    expected = {'a': [1, 2, 3], 'b': [[4,5],[6,7]]}
    assert result == expected

def test_serialize_np_dict_nested():
    obj = {
        'level1': {
            'level2': np.array([10, 20]),
            'list': [np.array(1), np.array(2)]
        }
    }
    result = serialize_np_dict(obj)
    expected = {
        'level1': {
            'level2': [10, 20],
            'list': [1, 2]
        }
    }
    assert result == expected

def test_serialize_np_dict_no_arrays():
    obj = {'x': 1, 'y': 'string', 'z': [1,2,3]}
    result = serialize_np_dict(obj)
    assert result == obj

def test_serialize_np_dict_np_scalar():
    obj = {'val': np.float32(3.14)}
    result = serialize_np_dict(obj)
    assert result['val'] == 3.14   # tolist() converts scalar to Python float

def test_serialize_np_dict_empty():
    assert serialize_np_dict({}) == {}
    assert serialize_np_dict([]) == []
    assert serialize_np_dict(np.array([])) == []   # empty array -> empty list

# -------------------- Tests for deserialize_np_dict -------------------- #
def test_deserialize_np_dict_simple_list():
    obj = [1, 2, 3]
    result = deserialize_np_dict(obj)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3]))

def test_deserialize_np_dict_empty_list():
    obj = []
    result = deserialize_np_dict(obj)
    assert isinstance(result, np.ndarray)
    assert result.size == 0

def test_deserialize_np_dict_nested_list():
    obj = [[1, 2], [3, 4]]
    result = deserialize_np_dict(obj)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.array_equal(result, np.array([[1, 2], [3, 4]]))

def test_deserialize_np_dict_mixed_types():
    obj = [1, 'a', 2.5]
    result = deserialize_np_dict(obj)
    # The function converts the list to a numpy array of strings
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

    assert result[0] == '1' or result[0] == 1   # but array will be strings

    assert result.tolist() == ['1', 'a', '2.5']

def test_deserialize_np_dict_dict():
    obj = {'a': [1, 2], 'b': {'c': [3, 4]}}
    result = deserialize_np_dict(obj)
    assert isinstance(result, dict)
    assert isinstance(result['a'], np.ndarray)
    assert np.array_equal(result['a'], np.array([1, 2]))
    assert isinstance(result['b']['c'], np.ndarray)
    assert np.array_equal(result['b']['c'], np.array([3, 4]))

def test_deserialize_np_dict_scalar():
    obj = 42
    result = deserialize_np_dict(obj)
    assert result == 42

# -------------------- Tests for print_evaluation_info -------------------- #
def test_print_evaluation_info_with_input_windows():
    out_dates = [pd.date_range('2020-01-01', periods=5), pd.date_range('2020-02-01', periods=5)]
    in_dates = [pd.date_range('2019-12-01', periods=5), pd.date_range('2020-01-15', periods=5)]
    df1 = pd.DataFrame({'A': [1,2]})
    df2 = pd.DataFrame({'B': [3,4]})
    
    captured = StringIO()
    sys.stdout = captured
    print_evaluation_info(out_win_date_cols=out_dates, in_win_date_cols=in_dates, metrics1=df1, metrics2=df2)
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert 'Models evaluated on:' in output
    assert 'Input Window Start' in output
    assert 'Out Window Start' in output
    assert 'METRICS1' in output
    assert 'METRICS2' in output
    assert '1' in output  # from df1

def test_print_evaluation_info_without_input_windows():
    out_dates = [pd.date_range('2020-01-01', periods=5)]
    df = pd.DataFrame({'X': [10]})
    captured = StringIO()
    sys.stdout = captured
    print_evaluation_info(out_win_date_cols=out_dates, in_win_date_cols=None, single_metric=df)
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert 'Models evaluated on:' in output
    assert 'Out Window Start' in output
    assert 'Input Window Start' not in output
    assert 'SINGLE METRIC' in output

# -------------------- Test reform_returns_w_dates -------------------- #
def test_reform_returns_w_dates():
    daily_returns = {
        'ModelA': [[0.1, 0.2], [0.3, 0.4]],
        'ModelB': [[0.5, 0.6]]
    }
    out_win_date_cols = [
        pd.date_range('2020-01-01', periods=2),
        pd.date_range('2020-01-03', periods=2)
    ]
    expected = {
        'ModelA': {
            '2020-01-01_2020-01-02': [0.1, 0.2],
            '2020-01-03_2020-01-04': [0.3, 0.4]
        },
        'ModelB': {
            '2020-01-01_2020-01-02': [0.5, 0.6]
        }
    }
    result = reform_returns_w_dates(daily_returns, out_win_date_cols)
    assert result == expected

def test_reform_returns_w_dates_mismatched_windows():
    daily_returns = {'ModelC': [[0.1]]}
    out_win_date_cols = [pd.date_range('2020-01-01', periods=1)]
    result = reform_returns_w_dates(daily_returns, out_win_date_cols)
    assert result['ModelC']['2020-01-01_2020-01-01'] == [0.1]

# -------------------- Tests for reformat_model_perfs -------------------- #
def test_reformat_model_perfs_with_weights():
    daily_returns = {
        'ModelA': [[0.1, 0.2], [0.3, 0.4]],
    }
    alloc_weights = {
        'ModelA': [[0.5, 0.5], [0.6, 0.4]],
    }
    out_win_date_cols = [
        pd.date_range('2020-01-01', periods=2),
        pd.date_range('2020-01-03', periods=2)
    ]
    result = reformat_model_perfs(daily_returns, alloc_weights, out_win_date_cols)
    expected = {
        'ModelA': {
            '2020-01-01_2020-01-02': {'returns': [0.1, 0.2], 'weights': [0.5, 0.5]},
            '2020-01-03_2020-01-04': {'returns': [0.3, 0.4], 'weights': [0.6, 0.4]}
        }
    }
    assert result == expected

def test_reformat_model_perfs_without_weights():
    daily_returns = {
        'Benchmark': [[0.1, 0.2]],
    }
    alloc_weights = {}  # no weights for benchmark
    out_win_date_cols = [pd.date_range('2020-01-01', periods=2)]
    result = reformat_model_perfs(daily_returns, alloc_weights, out_win_date_cols)
    expected = {
        'Benchmark': {
            '2020-01-01_2020-01-02': {'returns': [0.1, 0.2]}
        }
    }
    assert result == expected

def test_reformat_model_perfs_missing_model_weights():
    daily_returns = {'ModelA': [[0.1]], 'ModelB': [[0.2]]}
    alloc_weights = {'ModelA': [[0.9]]}  # ModelB missing weights
    out_win_date_cols = [pd.date_range('2020-01-01', periods=1)]
    result = reformat_model_perfs(daily_returns, alloc_weights, out_win_date_cols)
    # ModelB should have only returns (no weights)
    assert 'weights' not in result['ModelB']['2020-01-01_2020-01-01']
    assert result['ModelA']['2020-01-01_2020-01-01']['weights'] == [0.9]