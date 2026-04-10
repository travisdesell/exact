import pytest
import numpy as np
from src.utils.formatting import (
    split_col,
    extract_req_cols,
    reformat_hparams,
    split_combo_names,
    serialize_np_dict
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

# -------------------- Tests for reformat_hparams
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
    assert result['scheduler'] is None
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