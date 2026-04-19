import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.training.train_trad import TradModelsTrainer


@pytest.fixture
def sample_returns():
    """DataFrame of stock returns: 100 days, 3 stocks"""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    return returns

@pytest.fixture
def mock_model_lib():
    """Mock model library with two dummy model classes"""
    class DummyModel1:
        def __init__(self, **kwargs):
            pass  # ignore any hyperparameters
        def calculate_weights(self, cov, returns):
            return pd.Series([0.3, 0.3, 0.4], index=['A', 'B', 'C'])

    class DummyModel2:
        def __init__(self, **kwargs):
            pass
        def calculate_weights(self, corr, returns):
            return np.array([0.2, 0.5, 0.3])

    return {
        'Model1': DummyModel1,
        'Model2': DummyModel2,
    }

@pytest.fixture
def hparams_config():
    return {
        'rolling_windows': {'out_size': 20},
        'trad_models': {
            'Model1': {'param': 1},
            'Model2': {}
        }
    }

@pytest.fixture
def trainer(mock_model_lib, hparams_config):
    return TradModelsTrainer(
        model_lib=mock_model_lib,
        hparams_config=hparams_config,
        num_steps=3
    )

# -------------------- Tests for train_one_model -------------------- #
def test_train_one_model_success(trainer):
    model_name = 'Model1'
    model_class = trainer.model_lib[model_name]
    filtered_kwargs = {'cov': np.eye(3), 'returns': pd.DataFrame()}
    weights = trainer._train_one_model(model_name, model_class, filtered_kwargs)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3

def test_train_one_model_without_hparams(trainer):
    # Model2 has no hparams, should still work
    model_name = 'Model2'
    model_class = trainer.model_lib[model_name]
    filtered_kwargs = {'corr': np.eye(3), 'returns': pd.DataFrame()}
    weights = trainer._train_one_model(model_name, model_class, filtered_kwargs)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (3,)

# -------------------- Tests for _process_train_1_ds --------------------# 
def test_process_train_1_ds_success(trainer, sample_returns):
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        result = trainer._process_train_1_ds(sample_returns)
    assert 'Model1' in result
    assert 'Model2' in result
    assert isinstance(result['Model1'], np.ndarray)
    assert result['Model1'].shape == (3,)
    assert result['Model2'].shape == (3,)

def test_process_train_1_ds_error_handling(trainer, sample_returns):
    # Add a model that raises an exception
    class FaultyModel:
        def calculate_weights(self, cov, returns):
            raise ValueError("Test error")
    trainer.model_lib['Faulty'] = FaultyModel
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        result = trainer._process_train_1_ds(sample_returns)
    assert 'Faulty' not in result
    assert 'Model1' in result
    assert 'Model2' in result

def test_process_train_1_ds_no_valid_kwargs_raises(trainer, sample_returns):
    # Model that requires a parameter not in payload
    class StrangeModel:
        def calculate_weights(self, missing_arg):
            pass
    trainer.model_lib['Strange'] = StrangeModel
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        with pytest.raises(ValueError, match="Required parameters for Strange do not exist in payload"):
            trainer._process_train_1_ds(sample_returns)

def test_process_train_1_ds_converts_series_to_numpy(trainer, sample_returns):
    # Model1 returns Series, should become numpy array
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        result = trainer._process_train_1_ds(sample_returns)
    assert isinstance(result['Model1'], np.ndarray)

# -------------------- Tests for build_walk_slice -------------------- #
def test_build_walk_slice_first_step(trainer, sample_returns):
    init_rets_train = sample_returns.iloc[:50].copy()
    init_rets_split = sample_returns.iloc[50:].copy()
    with patch('src.training.train_trad.calc_current_idxs') as mock_calc:
        mock_calc.return_value = (0, 20)
        result = trainer._build_walk_slice(init_rets_train, init_rets_split, step=0)
    pd.testing.assert_frame_equal(result, init_rets_train)

def test_build_walk_slice_subsequent_step(trainer, sample_returns):
    init_rets_train = sample_returns.iloc[:50].copy()
    init_rets_split = sample_returns.iloc[50:].copy()
    with patch('src.training.train_trad.calc_current_idxs') as mock_calc:
        mock_calc.return_value = (20, 40)
        result = trainer._build_walk_slice(init_rets_train, init_rets_split, step=1)
    expected = pd.concat([init_rets_train, init_rets_split.iloc[:20]])
    pd.testing.assert_frame_equal(result, expected)

# -------------------- Tests for train_all -------------------- #
def test_train_all_success(trainer, sample_returns):
    init_rets_train = sample_returns.iloc[:50].copy()
    init_rets_split = sample_returns.iloc[50:].copy()
    trainer.num_steps = 2
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        with patch('src.training.train_trad.calc_current_idxs') as mock_calc:
            mock_calc.side_effect = [(0,20), (20,40)]
            result = trainer.train_all(init_rets_train, init_rets_split)
    assert 'Model1' in result
    assert 'Model2' in result
    assert result['Model1'].shape == (2, 3)
    assert result['Model2'].shape == (2, 3)
    assert isinstance(result['Model1'], np.ndarray)

def test_train_all_column_mismatch_raises(trainer, sample_returns):
    init_rets_train = sample_returns.iloc[:50].copy()
    init_rets_split = sample_returns.iloc[50:].copy()
    # Change the number of columns (e.g., drop one column)
    init_rets_split = init_rets_split.drop(columns=['A'])
    with pytest.raises(ValueError, match="Both dataframes must have equal number of columns"):
        trainer.train_all(init_rets_train, init_rets_split)

def test_train_all_empty_model_lib(sample_returns):
    trainer_empty = TradModelsTrainer(model_lib={}, hparams_config={'rolling_windows': {'out_size': 20}, 'trad_models': {}}, num_steps=1)
    init_rets_train = sample_returns.iloc[:50].copy()
    init_rets_split = sample_returns.iloc[50:].copy()
    with patch('src.training.train_trad.preprocessor2') as mock_preproc:
        mock_preproc.return_value = (np.eye(3), np.eye(3))
        result = trainer_empty.train_all(init_rets_train, init_rets_split)
    assert result == {}  # no models, empty dict

# -------------------- Tests for _stack_weights -------------------- #
def test_stack_weights(trainer):
    # Simulate weights collected for two steps
    trainer.all_alloc_weights = {
        'Model1': [np.array([0.1, 0.2, 0.7]), np.array([0.2, 0.3, 0.5])],
        'Model2': [np.array([0.4, 0.3, 0.3]), np.array([0.1, 0.1, 0.8])]
    }
    trainer._stack_weights()
    assert trainer.all_alloc_weights['Model1'].shape == (2, 3)
    assert trainer.all_alloc_weights['Model2'].shape == (2, 3)
    np.testing.assert_array_equal(trainer.all_alloc_weights['Model1'][0], [0.1, 0.2, 0.7])
    np.testing.assert_array_equal(trainer.all_alloc_weights['Model2'][1], [0.1, 0.1, 0.8])

def test_stack_weights_empty():
    trainer = TradModelsTrainer(model_lib={}, hparams_config={'rolling_windows': {'out_size': 20}, 'trad_models': {}}, num_steps=1)
    trainer.all_alloc_weights = {}
    trainer._stack_weights()
    assert trainer.all_alloc_weights == {}