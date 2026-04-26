import torch
import pytest
import numpy as np
import pandas as pd
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.data_processing.dataset import WindowDataset, Reshaper
from src.training.train_nn import (
    Trainer,
    Walker,
    MetricModel,
    Tuner,
    CandidatesGrid,
)

# Dummy model with trainable parameters (no shape errors)
class DummyModel(nn.Module):
    def __init__(self, input_size, num_stocks, max_seq_len, **kwargs):
        super().__init__()
        # Flatten the input (B, seq_len, input_size) to (B, seq_len*input_size)
        self.fc = nn.Linear(max_seq_len * input_size, num_stocks)
    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        logits = self.fc(x_flat)
        return torch.softmax(logits, dim=-1)

# ==================== Fixtures for Trainer ==================== #
@pytest.fixture
def dummy_model_class():
    return DummyModel

@pytest.fixture
def dummy_loss():
    def loss(weights, all_returns, pf_returns, **kwargs):
        return pf_returns.mean()
    return loss

@pytest.fixture
def default_train_hparams():
    return {
        'epochs': 5,
        'min_epochs': 0,
        'early_stop_patience': 2,
        'early_stop_min_delta': 1e-3,
        'early_stopping': True,
        'clip_grad_norm': 0.5,
        'train_batch_size': 4,
        'val_batch_size': 4,
    }

@pytest.fixture
def model_hparams():
    # Not used in dummy model, but required for Trainer
    return {'hidden_size': 16, 'num_layers': 1, 'dropout': 0.0}

@pytest.fixture
def optimizer_hparams():
    return {'lr': 0.001, 'weight_decay': 0.0}

@pytest.fixture
def scheduler_hparams():
    return {'factor': 0.5, 'patience': 5, 'min_lr': 1e-6}

@pytest.fixture
def loss_hparams():
    return {'lambda': 0.1}

@pytest.fixture
def trainer(dummy_model_class, dummy_loss, model_hparams, optimizer_hparams, default_train_hparams,
            scheduler_hparams, loss_hparams):
    return Trainer(
        model=dummy_model_class,
        loss=dummy_loss,
        model_hparams=model_hparams,
        optimizer_hparams=optimizer_hparams,
        train_hparams=default_train_hparams,
        in_size=251,          # number of features
        num_stocks=50,
        max_seq_len=120,
        device='cpu',
        scheduler_hparams=scheduler_hparams,
        loss_hparams=loss_hparams
    )


@pytest.fixture
def test_device_str():
    if torch.cuda.is_available():
        return 'cuda'
    
    elif torch.backends.mps.is_available():
        return 'mps'
    
    else:
        return 'cpu'

# ==================== Tests for Trainer ==================== #
def test_init_device_string(test_device_str):
    trainer = Trainer(
        model=DummyModel, loss=lambda x: x, model_hparams={}, optimizer_hparams={},
        train_hparams={'epochs': 1}, in_size=1, num_stocks=1, max_seq_len=1, device=test_device_str
    )
    assert trainer.device.type == test_device_str

def test_init_device_invalid_raises():
    with pytest.raises(ValueError, match="Incorrect type provided for torch device"):
        Trainer(
            model=DummyModel, loss=lambda x: x, model_hparams={}, optimizer_hparams={},
            train_hparams={'epochs': 1}, in_size=1, num_stocks=1, max_seq_len=1, device=123
        )

def test_init_loss_hparams_default():
    trainer = Trainer(
        model=DummyModel, loss=lambda x: x, model_hparams={}, optimizer_hparams={},
        train_hparams={'epochs': 1}, in_size=1, num_stocks=1, max_seq_len=1, device='cpu',
        scheduler_hparams=None, loss_hparams=None
    )
    assert trainer.loss_hparams == {}

def test_init_model_instantiated(trainer):
    assert isinstance(trainer.model, DummyModel)

# -------------------- Tests for _cal_pf_returns -------------------- #
def test_cal_pf_returns(trainer):
    weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
    returns = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)  # B=1, T=2, N=2
    expected = weights.unsqueeze(1) * returns  # (1,2,2) -> sum over last dim
    expected = expected.sum(dim=-1)  # (1,2)
    result = trainer._cal_pf_returns(weights, returns)
    assert torch.allclose(result, expected)

# -------------------- Tests for _init_optimizer -------------------- #
def test_init_optimizer(trainer):
    opt = trainer._init_optimizer()
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults['lr'] == 0.001
    assert opt.defaults['weight_decay'] == 0.0

# ----------------------------------------------------------------------
# _init_scheduler
# ----------------------------------------------------------------------
def test_init_scheduler(trainer):
    opt = trainer._init_optimizer()
    scheduler = trainer._init_scheduler(opt)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

def test_init_scheduler_none(trainer):
    trainer.scheduler_hparams = None
    opt = trainer._init_optimizer()
    scheduler = trainer._init_scheduler(opt)
    assert scheduler is None

# -------------------- Tests for train method (with mocking) -------------------- #
@patch('src.training.train_nn.DataLoader')
@patch('src.training.train_nn.time')
def test_train_basic_flow(mock_time, mock_dataloader, trainer):
    # Disable early stopping to run all epochs
    trainer.train_hparams['early_stopping'] = False
    # Mock time to return constant value to avoid StopIteration
    mock_time.time.return_value = 0.0
    mock_train_ds = MagicMock(spec=WindowDataset)
    mock_val_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(torch.randn(4, 120, 251), torch.randn(4, 60, 50))]
    mock_dataloader.return_value = mock_loader

    trainer.validate = MagicMock(return_value=0.5)

    trainer.train(mock_train_ds, mock_val_ds)

    assert len(trainer.train_losses) == 5
    assert len(trainer.val_losses) == 5
    assert trainer.best_model_state is not None
    assert trainer.best_epoch == 4

def test_train_early_stopping(trainer):
    trainer.train_hparams['epochs'] = 10
    trainer.train_hparams['early_stop_patience'] = 2
    trainer.train_hparams['min_epochs'] = 0

    mock_train_ds = MagicMock(spec=WindowDataset)
    mock_val_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(torch.randn(4, 120, 251), torch.randn(4, 60, 50))]
    with patch('src.training.train_nn.DataLoader', return_value=mock_loader):
        trainer.validate = MagicMock(side_effect=[1.0, 1.0, 1.0, 1.0, 1.0])
        trainer.train(mock_train_ds, mock_val_ds)
        # With patience=2, the observed behavior is 2 epochs total (epoch 0 and 1)
        assert len(trainer.train_losses) == 2

def test_train_no_validation(trainer):
    mock_train_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(torch.randn(4, 120, 251), torch.randn(4, 60, 50))]
    with patch('src.training.train_nn.DataLoader', return_value=mock_loader):
        trainer.train(mock_train_ds, val_ds=None)
    assert len(trainer.val_losses) == 0
    assert trainer.best_model_state is not None

def test_train_gradient_clipping(trainer):
    mock_train_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(torch.randn(4, 120, 251), torch.randn(4, 60, 50))]
    with patch('src.training.train_nn.DataLoader', return_value=mock_loader), \
         patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        trainer.train(mock_train_ds, val_ds=None)
        # Should be called once per epoch (5 times)
        assert mock_clip.call_count == 5
        for call_args in mock_clip.call_args_list:
            args, kwargs = call_args
            assert kwargs['max_norm'] == 0.5

# -------------------- Tests for validate method -------------------- #
def test_validate(trainer):
    mock_val_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    batch1 = (torch.randn(4, 120, 251), torch.randn(4, 60, 50))
    batch2 = (torch.randn(2, 120, 251), torch.randn(2, 60, 50))
    mock_loader.__iter__.return_value = [batch1, batch2]
    with patch('src.training.train_nn.DataLoader', return_value=mock_loader):
        avg_loss = trainer.validate(mock_val_ds)
    assert isinstance(avg_loss, float)
    assert not np.isnan(avg_loss)  # ensure it's a valid number

# -------------------- Tests for evaluate method -------------------- #
def test_evaluate(trainer):
    mock_split_ds = MagicMock(spec=WindowDataset)
    mock_loader = MagicMock()
    win1 = (torch.randn(1, 120, 251), torch.randn(1, 60, 50))
    win2 = (torch.randn(1, 120, 251), torch.randn(1, 60, 50))
    mock_loader.__iter__.return_value = [win1, win2]
    with patch('src.training.train_nn.DataLoader', return_value=mock_loader):
        trainer.evaluate(mock_split_ds)
    assert len(trainer.eval_losses) == 2
    assert len(trainer.eval_alloc_weights) == 2
    assert isinstance(trainer.avg_eval_loss, float)

# -------------------- Tests for get_eval_alloc_weights -------------------- #
def test_get_eval_alloc_weights(trainer):
    trainer.eval_alloc_weights = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.2, 0.8]])]
    result = trainer.get_eval_alloc_weights()
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.allclose(result, [[0.1, 0.9], [0.2, 0.8]])

def test_get_eval_alloc_weights_no_weights(trainer, capsys):
    trainer.eval_alloc_weights = []
    result = trainer.get_eval_alloc_weights()
    assert result is None
    captured = capsys.readouterr()
    assert "Model must be trained and validated." in captured.out

# -------------------- Tests get_best_losses and get_best_epoch -------------------- #
def test_get_best_losses(trainer):
    trainer.best_train_loss = 0.1
    trainer.best_val_loss = 0.2
    train_loss, val_loss = trainer.get_best_losses()
    assert train_loss == 0.1
    assert val_loss == 0.2

def test_get_best_epoch(trainer):
    trainer.best_epoch = 42
    assert trainer.get_best_epoch() == 42

# -------------------- Tests for device_cleanup -------------------- #
def test_device_cleanup(trainer, test_device_str):
    if test_device_str == 'cuda':
        with patch('torch.cuda.empty_cache') as mock_empty, \
             patch('torch.cuda.ipc_collect') as mock_ipc:
            trainer.device = torch.device('cuda')
            trainer.device_cleanup()
            mock_empty.assert_called_once()
            mock_ipc.assert_called_once()
    elif test_device_str == 'mps':
        with patch('torch.mps.empty_cache') as mock_empty:
            trainer.device = torch.device('mps')
            trainer.device_cleanup()
            mock_empty.assert_called_once()
    else:
        # CPU: no cleanup functions to call; just ensure no error
        trainer.device = torch.device('cpu')
        trainer.device_cleanup()  # should do nothing

def test_device_cleanup_cpu(trainer, capsys):
    trainer.device = torch.device('cpu')
    trainer.device_cleanup()  # should not error
    captured = capsys.readouterr()
    assert "MPS cleanup not available" not in captured.out

# =================== Fixtures for Walker ==================== #
@pytest.fixture
def mock_reshaper():
    reshaper = MagicMock(spec=Reshaper)
    reshaper.in_size = 120
    reshaper.out_size = 60
    # Mock reshape method: returns (X_train, y_train, None)
    reshaper.reshape.return_value = (np.zeros((10, 120, 251)), np.zeros((10, 60, 50)), None)
    # Mock transform_one_window: returns (1, 120, 251)
    reshaper.transform_one_window.return_value = np.zeros((120, 251))
    return reshaper

@pytest.fixture
def walker_hparams():
    return {
        'model': {'hidden_size': 32},
        'optimizer': {'lr': 0.001},
        'train': {'epochs': 5},
        'loss': {'lambda': 0.1}
    }

@pytest.fixture
def walker(mock_reshaper, walker_hparams):
    return Walker(
        num_steps=2,
        model_name='TestModel',
        model_cls=MagicMock(),
        loss_name='test_loss',
        loss_func=MagicMock(),
        hparams=walker_hparams,
        torch_device='cpu',
        reshaper=mock_reshaper,
        seed=42
    )

# -------------------- Tests for __init__ -------------------- #
def test_walker_init(walker, mock_reshaper, walker_hparams):
    assert walker.num_steps == 2
    assert walker.model_name == 'TestModel'
    assert walker.model_cls is not None
    assert walker.loss_name == 'test_loss'
    assert walker.loss_func is not None
    assert walker.hparams == walker_hparams
    assert walker.torch_device == 'cpu'
    assert walker.reshaper is mock_reshaper
    assert walker.seed == 42
    assert walker.in_size == 120
    assert walker.stride == 60
    assert walker.alloc_weights == []
    assert walker.train_eval_losses == []

# -------------------- Tests for _reshape_step_data -------------------- #
def test_reshape_step_data(walker, mock_reshaper):
    walk_train = np.random.randn(100, 251)
    walk_rets_train = np.random.randn(100, 50)
    walk_rets_val = np.random.randn(60, 50)

    X_train, y_train, X_val, y_val = walker._reshape_step_data(walk_train, walk_rets_train, walk_rets_val)

    # Check that reshaper.reshape was called with correct args
    mock_reshaper.reshape.assert_called_once_with(walk_train, walk_rets_train)
    call_args = mock_reshaper.transform_one_window.call_args[0][0]
    assert call_args.shape == walk_train.shape  # should be (100, 251)
    # Check output shapes
    assert X_val.shape == (1, 120, 251)
    assert y_val.shape == (1, 60, 50)

# -------------------- Tests for _train_eval_helper (with mocking Trainer) -------------------- #
@patch('src.training.train_nn.Trainer')
def test_train_eval_helper(mock_trainer_class, walker):
    # Setup mocks
    mock_trainer = MagicMock()
    mock_trainer.get_eval_alloc_weights.return_value = np.array([[0.5, 0.5]])
    mock_trainer.train_losses = [0.1, 0.2]
    mock_trainer.eval_losses = [0.3]
    mock_trainer_class.return_value = mock_trainer

    train_ds = MagicMock()
    infer_ds = MagicMock()
    X_train_shape = torch.Size([10, 120, 251])
    y_train_shape = torch.Size([10, 60, 50])

    alloc_weights, train_eval_losses = walker._train_eval_helper(train_ds, infer_ds, X_train_shape, y_train_shape)

    # Check Trainer instantiated with correct args
    mock_trainer_class.assert_called_once()
    call_args = mock_trainer_class.call_args[1]  # keyword arguments
    assert call_args['model'] == walker.model_cls
    assert call_args['loss'] == walker.loss_func
    assert call_args['model_hparams'] == walker.hparams['model']
    assert call_args['optimizer_hparams'] == walker.hparams['optimizer']
    assert call_args['train_hparams'] == walker.hparams['train']
    assert call_args['in_size'] == 251
    assert call_args['num_stocks'] == 50
    assert call_args['max_seq_len'] == 120
    assert call_args['loss_hparams'] == walker.hparams['loss']
    assert call_args['device'] == 'cpu'

    # Check methods called
    mock_trainer.train.assert_called_once_with(train_ds)
    mock_trainer.evaluate.assert_called_once_with(infer_ds)
    mock_trainer.device_cleanup.assert_called_once()

    # Check return values
    assert np.array_equal(alloc_weights, np.array([[0.5, 0.5]]))
    assert train_eval_losses == {'train': [0.1, 0.2], 'eval': [0.3]}

# -------------------- Tests for walk_1_model -------------------- #
@patch('src.training.train_nn.calc_current_idxs')
@patch('src.training.train_nn.set_seed')
def test_walk_1_model(mock_set_seed, mock_calc_idx, walker):
    # Mock calc_current_idxs to return (start, end)
    # For step 0: current_start=0, current_end=60
    # For step 1: current_start=60, current_end=120
    mock_calc_idx.side_effect = [(0, 60), (60, 120)]

    # Prepare input data
    train = np.random.randn(200, 251)
    rets_train = np.random.randn(200, 50)
    val = np.random.randn(120, 251)
    rets_val = np.random.randn(120, 50)

    # Mock _train_eval_helper to return fixed values
    walker._train_eval_helper = MagicMock()
    walker._train_eval_helper.return_value = (np.array([[0.6, 0.4]]), {'train': [0.1], 'eval': [0.2]})

    # Run walk_1_model
    result = walker.walk_1_model(train, rets_train, val, rets_val)

    # Check seed set
    mock_set_seed.assert_called_once_with(42)

    # Check number of steps
    assert walker._train_eval_helper.call_count == 2

    # For first step, walk_train and walk_rets_train should be initial copies
    # For second step, after first step, walk_train should have been concatenated with first walk_val
    calls = walker._train_eval_helper.call_args_list
    # First call: train_ds created from initial walk_train (copy of train)
    # Second call: train_ds created from concatenated walk_train (train + first walk_val slice)
    assert isinstance(calls[0][0][0], WindowDataset)
    assert isinstance(calls[1][0][0], WindowDataset)

    # Check that alloc_weights and train_eval_losses are stored and returned
    assert walker.alloc_weights.shape == (2, 2)  # 2 steps, 2 stocks
    assert len(walker.train_eval_losses) == 2
    np.testing.assert_array_equal(result, walker.alloc_weights)

def test_walk_1_model_no_seed():
    # Create a mock reshaper
    mock_reshaper = MagicMock()
    mock_reshaper.in_size = 120
    mock_reshaper.out_size = 60
    # For reshape, return a tuple of three arrays: X_train, y_train, None
    # Use dummy shapes (10 samples, 120 time steps, 251 features; 10 samples, 60 time steps, 50 assets)
    mock_reshaper.reshape.return_value = (
        np.zeros((10, 180, 251)),
        np.zeros((10, 60, 50)),
        None
    )
    # For transform_one_window, return an array of shape (120, 251)
    mock_reshaper.transform_one_window.return_value = np.zeros((180, 251))

    walker_no_seed = Walker(
        num_steps=1,
        model_name='Test',
        model_cls=MagicMock(),
        loss_name='loss',
        loss_func=MagicMock(),
        hparams={},
        torch_device='cpu',
        reshaper=mock_reshaper,
        seed=None
    )
    with patch('src.training.train_nn.calc_current_idxs') as mock_calc_idx, \
         patch('src.training.train_nn.set_seed') as mock_set_seed:
        mock_calc_idx.return_value = (0, 60)
        # Mock _train_eval_helper to avoid actual training
        walker_no_seed._train_eval_helper = MagicMock(return_value=(np.array([[0.5, 0.5]]), {}))
        # Run the walk
        walker_no_seed.walk_1_model(
            np.zeros((200, 251)), np.zeros((200, 50)),
            np.zeros((180, 251)), np.zeros((180, 50))
        )
        mock_set_seed.assert_not_called()

# -------------------- Test get_train_eval_losses -------------------- #
def test_get_train_eval_losses(walker):
    walker.train_eval_losses = [{'train': [1,2], 'eval': [3,4]}]
    assert walker.get_train_eval_losses() == [{'train': [1,2], 'eval': [3,4]}]

# ==================== Fixtures for Tuner ==================== #
@pytest.fixture
def sample_tune_metric():
    return {
        'sharpe': MetricModel(func=MagicMock(return_value=0.5), sign='+'),
        'cvar': MetricModel(func=MagicMock(return_value=0.1), sign='-')
    }

@pytest.fixture
def n_steps():
    return 10

@pytest.fixture
def sample_bench_rets(n_steps):
    # Shape: (n_steps, 60) - must match number of steps
    return np.random.randn(n_steps, 60)

@pytest.fixture
def sample_eval_winds(n_steps):
    # Shape: (n_steps, 60, 50)
    return np.random.randn(n_steps, 60, 50)

@pytest.fixture
def sample_ba_eval(n_steps):
    # Shape: (n_steps, 50)
    return np.random.randn(n_steps, 50)

@pytest.fixture
def tuner(sample_tune_metric, sample_bench_rets, sample_eval_winds, mock_reshaper, n_steps):
    return Tuner(
        tune_metric=sample_tune_metric,
        tune_bench_rets=sample_bench_rets,
        eval_winds=sample_eval_winds,
        n_steps=n_steps,
        n_trials=5,
        n_warmup_steps=2,
        n_jobs=1,
        reshaper=mock_reshaper,
        torch_device='cpu',
        ba_eval_winds=None
    )

# -------------------- Tests for __init__ -------------------- #
def test_init_validates_tune_metric(sample_tune_metric, sample_bench_rets, sample_eval_winds, mock_reshaper):
    tuner = Tuner(
        tune_metric=sample_tune_metric,
        tune_bench_rets=sample_bench_rets,
        eval_winds=sample_eval_winds,
        n_steps=5,
        n_trials=10,
        n_warmup_steps=1,
        n_jobs=2,
        reshaper=mock_reshaper,
        torch_device='cuda',
        ba_eval_winds=None
    )
    assert tuner.tune_metric == sample_tune_metric
    assert tuner.n_startup_trials == max(int(10 * 0.3), 20)  # 20 because min is 20
    assert tuner.direction == 'maximize'

def test_init_invalid_tune_metric_raises():
    with pytest.raises(Exception):  # TypeAdapter will raise validation error
        Tuner(
            tune_metric={'invalid': {}},
            tune_bench_rets=np.zeros(1),
            eval_winds=np.zeros((1,1,1)),
            n_steps=1,
            n_trials=1,
            n_warmup_steps=1,
            n_jobs=1,
            reshaper=MagicMock(),
            torch_device='cpu',
            ba_eval_winds=None
        )

# -------------------- _calc_pf_metrics_for_seed -------------------- #
# @patch('src.training.train_nn.Evaluator')
# def test_calc_pf_metrics_for_seed(mock_evaluator_class, tuner):
#     mock_evaluator = MagicMock()
#     mock_evaluator.calc_metric_performance.side_effect = [
#         MagicMock(item=MagicMock(return_value=0.5)),  # for sharpe
#         MagicMock(item=MagicMock(return_value=0.2))   # for cvar
#     ]
#     mock_evaluator_class.return_value = mock_evaluator

#     alloc_weights = np.random.randn(10, 50)
#     y_val = np.random.randn(100, 60, 50)  # dummy
#     result = tuner._calc_pf_metrics_for_seed('model', 'loss', 42, alloc_weights, y_val)
#     assert result == {'sharpe': 0.5, 'cvar': 0.2}
#     mock_evaluator.calc_pf_daily_rets.assert_called_once_with(alloc_weights, 'model-loss-42')
#     assert mock_evaluator.calc_metric_performance.call_count == 2

# -------------------- _calc_composite_scores -------------------- #
@patch('src.training.train_nn.Evaluator')
def test_calc_composite_scores(mock_evaluator_class, tuner):
    # Setup mock evaluator
    mock_evaluator = MagicMock()
    # get_rets_for_one returns (n_steps, 60) – correct shape
    mock_evaluator.get_rets_for_one.return_value = np.random.randn(tuner.n_steps, 60)
    import pandas as pd
    df_sharpe = pd.DataFrame({'model': [0.5] * tuner.n_steps})
    df_cvar = pd.DataFrame({'model': [0.1] * tuner.n_steps})
    mock_evaluator.calc_metric_performance.side_effect = [df_sharpe, df_cvar]
    mock_evaluator_class.return_value = mock_evaluator

    alloc_weights = np.random.randn(tuner.n_steps, 50)  # (n_steps, 50)
    result = tuner._calc_composite_scores('model-loss', alloc_weights)

    # Expected composite: sharpe (sign +) + (-cvar) because cvar sign is '-'
    # So per step: 0.5 - 0.1 = 0.4
    expected = np.full(tuner.n_steps, 0.4)
    np.testing.assert_array_equal(result, expected)
    mock_evaluator.calc_pf_daily_rets.assert_called_once_with(alloc_weights, 'model-loss')
    mock_evaluator.get_rets_for_one.assert_called_once_with('model-loss')
    mock_evaluator.update_rets_for_one.assert_called_once()
    assert mock_evaluator.calc_metric_performance.call_count == 2

# -------------------- calc_hinge_penalty -------------------- #
def test_calc_hinge_penalty_positive_gap(tuner):
    train_losses = [0.1, 0.2]
    val_losses = [0.2, 0.3]
    penalty = tuner.calc_hinge_penalty(train_losses, val_losses)
    # avg_train = 0.15, avg_val = 0.25, raw_gap = (0.25-0.15)/0.15 = 0.6666666666666666
    expected = 0.6666666666666666
    assert pytest.approx(penalty, 1e-6) == expected

def test_calc_hinge_penalty_negative_gap(tuner):
    train_losses = [0.2, 0.3]
    val_losses = [0.1, 0.2]
    penalty = tuner.calc_hinge_penalty(train_losses, val_losses)
    assert penalty == 0.0

def test_calc_hinge_penalty_zero_avg_train(tuner):
    train_losses = [0.0, 0.0]
    val_losses = [0.0, 0.0]
    penalty = tuner.calc_hinge_penalty(train_losses, val_losses, eps=1e-9)
    assert penalty == 0.0

# -------------------- _calc_tuning_objective -------------------- #
def test_calc_tuning_objective(tuner):
    composite_scores = [1.0, 2.0, 3.0]
    train_losses = [0.1, 0.1, 0.1]
    val_losses = [0.2, 0.2, 0.2]
    objective = tuner._calc_tuning_objective(composite_scores, train_losses, val_losses)
    # mean_score = 2.0, std = 1.0, n=3, t_val for 95% df=2 = 2.92, margin = 2.92*1/√3≈1.686
    # base_score = 2.0 - 1.686 = 0.314
    # gap_penalty = (0.2-0.1)/0.1 = 1.0
    # final = 0.314 - 1.0 = -0.686
    assert pytest.approx(objective, abs=0.01) == -0.686

def test_calc_tuning_objective_less_than_2_seeds(tuner):
    composite_scores = [1.0]
    train_losses = [0.1]
    val_losses = [0.2]
    objective = tuner._calc_tuning_objective(composite_scores, train_losses, val_losses)
    # Use approx to handle floating point
    assert pytest.approx(objective, abs=1e-8) == 0.0

# -------------------- _calc_tuning_objective_no_gap -------------------- #
def test_calc_tuning_objective_no_gap(tuner):
    composite_scores = np.array([1.0, 2.0, 3.0])
    objective = tuner._calc_tuning_objective_no_gap(composite_scores)
    assert pytest.approx(objective, abs=0.01) == 0.314

def test_calc_tuning_objective_no_gap_less_than_2_seeds(tuner):
    composite_scores = np.array([1.0])
    objective = tuner._calc_tuning_objective_no_gap(composite_scores)
    assert objective == 1.0

# -------------------- _run_tuning_study (with mocking) -------------------- #
@patch('src.training.train_nn.optuna.create_study')
@patch('src.training.train_nn.Walker')
def test_run_tuning_study(mock_walker_class, mock_create_study, tuner):
    # Create a dummy trial to simulate Optuna's trial object
    class DummyTrial:
        number = 0
        def suggest_float(self, name, low, high, log):
            return (low + high) / 2
        def suggest_int(self, name, low, high):
            return (low + high) // 2
        def suggest_categorical(self, name, choices):
            return choices[0]

    # Mock Walker
    mock_walker = MagicMock()
    mock_walker.walk_1_model.return_value = np.random.randn(tuner.n_steps, 50)
    mock_walker_class.return_value = mock_walker

    # Mock Optuna study
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study

    # Define a side effect for study.optimize to call the objective once
    def optimize_side_effect(objective_func, n_trials, n_jobs):
        # Call the objective with a dummy trial
        objective_func(DummyTrial())
    mock_study.optimize.side_effect = optimize_side_effect

    # Mock _calc_composite_scores and _calc_tuning_objective_no_gap
    tuner._calc_composite_scores = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
    tuner._calc_tuning_objective_no_gap = MagicMock(return_value=0.5)

    # Provide a valid model_cfg with required keys
    model_cfg = {
        'model': {},
        'optimizer': {},
        'train': {},
        'scheduler': None,
        'tuning': {'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True}}
    }
    loss_cfg = {}

    result = tuner._run_tuning_study(
        model_name='TestModel',
        model_class=MagicMock(),
        loss_name='test_loss',
        loss_func=MagicMock(),
        train_data=np.zeros((100, 251)),
        rets_train=np.zeros((100, 50)),
        val_data=np.zeros((100, 251)),
        rets_val=np.zeros((100, 50)),
        model_cfg=model_cfg,
        loss_cfg=loss_cfg
    )

    assert result == mock_study
    mock_create_study.assert_called_once_with(direction='maximize', study_name='TestModel-test_loss')
    mock_study.optimize.assert_called_once()
    # Now Walker should have been instantiated
    mock_walker_class.assert_called_once()
    # Also check that the mocks were called
    tuner._calc_composite_scores.assert_called_once()
    tuner._calc_tuning_objective_no_gap.assert_called_once()

# -------------------- set_tuning_direction -------------------- #
def test_set_tuning_direction(tuner):
    tuner.set_tuning_direction('minimize')
    assert tuner.direction == 'minimize'


# ==================== Fixtures for CandidatesGrid ==================== #
@pytest.fixture
def sample_model_lib():
    class DummyModel:
        pass
    return {
        'transformer': {
            'ModelA': DummyModel,
            'ModelB': DummyModel,
        },
        'lstm': {
            'ModelC': DummyModel,
        }
    }

@pytest.fixture
def sample_loss_lib():
    def dummy_loss(**kwargs):
        return 0.0
    return {
        'custom': {'__default__': {'custom_loss_1': dummy_loss, 'custom_loss_2': dummy_loss}},
        'objectives': {'__default__': {'sharpe': dummy_loss, 'sortino': dummy_loss}},
        'regularizers': {'structural': {'hhi': dummy_loss}, 'tail_risk': {'cvar': dummy_loss}}
    }

@pytest.fixture
def sample_hparams_config():
    return {
        'nn_models': {
            'ModelA': {
                'model': {'hidden_size': 32},
                'optimizer': {'lr': 0.001},
                'train': {'epochs': 5},
                'scheduler': None,
                'tuning': {}
            },
            'ModelB': {
                'model': {'hidden_size': 64},
                'optimizer': {'lr': 0.0001},
                'train': {'epochs': 10},
                'scheduler': None,
                'tuning': {}
            }
        },
        'losses': {
            'custom_loss_1': {'lambdas': {}},
            'sharpe': {'lambdas': {}}
        },
        'rolling_windows': {
            'in_size': 120,
            'out_size': 60,
            'stride': 60
        },
        'tuner': {
            'n_tuning_trials': 10,
            'n_warmup_steps': 2,
            'n_jobs': 1
        }
    }

@pytest.fixture
def sample_common_features():
    return ['sprtrn']

@pytest.fixture
def sample_train_data():
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    features = pd.DataFrame(np.random.randn(200, 251), index=dates, columns=[f'feat_{i}' for i in range(251)])
    returns = pd.DataFrame(np.random.randn(200, 50), index=dates, columns=[f'stock_{i}' for i in range(50)])
    return features, returns

@pytest.fixture
def sample_val_data():
    dates = pd.date_range('2020-07-20', periods=120, freq='D')
    features = pd.DataFrame(np.random.randn(120, 251), index=dates, columns=[f'feat_{i}' for i in range(251)])
    returns = pd.DataFrame(np.random.randn(120, 50), index=dates, columns=[f'stock_{i}' for i in range(50)])
    return features, returns

@pytest.fixture
def candidates_grid(sample_model_lib, sample_loss_lib, sample_hparams_config, sample_common_features):
    return CandidatesGrid(
        model_lib=sample_model_lib,
        loss_lib=sample_loss_lib,
        hparams_config=sample_hparams_config,
        num_steps=2,
        common_features=sample_common_features,
        torch_device='cpu',
        loss_mode='custom',
        tune=False,
        tuner_eval_items=None,
        mpi=False,
        temp_dir=None,
        enable_diagnostics=False
    )

# -------------------- Tests for __init__ and inherited methods -------------------- #
def test_init(candidates_grid, sample_model_lib, sample_loss_lib, sample_hparams_config, sample_common_features):
    assert candidates_grid.model_lib == sample_model_lib
    assert candidates_grid.loss_lib == sample_loss_lib
    assert candidates_grid.hparams_config == sample_hparams_config
    assert candidates_grid.num_steps == 2
    assert candidates_grid.torch_device == 'cpu'
    assert candidates_grid.loss_mode == 'custom'
    assert candidates_grid.tune is False
    assert candidates_grid.tuner is None
    assert candidates_grid.mpi is False
    assert candidates_grid.temp_dir is None
    assert isinstance(candidates_grid.reshaper, Reshaper)

def test_init_invalid_loss_mode(sample_model_lib, sample_loss_lib, sample_hparams_config, sample_common_features):
    with pytest.raises(ValueError, match="Incorrect Loss Mode"):
        CandidatesGrid(
            model_lib=sample_model_lib,
            loss_lib=sample_loss_lib,
            hparams_config=sample_hparams_config,
            num_steps=2,
            common_features=sample_common_features,
            torch_device='cpu',
            loss_mode='invalid',
            tune=False,
            tuner_eval_items=None,
            mpi=False,
            temp_dir=None,
            enable_diagnostics=False
        )

# -------------------- Tests for _tuner_setup -------------------- #
def test_tuner_setup_with_tune(candidates_grid):
    candidates_grid.tune = True
    # Create proper MetricModel objects
    metric_model = MetricModel(func=MagicMock(), sign='+')
    tuner_eval_items = {
        'metric': {'sharpe': metric_model, 'cvar': metric_model},
        'bench_rets': np.random.randn(2, 60),
        'eval_winds': np.random.randn(2, 60, 50),
        'ba_eval_winds': np.random.randn(2, 50)
    }
    tuner = candidates_grid._tuner_setup(tuner_eval_items)
    assert tuner is not None

def test_tuner_setup_without_tune(candidates_grid):
    candidates_grid.tune = False
    tuner = candidates_grid._tuner_setup(None)
    assert tuner is None

def test_tuner_setup_missing_metric_raises(candidates_grid):
    candidates_grid.tune = True
    tuner_eval_items = {'bench_rets': np.zeros((2,60)), 'eval_winds': np.zeros((2,60,50))}
    with pytest.raises(ValueError, match="Provide Tuning metric if tune = True"):
        candidates_grid._tuner_setup(tuner_eval_items)

def test_tuner_setup_missing_bench_rets_raises(candidates_grid):
    candidates_grid.tune = True
    # Need to provide metric to avoid earlier check
    metric_model = MetricModel(func=MagicMock(), sign='+')
    tuner_eval_items = {'metric': {'sharpe': metric_model}, 'eval_winds': np.zeros((2,60,50))}
    with pytest.raises(ValueError, match="Provide tuning benchmark"):
        candidates_grid._tuner_setup(tuner_eval_items)

def test_tuner_setup_missing_eval_winds_raises(candidates_grid):
    candidates_grid.tune = True
    metric_model = MetricModel(func=MagicMock(), sign='+')
    tuner_eval_items = {'metric': {'sharpe': metric_model}, 'bench_rets': np.zeros((2,60))}
    with pytest.raises(ValueError, match="Provide evaluation windows"):
        candidates_grid._tuner_setup(tuner_eval_items)

# -------------------- _build_losses_to_use -------------------- #
def test_build_losses_to_use_custom_mode(candidates_grid):
    candidates_grid.loss_mode = 'custom'
    losses = candidates_grid._build_losses_to_use()
    assert set(losses.keys()) == {'custom_loss_1', 'custom_loss_2'}

def test_build_losses_to_use_all_mode(candidates_grid):
    candidates_grid.loss_mode = 'all'
    losses = candidates_grid._build_losses_to_use()
    expected = {'custom_loss_1', 'custom_loss_2', 'sharpe', 'sortino'}
    assert set(losses.keys()) == expected

# -------------------- _build_combos -------------------- #
def test_build_combos_all_mode(candidates_grid):
    losses_to_use = {'loss1': MagicMock(), 'loss2': MagicMock()}
    combos = candidates_grid._build_combos(losses_to_use, 'all')
    assert len(combos) == 6
    for combo in combos:
        assert len(combo) == 4
        assert combo[0] in losses_to_use
        assert isinstance(combo[2], str)

def test_build_combos_one_model_mode(candidates_grid):
    losses_to_use = {'loss1': MagicMock()}
    model_name = 'ModelA'
    model_class = MagicMock()
    combos = candidates_grid._build_combos(losses_to_use, 'one_model', model_name, model_class)
    assert len(combos) == 1
    assert combos[0][0] == 'loss1'
    assert combos[0][2] == 'ModelA'
    assert combos[0][3] == model_class

def test_build_combos_invalid_mode(candidates_grid):
    with pytest.raises(ValueError, match="Incorrect grid mode"):
        candidates_grid._build_combos({}, 'invalid')

# -------------------- _walker_helper (with mocks) -------------------- #
@patch('src.training.train_nn.reformat_hparams')
@patch('src.training.train_nn.Walker')
def test_walker_helper_without_tune(mock_walker_class, mock_reformat, candidates_grid):
    mock_reformat.return_value = {'model': {}, 'optimizer': {}, 'train': {}, 'loss': {}}
    mock_walker = MagicMock()
    mock_walker.walk_1_model.return_value = np.zeros((2,50))
    mock_walker.get_train_eval_losses.return_value = {'train': [1,2], 'eval': [3,4]}
    mock_walker_class.return_value = mock_walker

    model_name = 'ModelA'
    loss_name = 'custom_loss_1'
    train_data = np.zeros((200,251))
    rets_train = np.zeros((200,50))
    val_data = np.zeros((120,251))
    rets_val = np.zeros((120,50))

    alloc_weights, losses, opt_hparams = candidates_grid._walker_helper(
        model_name, MagicMock(), loss_name, MagicMock(),
        train_data, rets_train, val_data, rets_val
    )
    mock_reformat.assert_called_once()
    mock_walker_class.assert_called_once()
    mock_walker.walk_1_model.assert_called_once_with(train_data, rets_train, val_data, rets_val)
    assert alloc_weights.shape == (2,50)
    assert losses == {'train': [1,2], 'eval': [3,4]}
    assert opt_hparams is None

@patch('src.training.train_nn.reformat_hparams')
@patch('src.training.train_nn.Walker')
@patch('src.training.train_nn.Tuner')
def test_walker_helper_with_tune(mock_tuner_class, mock_walker_class, mock_reformat, candidates_grid):
    candidates_grid.tune = True
    # Create a mock tuner instance
    mock_tuner = MagicMock()
    mock_tuner_class.return_value = mock_tuner
    # Set the tuner attribute on the instance
    candidates_grid.tuner = mock_tuner

    mock_study = MagicMock()
    mock_study.best_params = {'lr': 0.001}
    mock_tuner._run_tuning_study.return_value = mock_study
    mock_reformat.return_value = {'model': {}, 'optimizer': {}, 'train': {}, 'loss': {}}
    mock_walker = MagicMock()
    mock_walker.walk_1_model.return_value = np.zeros((2,50))
    mock_walker.get_train_eval_losses.return_value = {}
    mock_walker_class.return_value = mock_walker

    model_name = 'ModelA'
    loss_name = 'custom_loss_1'
    train_data = np.zeros((200,251))
    rets_train = np.zeros((200,50))
    val_data = np.zeros((120,251))
    rets_val = np.zeros((120,50))

    alloc_weights, losses, opt_hparams = candidates_grid._walker_helper(
        model_name, MagicMock(), loss_name, MagicMock(),
        train_data, rets_train, val_data, rets_val
    )
    mock_tuner._run_tuning_study.assert_called_once()
    mock_walker_class.assert_called_once()
    assert opt_hparams is not None

# -------------------- train_eval_one_model (non-MPI) -------------------- #
def test_train_eval_one_model_non_mpi(candidates_grid, sample_train_data, sample_val_data):
    # Set up mocks
    candidates_grid.reshaper.extract_features = MagicMock()
    candidates_grid._data_check = MagicMock()
    candidates_grid._trained_check = MagicMock()
    candidates_grid._walker_helper = MagicMock(return_value=(np.zeros((2,50)), {'train': [1,2], 'eval': [3,4]}, {'model': {}}))

    train_features, train_rets = sample_train_data
    val_features, val_rets = sample_val_data

    result = candidates_grid.train_eval_one_model(
        model_name='ModelA',
        train_data=train_features,
        rets_train=train_rets,
        val_data=val_features,
        rets_val=val_rets
    )
    # 2 custom losses
    assert candidates_grid._walker_helper.call_count == 2
    assert len(candidates_grid.all_alloc_weights) == 2
    assert result is candidates_grid.all_alloc_weights

# -------------------- train_eval_one -------------------- #
def test_train_eval_one(candidates_grid, sample_train_data, sample_val_data):
    candidates_grid.reshaper.extract_features = MagicMock()
    candidates_grid._data_check = MagicMock()
    candidates_grid._trained_check = MagicMock()
    candidates_grid._walker_helper = MagicMock(return_value=(np.zeros((2,50)), {'train': [1,2], 'eval': [3,4]}, {'model': {}}))

    train_features, train_rets = sample_train_data
    val_features, val_rets = sample_val_data

    result = candidates_grid.train_eval_one(
        model_name='ModelA',
        loss_name='custom_loss_1',
        train_data=train_features,
        rets_train=train_rets,
        val_data=val_features,
        rets_val=val_rets
    )
    candidates_grid._walker_helper.assert_called_once()
    assert len(candidates_grid.all_alloc_weights) == 1
    assert result is candidates_grid.all_alloc_weights

# -------------------- get_optimized_hparams and get_train_val_losses -------------------- #
def test_get_optimized_hparams(candidates_grid):
    candidates_grid.optimized_hparams = {'modelA-loss1': {'lr': 0.001}}
    assert candidates_grid.get_optimized_hparams() == {'modelA-loss1': {'lr': 0.001}}

def test_get_train_val_losses(candidates_grid):
    candidates_grid.train_eval_losses = {
        'modelA-loss1': [{'train': [0.1,0.2], 'eval': [0.3]}, {'train': [0.4], 'eval': [0.5]}]
    }
    result = candidates_grid.get_train_val_losses()
    expected = {
        'modelA-loss1': {
            'train': [[0.1,0.2], [0.4]],
            'eval': [0.3, 0.5]
        }
    }
    assert result == expected

def test_get_train_val_losses_not_trained_raises(candidates_grid):
    candidates_grid.train_eval_losses = {}
    with pytest.raises(RuntimeError, match="Models not trained yet"):
        candidates_grid.get_train_val_losses()

# -------------------- Utility methods -------------------- #
def test_search_model_found(candidates_grid):
    model_class = candidates_grid._search_model('ModelA')
    assert model_class is not None
    assert model_class.__name__ == 'DummyModel'

def test_search_model_not_found(candidates_grid):
    model_class = candidates_grid._search_model('NonExistent')
    assert model_class is None

def test_search_loss_func_found(candidates_grid):
    loss_func = candidates_grid._search_loss_func('custom_loss_1')
    assert loss_func is not None

def test_search_loss_func_not_found(candidates_grid):
    loss_func = candidates_grid._search_loss_func('missing')
    assert loss_func is None

def test_convert_datasets_to_np(candidates_grid):
    df1 = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    df2 = pd.DataFrame({'c': [5,6], 'd': [7,8]})
    df3 = pd.DataFrame({'e': [9,10]})
    df4 = pd.DataFrame({'f': [11,12]})
    arr1, arr2, arr3, arr4 = candidates_grid._convert_datasets_to_np(df1, df2, df3, df4)
    assert isinstance(arr1, np.ndarray)
    assert arr1.shape == (2,2)
    assert arr2.shape == (2,2)
    assert arr3.shape == (2,1)
    assert arr4.shape == (2,1)

def test_data_check_valid(candidates_grid):
    train = pd.DataFrame(np.zeros((120, 251)))
    rets_val = pd.DataFrame(np.zeros((120, 50)))  # 120 days -> 2 steps (60 each)
    candidates_grid._data_check(train, rets_val)  # should not raise

def test_data_check_invalid_steps(candidates_grid):
    train = pd.DataFrame(np.zeros((120, 251)))
    rets_val = pd.DataFrame(np.zeros((100, 50)))  # 100/60 = 1 step, but num_steps=2
    with pytest.raises(ValueError, match="does not match actual number of full windows"):
        candidates_grid._data_check(train, rets_val)

def test_trained_check_empty(candidates_grid):
    candidates_grid.all_alloc_weights = {}
    candidates_grid._trained_check()

def test_trained_check_not_empty(candidates_grid):
    candidates_grid.all_alloc_weights = {'some': np.zeros(1)}
    with pytest.raises(RuntimeError, match="Allocation weights already predicted"):
        candidates_grid._trained_check()

# -------------------- MPI methods -------------------- #
def test_select_ranks_combos():
    combos = list(range(10))
    result = CandidatesGrid._select_ranks_combos(combos, global_rank=1, size=3)
    assert result == [4,5,6]

def test_mpi_setup_check():
    CandidatesGrid._mpi_setup_check([1,2,3])
    with pytest.raises(ValueError, match="All necessary MPI values must be provided"):
        CandidatesGrid._mpi_setup_check([1, None, 3])

# -------------------- _merge_all_results -------------------- #
@patch('src.training.train_nn.load_pickle_temp')
@patch('src.training.train_nn.delete_file')
def test_merge_all_results(mock_delete, mock_load, candidates_grid, tmp_path):
    candidates_grid.temp_dir = tmp_path
    size = 2
    temps_wts_prefix = 'test_weights'
    temp_losses_prefix = 'test_losses'
    temp_hparams_prefix = 'test_hparams'
    mock_load.side_effect = [
        {'modelA-loss1': np.zeros((2,50))},
        {'modelA-loss2': np.zeros((2,50))},
        {'modelA-loss1': [{'train':[1]}]},
        {'modelA-loss2': [{'train':[2]}]},
        {'modelA-loss1': {'lr':0.001}},
        {'modelA-loss2': {'lr':0.002}}
    ]
    candidates_grid._merge_all_results(size, temps_wts_prefix, temp_losses_prefix, temp_hparams_prefix)
    assert len(candidates_grid.all_alloc_weights) == 2
    assert len(candidates_grid.train_eval_losses) == 2
    assert len(candidates_grid.optimized_hparams) == 2
    assert mock_delete.call_count == 6