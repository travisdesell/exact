import pytest
import torch
import numpy as np
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.data_processing.dataset import WindowDataset, Reshaper
from src.training.train_nn import (
    Trainer,
    Walker
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

# ==================== Fixtures ==================== #
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