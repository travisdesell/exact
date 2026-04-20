import pytest
import torch
import numpy as np
import torch.nn as nn
from src.training.train_nn import Trainer
from unittest.mock import MagicMock, patch
from src.data_processing.dataset import WindowDataset

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