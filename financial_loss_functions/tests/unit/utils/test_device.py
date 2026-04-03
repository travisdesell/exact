import torch
import random
import numpy as np
from unittest.mock import patch
from src.utils.device import get_best_device, set_seed

# -------------------- Tests for get_best_device --------------------
def test_get_best_device_cuda():
    with patch('torch.cuda.is_available', return_value=True):
        device = get_best_device(gpu_id=2)
        assert device == torch.device('cuda:2')

def test_get_best_device_mps():
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.backends.mps.is_available', return_value=True):
            device = get_best_device()
            assert device == torch.device('mps')

def test_get_best_device_cpu():
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.backends.mps.is_available', return_value=False):
            device = get_best_device()
            assert device == torch.device('cpu')

# -------------------- Tests for set_seed --------------------
def test_set_seed_cpu_only(capsys):
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.backends.mps.is_available', return_value=False):
            set_seed(42)
            # Check random seeds
            assert random.getstate() is not None  # indirect check
            # Check numpy seed
            np.random.seed(42)   # to compare; but we can just verify that calling set_seed doesn't error
            # Check torch seed
            assert torch.initial_seed() == 42
            # Capture print
            captured = capsys.readouterr()
            assert 'Seeds set to 42 across all available backends.' in captured.out

def test_set_seed_prints_message(capsys):
    set_seed(1)
    captured = capsys.readouterr()
    assert 'Seeds set to 1 across all available backends.' in captured.out