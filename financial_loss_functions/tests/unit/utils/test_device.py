import os
import sys
import torch
import pytest
import random
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils.device import get_best_device, set_seed, mpi_setup

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
            np.random.seed(42)   # to compare
            # Check torch seed
            assert torch.initial_seed() == 42
            # Capture print
            captured = capsys.readouterr()
            assert 'Seeds set to 42 across all available backends.' in captured.out

def test_set_seed_prints_message(capsys):
    set_seed(1)
    captured = capsys.readouterr()
    assert 'Seeds set to 1 across all available backends.' in captured.out

# -------------------- Tests for mpi setup -------------------- #
# Mock mpi4py at the module level to avoid import errors
mock_mpi4py = MagicMock()
mock_mpi4py.MPI = MagicMock()
sys.modules['mpi4py'] = mock_mpi4py
sys.modules['mpi4py.MPI'] = mock_mpi4py.MPI

class TestMpiSetup:
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_mpi_setup_cuda(self, mock_cuda_count, mock_cuda_available):
        mock_comm = MagicMock()
        mock_mpi4py.MPI.COMM_WORLD = mock_comm
        mock_comm.Get_rank.return_value = 5
        mock_comm.Get_size.return_value = 10

        with patch.dict(os.environ, {'SLURM_LOCALID': '2', 'SLURM_CPUS_PER_TASK': '8'}):
            comm, rank, size, gpu_id, cpus = mpi_setup()
        assert comm is mock_comm
        assert rank == 5
        assert size == 10
        assert gpu_id == 2 % 2
        assert cpus == 8

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    def test_mpi_setup_gpu_id_calculation(self, mock_cuda_count, mock_cuda_available):
        mock_comm = MagicMock()
        mock_mpi4py.MPI.COMM_WORLD = mock_comm
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        with patch.dict(os.environ, {'SLURM_LOCALID': '3', 'SLURM_CPUS_PER_TASK': '2'}):
            comm, rank, size, gpu_id, cpus = mpi_setup()
        assert gpu_id == 3 % 4
        assert cpus == 2

    @patch('torch.cuda.is_available', return_value=False)
    def test_mpi_setup_no_cuda_raises(self, mock_cuda_available):
        with pytest.raises(RuntimeError, match="CUDA is required to run MPI version!"):
            mpi_setup()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_mpi_setup_default_cpus(self, mock_cuda_count, mock_cuda_available):
        mock_comm = MagicMock()
        mock_mpi4py.MPI.COMM_WORLD = mock_comm
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        with patch.dict(os.environ, {'SLURM_LOCALID': '0'}):
            comm, rank, size, gpu_id, cpus = mpi_setup()
        assert cpus == 1