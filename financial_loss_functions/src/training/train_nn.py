import gc
import os
import sys
import time
import copy
import torch
import optuna
import psutil
import traceback
import numpy as np
import pandas as pd
from torch import Tensor
from abc import ABC, abstractmethod
from src.utils.device import set_seed
from typing import Callable, Type, Any
from torch.utils.data import DataLoader
from src.utils.constants import MODEL_LOSS_SEP
from src.utils.window import calc_current_idxs
from src.data_processing.dataset import WindowDataset, Reshaper
from src.utils.formatting import reformat_hparams, split_combo_names
from src.utils.io import save_pickle_temp, load_pickle_temp, delete_file

from src.evaluation.evaluator import Evaluator
from pydantic import BaseModel, TypeAdapter
from typing import Callable, Dict, Literal
from scipy import stats

optuna.logging.set_verbosity(optuna.logging.INFO)

class Trainer:
    """
    Class to train provided models with provided hyperparameters.
    """
    def __init__(
        self, 
        model,  # Model class, not instance
        loss: Callable,
        model_hparams: dict[str, Any],      # Specific to model architecture
        optimizer_hparams: dict[str, Any],  # Specific to optimizer
        train_hparams: dict[str, Any],      # Generic training params (epochs, batch_size, etc.)
        in_size: int,
        num_stocks: int,
        max_seq_len: int,
        device: torch.device | str,
        scheduler_hparams: dict[str, Any] | None = None,
        loss_hparams: dict[str, Any] | None = None
    ):
        """
        Initialize Trainer instance to any given PyTorch model.

        Args:
            model (torch.nn.Module): Pytorch neural network class to be trained and evaluated.
            loss (Callable): Custom loss function.
            model_hparams (dict): Hyperparameters for model initialization.
            optimizer_hparams (dict): Hyperparameters for optimizer initialization.
            train_hparams (dict): Hyperparameters for training.
            loss_hparams (dict | None): Hyperparameters for loss functions. 
                Default = None.
            in_size (int): Size of input window.
            num_stocks (int): Number of stocks in the dataset (i.e., number of output nodes).
            max_seq_len (int): Length of input window.
            device (torch.device | str): GPU or CPU device to run the PyTorch model on.
            scheduler_hparams (dict[str, Any] | None): Hyperparamaters for LR Scheduler (ReduceLROnPlateau).
                Default = None.
            loss_hparams (dict[str, Any] | None): Hyperparameters for the loss function (i.e., lambdas).
                Default = None
        
        Raises:
            ValueError: Incorrect torch device provided.
        """

        if isinstance(device, torch.device) :
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            raise ValueError(
                'Incorrect type provided for torch device, must be `str` or `torch.device`.'
            )
        
        print('Model hyperparameters:\n', model_hparams)
        print('Optimizer hyperparameters:\n', optimizer_hparams)
        print('Training hyperparameters:\n', train_hparams)
        print('Scheduler hyperparameters:\n', scheduler_hparams)
        print('Loss Function hyperparameters:\n', loss_hparams)
        
        # Initialize model with its specific hyperparameters
        self.model = model(
            input_size=in_size,
            num_stocks=num_stocks,
            max_seq_len = max_seq_len,
            **model_hparams  # Unpack all model-specific hyperparams
        ).to(self.device)

        self.loss = loss

        self.optimizer_hparams = optimizer_hparams
        self.train_hparams = train_hparams
        self.loss_hparams = loss_hparams or {}
        self.scheduler_hparams = scheduler_hparams
        
        self.train_losses = [] # Stores average losses, for plotting
        self.val_losses = [] # Stores average losses, for plotting
        
        self.eval_losses = [] # For out of sample eval. Stores batch losses, not average
        
        self.avg_train_loss = None
        self.avg_eval_loss = None

        self.eval_alloc_weights = []

        # For Early Stopping
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.patience_counter = 0
    
    def _cal_pf_returns(self, weights: Tensor, returns: Tensor) -> Tensor:
        """
        Calculates returns of a portfolio for every time step by 
        multiplying its weights with the returns of all stocks.

        Args:
            weights (Tensor): (B, N) Portfolio allocation weights (normalized).
            returns (Tensor): (B, T, N) Output (future) window of raw returns to calculate the loss term on.

        Returns:
            port_returns (Tensor): (B, T_out) Returns of a portfolio
        
        """
        port_returns = (weights.unsqueeze(1) * returns).sum(dim=-1) 
        return port_returns

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize optimizer with its specific hyperparameters.
        Optimizer used here is AdamW: Adam optimizer with decoupled weight decay.
        
        Returns:
            optimizer (torch.optim.Optimizer): Optimizer object for AdamW
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.optimizer_hparams
        )
        return optimizer

    def _init_scheduler(
            self, optimizer: torch.optim.Optimizer
        ) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        """
        Initialize LR Scheduler if scheduler hyperparameters are provided.
        LR Scheduler used here is ReduceLROnPlateau.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer object.
        
        Returns:
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau | None):
                LR Scheduler object.
        """
        if self.scheduler_hparams:
            # 2. Initialize Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',       # We want to minimize loss
                **self.scheduler_hparams
            )
        else:
            scheduler = None
        
        return scheduler

    def train(
            self, train_ds: WindowDataset, val_ds: WindowDataset | None = None
        ):
        """
        Train initialized model using the train data split.
        Validate during training if validation data is provided.

        Args:
            train_ds (WindowDataset): Training data split converted to windowed dataset tensors.
            val_ds (WindowDataset | None): Validation data split for validation during training, 
                converted to windowed dataset tensors. 
                If None, validation will not be done during training. Default = None.
        """
        start_time = time.time()

        # Pull hyperparameters with sensible defaults
        n_epochs = self.train_hparams['epochs']
        min_epochs = self.train_hparams.get('min_epochs', 0)
        patience = self.train_hparams.get('early_stop_patience', 20)
        min_delta = self.train_hparams.get('early_stop_min_delta', 1e-3)
        early_stopping = self.train_hparams.get('early_stopping', True)
        clip_grad_norm = self.train_hparams.get('clip_grad_norm', 0.5)
        
        # Initialize optimizer and scheduler
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_hparams['train_batch_size'],
            shuffle=True
        )

        for epoch in range(n_epochs):
            epoch_start = time.time()
            self.model.train()
            total_loss_sum = 0.0
            total_samples = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                weights = self.model(xb)  # (B, N)

                port_returns = self._cal_pf_returns(weights, yb)
                loss = self.loss(
                    weights = weights,
                    all_returns = yb,
                    pf_returns = port_returns,
                    **self.loss_hparams
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()

                batch_size = xb.size(0)

                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

            epoch_avg_loss = total_loss_sum / total_samples
            self.train_losses.append(epoch_avg_loss)
            status_msg = f'Epoch {epoch} | Train Loss: {epoch_avg_loss:.5f}'
            self.avg_train_loss = epoch_avg_loss

            # --- Validation & Early Stopping Logic ---
            if val_ds is not None:
                avg_val_loss = self.validate(val_ds)
                self.val_losses.append(avg_val_loss)

                # --- STEP THE SCHEDULER HERE ---
                # It takes the current validation loss to decide if it should drop the LR
                if scheduler is not None:
                    scheduler.step(avg_val_loss)
                
                # Only allow state-saving after warmup to avoid "Lucky Epoch 0"
                # 1. Initialize baseline if it's the very first validation
                if self.best_val_loss == float('inf'):
                    self.best_val_loss = avg_val_loss
                    self.best_train_loss = epoch_avg_loss

                is_improving = avg_val_loss < (self.best_val_loss - min_delta)
                is_past_warmup = epoch >= min_epochs

                if is_improving:
                    self.best_val_loss = avg_val_loss
                    self.best_train_loss = epoch_avg_loss
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    self.patience_counter = 0 
                else:
                    # ONLY start the "timer" after the warmup is over
                    if is_past_warmup:
                        self.patience_counter += 1

                # 2. THE EARLY STOPPING
                if early_stopping and is_past_warmup and self.patience_counter >= patience:
                    status_msg = status_msg + \
                        f' | Val Loss: {avg_val_loss:.5f}'
                    print(status_msg + f' | Time: {round(time.time() - epoch_start, 3)}s')
                    print(f'----- Early Stopping Triggered at Epoch {epoch} -----\n')
                    break
                        
                else:
                    status_msg = status_msg + f' | Val Loss: {avg_val_loss:.5f}'
            
            print(status_msg + f' | Time: {round(time.time() - epoch_start, 3)}s')
        
        # After the training loop
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            # No improvement ever after warm-up; fall back to final model
            if val_ds is not None:
                self.best_val_loss = avg_val_loss

            # Save bests
            self.best_epoch = epoch
            self.best_train_loss = self.avg_train_loss  # from the last epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            
        print(f'Training Complete. Best Val Loss: {self.best_val_loss:.5f}')
            
        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        print(f'Best Train Loss: {self.best_train_loss:.5f}, Time Taken: {time_taken}s')

    def validate(self, val_ds: WindowDataset) -> float:
        """
        Validation method to run during each training epoch.
        
        Args:
            val_ds (WindowDataset): Validation data split for validation during training.
        
        Returns:
            avg_val_loss (float): Average validation loss over a batch
        """
        val_loader = DataLoader(
            val_ds,
            batch_size=self.train_hparams['val_batch_size'],
            shuffle=False
        )

        self.model.eval()
        with torch.no_grad():
            total_loss, total_samples = 0.0, 0

            for xb, yb in val_loader:
                b = xb.size(0)
                xb, yb = xb.to(self.device), yb.to(self.device)
                weights = self.model(xb)

                port_returns = self._cal_pf_returns(weights, yb)
                loss = self.loss(
                    weights = weights,
                    all_returns = yb,
                    pf_returns = port_returns,
                    **self.loss_hparams
                )

                # --- store per-batch loss ---
                # self.eval_losses.append(loss.item())

                # --- accumulate weighted sum for overall avg ---
                total_loss += loss.item() * b
                total_samples += b

            # --- weighted average over all samples ---
            avg_val_loss = total_loss / total_samples
        
        return avg_val_loss

    def evaluate(self, split_ds: WindowDataset):
        """
        Evaluate the trained model using a split, can be validation or test split.
        This method must not be used during training because we move portfolio 
        allocation weights to the CPU.
        
        Args:
            split_ds (WindowDataset): Validation or test data split converted to 
                windowed dataset tensors.
        """
        start_time = time.time()
        eval_loader = DataLoader(
            split_ds,
            # batch_size=self.train_hparams['val_batch_size'],
            batch_size=1, # To iterate over each evaluation sample
            shuffle=False
        )

        # --- evaluation ---
        self.model.eval()
        with torch.no_grad():
            self.eval_losses = []
            total_loss, total_samples = 0.0, 0

            for xb, yb in eval_loader:
                b = xb.size(0)
                xb, yb = xb.to(self.device), yb.to(self.device)
                weights = self.model(xb)

                # Calculating portfolio returns first
                port_returns = self._cal_pf_returns(weights, yb)
                loss = self.loss(
                    weights = weights,
                    all_returns = yb,
                    pf_returns = port_returns,
                    **self.loss_hparams
                )

                # detach & move to CPU BEFORE appending
                self.eval_alloc_weights.append(weights.detach().cpu()) 

                # --- store per-batch loss ---
                batch_loss = loss.item()
                self.eval_losses.append(batch_loss)

                # --- accumulate weighted sum for overall avg ---
                total_loss += batch_loss * b
                total_samples += b

            # --- weighted average over all samples ---
            self.avg_eval_loss = total_loss / total_samples
        
        end_time = time.time()
        time_taken = round(end_time-start_time, 3)
        print(f'Average Eval Loss: {self.avg_eval_loss:.5f}, Time Taken: {time_taken}')

    def get_eval_alloc_weights(self) -> np.ndarray:
        """
        Getter for allocation weights for every output window as numpy array.
        
        Returns:
            (np.ndarray | None): Portfolio allocation weights for each validation window.
                None if model has not been used to evaluate using `Trainer.evaluate()`.
        """
        if self.eval_alloc_weights:
            wt_array = []
            for w in self.eval_alloc_weights:
                wt_array.append(w.numpy())
            return np.vstack(wt_array)
        else:
            print('Model must be trained and validated.')
            return None
    
    def get_best_losses(self) -> tuple[float, float]:
        """
        Get the best train and validation losses.

        Returns:
            tuple[float, float]: Best train loss and best validation loss.
        """
        return self.best_train_loss, self.best_val_loss
    
    def get_best_epoch(self) -> int:
        """
        Get the best epoch which has the lowest validation loss, when using early stopping.

        Returns:
            best_epoch (int): Best epoch at which the validation loss is the lowest.
        """
        return self.best_epoch
    
    def device_cleanup(self):
        """
        Device cleanup method to clean up caches in either CUDA device or MPS device.
        """
        if self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except Exception as e:
                print(f'MPS cleanup not available. Error: {e}')
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class Walker:
    """
    Class to handle expanding window walk-forward for a single model-loss combination. 

    The Walker manages the entire process: expanding training data, reshaping,
    training a model, and evaluating it at each step of the walk-forward.
    """

    def __init__(
            self,
            num_steps: int,
            model_name: str,
            model_cls: Type,
            loss_name: str,
            loss_func: Callable,
            hparams: dict,
            torch_device: torch.device | str,
            reshaper: Reshaper,
            seed: int | None = None
        ):
        """Initializes the Walker.

        Args:
            num_steps (int): Number of walk-forward steps.
            model_name (str): Name of the model (for logging).
            model_cls (Type): Model class (not instance).
            loss_name (str): Name of the loss function.
            loss_func (Callable): The loss function itself.
            hparams (dict): Hyperparameters for model, optimizer, training, loss.
            torch_device (torch.device or str): Device to run on.
            reshaper (Reshaper): Helper for creating rolling windows.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.num_steps = num_steps
        self.model_name = model_name
        self.model_cls = model_cls
        self.loss_name = loss_name
        self.loss_func = loss_func
        self.hparams = hparams
        self.torch_device = torch_device
        self.reshaper = reshaper
        self.seed = seed

        # get window sizes from reshaper object
        self.in_size = reshaper.in_size
        self.stride = reshaper.out_size

        self.alloc_weights = []
        self.train_eval_losses = []
    
    def _reshape_step_data(
            self,
            walk_train: np.ndarray,
            walk_rets_train: np.ndarray,
            walk_rets_val: np.ndarray    
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reshapes training data and the current inference window.

        Uses the reshaper to create training samples (rolling windows) from the
        training set, and extracts the last `in_size` rows of features
        as the inference input for the next output period.

        Args:
            walk_train (np.ndarray): Cumulative features of the expanded training set.
            walk_rets_train (np.ndarray): Cumulative asset returns of the expanded training set.
            walk_rets_val (np.ndarray): Asset returns of the current validation/holding period.

        Returns:
            tuple: (X_train, y_train, X_val, y_val) where:
                - X_train: training input windows (N, in_size, features)
                - y_train: training target returns (N, out_size, assets)
                - X_val: inference input (1, in_size, features)
                - y_val: inference target returns (1, out_size, assets)
        """
        # Reshaping entire training data
        X_train, y_train, _ = self.reshaper.reshape(walk_train, walk_rets_train)

        # Reshaping only last window
        X_val = self.reshaper.transform_one_window(walk_train[-self.in_size:])
        X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1])

        y_val = walk_rets_val.reshape(
            1, walk_rets_val.shape[0], walk_rets_val.shape[1]
        )

        return X_train, y_train, X_val, y_val

    def _train_eval_helper(
            self,
            train_ds: 'WindowDataset',
            infer_ds: 'WindowDataset',
            X_train_shape: torch.Size,
            y_train_shape: torch.Size
        ):
        """
        Trains the model on the training dataset and evaluates on the inference window.
        Args:
            train_ds (WindowDataset): Dataset containing training windows.
            infer_ds (WindowDataset): Dataset containing the single inference window.
            X_train_shape (torch.Size): Shape of the input tensor (used for Trainer).
            y_train_shape (torch.Size): Shape of the target tensor.

        Returns:
            tuple: (alloc_weights, train_eval_losses) where:
                - alloc_weights (np.ndarray): Portfolio weights for the evaluation window.
                - train_eval_losses (dict): Contains 'train' (epoch losses) and 'eval' losses.
        """

        #### FOR TESTING ####
        # best_hparams['train']['epochs'] = 5
        ####################
        trainer = Trainer(
            model=self.model_cls,
            loss=self.loss_func,
            model_hparams=self.hparams['model'],
            optimizer_hparams=self.hparams['optimizer'],
            train_hparams=self.hparams['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2],
            max_seq_len=X_train_shape[1],
            # scheduler_hparams=self.hparams['scheduler'],
            loss_hparams=self.hparams['loss'],
            device=self.torch_device
        )
        
        trainer.train(train_ds)
        trainer.evaluate(infer_ds)

        # Logging and Diagnostics
        train_eval_losses = {
            'train': trainer.train_losses,
            'eval': trainer.eval_losses
        }

        alloc_weights = trainer.get_eval_alloc_weights()

        trainer.device_cleanup()
        del trainer

        return alloc_weights, train_eval_losses

    def walk_1_model(
            self,
            train: np.ndarray,
            rets_train: np.ndarray,
            val: np.ndarray,
            rets_val: np.ndarray
        ):
        """Executes the full walk-forward for one model-loss combination.

        Args:
            train (np.ndarray): Initial training features (shape (train_days, features)).
            rets_train (np.ndarray): Initial training returns (shape (train_days, assets)).
            val (np.ndarray): Full validation/test features (shape (val_days, features)).
            rets_val (np.ndarray): Full validation/test returns (shape (val_days, assets)).

        Returns:
            np.ndarray: Portfolio allocation weights for each walk-forward step,
                stacked vertically, shape (num_steps, num_assets).
        """


        if self.seed: # set a fixed seed if provided
            set_seed(self.seed)
        
        walk_train = train.copy()
        walk_rets_train = rets_train.copy()
        walk_val = None
        walk_rets_val = None
        
        # Take steps
        steps_alloc_weights = []
        steps_train_eval_losses = []
        for step in range(self.num_steps):
            print(
                '\n', '='*10,
                f' WFV: {self.model_name} - {self.loss_name}, Step: {step}/{self.num_steps-1}',
                '='*10
            )

            current_start, current_end = calc_current_idxs(step+1, self.stride)

            if current_start > 0:
                walk_train = np.concatenate([walk_train, walk_val], axis=0)
                walk_rets_train = np.concatenate([walk_rets_train, walk_rets_val], axis=0)
            
            # Save walk_val and returns val at every step
            walk_val = val[current_start : current_end] # To be added to train data later
            walk_rets_val = rets_val[current_start : current_end]

            # # Median Scale at every walk step
            # robust_scaler = RobustScaler()
            # walk_train_scaled = robust_scaler.fit_transform(walk_train)
            
            X_train, y_train, infer_in, infer_out = self._reshape_step_data(
                walk_train, walk_rets_train, walk_rets_val
            )

            train_ds = WindowDataset(X_train, y_train)
            infer_ds = WindowDataset(infer_in, infer_out)
            X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
            
            alloc_weights, train_eval_losses = self._train_eval_helper(
                train_ds,
                infer_ds,
                X_train_shape,
                y_train_shape
            )

            steps_alloc_weights.append(alloc_weights)
            steps_train_eval_losses.append(train_eval_losses)

        self.alloc_weights = np.vstack(steps_alloc_weights)
        self.train_eval_losses = steps_train_eval_losses

        return self.alloc_weights
    
    def get_train_eval_losses(self) -> dict:
        """Returns the training and evaluation loss histories for each step.

        Returns:
            dict: A dictionary where keys are step indices (or models) and values
                are the loss dictionaries returned by _train_eval_helper.
        """
        return self.train_eval_losses

class MetricModel(BaseModel):
    """
    Must use this data model to provide composite score metric for 
    hyperparameter tuning to `CandidatesGrid`. 
    
    Tuning metric dict must be defined as:
    tune_metric = {
    `<metric name>`: MetricModel(func=`<callable function>`, sign=`<sign>`),
    }
    """
    func: Callable
    sign: Literal['+', '-']

class Tuner:
    """Hyperparameter tuner using Optuna with a composite objective based on walk-forward performance.

    The tuner runs an Optuna study for a given model-loss combination. Each trial evaluates a
    hyperparameter configuration by running a full walk-forward simulation (via Walker),
    computes composite scores (e.g. Information Ratio) per step, and returns a robust objective
    (95% lower confidence bound of the mean composite score). The best hyperparameters
    are then used for final training.
    """
    direction = 'maximize'
    max_seed = 1000000
    n_startup_perc = 0.3
    min_n_startup = 20

    def __init__(
            self,
            tune_metric: dict[str, MetricModel],
            tune_bench_rets: np.ndarray,
            eval_winds: np.ndarray,
            n_steps: int,
            n_trials: int,
            n_warmup_steps: int, 
            n_jobs: int,
            reshaper: Reshaper,
            torch_device: torch.device | str,
            ba_eval_winds: np.ndarray | None
        ):
        """Initializes the Tuner.

        Args:
            tune_metric (dict[str, MetricModel]): Dictionary mapping metric names to MetricModel
                objects (function and sign). The composite score is a linear combination of these.
            tune_bench_rets (np.ndarray): Benchmark returns (e.g., equal weight) used to compute
                excess returns for each walk-forward step. Shape (n_steps, out_size).
            eval_winds (np.ndarray): Asset returns for each walk-forward step.
                Shape (n_steps, out_size, num_assets).
            n_steps (int): Number of walk-forward steps.
            n_trials (int): Number of Optuna trials.
            n_warmup_steps (int): Warmup steps for the pruner (unused in this code).
            n_jobs (int): Number of parallel jobs for Optuna.
            reshaper (Reshaper): Helper for reshaping data into windows.
            torch_device (torch.device or str): Device for training.
            ba_eval_winds (np.ndarray | None): Bid-ask spread data for transaction costs,
                shape (n_steps, num_assets).
        """
        self.tune_metric = TypeAdapter(
            Dict[str, MetricModel]
        ).validate_python(tune_metric)

        self.tune_bench_rets = tune_bench_rets # benchmark returns for information ratio style metrics
        self.eval_winds = eval_winds
        self.n_steps = n_steps
        self.n_trials = n_trials
        self.n_warmup_steps = n_warmup_steps
        self.n_jobs = n_jobs
        self.reshaper = reshaper
        self.torch_device = torch_device
        self.ba_eval_winds = ba_eval_winds

        self.n_startup_trials = max(
            int(self.n_trials * self.n_startup_perc),
            self.min_n_startup
        )
    
    def _calc_composite_scores(
            self, model_loss_name,alloc_weights: np.ndarray
        ) -> np.ndarray:
        """Computes composite scores per walk-forward step for a given set of allocation weights.

        For each step, the function calculates portfolio daily returns (with transaction costs),
        subtracts the benchmark returns to obtain excess returns, and then applies each metric
        in `tune_metric` to the excess returns. The results are combined according to the metric
        signs (+ or -) to produce a single composite score per step.

        Args:
            model_loss_name (str): Identifier for the current model-loss combination (used only
                for internal logging within the Evaluator).
            alloc_weights (np.ndarray): Portfolio allocation weights for each walk-forward step.
                Shape (n_steps, num_assets).

        Returns:
            np.ndarray: Composite score for each step, shape (n_steps,).
        """
        evaluator = Evaluator(self.eval_winds, self.ba_eval_winds, None)
        # Calculate daily returns for this particular portfolio
        evaluator.calc_pf_daily_rets(alloc_weights, model_loss_name)
        model_rets = evaluator.get_rets_for_one(model_loss_name)

        # Calculate excess returns compared to the benchmark
        excess_rets = model_rets - self.tune_bench_rets
        evaluator.update_rets_for_one(model_loss_name, excess_rets)

        # Calculate compossite score for each window (Information Ratio style)
        composite_scores = np.zeros(self.n_steps)
        for _, met_dict in self.tune_metric.items():
            metric_values = evaluator.calc_metric_performance(met_dict.func, mean=False)
            metric_values = metric_values.iloc[:,0].values # Since there is 1 model but multiple steps
            
            if met_dict.sign == '+':
                composite_scores += metric_values
            elif met_dict.sign == '-':
                composite_scores -= metric_values
            else:
                raise ValueError(
                    'Provide only linear operators like + or -. \
                        System designed to take only linear formulas as of now'
                    )
        
        del evaluator
        
        return composite_scores

    def calc_hinge_penalty(
            self, seed_train_losses: list[float], seed_val_losses: list[float],
            eps: float = 1e-9
        ) -> float:
        """Calculates a penalty based on the gap between average train and validation loss.

        Penalises configurations where the average validation loss exceeds the average train loss
        (overfitting). The penalty is the relative excess, clamped at zero for negative gaps.

        Args:
            seed_train_losses (list[float]): List of best training losses (one per seed).
            seed_val_losses (list[float]): List of corresponding best validation losses.
            eps (float): Small constant to avoid division by zero.

        Returns:
            float: Penalty value (≥0).
        """
        avg_train_loss = np.mean(seed_train_losses)
        avg_val_loss = np.mean(seed_val_losses)

        # We calculate how much larger Val Loss is than Train Loss
        # If Val < Train (Healthy/Dropout), raw_gap is negative.
        raw_gap = (avg_val_loss - avg_train_loss) / (avg_train_loss + eps)

        # max(0, raw_gap) ensures we ONLY penalize if Val > Train (Overfitting)
        gap_penalty = max(0, raw_gap)

        return gap_penalty

    def _calc_tuning_objective(
            self, composite_scores: list[float],
            seed_train_losses: list[float],
            seed_val_losses: list[float]
        ) -> float:
        """
        Computes the final tuning objective using a confidence bound and a gap penalty.

        For 2 or more seeds, calculates the 95% one-sided lower confidence bound of the mean
        composite score, then subtracts the hinge penalty. For fewer seeds, falls back to the
        mean composite score minus the penalty.

        Args:
            composite_scores (list[float]): List of composite scores (one per seed).
            seed_train_losses (list[float]): Best training loss per seed.
            seed_val_losses (list[float]): Best validation loss per seed.

        Returns:
            float: Objective value to maximise (higher is better).
        """
        mean_score = np.mean(composite_scores)
        n = len(composite_scores)
        if n < 2:
            # Not enough seeds for variance estimate; fall back to mean
            base_score = mean_score
        else:
            # For statistical consistency across seeds
            std_score = np.std(composite_scores, ddof=1)
            # 95% one‑sided lower bound (t‑distribution)
            t_val = stats.t.ppf(0.95, df=n-1)
            margin = t_val * std_score / np.sqrt(n)

            base_score = mean_score - margin
        
        gap_penalty = self.calc_hinge_penalty(
            seed_train_losses, seed_val_losses
        )
    
        return base_score - gap_penalty
    
    def _calc_tuning_objective_no_gap(
            self, composite_scores: np.ndarray
        ) -> float:
        """
        Computes the 95% lower confidence bound of the mean composite score (or just the mean
        if fewer than 2 walk steps).

        Args:
            composite_scores (np.ndarray): Composite scores per walk step, shape (steps,).

        Returns:
            float: Objective value.
        """
        mean_score = np.mean(composite_scores)
        n = len(composite_scores)
        if n < 2:
            # Not enough seeds for variance estimate; fall back to mean
            final_objective = mean_score
        else:
            # For statistical consistency across seeds
            std_score = np.std(composite_scores, ddof=1)
            # 95% one‑sided lower bound (t‑distribution)
            t_val = stats.t.ppf(0.95, df=n-1)
            margin = t_val * std_score / np.sqrt(n)

            final_objective = mean_score - margin
        
        return final_objective

    def _run_tuning_study(
            self,
            model_name: str,
            model_class: Type,
            loss_name: str,
            loss_func: Callable, 
            train_data: np.ndarray,
            rets_train: np.ndarray, 
            val_data: np.ndarray,
            rets_val: np.ndarray,
            model_cfg: dict,
            loss_cfg: dict
        ) -> optuna.Study:
        """Runs an Optuna hyperparameter study for a single model-loss combination.

        Creates an Optuna study, defines the objective function (which builds a Walker,
        runs walk-forward, computes composite scores, and returns the objective value),
        and executes the optimization.

        Args:
            model_name (str): Name of the model.
            model_class (Type): Model class.
            loss_name (str): Name of the loss function.
            loss_func (Callable): Loss function.
            train_data (np.ndarray): Training features.
            rets_train (np.ndarray): Training returns.
            val_data (np.ndarray): Validation features.
            rets_val (np.ndarray): Validation returns.
            model_cfg (dict): Model hyperparameter configuration (includes default and tuning space).
            loss_cfg (dict): Loss hyperparameter configuration (includes lambdas and tuning).

        Returns:
            optuna.Study: Completed Optuna study containing all trials and the best parameters.

        Raises:
            ValueError: If the model configuration contains no tuning space.
        """
        model_loss_name = f'{model_name}{MODEL_LOSS_SEP}{loss_name}'

        # # # Calculate equal weight portfolio & its returns as benchmark
        # if self.benchmark_rets is None:
        #     eq_wt_calc = EqualWeightCalculator(y_val)
        #     self.benchmark_rets = eq_wt_calc.calc_eq_wt_daily_rets()
        default_hparams = reformat_hparams(model_cfg, loss_cfg)

        # extract tuning ranges
        model_tuning_space = model_cfg.get('tuning', {})        
        loss_tuning_space = loss_cfg.get('tuning', {}) if loss_cfg else {}
        combined_tuning = model_tuning_space | loss_tuning_space

        def _objective(trial):
            # 1. Start with base hparams from JSON
            
            trial_hparams = copy.deepcopy(default_hparams)
            
            # 2. Dynamically update hparams from the JSON tuning space
            for param_name, space in combined_tuning.items():
                stype = space['type']
                if stype == 'float':
                    value = trial.suggest_float(
                        param_name, space['low'], space['high'], log=space.get('log', False)
                    )
                elif stype == 'int':
                    value = trial.suggest_int(param_name, space['low'], space['high'])
                elif stype == 'categorical':
                    value = trial.suggest_categorical(param_name, space['choices'])
                
                # Map the suggested value back to the correct dictionary
                for cat, values_dict in trial_hparams.items():
                    if values_dict:
                        if param_name in values_dict:
                            trial_hparams[cat][param_name] = value

            print(
                '+'*20,
                f'Trial {trial.number}, {model_loss_name}',
                '+'*20
            )
            final_walker = Walker(
                self.n_steps,
                model_name,
                model_class,
                loss_name,
                loss_func,
                trial_hparams,
                self.torch_device,
                self.reshaper
            )

            alloc_weights = final_walker.walk_1_model(
                train_data,
                rets_train, 
                val_data,
                rets_val
            )

            # train_val_losses = final_walker.get_train_eval_losses()

            # Calculate composite scores from allocation weights
            composite_scores = self._calc_composite_scores(model_loss_name, alloc_weights)

            # Calculate final objective score for the trial
            final_objective = self._calc_tuning_objective_no_gap(composite_scores)
        
            print(
                f'Composite Score for trial {trial.number} = {final_objective}'
            )
            #     composite_scores.append(composite_score)

            #     # --- PRUNING LOGIC START ---
            #     # Report the score of the CURRENT seed (i)
            #     # Optuna tracks "step i" across all trials
            #     trial.report(composite_score, step=i)

            #     # Check if this trial should be killed
            #     if trial.should_prune():
            #         print(f'!!!! Trial {trial.number} pruned at seed {i+1} !!!!')
            #         raise optuna.exceptions.TrialPruned()
            #     # --- PRUNING LOGIC END ---
                
            
            
            # final_objective = self._calc_tuning_objective(
            #     composite_scores, seed_train_losses, seed_val_losses
            # )

            return final_objective
        
        if model_tuning_space:
            # pruner = optuna.pruners.MedianPruner(
            #     n_startup_trials=self.n_startup_trials,
            #     n_warmup_steps=self.n_warmup_steps
            # )

            study = optuna.create_study(
                direction=self.direction,
                # pruner=pruner,
                study_name=model_loss_name
            )
            study.optimize(
                _objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs
            )

            # # GUARD: Check if we actually found a completed trial
            # completed_trials = [
            #     t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            # ]
            # if not completed_trials:
            #     print('WARNING: All trials were pruned. Returning the best pruned trial or default.')
            
            return study
        else:
            raise ValueError(
            f'Tuning enabled but no ranges found for {model_name}-{loss_name} or no oos evaluation data found'
        )
    
    def set_tuning_direction(self, direction: str):
        """Sets the optimization direction for the study.

        Args:
            direction (str): Either 'maximize' or 'minimize'.
        """
        self.direction = direction


class WalkerGridUtilities(ABC):
    """
    Abstract class to encapsulate all common walk-forward grid functionality.

    Attributes:
        mdls_hparams_name (str): Key for the default model hyperparameters and their 
            tuning ranges in the hparams_config
        ls_hparams_name (str): Key for the default loss function hyperparameters and 
            their tuning ranges in the hparams_config.
    """
    mdls_hparams_name = 'nn_models'
    ls_hparams_name = 'losses'

    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            num_steps: int,
            common_features: list[str],
            torch_device: torch.device | str,
            mpi: bool = False,
            temp_dir: str | None = None
        ):
        """
        Abstract class to encapsulate all common walk-forward grid functionality.

        Args:
            model_lib (dict[str, dict[str, Type]]): Neural network model architecture 
                library as a dict.
            loss_lib (dict[str, dict[str, dict[str, Callable]]]): Loss functions library 
                as a dict.
            hparams_config (dict[str, Any]): Dictionary containing default hyperparameters 
                and tuning ranges.
            num_steps: (int): Number of walk forward steps to be taken.
            common_features (list[str]): List of common features in the dataset, eg., S&P500 returns.
                This is used for reshaping for different types of broadcasting + reshaping.
            torch_device (torch.device | str): Device to run the PyTorch models.
            mpi (bool): Toggle the use of mpi for distributed evaluation of model-loss combinations.
            temp_dir: (str | None): Directory to save temporary files after a rank has completed its
                work.
        """
        self.model_lib = model_lib
        self.loss_lib = loss_lib
        self.hparams_config = hparams_config # Default config. Not optimized
        self.num_steps = num_steps
        self.torch_device = torch_device
        self.mpi = mpi
        self.temp_dir = temp_dir
        
        self._temp_dir_check()

        # Reshaper setup
        self.reshaper = self._reshaper_setup(common_features)

        # Initialize training data storage
        self.all_alloc_weights: dict[str, np.ndarray] = {}
        self.train_eval_losses: dict[str, dict[str, list[float]]] = {}

    def _temp_dir_check(self):
        """
        Check for the existence of the temporaray directory.

        Raises:
            FileNotFoundError: If temporaray directory does not exist.
        """
        if self.mpi:
            if not self.temp_dir or not os.path.exists(self.temp_dir):
                raise FileNotFoundError(
                    f'Temp directory not found: {self.temp_dir}'
                )
    
    def set_temp_directory(self, temp_dir: str):
        self.temp_dir = temp_dir
    
    # -------------------- Library Searches -------------------- #
    def _search_model(self, model_name: str) -> Type | None:
        """
        Search for required model in the NNModelLibrary registry.
        
        Args:
            model_name (str): Name of the neural network architecture in the library.
        
        Returns:
            Type | None: Class (not initialized) of the required model, or None if not found.
        """
        for _, models_dict in self.model_lib.items():
            if model_name in models_dict:
                return models_dict[model_name]
        return None
    
    def _search_loss_func(self, loss_name: str) -> Callable | None:
        """
        Search for required loss function in the LossFunctionLibrary registry.
        
        Args:
            loss_name (str): Name of the loss function in the library.
        
        Returns:
            Callable | None: Loss function or None if not found.
        """
        for _, cat_dict in self.loss_lib.items():
            for _, sub_cat_dict in cat_dict.items():
                if loss_name in sub_cat_dict:
                    return sub_cat_dict[loss_name]
        return None
    
    # -------------------- Monitoring Methods -------------------- #
    def _memory_diagnostics(self):
        """
        Print memory usage diagnostics wherever this method is called.
        """
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 ** 3
        
        print(f'  Process memory: {mem_gb:.2f} GB')
        
        # if DEVICE == 'cuda':
        #     print(f"  GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        #     print(f"  GPU cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Count Trainer instances
        trainer_count = 0
        for obj in gc.get_objects():
            if type(obj).__name__ == 'Trainer':
                trainer_count += 1
        
        if trainer_count > 0:
            print(f'  WARNING: {trainer_count} Trainer instances still in memory!')
    
    # -------------------- MPI Methods -------------------- #
    @staticmethod
    def _mpi_setup_check(mpi_items: list):
        """
        Check if mpi items like size, comm, rank exists

        Args:
            mpi_items list: List of mpi items to be checked before running mpi methods.

        Raises:
            ValueError: If necessary mpi items are not found.
        """
        for i in mpi_items:
            if i is None:
                raise ValueError(f'All necessary MPI values must be provided. {i} not provided.')
    
    @staticmethod
    def _select_ranks_combos(
        all_combos: list, global_rank: int, size: int
    ) -> list[tuple[str, str]]:
        """
        Distribute model-loss combinations across ranks. This method calculates the indexes 
        of the model-loss combos to run on the current rank.

        Args:
            all_combos (list[tuple[str, str]]): List of tuples containing the model and loss names.
        """
        # Distribute combos evenly across ranks
        chunk_size = len(all_combos) // size
        remainder = len(all_combos) % size
        start = global_rank * chunk_size + min(global_rank, remainder)
        end = start + chunk_size + (1 if global_rank < remainder else 0)
        this_ranks_combos = all_combos[start:end]

        return this_ranks_combos

    # -------------------- Setup Methods -------------------- #
    def _reshaper_setup(self, common_features: list[str]) -> Reshaper:
        """
        Set up method to initialize the Reshaper class using the rolling window sizes.

        Args:
            common_features (list[str]): List of common features that will be placed at the end 
                of the columns or broadcasted if needed.
        
        Returns:
            Reshaper: Reshaper object.
        """
        return Reshaper(
            self.hparams_config['rolling_windows']['in_size'],
            self.hparams_config['rolling_windows']['out_size'],
            self.hparams_config['rolling_windows']['stride'],
            common_features
        )
    
    @staticmethod
    def _convert_datasets_to_np(
        train_data: pd.DataFrame,
        rets_train: pd.DataFrame,
        eval_data: pd.DataFrame,
        rets_eval: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert all dataframes to numpy arrays
        
        Args:
            train_data (pd.DataFrame): Train data split that contain all features.
            rets_train (pd.DataFrame): Train data split that contains only returns 
                data for all stocks.
            eval_data (pd.DataFrame): Validation/Test data split that contains all features.
            rets_eval (pd.DataFrame): Validation/Test data split thay comtains only returns 
                data for all stocks.
        
        Returns:
            tuple: (train_data, rets_train, eval_data, rets_evals)
        """
        return train_data.values, rets_train.values, eval_data.values, rets_eval.values
    
    # -------------------- Integrity Check Methods -------------------- #
    def _data_check(self, train: pd.DataFrame, rets_eval: pd.DataFrame):
        """
        Checks data consistency to ensure number of steps in the 
        walk forward is correctly divides given evaluation data.
        
        Args:
            train (pd.DataFrame): Train data that contains all features.
            rets_eval (pd.DataFrame): Evaluation data that contains only 
                returns data for all stocks.

        Raises:
            ValueError: If provided number of steps does not match the length of 
                windows in the evaulation data.
            ValueError: If evaluation data is too short.
            ValueError: If initial train data is too short.
        """
        walk_stride = self.hparams_config['rolling_windows']['out_size']
        total_val_days = len(rets_eval)
        expected_steps = total_val_days // walk_stride
        if expected_steps != self.num_steps:
            raise ValueError(
                f'Provided num_steps ({self.num_steps}) does not match actual '
                f'number of full windows of length {walk_stride} in evaluation set '
                f'({expected_steps}). Adjust num_steps or data.'
            )
        if total_val_days < walk_stride:
            raise ValueError(
                f'Evaluation data set too short: need at least {walk_stride} days, '
                f'got {total_val_days}.'
            )
        # Eensure initial training set has at least in_size days for first inference
        if len(train) < walk_stride:
            raise ValueError(
                f'Initial training set must have at least {walk_stride} days to form the first inference window, '
                f'got {len(train)} days.'
            )
    
    def _trained_check(self):
        """
        Check if training has been run before. 
        New instance of the class must be created for every training grid.

        Raises:
            RuntimeError: If the instance reties to train models more than once.
        """
        if len(self.all_alloc_weights) != 0:
            raise RuntimeError(
                'Allocation weights already predicted. Create new instance of this class.'
            )
        
    # -------------------- Abstract Methods -------------------- #
    @abstractmethod
    def _merge_all_results(self) -> None:
        pass

    @abstractmethod
    def _walker_helper(
        self,
        model_name: str,
        model_class: Type,
        loss_name: str,
        loss_func: Callable,
        train_data: np.ndarray,
        rets_train: np.ndarray, 
        split_data: np.ndarray,
        rets_split: np.ndarray
    ) -> tuple[np.ndarray, dict[str, list], dict | None]:
        pass

    @abstractmethod
    def _build_combos() ->list[tuple]: pass

    @abstractmethod
    def get_train_val_losses(self) -> dict: pass


class CandidatesGrid(WalkerGridUtilities):
    """
    Class to tune and/or train a neural network model with all available loss functions on 
    the validation data. This can run sequentially or in parallel to tune + train model-loss 
    combinations.

    Attributes:
        temp_wts_prefix_stem (str): This is the prefix stem to be used while saving temporary portfolio 
            allocation weight files used during distributed tuning + training using mpi.
        temp_losses_prefix_stem (str): This is the prefix stem to be used while saving temporary model 
            train-eval loss curve files used during distributed tuning + training using mpi.
        temp_hparams_prefix_stem (str): This is the prefix stem to be used while saving temporary
        hyperparameters used during the distributed tuning + training.
    """
    temp_wts_prefix_stem = 'temp_alloc_wts'
    temp_losses_prefix_stem = 'temp_losses'
    temp_hparams_prefix_stem = 'temp_hparams'

    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            num_steps: int,
            common_features: list[str],
            torch_device: torch.device | str,
            loss_mode: str = 'custom',
            tune: bool = False,
            tuner_eval_items: dict[str, dict[str, MetricModel] | np.ndarray] = None,
            mpi: bool = False,
            temp_dir: str | None = None,
            enable_diagnostics: bool = False
        ):
        """
        Initialize CandidatesGrid class for tuning and training of a model with its loss functions.

        train_eval_* methods can be run only once with each instance

        Args:
            model_lib (dict[str, dict[str, Type]]): Neural network model architecture 
                library as a dict.
            loss_lib (dict[str, dict[str, dict[str, Callable]]]): Loss functions library 
                as a dict.
            hparams_config (dict[str, Any]): Dictionary containing default hyperparameters 
                and tuning ranges.
            num_steps: (int): Number of walk forward steps to be taken.
            common_features (list[str]): List of common features in the dataset, eg., S&P500 returns.
                This is used for reshaping for different types of broadcasting + reshaping.
            torch_device (torch.device | str): Device to run the PyTorch models.
            loss_mode (str): This is used to determine if we need to only custom losses or all loss 
                functions . Default = 'custom'
            mpi (bool): Toggle the use of mpi for distributed evaluation of model-loss combinations.
            tune (bool): Toggle if we need to tune models before finally running them on the validation 
                set or use default values. Default = False,
            tuner_eval_items (dict[str, dict[str, MetricModel] | np.ndarray]): Dictionary containing items 
                to be passed on to the tuner object. Default = None.
            temp_dir (str | None): Directory to save temporary files after a rank has completed its
                work.
            enable_diagnostics (bool): Toggle to print statements about memory usuage during 
                train_eval_* methods. Default = False.
        """
        super().__init__(
            model_lib, loss_lib, hparams_config, num_steps, 
            common_features, torch_device, mpi, temp_dir
        )

        if loss_mode not in ['all', 'custom']:
            raise ValueError('Incorrect Loss Mode. Mode must be `all` or `custom`')
        else:
            self.loss_mode = loss_mode
        
        self.tune = tune
        
        # Tuner setup
        self.tuner = self._tuner_setup(tuner_eval_items)
        
        self.enable_diagnostics = enable_diagnostics

        self.optimized_hparams = {} # Will be filled if tuned
    
    def _tuner_setup(self, tuner_eval_items: dict) -> Tuner | None:
        """
        Method to initalize and setup tuner object if tune flag is True. Use items like tuning metric, 
        benchmark returns and BA spreads for the evaluation windows, to inistalize the tuner.

        Args:
            tuner_eval_items dict[str, dict[str, MetricModel] | np.ndarray]: Dictionary containing items 
                to be passed on to the tuner object. Dictionary must contain 'metric', 'bench_rets', 
                'eval_winds' and 'ba_eval_winds'.
        
        Returns:
            tuner (Tuner): Tuner object (intialized) for None tune flag is set to False.

        Raises:
            ValueError: If tune metric is not provided.
            ValueError: If benchmark returns is not provided.
            ValueError: If evaluation windows (returns only) is not provided.
        """
        if self.tune:
            tune_metric = tuner_eval_items.get('metric')
            tune_bench_rets = tuner_eval_items.get('bench_rets')
            tune_eval_winds = tuner_eval_items.get('eval_winds')
            tune_ba_eval_winds = tuner_eval_items.get('ba_eval_winds')
            if tune_ba_eval_winds is None:
                print('No BA Spread data provided. Tuner not accounting for trading costs.')
            
            # Tuner configuration
            if tune_metric and tune_bench_rets is not None and tune_eval_winds is not None:
                tuner_config = self.hparams_config.get('tuner', {})
                print('Tuner Config:\n', tuner_config)
                
                n_trials = tuner_config.get('n_tuning_trials', 30)
                n_warmup_steps = tuner_config.get('n_warmup_steps', 2)
                n_jobs = tuner_config.get('n_jobs', 1)
                print(
                    f'Tuning all models with {n_trials} trials, across {self.num_steps} steps.'
                )
                tuner = Tuner(
                    tune_metric,
                    tune_bench_rets,
                    tune_eval_winds,
                    self.num_steps,
                    n_trials,
                    n_warmup_steps,
                    n_jobs,
                    self.reshaper,
                    self.torch_device,
                    tune_ba_eval_winds
                )
            
            elif not tune_metric:
                raise ValueError(
                    'Provide Tuning metric if tune = True.',
                    "In the tuner_eval_items dict, add 'metric': {'<metric>': 'func': Callable, 'sign': '<sign>'}"
                )

            elif tune_bench_rets is None:
                raise ValueError(
                    'Provide tuning benchmark (eg. S&P500 or Equal Weight returns) if tune = True.',
                    "In the tuner_eval_items dict, add 'bench_rets': np.ndarray"
                )
            elif tune_eval_winds is None:
                raise ValueError(
                    'Provide evaluation windows reshaped and sliced from validation returns if tune = True.',
                    "In the tuner_eval_items dict, add 'eval_winds': np.ndarray"
                )
            
        else:
            tuner = None
        
        return tuner
    
    
    def _map_new_params(self, best_config: dict, new_hparams: dict) -> dict:
        """
        Map best found hyperparameters to the reformatted hyperparameters dictionary.
        
        Args:
            best_config (dict): Dictionary containing the old (default) hyperparameters.
            new_hparams (dict): Dictionary containing the new found hyperparameters.
        
        Returns:
            best_config (dict): Dictionary containing all hyperparameters, including the 
                new tuned hyperparameters.
        """
        # Map the best Optuna parameters into our new dictionary
        for k, v in new_hparams.items():
            for cat, cat_dict in best_config.items():
                if k in cat_dict: best_config[cat][k] = v

        return best_config
    
    def _add_median_epoch(self, best_config: dict, best_epochs: list[int]):
        """
        Add the median of the best epochs from multiple seed runs, when early stopping is used.

        `Currently, this is not in use.`

        Args:
            best_config (dict): Dictionary containing all hyperparameters, including the 
                new tuned hyperparameters.
            best_epochs (list[str]): List of the best epochs at which early stopping was 
                triggered at across multiple seeds.
        
        Returns:
            best_config (dict): Dictionary containing all hyperparameters, and the median 
                of the best epochs from each seed.
        """
        best_config['train']['median_epochs'] = int(np.median(best_epochs)) + 1
        # 1 is added because the epoch saved starts from 0 in the training loop
        return best_config

    def _walker_helper(
        self,
        model_name: str,
        model_class: Type,
        loss_name: str,
        loss_func: Callable,
        train_data: np.ndarray,
        rets_train: np.ndarray, 
        val_data: np.ndarray,
        rets_val: np.ndarray
    ) -> tuple[np.ndarray, dict[str, list], dict | None]:
        """
        Method to tune a model-loss combination, train it, and then evaulate it on the validation 
        data, or just evaulate it on the validation data. This method runs the Tuner._run_tuning_study 
        method if tune flag is True. Memory diagnostics can be monitored in this method if 
        CandidatesGrid.enable_diagnostics = True.

        Args:
            model_name (str): Name of the neural network model architecture.
            model_class (Type): Class of the neural network model architecture.
            loss_name (str): Name of the loss function to be used for the training.
            loss_func (Callable): Loss function to be used for the training.
            train_data (np.ndarray): Train data split that contains all the features.
            rets_train (np.ndarray): Train data split that contains only returns data for all stocks.
            val_data (np.ndarray): Validation data split that contains all the features.
            rets_val (np.ndarray): Validation data split that contains only the returns data for all stocks.

        Returns:
            tuple (tuple[np.ndarray, dict[str, list], dict | None]):
                - alloc_weights: Portfolio allocation weights for each window in this walk forward on the 
                    validation data
                - train_val_losses: Train val loss curves data at each walk step.
                - optimized_hparams: Best found (optimized) hyperparameters from the Optuna study.
                    This can be None if no tuning was done.
        """
        # Extract base configs
        model_cfg = self.hparams_config[self.mdls_hparams_name][model_name]
        loss_cfg = self.hparams_config[self.ls_hparams_name].get(loss_name, {})
        
        if self.enable_diagnostics:
            print(f'\n[Before training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        # --- Construct the NEW Best Params Dictionary --- 
        # Build new structure to save each combo. Uses defaults if not tuned
        best_config = reformat_hparams(model_cfg, loss_cfg)

        # --- Run the Study ---
        if self.tune:
            print(f'Tuning Hyperparameters for {model_name}-{loss_name}...')
            study = self.tuner._run_tuning_study(
                model_name,
                model_class,
                loss_name,
                loss_func, 
                train_data,
                rets_train, 
                val_data,
                rets_val,
                model_cfg,
                loss_cfg
            )
            best_found_params = study.best_params
            # best_epochs = study.best_trial.user_attrs.get('best_epochs')
            
            best_config = self._map_new_params(best_config, best_found_params)
            # best_config = self._add_median_epoch(best_config, best_epochs)
            
            del study
        else:
            # If not tuning, we just use the original values and use the first seed in the seed list
            # set_seed(self.seed_list[0])
            # No set seed
            pass
        
        if self.tune:
            optimized_hparams = best_config
        else:
            optimized_hparams = None
        
        # --- 3. Final Walk Training --- #
        final_walker = Walker(
            self.num_steps,
            model_name,
            model_class,
            loss_name,
            loss_func,
            best_config,
            self.torch_device,
            self.reshaper
        )

        alloc_weights = final_walker.walk_1_model(
            train_data,
            rets_train, 
            val_data,
            rets_val
        )

        train_val_losses = final_walker.get_train_eval_losses()
        
        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        return alloc_weights, train_val_losses, optimized_hparams

    def _count_only_models(self) -> int:
        n_models = sum(len(models_dict) for models_dict in self.model_lib.values())
        return n_models

    def _build_losses_to_use(self) -> dict[str, Callable]:
        # Build losses that should be used based on loss_mode
        losses_to_use = {}
        custom_combos = self.loss_lib['custom']['__default__'] # Custom combos have no category
        # Grid with custom loss functions
        if self.loss_mode == 'custom':           
            print('Training all models with all custom loss functions...')
            losses_to_use = custom_combos
        
        # Grid with custom loss functions + only objectives
        else: # loss_mode = 'all'
            losses_to_use = custom_combos

            loss_objectives = self.loss_lib['objectives']['__default__'] # Objectives have no category
            for objtiv_name, objtiv_func in loss_objectives.items():
                if objtiv_name not in losses_to_use:
                    losses_to_use[objtiv_name] = objtiv_func
                else:
                    print(f'WARNING: Objective {objtiv_name} already exists! Skipping duplicate.')
                    continue
        
        return losses_to_use
    
    def _merge_all_results(
            self,
            size,
            temps_wts_prefix,
            temp_losses_prefix,
            temp_hparams_prefix
        ):
        """Merge all results into one dict if rank is 0, i.e., main process."""
        """
        Merge all temporary allocation weights, train-eval losses and optimized hyperparameters
        into three combined dicts, if rank is 0, i.e., main process. After the merging is done, 
        it deletes the temporary pkl files.

        Args:
            size (int): Size of the mpi comm world.
            temp_wts_prefix (str): The prefix used while saving temporary portfolio 
                allocation weight files.
            temp_losses_prefix (str): The prefix used while saving temporary model 
                train-eval loss curve files.
            temp_hparams_prefix (str): The prefix used while saving temporary optimized 
                hyperparameter files.
        """
        self.all_alloc_weights = {}
        self.train_eval_losses = {}
        self.optimized_hparams = {} #### Must be same as in constructor ####
        for r in range(size):
            # Load all temp alloc wt files
            rank_alloc_weights = load_pickle_temp(
                self.temp_dir / f'{temps_wts_prefix}_{r}.pkl'
            )
            # Merge into self.all_alloc_weights
            for model_loss, models_dict in rank_alloc_weights.items():
                self.all_alloc_weights[model_loss] = models_dict
            
            rank_losses = load_pickle_temp(
                self.temp_dir / f'{temp_losses_prefix}_{r}.pkl'
            )
            # Merge into self.train_val_losses
            for model_loss, losses_dict in rank_losses.items():
                self.train_eval_losses[model_loss] = losses_dict
            
            rank_hparams = load_pickle_temp(
                self.temp_dir / f'{temp_hparams_prefix}_{r}.pkl'
            )
            # Merge into self.optimized_hparams
            for model_loss, hparms_dict in rank_hparams.items():
                self.optimized_hparams[model_loss] = hparms_dict

        
        # Delete all temp files
        for r in range(size):
            delete_file(self.temp_dir / f'{temps_wts_prefix}_{r}.pkl')
            delete_file(self.temp_dir / f'{temp_losses_prefix}_{r}.pkl')
            delete_file(self.temp_dir / f'{temp_hparams_prefix}_{r}.pkl')

        print('All temp files merged and then deleted.')
    
    def _build_combos(
            self, 
            losses_to_use: dict[str, Callable], 
            grid_mode: str,
            model_name: str | None = None,
            model_class: Type | None = None
        ) -> list[tuple[str, Callable, str, Type]]:
        """
        Build a list of tuples with all the names, classes, and functions for models and 
        loss functions.

        Args:
            losses_to_use (dict[str, Callable]): Dictionary containing the loss names and the 
                callable functions
            grid_mode (str): Grid mode being used in this instance. Must `all` or `one_model`.
            model_name (str | None): Name of the model is required if grid mode is one_model.
            model_class: (Type | None): Model class is required if the grid mode is one_model.
        
        Returns:
            all_combos (list[tuple[str, Type, str, Callable]]): List of tuples containing 
                the loss name, loss function, model name and model class.
        
        Raises:
            ValueError: If provided grid mode is `one`. This method cannot be used with `one`.
        """
        all_combos = []
        if grid_mode == 'all':
            for loss_name, loss_func in losses_to_use.items():
                for category, models_dict in self.model_lib.items():
                    for model_name, model_class in models_dict.items():
                        all_combos.append((loss_name, loss_func, model_name, model_class))
        
        elif grid_mode == 'one_model' and model_name and model_class:
            for loss_name, loss_func in losses_to_use.items():
                all_combos.append((loss_name, loss_func, model_name, model_class))

        else:
            raise ValueError(
                'Incorrect grid mode provided to build combos.',
                'Must be `all` or `one_model`.',
                '`model_name` and `model_class` must be provided for `one_model`.'
            )

        return all_combos
    
    def train_eval_one_model(
            self,
            model_name: str,
            train_data: pd.DataFrame,
            rets_train: pd.DataFrame, 
            val_data: pd.DataFrame,
            rets_val: pd.DataFrame,
            comm = None,
            global_rank = None,
            size = None
        ) -> dict[str, dict[str, np.ndarray]] | None:
        """
        Run hyperparameter tuning and training of one neural network model architecture with all
        available loss functions. This is an entry point method for this process. It can run 
        sequentially as well as in parallel using mpi to distribute the model-loss combinations 
        across nodes and gpus.
        
        Args:
            model_name (str): Name of the neural network architecture to be used in this instance.
            train_data (pd.DataFrame): Train data split that contains all the features. 
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_train (pd.DataFrame): Train data split that contains only returns data for all stocks.
                Returns columns must be in the format <ticker>.
            val_data (pd.DataFrame): Validation data split that contains all the features.
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_val (pd.DataFrame): Validation data split that contains only the returns data for all stocks.
                Returns columns must be in the format <ticker>.
            comm: MPI Communication object. Default = None.
            global_rank: Global rank of the current rank this is being executed on. Default = None.
            size: Size of the mpi communication world, i.e., number of ranks. Default = None.

        Returns:
            all_alloc_weights (dict[str, dict[str, np.ndarray]] | None): Portfolio allocation weights
            for all the portfolio optimizer models and for all output windows. 
            Return is None only for non-zero ranks.
        """
        
        self._data_check(train_data, rets_val)
        self._trained_check()

        # Extract feature data
        self.reshaper.extract_features(train_data.columns)
        
        # Convert all dataframes to arrays
        train_data, rets_train, val_data, rets_val = self._convert_datasets_to_np(
            train_data, rets_train, val_data, rets_val
        )

        # Search for model
        model_class = self._search_model(model_name)
        if not model_class: # model not found
            raise RuntimeError(f'{model_name} MODEL NOT FOUND IN LIBRARY!')

        losses_to_use = self._build_losses_to_use()
        
        # Calculate number of models to train (model + loss combinations)
        total_train_count = len(losses_to_use)
        
        print(
            f'\nTraining {total_train_count} models.',
            f'Running all models with {self.loss_mode} losses.'
        )

        all_combos = self._build_combos(
            losses_to_use, 'one_model', model_name, model_class
        )

        # Not using MPI for distributed computing
        if self.mpi == False:
            
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(all_combos, 1):
                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {idx}/{total_train_count}',
                    '-'*10
                )
                try:        
                    alloc_weights, train_val_losses, optimized_hparams = self._walker_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_data,
                        rets_train, 
                        val_data,
                        rets_val
                    )
                    self.all_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = alloc_weights
                    self.train_eval_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = train_val_losses
                    self.optimized_hparams[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = optimized_hparams

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue

            return self.all_alloc_weights
        
        # If MPI is true for distributed computing
        else:
            self._mpi_setup_check([comm, global_rank, size])

            this_ranks_combos = self._select_ranks_combos(all_combos, global_rank, size)

            # Print summary on each rank
            print(f'Rank {global_rank}: tuning & training {len(this_ranks_combos)} combos.')
            sys.stdout.flush()

            # Local results dictionary
            local_alloc_weights = {}
            local_train_val_losses = {}
            local_optimized_hparams = {}
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(this_ranks_combos, 1):
                print(f'\nRank {global_rank}: {idx}/{len(this_ranks_combos)} - {model_name} - {loss_name}')
                try:
                    alloc_weights, train_val_losses, optimized_hparams = self._walker_helper(
                        model_name,
                        model_class,
                        loss_name,
                        loss_func,
                        train_data,
                        rets_train, 
                        val_data,
                        rets_val
                    )
                    
                    local_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = alloc_weights
                    local_train_val_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = train_val_losses
                    local_optimized_hparams[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = optimized_hparams
                
                except Exception as e:
                    print(f'Rank {global_rank}: Error on {model_name} - {loss_name}: {e}')
                    traceback.print_exc()
                    continue
            
            temp_wts_prefix = f'{model_name}_{self.temp_wts_prefix_stem}'
            temp_losses_prefix = f'{model_name}_{self.temp_losses_prefix_stem}'
            temp_hparams_prefix = f'{model_name}_{self.temp_hparams_prefix_stem}'
            
            # Save local results to a rank‑specific file
            save_pickle_temp(
                local_alloc_weights,
                self.temp_dir / f'{temp_wts_prefix}_{global_rank}.pkl'
            )
            save_pickle_temp(
                local_train_val_losses,
                self.temp_dir / f'{temp_losses_prefix}_{global_rank}.pkl'
            )
            save_pickle_temp(
                local_optimized_hparams,
                self.temp_dir / f'{temp_hparams_prefix}_{global_rank}.pkl'
            )
            
            # Wait for all ranks to finish
            comm.Barrier()

            # Rank 0 collects and merges all files
            if global_rank == 0:
                self._merge_all_results(
                    size,
                    temp_wts_prefix,
                    temp_losses_prefix,
                    temp_hparams_prefix
                )
                
                return self.all_alloc_weights
            else:
                return None

    def train_eval_one(
            self,
            model_name: str, 
            loss_name: str,
            train_data: pd.DataFrame,
            rets_train: pd.DataFrame, 
            val_data: pd.DataFrame,
            rets_val: pd.DataFrame,
        ) -> dict[str, dict[str, np.ndarray]]:
        """
        Run hyperparameter tuning and evaluation of one model-loss combination on the 
        validation data. This is an entry point method for this process. 
        It can run only sequentially.
        
        Args:
            model_name (str): Name of the neural network model architecture.
            loss_name (str): Name of the loss function to be used with the neural network model.
            train_data (pd.DataFrame): Train data split that contains all the features. 
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_train (pd.DataFrame): Train data split that contains only returns data for all stocks.
                Returns columns must be in the format <ticker>.
            val_data (pd.DataFrame): Validation data split that contains all the features.
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_val (pd.DataFrame): Validation data split that contains only the returns data for all stocks.
                Returns columns must be in the format <ticker>.

        Returns:
            all_alloc_weights (dict[str, dict[str, np.ndarray]]): Portfolio allocation weights for one 
                model-loss portfolio optimizer for all output windows.
        """
        self._data_check(train_data, rets_val)
        self._trained_check()

        # Extract feature data
        self.reshaper.extract_features(train_data.columns)
        
        # Convert all dataframes to arrays
        train_data = train_data.values
        rets_train = rets_train.values
        val_data = val_data.values
        rets_val = rets_val.values

        model_class = self._search_model(model_name)
        if model_class is None:
            raise KeyError(f'Model {model_name} not found.')
        
        loss_func = self._search_loss_func(loss_name)
        if loss_func is None:
            raise KeyError(f'Loss Function {loss_name} not found.')

        try:
            print('\n', '-'*10, f' Training {model_name}-{loss_name} ', '-'*10)
            alloc_weights, train_val_losses, optimized_hparams = self._walker_helper(
                model_name,
                model_class, 
                loss_name,
                loss_func,
                train_data,
                rets_train, 
                val_data,
                rets_val
            )
            self.all_alloc_weights[
                f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
            ] = alloc_weights
            self.train_eval_losses[
                f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
            ] = train_val_losses
            self.optimized_hparams[
                f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
            ] = optimized_hparams

        except Exception as e:
            print(f'DEBUG: Error while training {model_name}. Not training.', e)
            traceback.print_exc()
        
        return self.all_alloc_weights
    
    def get_optimized_hparams(self) -> dict:
        """
        Get the optimzed hyperparameters from tuning hyperparameters.
        """
        return self.optimized_hparams
    
    def get_train_val_losses(self) -> dict[str, dict[str, list[float]]]:
        """
        Get the train-eval loss curves from training at each walk step. 
        This method reformats the internal dictionary.

        Returns:
            reformatted_dict (dict[str, dict[str, list[float]]]): Dictionary containing the
                train-eval losses at each walk step.
        
        Raises:
            RunTimeError: If models are not yet training and evaluated on the test data
        """
        if self.train_eval_losses:
            reformatted_dict = {}
            for model_loss, step_losses in self.train_eval_losses.items():
                train_losses = []
                eval_losses = []
                for step in step_losses:
                    train_losses.append(step['train'])
                    eval_losses.append(step['eval'][0]) # 0 since all evaulation is done on single windows
                
                reformatted_dict[model_loss] = {
                    'train': train_losses,
                    'eval': eval_losses
                }
        else:
            raise RuntimeError('Models not trained yet. Run training first.')
        
        return reformatted_dict
    


class WalkForwardValidator(WalkerGridUtilities):
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            common_features: list[str],
            torch_device: torch.device | str,
            num_steps: int,
            filtered_models: list[str] | None = None,
            mpi: bool = False,
            temp_dir: str | None = None,
            optimized_hparams = None
        ):
        """
        DEPRECATED
        """
        print('DEPRECATED: WalkForwardValidator as a separate pipeline is removed.')

        super().__init__(
            model_lib, loss_lib, hparams_config, torch_device, mpi, temp_dir
        )
        self.num_steps = num_steps
        if filtered_models:
            self.filtered_models = split_combo_names(filtered_models, '-')
            # self.filtered_models = [self.filtered_models[0]] #### SLICE FOR TESTING ####
        else:
            self.filtered_models = filtered_models
        
        self.optimized_hparams = optimized_hparams # Optimized

        self.in_size = self.hparams_config['rolling_windows']['in_size']
        out_size = self.hparams_config['rolling_windows']['out_size']
        self.stride =  out_size # Output size by default

        self.reshaper = Reshaper(
            self.in_size,
            out_size,
            self.hparams_config['rolling_windows']['stride'],
            common_features
        )

        self.all_alloc_weights: dict[str, np.ndarray] = {}
        self.train_infer_losses: dict[str, list[dict[str, list[float]]]] = {}

    def update_stride(self, stride: int):
        self.stride = stride
    
    def _collected_combos(self) -> list[tuple[str, Callable, str, Type]]:
        relevant_temp = {'models': {}, 'losses': {}}
        all_combos = []
        for model_loss in self.filtered_models:
            model_name = model_loss[0]
            loss_name = model_loss[1]

            if model_name not in relevant_temp['models']:
                relevant_temp['models'][model_name] = self._search_model(model_name)
            
            if loss_name not in relevant_temp['losses']:
                relevant_temp['losses'][loss_name] = self._search_loss_func(loss_name)

            all_combos.append(
                (
                    loss_name,
                    relevant_temp['losses'][loss_name],
                    model_name,
                    relevant_temp['models'][model_name]
                )
            )
        return all_combos

    def _reshape_step_data(
            self,
            walk_train: pd.DataFrame,
            walk_rets_train: pd.DataFrame,
            walk_rets_val: pd.DataFrame    
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Reshaping entire training data
        X_train, y_train, _ = self.reshaper.reshape(walk_train, walk_rets_train)

        # Reshaping only last window
        infer_in = self.reshaper.transform_one_window(walk_train.iloc[-self.in_size:])
        infer_in = infer_in.reshape(1, infer_in.shape[0], infer_in.shape[1])

        infer_out = walk_rets_val.values
        infer_out = infer_out.reshape(1, infer_out.shape[0], infer_out.shape[1])

        return X_train, y_train, infer_in, infer_out

    def _replace_epochs(self, model_name: str, loss_name: str):
        """
        Replace number of epochs with median number of epochs if it exists.

        Args:
            model_loss_name (str): Name of the model loss combination in the format: <Model>-<Loss>.
        """
        model_loss_name = f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
        if self.optimized_hparams and model_loss_name in self.optimized_hparams:
            median_epochs = self.optimized_hparams[
                model_loss_name
            ]['train'].get('median_epochs')
            if median_epochs:
                self.optimized_hparams[model_loss_name]['train']['epochs'] = median_epochs
            else:
                print(
                    f'WARNING: No median epochs found or is 0, {median_epochs}.',
                    f'Using default number of epochs for {model_loss_name}!'
                )

    def _train_eval_helper(
            self,
            model_name: str,
            model_cls: Type,
            loss_name: str,
            loss_func: Callable, 
            train_ds: 'WindowDataset',
            infer_ds: 'WindowDataset',
            X_train_shape: torch.Size,
            y_train_shape: torch.Size
        ):
        model_loss_name = f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
        
        # Gather best hyperparameters or use defaults
        if self.optimized_hparams and model_loss_name in self.optimized_hparams:
            best_hparams = self.optimized_hparams[model_loss_name]
            
        else:
            print('!No optimized hyperparameters provided, using defaults!')
            model_cfg = self.hparams_config[self.mdls_hparams_name][model_name]
            loss_cfg = self.hparams_config[self.ls_hparams_name].get(loss_name, {})

            best_hparams = reformat_hparams(model_cfg, loss_cfg)

        #### FOR TESTING ####
        # best_hparams['train']['epochs'] = 5
        ####################
        trainer = Trainer(
            model=model_cls,
            loss=loss_func,
            model_hparams=best_hparams['model'],
            optimizer_hparams=best_hparams['optimizer'],
            train_hparams=best_hparams['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2],
            max_seq_len=X_train_shape[1],
            scheduler_hparams=best_hparams['scheduler'],
            loss_hparams=best_hparams['loss'],
            device=self.torch_device
        )
        
        trainer.train(train_ds)
        trainer.evaluate(infer_ds)

        # Logging and Diagnostics
        train_infer_losses = {
            'train': trainer.train_losses,
            'eval': trainer.eval_losses
        }

        alloc_weights = trainer.get_eval_alloc_weights()

        trainer.device_cleanup()
        del trainer

        return alloc_weights, train_infer_losses

    def _walk_1_model(
            self,
            init_train: pd.DataFrame,
            init_rets_train: pd.DataFrame,
            init_val: pd.DataFrame,
            init_rets_val: pd.DataFrame,
            model_name: str,
            model_cls: Type,
            loss_name: str,
            loss_func: Callable
        ) -> tuple[list[np.ndarray], list[dict[str, list[float]]]]:
        walk_train = init_train.copy()
        walk_rets_train = init_rets_train.copy()
        walk_val = None
        walk_rets_val = None
        
        # Take steps
        steps_alloc_weights = []
        steps_train_infer_losses = []
        for step in range(self.num_steps):
            print(
                '\n', '='*10,
                f' WFV: {model_name} - {loss_name}, Step: {step}/{self.num_steps-1}',
                '='*10
            )

            current_start, current_end = calc_current_idxs(step+1, self.stride)

            if current_start > 0:
                walk_train = pd.concat([walk_train, walk_val], axis=0)
                walk_rets_train = pd.concat([walk_rets_train, walk_rets_val], axis=0)
            
            # Save walk_val and returns val at every step
            walk_val = init_val.iloc[current_start : current_end] # To be added to train data later
            walk_rets_val = init_rets_val.iloc[current_start : current_end]

            X_train, y_train, infer_in, infer_out = self._reshape_step_data(
                walk_train, walk_rets_train, walk_rets_val
            )

            train_ds = WindowDataset(X_train, y_train)
            infer_ds = WindowDataset(infer_in, infer_out)
            X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

            self._replace_epochs(model_name, loss_name)
            
            alloc_weights, train_infer_losses = self._train_eval_helper(
                model_name,
                model_cls,
                loss_name,
                loss_func,
                train_ds,
                infer_ds,
                X_train_shape,
                y_train_shape
            )

            steps_alloc_weights.append(alloc_weights)
            steps_train_infer_losses.append(train_infer_losses)

        return np.vstack(steps_alloc_weights), steps_train_infer_losses

    
            
    def validate_grid(
            self,
            init_train: pd.DataFrame,
            init_rets_train: pd.DataFrame,
            init_val: pd.DataFrame,
            init_rets_val: pd.DataFrame,
            comm = None, global_rank = None, size = None
        ) -> dict[str, np.ndarray]:

        self._data_check(init_train, init_rets_val)

        self.reshaper.extract_features(init_train.columns)
        
        # Search and prepare models combos
        validate_count = len(self.filtered_models)
        print(
            f'\nWalk-forward validating {validate_count} models, over {self.num_steps} steps.'
        )
        all_combos = self._collected_combos()
        
        if self.mpi == False:
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(all_combos, 1):
                print(
                    '\n', '-'*20,
                    f' WFV: {model_name} - {loss_name}, Model: {idx}/{validate_count}',
                    '-'*20
                )
                try:
                    steps_alloc_weights, steps_train_infer_losses = self._walk_1_model(
                        init_train,
                        init_rets_train,
                        init_val,
                        init_rets_val,
                        model_name,
                        model_class,
                        loss_name,
                        loss_func
                    )

                    self.all_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = steps_alloc_weights
                    self.train_infer_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = steps_train_infer_losses
                
                except Exception as error:
                    print(
                        f'DEBUG: Error while walk-forward validating {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue

            return self.all_alloc_weights
        
        else:
            self._data_check(init_train, init_rets_val)
            self._mpi_setup_check([comm, global_rank, size])

            this_ranks_combos = self._select_ranks_combos(all_combos, global_rank, size)

            # Print summary on each rank
            print(f'Rank {global_rank}: tuning & training {len(this_ranks_combos)} combos.')
            sys.stdout.flush()

            # Local results dictionary
            local_alloc_weights = {}
            local_train_infer_losses = {}
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(this_ranks_combos, 1):
                print(f'\nRank {global_rank}: {idx}/{len(this_ranks_combos)} - {model_name} - {loss_name}')

                try:
                    steps_alloc_weights, steps_train_infer_losses = self._walk_1_model(
                        init_train,
                        init_rets_train,
                        init_val,
                        init_rets_val,
                        model_name,
                        model_class,
                        loss_name,
                        loss_func
                    )

                    local_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = steps_alloc_weights
                    local_train_infer_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = steps_train_infer_losses
                
                except Exception as e:
                    print(f'Rank {global_rank}: Error on {model_name} - {loss_name}: {e}')
                    traceback.print_exc()
                    continue
            
            temp_wts_prefix = f'all_wf_temp_alloc_wts'
            temp_losses_prefix = f'all_wf_temp_losses'

            # Save local results to a rank‑specific file
            save_pickle_temp(
                local_alloc_weights,
                self.temp_dir / f'{temp_wts_prefix}_{global_rank}.pkl'
            )
            save_pickle_temp(
                local_train_infer_losses,
                self.temp_dir / f'{temp_losses_prefix}_{global_rank}.pkl'
            )
            
            # Wait for all ranks to finish
            comm.Barrier()

            # Rank 0 collects and merges all files
            if global_rank == 0:
                self._merge_all_results(
                    size,
                    temp_wts_prefix,
                    temp_losses_prefix
                )
                
                return self.all_alloc_weights
            else:
                return None

    def validate_one(
            self,
            model_name: str, 
            loss_name: str,
            init_train: pd.DataFrame,
            init_rets_train: pd.DataFrame,
            init_val: pd.DataFrame,
            init_rets_val: pd.DataFrame
        ):
        self._data_check(init_train, init_rets_val)
        
        self.reshaper.extract_features(init_train.columns)
        
        model_class = self._search_model(model_name)
        if model_class is None:
            raise KeyError(f'Model {model_name} not found.')
        
        loss_func = self._search_loss_func(loss_name)
        if loss_func is None:
            raise KeyError(f'Loss Function {loss_name} not found.')
        
        try:
            steps_alloc_weights, steps_train_infer_losses = self._walk_1_model(
                init_train,
                init_rets_train,
                init_val,
                init_rets_val,
                model_name,
                model_class,
                loss_name,
                loss_func
            )

            self.all_alloc_weights[
                f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
            ] = steps_alloc_weights
            self.train_infer_losses[
                f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
            ] = steps_train_infer_losses
       
        except Exception as e:
            print(f'DEBUG: Error while walk-forward validating {model_name}. Not Validating.', e)
            traceback.print_exc()
        
        return self.all_alloc_weights

    def get_train_infer_losses(self) -> dict[str, list]:
        reformatted_dict = {}
        for model_loss, step_losses in self.train_infer_losses.items():
            train_losses = []
            eval_losses = []
            for step in step_losses:
                train_losses.append(step['train'])
                eval_losses.append(step['eval'][0]) # 0 since all evaulation is done on single windows
            
            reformatted_dict[model_loss] = {
                'train': train_losses,
                'eval': eval_losses
            }
        
        return reformatted_dict