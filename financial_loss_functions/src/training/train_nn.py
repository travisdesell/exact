import gc
import os
import sys
import time
import copy
import torch
import optuna
import psutil
import random
import traceback
import numpy as np
import pandas as pd
from torch import Tensor
from abc import ABC, abstractmethod
from src.utils.device import set_seed
from typing import Callable, Type, Any
from torch.utils.data import DataLoader
from src.utils.formatting import reformat_hparams
from src.data_processing.dataset import WindowDataset, Reshaper, calc_current_idxs
from src.utils.io import save_pickle_temp, load_pickle_temp, delete_file

from src.evaluation.evaluator import Evaluator, EqualWeightCalculator
from pydantic import BaseModel, TypeAdapter
from typing import Callable, Dict, Literal
from scipy import stats

optuna.logging.set_verbosity(optuna.logging.INFO)
import warnings # Temporary

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
        return self.best_train_loss, self.best_val_loss
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
    
    def device_cleanup(self):
        if self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except Exception as e:
                print(f'MPS cleanup not available. Error: {e}')
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
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
    direction = 'maximize'
    max_seed = 1000000
    n_startup_perc = 0.3

    def __init__(
            self,
            tune_metric: dict[str, MetricModel],
            n_seeds: int,
            n_trials: int,
            n_warmup_steps: int, 
            n_jobs: int,
            torch_device: torch.device | str
        ):
        if tune_metric is not None:
            self.tune_metric = TypeAdapter(
                Dict[str, MetricModel]
            ).validate_python(tune_metric)
        else:
            self.tune_metric = tune_metric
        
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.n_warmup_steps = n_warmup_steps
        self.n_jobs = n_jobs
        self.torch_device = torch_device

        self.n_startup_trials = int(self.n_trials * self.n_startup_perc)

        self.benchmark_rets = None # benchmark returns for information ratio style metrics
    
    def _calc_pf_metrics_for_seed(
            self, model_name: str, loss_name: str, seed: int,
            alloc_weights: np.ndarray, y_val: np.ndarray
        ) -> dict[str, float]:
        evaluator = Evaluator(y_val, None)
        evaluator.calc_pf_daily_rets(alloc_weights, f'{model_name}-{loss_name}-{seed}')

        seed_metrics = {}
        for met_name, met_dict in self.tune_metric.items():
            metric_mean = evaluator.calc_metric_performance(met_dict.func, mean=True)
            seed_metrics[met_name] = metric_mean.item() # .item() because we calculate only 1 value
        
        return seed_metrics
    
    def _calc_excess_returns(self, model_rets: np.ndarray) -> np.ndarray:
        return model_rets - self.benchmark_rets

    def _calc_composite_score(
            self, model_loss_name,
            alloc_weights: np.ndarray, y_val: np.ndarray
        ) -> float:
        
        evaluator = Evaluator(y_val, None)
        evaluator.calc_pf_daily_rets(alloc_weights, model_loss_name)
        model_rets = evaluator.get_rets_for_one(model_loss_name)
        excess_rets = self._calc_excess_returns(model_rets)
        evaluator.update_rets_for_one(model_loss_name, excess_rets)

        composite_score = 0
        for _, met_dict in self.tune_metric.items():
            metric_mean = evaluator.calc_metric_performance(met_dict.func, mean=True)
            if met_dict.sign == '+':
                composite_score += metric_mean.item() # .item() since the series will have only 1 value
            elif met_dict.sign == '-':
                composite_score -= metric_mean.item()
            else:
                raise ValueError(
                    'Provide only linear operators like + or -. \
                        System designed to take only linear formulas as of now'
                    )
        
        if composite_score == 0:
            print('DEBUG: Got a 0 score. Something must be wrong.')
        
        return composite_score

    def calc_hinge_penalty(
            self, seed_train_losses: list[float], seed_val_losses: list[float],
            eps: float = 1e-9
        ) -> float:

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
        calculate tuing objective from statistics of composite scores across seeds
        and gap penalty from train - val losses.
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

    def _run_tuning_study(
            self,
            model_name: str,
            model_class: Type,
            loss_name: str,
            loss_func: Callable, 
            train_ds: 'WindowDataset',
            val_ds: 'WindowDataset',
            X_train_shape: torch.Size,
            y_train_shape: torch.Size,
            y_val: np.ndarray,
            model_cfg: dict,
            loss_cfg: dict
        ):
        
        model_loss_name = f'{model_name}-{loss_name}'

        # # Calculate equal weight portfolio & its returns as benchmark
        if self.benchmark_rets is None:
            eq_wt_calc = EqualWeightCalculator(y_val)
            self.benchmark_rets = eq_wt_calc.calc_eq_wt_daily_rets()

        model_tuning_space = model_cfg.get('tuning', {})
        
        loss_lambdas = loss_cfg.get('lambdas') if loss_cfg else {}
        loss_tuning_space = loss_cfg.get('tuning', {}) if loss_cfg else {}

        def _objective(trial):
            # 1. Start with base hparams from JSON
            
            m_hparams = copy.deepcopy(model_cfg['model'])
            o_hparams = copy.deepcopy(model_cfg['optimizer'])
            l_hparams = copy.deepcopy(loss_lambdas)
            
            # 2. Dynamically update hparams from the JSON tuning space
            for param_name, space in model_tuning_space.items():
                stype = space['type']
                if stype == 'float':
                    val = trial.suggest_float(
                        param_name, space['low'], space['high'], log=space.get('log', False)
                    )
                elif stype == 'int':
                    val = trial.suggest_int(param_name, space['low'], space['high'])
                elif stype == 'categorical':
                    val = trial.suggest_categorical(param_name, space['choices'])
                
                # Map the suggested value back to the correct dictionary
                if param_name in m_hparams:
                    m_hparams[param_name] = val
                elif param_name in o_hparams:
                    o_hparams[param_name] = val
            
            if l_hparams:
                for lambda_name, space in loss_tuning_space.items():
                    stype = space['type']
                    if stype == 'float':
                        val = trial.suggest_float(
                            lambda_name, space['low'], space['high'], log=space.get('log', False)
                        )
                    elif stype == 'int':
                        val = trial.suggest_int(lambda_name, space['low'], space['high'])
                    elif stype == 'categorical':
                        val = trial.suggest_categorical(lambda_name, space['choices'])
                    
                    if lambda_name in l_hparams:
                        l_hparams[lambda_name] = val
                
            # Cross-seed training
            composite_scores = []
            seed_train_losses = []
            seed_val_losses = []
            seed_best_epochs = []

            # loop over list of random seeds
            rng = random.Random(trial.number) # To avoid global random state and for reproducibilty
            rnd_seeds = [rng.randint(0, self.max_seed) for _ in range(self.n_seeds)]
            print(
                '+'*20,
                f'Trial {trial.number}, seeds: {rnd_seeds}',
                '+'*20
            )
            for i, seed in enumerate(rnd_seeds):
                # IMPORTANT: Reset the world to this specific seed
                print(
                    '='*20,
                    f'Trial {trial.number}, seed {i+1}/{self.n_seeds} (seed={seed})',
                    '='*20
                )
                set_seed(seed)

                trainer = Trainer(
                    model=model_class,
                    loss=loss_func,
                    model_hparams=m_hparams,
                    optimizer_hparams=o_hparams,
                    train_hparams=model_cfg['train'],
                    in_size=X_train_shape[2],
                    num_stocks=y_train_shape[2],
                    max_seq_len=X_train_shape[1],
                    scheduler_hparams=model_cfg.get('scheduler'),
                    loss_hparams=l_hparams,
                    device=self.torch_device
                )
                
                trainer.train(train_ds, val_ds)

                # We grab the losses from the trainer's "Best" epoch
                best_train_loss, best_val_loss = trainer.get_best_losses()
                seed_train_losses.append(best_train_loss)
                seed_val_losses.append(best_val_loss)

                seed_best_epochs.append(trainer.get_best_epoch())

                # Evaluate the get all the portfolio weights for eah window
                trainer.evaluate(val_ds)
                alloc_weights = trainer.get_eval_alloc_weights()
                
                # Calculate composite scores from allocation weights
                composite_score = self._calc_composite_score(
                    model_loss_name,
                    alloc_weights,
                    y_val
                )
                print(
                    f'Composite Score for trial: {trial.number}, seed: {seed} = {composite_score}'
                )
                composite_scores.append(composite_score)

                # --- PRUNING LOGIC START ---
                # Report the score of the CURRENT seed (i)
                # Optuna tracks "step i" across all trials
                trial.report(composite_score, step=i)

                # Check if this trial should be killed
                if trial.should_prune():
                    print(f'!!!! Trial {trial.number} pruned at seed {i+1} !!!!')
                    raise optuna.exceptions.TrialPruned()
                # --- PRUNING LOGIC END ---
                
                trainer.device_cleanup()
                del trainer
            
            # Inside _objective, after the seed loop
            trial.set_user_attr('best_epochs', seed_best_epochs)
            
            final_objective = self._calc_tuning_objective(
                composite_scores, seed_train_losses, seed_val_losses
            )

            return final_objective
        
        if model_tuning_space and y_val is not None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps
            )

            study = optuna.create_study(
                direction=self.direction,
                pruner=pruner,
                study_name=model_loss_name
            )
            study.optimize(
                _objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs
            )

            # GUARD: Check if we actually found a completed trial
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                print('WARNING: All trials were pruned. Returning the best pruned trial or default.')
            
            return study
        else:
            raise ValueError(
            f'Tuning enabled but no ranges found for {model_name}-{loss_name} or no oos evaluation data found'
        )
    
    def set_tuning_direction(self, direction: str):
        self.direction = direction


class GridUtilities(ABC):
    mdls_hparams_name = 'nn_models'
    ls_hparams_name = 'losses'

    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            torch_device: torch.device | str,
            mpi: bool = False,
            temp_dir: str | None = None
        ):
        self.model_lib = model_lib
        self.loss_lib = loss_lib
        self.hparams_config = hparams_config # Default config. Not optimized
        self.torch_device = torch_device
        self.mpi = mpi
        self.temp_dir = temp_dir
        
        self._temp_dir_check()

    def _temp_dir_check(self):
        if self.mpi:
            if not self.temp_dir or not os.path.exists(self.temp_dir):
                raise FileNotFoundError(
                    f'Temp directory not found: {self.temp_dir}'
                )
    
    def set_temp_directory(self, temp_dir: str):
        self.temp_dir = temp_dir
    
    def _search_model(self, model_name: str) -> Type | None:
        """Search for required model"""
        for _, models_dict in self.model_lib.items():
            if model_name in models_dict:
                return models_dict[model_name]
        return None
    
    def _search_loss_func(self, loss_name: str) -> Callable | None:
        """Search for required loss function"""
        for _, cat_dict in self.loss_lib.items():
            for _, sub_cat_dict in cat_dict.items():
                if loss_name in sub_cat_dict:
                    return sub_cat_dict[loss_name]
        
        # objectives = self.loss_lib['objectives']['__default__']
        # if loss_name in objectives:
        #     return objectives[loss_name]
        
        # custom_combos = self.loss_lib['custom']['__default__']
        # if loss_name in custom_combos:
        #     return custom_combos[loss_name]
        
        return None
    
    def _memory_diagnostics(self):
        """Print memory usage diagnostics"""
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 ** 3
        
        print(f"  Process memory: {mem_gb:.2f} GB")
        
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
    
    @staticmethod
    def _mpi_setup_check(mpi_items: list):
        for i in mpi_items:
            if i is None:
                raise ValueError(f'All necessary MPI values must be provided. {i} not provided.')

    @abstractmethod
    def _train_eval_helper(self):
        pass


class CandidatesGrid(GridUtilities):

    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            torch_device: torch.device | str,
            loss_mode: str = 'custom',
            tune: bool = False,
            tune_metric: dict[str, MetricModel] | None = None,
            mpi: bool = False,
            temp_dir: str | None = None,
            enable_diagnostics: bool = False
        ):
        """
        This runs either all models and loss functions or 
        one model with all loss functions or all models with on loss function.
        It should not run one model for one loss function as this class is for a grid.

        train_eval_* methods can be run only once with each instance
        """

        super().__init__(
            model_lib, loss_lib, hparams_config, torch_device, mpi, temp_dir
        )

        if loss_mode not in ['all', 'custom']:
            raise ValueError('Incorrect Loss Mode. Mode must be `all` or `custom`')
        else:
            self.loss_mode = loss_mode
        
        self.tune = tune
        
        # Tuner configuration
        if self.tune and tune_metric:
            tuner_config = self.hparams_config.get('tuner', {})
            print('Tuner Config:\n', tuner_config)
            
            n_seeds = tuner_config.get('n_seeds', 5)
            n_trials = tuner_config.get('n_tuning_trials', 30)
            n_warmup_steps = tuner_config.get('n_warmup_steps', 2)
            n_jobs = tuner_config.get('n_jobs', 1)
            print(
                f'Tuning all models with {n_trials} trials, across {n_seeds} seeds.'
            )
            self.tuner = Tuner(
                tune_metric,
                n_seeds,
                n_trials,
                n_warmup_steps,
                n_jobs,
                self.torch_device
            )
        
        elif self.tune and not tune_metric:
            raise ValueError(
                'Provide Tuning metric if tune = True.',
                "In the format {'<metric>': 'func': Callable, 'sign': '<sign>'}"
            )

        else:
            self.tuner = None
        
        self.enable_diagnostics = enable_diagnostics

        self.all_alloc_weights: dict[str, np.ndarray] = {}
        self.train_val_losses: dict[str, dict[str, list[float]]] = {}

        self.optimized_hparams = {} # Will be filled if tuned
    
    def _map_new_params(self, best_config: dict, new_hparams: dict) -> dict:
        
        # Map the best Optuna parameters into our new dictionary
        for k, v in new_hparams.items():
            if k in best_config['model']: best_config['model'][k] = v
            elif k in best_config['optimizer']: best_config['optimizer'][k] = v
            elif k in best_config['loss']: best_config['loss'][k] = v

        return best_config
    
    def _add_median_epoch(self, best_config: dict, best_epochs: list[int]):
        best_config['train']['median_epochs'] = int(np.median(best_epochs)) + 1
        # 1 is added because the epoch saved starts from 0 in the training loop
        return best_config

    def _train_eval_helper(
        self,
        model_name: str,
        model_class: Type,
        loss_name: str,
        loss_func: Callable, 
        train_ds: 'WindowDataset',
        val_ds: 'WindowDataset',
        X_train_shape: torch.Size,
        y_train_shape: torch.Size,
        y_val: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, list], dict | None]:
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
                train_ds,
                val_ds,
                X_train_shape,
                y_train_shape,
                y_val,
                model_cfg,
                loss_cfg,
            )
            best_found_params = study.best_params
            best_epochs = study.best_trial.user_attrs.get('best_epochs')
            
            best_config = self._map_new_params(best_config, best_found_params)
            best_config = self._add_median_epoch(best_config, best_epochs)
            
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
        
        # --- 3. Final Training with the Captured Params ---
        final_trainer = Trainer(
            model=model_class,
            loss=loss_func,
            model_hparams=best_config['model'],
            optimizer_hparams=best_config['optimizer'],
            train_hparams=best_config['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2],
            max_seq_len=X_train_shape[1],
            scheduler_hparams=best_config['scheduler'],
            loss_hparams=best_config['loss'],
            device=self.torch_device
        )
        
        final_trainer.train(train_ds, val_ds)
        final_trainer.evaluate(val_ds)

        # Logging and Diagnostics
        train_val_losses = {
            'train': final_trainer.train_losses,
            'val': final_trainer.val_losses,
            'eval': final_trainer.eval_losses
        }

        alloc_weights = final_trainer.get_eval_alloc_weights()

        final_trainer.device_cleanup()
        del final_trainer

        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        return alloc_weights, train_val_losses, optimized_hparams

    def _trained_check(self):
        """
        Check if training has been run before. 
        New instance of the class must be created for every training grid.
        """
        if len(self.all_alloc_weights) != 0:
            raise RuntimeError(
                'Allocation weights already predicted. Create new instance of this class.'
            )

    def _count_only_models(self) -> int:
        n_models = sum(len(models_dict) for models_dict in self.model_lib.values())
        return n_models

    def _build_losses_to_use(self) -> dict[str, Callable]:
        # Build losses that should be used base on loss_mode
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

    @staticmethod
    def _select_ranks_combos(all_combos: list, global_rank, size):
        # Distribute combos evenly across ranks
        chunk_size = len(all_combos) // size
        remainder = len(all_combos) % size
        start = global_rank * chunk_size + min(global_rank, remainder)
        end = start + chunk_size + (1 if global_rank < remainder else 0)
        this_ranks_combos = all_combos[start:end]

        return this_ranks_combos
    
    def _merge_all_results(
            self,
            size,
            temps_wts_prefix,
            temp_losses_prefix,
            temp_hparams_prefix
        ):
        """Merge all results into one dict if rank is 0, i.e., main process."""
        self.all_alloc_weights = {}
        self.train_val_losses = {}
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
                self.train_val_losses[model_loss] = losses_dict
            
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

    def train_eval_grid(
            self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray,
            comm = None,
            global_rank = None,
            size = None
        ) -> dict[str, dict[str, np.ndarray]]:
        """Loops over Loss functions first with a nested loop for models"""
        self._trained_check()

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
        losses_to_use = self._build_losses_to_use()
        
        # Calculate number of models to train (model + loss combinations)
        n_models = self._count_only_models()
        total_train_count = len(losses_to_use) * n_models
        
        print(
            f'\nTraining {total_train_count} models.',
            f'Running all models with {self.loss_mode} losses.'
        )
        
        # Not using MPI for distributed computing
        if self.mpi == False:
            progress_count = 1
            # Loop over loss functions
            for loss_name, loss_func in losses_to_use.items():

                for category, models_dict in self.model_lib.items():
                    # Loop over models
                    for model_name, model_class in models_dict.items():
                        print(
                            '\n', '-'*10,
                            f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                            '-'*10
                        )
                        try: 
                            
                            alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                                model_name,
                                model_class, 
                                loss_name,
                                loss_func,
                                train_ds,
                                val_ds,
                                X_train_shape,
                                y_train_shape,
                                y_val
                            )
                            self.all_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
                            self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                            self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

                        except Exception as error:
                            print(
                                f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.', error
                            )
                            traceback.print_exc()
                            continue
                        finally:
                            progress_count += 1
            
            return self.all_alloc_weights
        
        # If MPI is true for distributed computing
        else:
            self._mpi_setup_check([comm, global_rank, size])

            all_combos = []
            for loss_name, loss_func in losses_to_use.items():
                for category, models_dict in self.model_lib.items():
                    for model_name, model_class in models_dict.items():
                        all_combos.append((loss_name, loss_func, model_name, model_class))

            this_ranks_combos = self._select_ranks_combos(all_combos, global_rank, size)

            # Print summary on each rank
            print(f'Rank {global_rank}: tuning & training {len(this_ranks_combos)} combos.')
            sys.stdout.flush()

            # Local results dictionary
            local_alloc_weights = {}
            local_train_val_losses = {}
            local_optimized_hparams = {}
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(this_ranks_combos):
                print(f'\nRank {global_rank}: {idx+1}/{len(this_ranks_combos)} - {model_name} - {loss_name}')
                try:
                    alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                        model_name,
                        model_class,
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape,
                        y_val
                    )
                    
                    local_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
                    local_train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                    local_optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams
                
                except Exception as e:
                    print(f'Rank {global_rank}: Error on {model_name} - {loss_name}: {e}')
                    traceback.print_exc()
                    continue
            
            temps_wts_prefix = 'all_temp_alloc_wts'
            temp_losses_prefix = 'all_temp_losses'
            temp_hparams_prefix = 'all_temp_hparams'
            
            # Save local results to a rank‑specific file
            save_pickle_temp(
                local_alloc_weights,
                self.temp_dir / f'{temps_wts_prefix}_{global_rank}.pkl'
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
                    temps_wts_prefix,
                    temp_losses_prefix,
                    temp_hparams_prefix
                )
                
                return self.all_alloc_weights
            else:
                return None    
                
    def train_eval_one_model(
            self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray,
            comm = None,
            global_rank = None,
            size = None
        ) -> dict[str, dict[str, np.ndarray]]:

        self._trained_check()

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)
        
        # Search for model
        model_class = self._search_model(model_name)
        if not model_class: # model not found
            raise RuntimeError(f'{model_name} MODEL NOT FOUND IN LIBRARY!')
        
        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

        losses_to_use = self._build_losses_to_use()
        
        # Calculate number of models to train (model + loss combinations)
        total_train_count = len(losses_to_use)
        
        print(
            f'\nTraining {total_train_count} models.',
            f'Running all models with {self.loss_mode} losses.'
        )

        # Not using MPI for distributed computing
        if self.mpi == False:
            progress_count = 1
            # Loop over loss functions
            for loss_name, loss_func in losses_to_use.items():

                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                    '-'*10
                )
                try:        
                    alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape,
                        y_val
                    )
                    self.all_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
                    self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                    self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue
                finally:
                    progress_count += 1

            return self.all_alloc_weights
        
        # If MPI is true for distributed computing
        else:
            self._mpi_setup_check([comm, global_rank, size])

            all_combos = []
            for loss_name, loss_func in losses_to_use.items():
                all_combos.append((loss_name, loss_func, model_name, model_class))

            this_ranks_combos = self._select_ranks_combos(all_combos, global_rank, size)

            # Print summary on each rank
            print(f'Rank {global_rank}: tuning & training {len(this_ranks_combos)} combos.')
            sys.stdout.flush()

            # Local results dictionary
            local_alloc_weights = {}
            local_train_val_losses = {}
            local_optimized_hparams = {}
            for idx, (loss_name, loss_func, model_name, model_class) in enumerate(this_ranks_combos):
                print(f'\nRank {global_rank}: {idx+1}/{len(this_ranks_combos)} - {model_name} - {loss_name}')
                try:
                    alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                        model_name,
                        model_class,
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape,
                        y_val
                    )
                    
                    local_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
                    local_train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                    local_optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams
                
                except Exception as e:
                    print(f'Rank {global_rank}: Error on {model_name} - {loss_name}: {e}')
                    traceback.print_exc()
                    continue
            
            temps_wts_prefix = f'{model_name}_temp_alloc_wts'
            temp_losses_prefix = f'{model_name}_temp_losses'
            temp_hparams_prefix = f'{model_name}_temp_hparams'
            
            # Save local results to a rank‑specific file
            save_pickle_temp(
                local_alloc_weights,
                self.temp_dir / f'{temps_wts_prefix}_{global_rank}.pkl'
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
                    temps_wts_prefix,
                    temp_losses_prefix,
                    temp_hparams_prefix
                )
                
                return self.all_alloc_weights
            else:
                return None

    def train_eval_one_loss(
            self, loss_name: str, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray
        ) -> dict[str, dict[str, np.ndarray]]:

        # Deprecation warning
        warnings.warn(
            "CandidatesGrid.train_eval_one_loss() is deprecated and will be removed soon. "
            "Use CandidatesGrid.train_eval_grid() or CandidatesGrid.train_eval_one_model() instead.",
            category=FutureWarning,
            stacklevel=2
        )
        
        self._trained_check()

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)
    
        loss_func = self._search_loss_func(loss_name)
        
        if not loss_func: # loss function not found
            raise RuntimeError(f'{loss_name} LOSS FUNCTION NOT FOUND IN LIBRARY!')

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

        # Calculate number of models to train (model + loss combinations)
        total_train_count = self._count_only_models()
        print(
            f'\nTraining {total_train_count} models.',
        )
        progress_count = 1
        
        for category, models_dict in self.model_lib.items():
            # Loop over models
            for model_name, model_class in models_dict.items():
                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                    '-'*10
                )
                try: 
                    
                    alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape,
                        y_val
                    )
                    self.all_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
                    self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                    self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue
                finally:
                    progress_count += 1

        return self.all_alloc_weights

    def train_eval_one(
            self,
            model_name: str, 
            loss_name: str,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray
        ):

        self._trained_check()

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)

        model_class = self._search_model(model_name)
        if model_class is None:
            raise KeyError(f'Model {model_name} not found.')
        
        loss_func = self._search_loss_func(loss_name)
        if loss_func is None:
            raise KeyError(f'Loss Function {loss_name} not found.')
        
        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

        try:
            print('\n', '-'*10, f' Training {model_name}-{loss_name} ', '-'*10)
            alloc_weights, train_val_losses, optimized_hparams = self._train_eval_helper(
                model_name,
                model_class, 
                loss_name,
                loss_func,
                train_ds,
                val_ds,
                X_train_shape,
                y_train_shape,
                y_val
            )
            self.all_alloc_weights[f'{model_name}-{loss_name}'] = alloc_weights
            self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
            self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

        except Exception as e:
            print(f'DEBUG: Error while training {model_name}. Not training.', e)
            traceback.print_exc()
        
        return self.all_alloc_weights

    def get_train_val_losses(self) -> dict[str, dict[str, list[float]]]:
        return self.train_val_losses
    
    def get_optimized_hparams(self) -> dict:
        return self.optimized_hparams
    
class WalkForwardValidator(GridUtilities):
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            common_features: list[str],
            torch_device: torch.device | str,
            num_steps: int,
            filtered_models: list[tuple[str, str]] | None = None,
            mpi: bool = False,
            temp_dir: str | None = None,
            optimized_hparams = None
        ):

        super().__init__(
            model_lib, loss_lib, hparams_config, torch_device, mpi, temp_dir
        )
        self.num_steps = num_steps
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
    
    def _collected_combos(self):
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
                    model_name,
                    relevant_temp['models'][model_name],
                    loss_name,
                    relevant_temp['losses'][loss_name]
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
        model_loss_name = f'{model_name}-{loss_name}'
        
        # Gather best hyperparameters or use defaults
        if isinstance(self.optimized_hparams, dict) and model_loss_name in \
            self.optimized_hparams:
            best_hparams = self.optimized_hparams.get(model_loss_name)
            median_epochs = best_hparams.get('median_epochs')
            if median_epochs:
                best_hparams['epochs'] = median_epochs
            else:
                print(
                    f'WARNING: No median epochs found or is 0, {median_epochs}. Using default number of epochs!'
                )
        else:
            print('!No optimized hyperparameters provided, using defaults!')
            model_cfg = self.hparams_config[self.mdls_hparams_name][model_name]
            loss_cfg = self.hparams_config[self.ls_hparams_name].get(loss_name, {})

            best_hparams = reformat_hparams(model_cfg, loss_cfg)

        ### FOR TESTING ####
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
        for step in range(1, self.num_steps+1):
            print(
                '\n', '='*10,
                f' WFV: {model_name} - {loss_name}, Step: {step}/{self.num_steps}',
                '='*10
            )
            current_start, current_end = calc_current_idxs(step, self.stride)

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

        return np.vstack(steps_alloc_weights), np.vstack(steps_train_infer_losses)

    def validate_grid(
            self,
            init_train: pd.DataFrame,
            init_rets_train: pd.DataFrame,
            init_val: pd.DataFrame,
            init_rets_val: pd.DataFrame
        ) -> dict[str, np.ndarray]:

        self.reshaper.extract_features(init_train.columns)
        
        # Search and prepare models combos
        validate_count = len(self.filtered_models)
        print(
            f'\nWalk-forward validating {validate_count} models, over {self.num_steps} steps.'
        )
        all_combos = self._collected_combos()
        
        for idx, (model_name, model_cls, loss_name, loss_func) in enumerate(all_combos, 1):
            print(
                '\n', '-'*20,
                f' WFV: {model_name} - {loss_name}, Model: {idx}/{validate_count}',
                '-'*20
            )
            steps_alloc_weights, steps_train_infer_losses = self._walk_1_model(
                init_train,
                init_rets_train,
                init_val,
                init_rets_val,
                model_name,
                model_cls,
                loss_name,
                loss_func
            )

            self.all_alloc_weights[f'{model_name}-{loss_name}'] = steps_alloc_weights
            self.train_infer_losses[f'{model_name}-{loss_name}'] = steps_train_infer_losses

        return self.all_alloc_weights