import gc
import os
import sys
import time
import copy
import torch
import optuna
import psutil
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable, Type, Any, Optional
from src.utils.formatting import reformat_hparams
from src.data_processing.dataset import WindowDataset
from src.utils.device import set_seed, get_best_device
from src.utils.io import save_pickle_temp, load_pickle_temp

from src.evaluation.evaluator import Evaluator, EqualWeightCalculator
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
        optimizer: torch.optim.Optimizer,
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
        Initialize Trainer instance to train given model.

        @param model torch.nn.Module
            Pytorch neural network class to be trained and evaluated
        @param optimizer torch.optim
            Pytorch optimization class to be used to loss optimization
        @param loss Callable
            Custom loss function
        @param model_hparams dict
            Dictionary containing hyperparameters required for model initialization
        @param optimizer_hparams dict
            Dictionary containing hyperparameters required for optimizer initialization
        @param train_hparams Dict
            Dictionary containing hyperparameters required for training
        @param loss_hparams dict | None
            Dictionary containing hyperparameters for loss functions. Default = None
        @param in_size int
            Size of input window
        @param num_stocks int
            Number of stocks, i.e, number of output nodes 
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
        
        # Initialize optimizer with its specific hyperparameters
        self.optimizer = optimizer(
            self.model.parameters(),
            **optimizer_hparams
        )

        if scheduler_hparams:
            # 2. Initialize Scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',       # We want to minimize loss
                **scheduler_hparams
            )

            self.lr_schedule = True
        else:
            self.lr_schedule = False

        self.loss = loss

        self.train_hparams = train_hparams
        self.loss_hparams = loss_hparams or {}
        
        self.train_losses = [] # Stores average losses, for plotting
        self.val_losses = [] # Stores average losses, for plotting
        
        self.eval_losses = [] # For out of sample eval. Stores batch losses, not average
        
        self.avg_train_loss = None
        self.avg_eval_loss = None

        self.eval_alloc_weights = []

        # For Early Stopping
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
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

    def train(
            self, train_ds: WindowDataset, val_ds: Optional[WindowDataset] = None
        ):
        """
        Train inistalized model using a train data split.

        @param train_ds WindowDataset
            Training data split converted to windowed dataset tensors
        """
        start_time = time.time()

        # Pull hyperparameters with sensible defaults
        n_epochs = self.train_hparams['epochs']
        min_epochs = self.train_hparams.get('min_epochs', 0)
        patience = self.train_hparams.get('early_stop_patience', 20)
        min_delta = self.train_hparams.get('early_stop_min_delta', 1e-3)
        early_stopping = self.train_hparams.get('early_stopping', True)

        clip_grad_norm = self.train_hparams.get('clip_grad_norm', 0.5)
        
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

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
                self.optimizer.step()

                batch_size = xb.size(0)

                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

            epoch_avg_loss = total_loss_sum / total_samples
            self.train_losses.append(epoch_avg_loss)
            status_msg = f'Epoch {epoch} | Train Loss: {epoch_avg_loss:.4f}'
            self.avg_train_loss = epoch_avg_loss

            # --- Validation & Early Stopping Logic ---
            if val_ds is not None:
                avg_val_loss = self.validate(val_ds)
                self.val_losses.append(avg_val_loss)

                # --- STEP THE SCHEDULER HERE ---
                # It takes the current validation loss to decide if it should drop the LR
                if self.lr_schedule:
                    self.scheduler.step(avg_val_loss)
                
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
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    self.patience_counter = 0 
                else:
                    # ONLY start the "timer" after the warmup is over
                    if is_past_warmup:
                        self.patience_counter += 1

                # 2. THE EARLY STOPPING
                if early_stopping and is_past_warmup and self.patience_counter >= patience:
                    status_msg = status_msg + \
                        f' | Val Loss: {avg_val_loss:.4f}'
                    print(status_msg + f' | Time: {round(time.time() - epoch_start, 3)}s')
                    print(f'----- Early Stopping Triggered at Epoch {epoch} -----\n')
                    break
                        
                else:
                    status_msg = status_msg + f' | Val Loss: {avg_val_loss:.4f}'
            
            print(status_msg + f' | Time: {round(time.time() - epoch_start, 3)}s')
        
        # After the training loop
        if self.best_model_state is None:
            # No improvement ever after warm-up; fall back to final model
            self.best_val_loss = avg_val_loss
            self.best_train_loss = self.avg_train_loss  # from the last epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        else:
            self.model.load_state_dict(self.best_model_state)
            
        print(f'Training Complete. Best Val Loss: {self.best_val_loss:.4f}')
            
        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        print(f'Best Train Loss: {self.best_train_loss:.4f}, Time Taken: {time_taken}s')

    def validate(self, val_ds: WindowDataset):
        """
        Validation method to run on each training epoch
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
        Evaluate the trained model using a validation data split.
        
        @param val_ds WindowDataset
            Validation data split converted to windowed dataset tensors
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
        print(f'Average Eval Loss: {self.avg_eval_loss:.4f}, Time Taken: {time_taken}')

    def get_eval_alloc_weights(self) -> np.ndarray:
        """
        Getter for allocation weights as numpy array
        
        @return np.ndarray Portfolio allocation weights for each validation window
        """
        if self.eval_alloc_weights:
            wt_array = []
            for w in self.eval_alloc_weights:
                wt_array.append(w.numpy())
            return np.vstack(wt_array)
        else:
            print('Model must be trained and validated.')
            return None
    
    # def device_cleanup(self):
    #     if self.device_name == 'mps':
    #         try:
    #             # Empty MPS cache
    #             torch.mps.empty_cache()
            
    #         except Exception as e:
    #             print(f'MPS cleanup not available. Error: {e}')
            
    #     elif self.device_name == 'cuda':
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()

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

    def __init__(
            self,
            tune_metric: dict[str, MetricModel],
            seed_list: list[int],
            n_trails: int,
            n_jobs: int,
            torch_device: torch.device | str
        ):
        if tune_metric is not None:
            self.tune_metric = TypeAdapter(
                Dict[str, MetricModel]
            ).validate_python(tune_metric)
        else:
            self.tune_metric = tune_metric
        
        self.seed_list = seed_list
        self.n_trials = n_trails
        self.n_jobs = n_jobs
        self.torch_device = torch_device

        self.n_seeds = len(self.seed_list)

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
            self,model_name: str, loss_name: str,
            alloc_weights: np.ndarray, y_val: np.ndarray
        ) -> float:
        model_loss_name = f'{model_name}-{loss_name}'
        
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
            for i, seed in enumerate(self.seed_list):
                # IMPORTANT: Reset the world to this specific seed
                print(
                    '='*20,
                    f'Trial {trial.number}, seed {i+1}/{self.n_seeds} (seed={seed})',
                    '='*20
                )
                set_seed(seed)

                trainer = Trainer(
                    model=model_class,
                    optimizer=torch.optim.AdamW,
                    loss=loss_func,
                    model_hparams=m_hparams,
                    optimizer_hparams=o_hparams,
                    train_hparams=model_cfg['train'],
                    in_size=X_train_shape[2],
                    num_stocks=y_train_shape[2],
                    max_seq_len=X_train_shape[1],
                    scheduler_hparams=model_cfg.get('scheduler'),
                    loss_hparams=l_hparams,
                    device=self.torch_device if not model_name == 'DeformTime' else torch.device('cpu')
                )
                
                trainer.train(train_ds, val_ds)

                # We grab the losses from the trainer's "Best" epoch
                seed_train_losses.append(trainer.best_train_loss)
                seed_val_losses.append(trainer.best_val_loss)

                # Evaluate the get all the portfolio weights for eah window
                trainer.evaluate(val_ds)
                alloc_weights = trainer.get_eval_alloc_weights()
                
                # Calculate composite scores from allocation weights
                composite_score = self._calc_composite_score(
                    model_name,
                    loss_name,
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
                
                del trainer
            
            final_objective = self._calc_tuning_objective(
                composite_scores, seed_train_losses, seed_val_losses
            )

            return final_objective
        
        if model_tuning_space and y_val is not None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

            study = optuna.create_study(direction=self.direction, pruner=pruner)
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

class CandidatesGrid:
    models_hparams = 'nn_models'
    losses_hparams = 'losses'
    
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            loss_mode: str = 'custom',
            tune: bool = False,
            tune_metric: dict[str, MetricModel] | None = None,
            enable_diagnostics: bool = False,
            temp_dir: str | None = None
        ):
        """
        This runs either all models and loss functions or 
        one model with all loss functions or all models with on loss function.
        It should not run one model for one loss function as this class is for a grid.

        train_eval_* methods can be run only once with each instance
        """
        self.model_lib = model_lib
        self.loss_lib = loss_lib
        self.hparams_config = hparams_config
        self.torch_device = None

        if loss_mode not in ['all', 'custom']:
            raise ValueError('Incorrect Loss Mode. Mode must be `all` or `custom`')
        else:
            self.loss_mode = loss_mode
        
        self.tune = tune

        self.torch_device = get_best_device()
        
        # Tuner configuration
        self.seed_list = self.hparams_config.get('seed_list')
        if self.tune and tune_metric:
            tuner_config = self.hparams_config.get('tuner', {})
            print('Tuner Config:\n', tuner_config)
            
            n_trails = tuner_config.get('n_tuning_trials', 20)
            n_jobs = tuner_config.get('n_jobs', 1)
            print(
                f'Tuning all models with {n_trails} trials, across seeds: {self.seed_list}.'
            )
            self.tuner = Tuner(
                tune_metric,
                self.seed_list,
                n_trails,
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
        self.temp_dir = temp_dir

        self.all_alloc_weights: dict[str, dict[str, np.ndarray]] = {}
        self.train_val_losses: dict[str, dict[str, list[float]]] = {}

        self.optimized_hparams = {} # Will be filled if tuned
    
    def required_reshapes(self, train_data, returns_train, val_data, returns_val):
        # Implement different reshaping for different models if needed.
        # Move Reshaper instance from pipeline.py to here
        pass
    
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
        model_cfg = self.hparams_config[self.models_hparams][model_name]
        loss_cfg = self.hparams_config[self.losses_hparams].get(loss_name, {})
        
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
            # Map the best Optuna parameters into our new dictionary
            for k, v in best_found_params.items():
                if k in best_config['model']: best_config['model'][k] = v
                elif k in best_config['optimizer']: best_config['optimizer'][k] = v
                elif k in best_config['loss']: best_config['loss'][k] = v
            
            del study
        else:
            # If not tuning, we just use the original values and use the first seed in the seed list
            set_seed(self.seed_list[0])
        
        if self.tune:
            optimized_hparams = best_config
        else:
            optimized_hparams = None
        
        # --- 3. Final Training with the Captured Params ---
        final_trainer = Trainer(
            model=model_class,
            optimizer=torch.optim.AdamW,
            loss=loss_func,
            model_hparams=best_config['model'],
            optimizer_hparams=best_config['optimizer'],
            train_hparams=best_config['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2],
            max_seq_len=X_train_shape[1],
            scheduler_hparams=best_config['scheduler'],
            loss_hparams=best_config['loss'],
            device=self.torch_device if not model_name == 'DeformTime' else torch.device('cpu')
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

        del final_trainer

        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        return alloc_weights, train_val_losses, optimized_hparams
    
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
            print(f"  WARNING: {trainer_count} Trainer instances still in memory!")

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

    def train_eval_grid(
            self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray
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

        progress_count = 1
        # Loop over loss functions
        for loss_name, loss_func in losses_to_use.items():
            self.all_alloc_weights.setdefault(loss_name, {})

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
                        self.all_alloc_weights[loss_name][model_name] = alloc_weights
                        self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                        self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

                    except Exception as error:
                        print(
                            f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.', error
                        )
                        continue
                    finally:
                        progress_count += 1
        
        return self.all_alloc_weights    
    
    def set_temp_directory(self, temp_dir: str):
        self.temp_dir = temp_dir

    def train_eval_grid_mpi(
            self, 
            X_train: np.ndarray,
            y_train: np.ndarray, 
            X_val: np.ndarray,
            y_val: np.ndarray,
            comm,
            global_rank,
            size,
            local_rank
        ) -> dict[str, dict[str, np.ndarray]]:

        if not self.temp_dir or not os.path.exists(self.temp_dir):
            raise FileNotFoundError(
                f'Temp directory not found: {self.temp_dir}'
            )
        
        self.torch_device = get_best_device(local_rank)
        self.tuner.torch_device = self.torch_device ########## TEMPORARY FIX. NEEDS REFACTOR

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
        losses_to_use = self._build_losses_to_use()
        
        # Calculate number of models to train (model + loss combinations)
        n_models = self._count_only_models()
        total_train_count = len(losses_to_use) * n_models
        
        print(
            f'\nTraining {total_train_count} models in total.',
            f'Running all models with {self.loss_mode} losses.'
        )

        all_combos = []
        for loss_name, loss_func in losses_to_use.items():
            for category, models_dict in self.model_lib.items():
                for model_name, model_class in models_dict.items():
                    all_combos.append((loss_name, loss_func, model_name, model_class))

        # Distribute combos evenly across ranks
        chunk_size = len(all_combos) // size
        remainder = len(all_combos) % size
        start = global_rank * chunk_size + min(global_rank, remainder)
        end = start + chunk_size + (1 if global_rank < remainder else 0)
        this_ranks_combos = all_combos[start:end]

        # Print summary on each rank (optional)
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
                # Store in local dict
                if loss_name not in local_alloc_weights:
                    local_alloc_weights[loss_name] = {}
                
                local_alloc_weights[loss_name][model_name] = alloc_weights
                local_train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                local_optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams
            
            except Exception as e:
                print(f'Rank {global_rank}: Error on {model_name} - {loss_name}: {e}')
                continue
        
        temps_wts_prefix = 'temp_alloc_wts'
        temp_losses_prefix = 'temp_losses'
        temp_hparams_prefix = 'temp_hparams'
        
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
            self.all_alloc_weights = {}
            self.train_val_losses = {}
            for r in range(size):
                # Load all temp alloc wt files
                rank_alloc_weights = load_pickle_temp(
                    self.temp_dir / f'{temps_wts_prefix}_{r}.pkl'
                )
                # Merge into self.all_alloc_weights
                for loss_name, models_dict in rank_alloc_weights.items():
                    self.all_alloc_weights.setdefault(loss_name, {})
                    self.all_alloc_weights[loss_name].update(models_dict)
                
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
                for model_loss, hparms_dict, in rank_hparams.items():
                    self.optimized_hparams[model_loss] = hparms_dict
        else:
            return None
                
    def _search_model(self, model_name: str) -> Type | None:
        """Search for required model"""
        for _, models_dict in self.model_lib.items():
            if model_name in models_dict:
                return models_dict[model_name]
        return None

    def train_eval_one_model(
            self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray
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

        progress_count = 1
        # Loop over loss functions
        for loss_name, loss_func in losses_to_use.items():
            self.all_alloc_weights.setdefault(loss_name, {})

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
                self.all_alloc_weights[loss_name][model_name] = alloc_weights
                self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

            except Exception as error:
                print(
                    f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                    error
                )
                continue
            finally:
                progress_count += 1

        return self.all_alloc_weights

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

    def train_eval_one_loss(
            self, loss_name: str, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray
        ) -> dict[str, dict[str, np.ndarray]]:
        
        self._trained_check()

        # Converting to pytorch tensors
        train_ds = WindowDataset(X_train, y_train)
        val_ds   = WindowDataset(X_val, y_val)
    
        loss_func = self._search_loss_func(loss_name)
        
        if not loss_func: # loss function not found
            raise RuntimeError(f'{loss_name} LOSS FUNCTION NOT FOUND IN LIBRARY!')

        self.all_alloc_weights.setdefault(loss_name, {})

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
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights
                    self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
                    self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
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

        self.all_alloc_weights.setdefault(loss_name, {})

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
            self.all_alloc_weights[loss_name][model_name] = alloc_weights
            self.train_val_losses[f'{model_name}-{loss_name}'] = train_val_losses
            self.optimized_hparams[f'{model_name}-{loss_name}'] = optimized_hparams

        except Exception as e:
            print(f'DEBUG: Error while training {model_name}. Not training.', e)
        
        return self.all_alloc_weights

    def get_train_val_losses(self) -> dict[str, dict[str, list[float]]]:
        return self.train_val_losses
    
    def get_optimized_hparams(self) -> dict:
        return self.optimized_hparams