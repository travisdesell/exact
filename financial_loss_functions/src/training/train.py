import gc
import os
import time
import torch
import psutil
import numpy as np
import pandas as pd
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from src.data_processing.dataset import Reshaper
from typing import List, Dict, Callable, Type, Any
from src.data_processing.dataset import WindowDataset

if torch.mps.is_available():
    DEVICE = 'mps'
    # DEVICE = torch.device(device_name)
    print('Using mps for GPU acceleration.')
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    # DEVICE = torch.device(device_name)
    print('Using cuda for GPU acceleration.')
else:
    DEVICE = 'cpu'
    # DEVICE = torch.device('cpu')
    print('No GPU acceleration. Using CPU.')


class Trainer:
    """
    Class to train provided models with provided hyperparameters.
    """
    def __init__(
        self, 
        model,  # Model class, not instance
        optimizer,
        loss: Callable,
        model_hparams: Dict,      # Specific to model architecture
        optimizer_hparams: Dict,  # Specific to optimizer
        train_hparams: Dict,      # Generic training params (epochs, batch_size, etc.)
        in_size: int,
        num_stocks: int,
        device_name: str = DEVICE
    ):
        """
        Initialize Trainer instance to train given model.

        @param model torch.nn.Module
            Pytorch neural network class to be trained and evaluated
        @param optimizer torch.optim
            Pytorch optimization class to be used to loss optimization
        @param loss Callable
            Custom loss function
        @param model_hparams Dict
            Dictionary containing hyperparameters required for model initialization
        @param optimizer_hparams Dict
            Dictionary containing hyperparameters required for optimizer initialization
        @param train_hparams Dict
            Dictionary containing hyperparameters required for training
        @param in_size int
            Size of input window
        @param num_stocks int
            Number of stocks, i.e, number of output nodes 
        """
        self.device_name = device_name
        self.device = torch.device(device_name)
        print('Model hyperparameters:\n', model_hparams)
        print('Optimizer hyperparameters:\n', optimizer_hparams)
        print('Training hyperparameters:\n', train_hparams)
        
        # Initialize model with its specific hyperparameters
        self.model = model(
            input_size=in_size,
            num_stocks=num_stocks,
            **model_hparams  # Unpack all model-specific hyperparams
        ).to(self.device)
        
        # Initialize optimizer with its specific hyperparameters
        self.optimizer = optimizer(
            self.model.parameters(),
            **optimizer_hparams
        )
        self.loss = loss

        self.train_hparams = train_hparams
        
        self.train_losses = []
        self.val_losses = []
        self.avg_train_loss = None
        self.avg_val_loss = None

        self.val_alloc_weights = []
    
    def train(self, train_ds: WindowDataset):
        """
        Train inistalized model using a train data split.

        @param train_ds WindowDataset
            Training data split converted to windowed dataset tensors
        """
        start_time = time.time()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_hparams['train_batch_size'],
            shuffle=False
        )

        for epoch in range(self.train_hparams['epochs']):
            epoch_start = time.time()
            self.model.train()
            total_loss_sum = 0.0
            total_samples = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                weights = self.model(xb)  # (B, N)
                loss = self.loss(weights, yb)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.train_hparams.get('clip_grad_norm', 0.5)
                )
                self.optimizer.step()

                batch_size = xb.size(0)

                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

            epoch_end = time.time()
            epoch_time = round(epoch_end - epoch_start, 3)
            
            epoch_avg_loss = total_loss_sum / total_samples
            self.train_losses.append(epoch_avg_loss)
            print(f'Epoch {epoch} | Train Loss: {epoch_avg_loss:.4f} | Took: {epoch_time}s')

            self.avg_train_loss = epoch_avg_loss

        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        print(f'Average Train Loss: {self.avg_train_loss:.4f}, Time Taken: {time_taken}s')

    def evaluate(self, val_ds: WindowDataset):
        """
        Evaluate the trained model using a validation data split.
        
        @param val_ds WindowDataset
            Validation data split converted to windowed dataset tensors
        """
        start_time = time.time()
        val_loader = DataLoader(
            val_ds,
            batch_size=self.train_hparams['val_batch_size'],
            shuffle=False
        )

        # --- validation ---
        self.model.eval()
        with torch.no_grad():
            self.val_losses = []
            total_loss, total_samples = 0.0, 0

            for xb, yb in val_loader:
                b = xb.size(0)
                xb, yb = xb.to(self.device), yb.to(self.device)
                weights = self.model(xb)
                loss = self.loss(weights, yb)

                # detach & move to CPU BEFORE appending
                self.val_alloc_weights.append(weights.detach().cpu()) 

                # --- store per-batch loss ---
                self.val_losses.append(loss.item())

                # --- accumulate weighted sum for overall avg ---
                total_loss += loss.item() * b
                total_samples += b

            # --- weighted average over all samples ---
            self.avg_val_loss = total_loss / total_samples
        
        end_time = time.time()
        time_taken = round(end_time-start_time, 3)
        print(f'Average Val Loss: {self.avg_val_loss:.4f}, Time Taken: {time_taken}')

    def get_val_alloc_weights(self) -> np.ndarray:
        """
        Getter for allocation weights as numpy array
        
        @return np.ndarray Portfolio allocation weights for each validation window
        """
        if self.val_alloc_weights:
            wt_array = []
            for w in self.val_alloc_weights:
                wt_array.append(w.numpy())
            return np.vstack(wt_array)
        else:
            print('Model must be trained and validated.')
            return None
    
    def device_cleanup(self):
        if self.device_name == 'mps':
            try:
                # Empty MPS cache
                torch.mps.empty_cache()
            
            except Exception as e:
                print(f'MPS cleanup not available. Error: {e}')
            
        elif self.device_name == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
def train_val_losses_plot(
    train_losses: List[float],
    val_losses: List[float],
    title: str,
    output_path: str,
    plot: bool = False,
    sharey: bool = False,          # set True to use same y-axis for easier comparison
    figsize: tuple = (12, 4)
):
    """Plot training and validation loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=sharey)

    # Left: train loss
    ax1.plot(train_losses, linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Loss')
    ax1.grid(True)

    # Right: validation loss
    ax2.plot(val_losses, linestyle='-')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation Loss')
    ax2.grid(True)

    # Overall title centered above subplots
    fig.suptitle(title)

    # Tight layout so title and labels don't overlap
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and optionally show
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()

    plt.close('all')
    plt.clf()
    plt.cla()


class Evaluator:
    """
    Class to evaulate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks.
    """
    def __init__(self, eval_returns: np.ndarray):
        """
        Initialize Evaluator instance to evaulate and compare all generated weights.

        @param eval_returns np.ndarray
            Daily returns which are used to evulate all methods/models
        """
        # Returns by window
        self.eval_returns = eval_returns

        # Different Weights
        self.eval_weights = None
        self.eq_weights = None
        
        # Returns for each window
        self.all_daily_returns = {} # Add all returns for every window

    @staticmethod
    def _equal_weight_pf(num_tickers: int) -> np.array:
        """
        Calculates simple equal weights for a portfolio
        weight for each stock = 1/num_tickers
        
        @param num_tickers int number of tickers in the dataset

        @return np.array equal weight portfolio allocation weights
        """
        return np.full((num_tickers), 1/num_tickers)

    @staticmethod
    def _cumulative_return(returns_arr: np.array) -> np.float64:
        """Calculate cummulative returns for given window"""
        return np.prod(1 + returns_arr) - 1
    
    @staticmethod
    def _basic_sharpe(
            returns_arr: np.array, risk_free_rate: float = 0.0
        ) -> np.float64:
        """
        Calculates non-annualized sharpe for given window.
        
        @param returns_arr np.array (n,)
            array of discrete returns for each time step
        @param risk_free_rate float
            Risk free rate for window used for returns. Default = 0.0
        """
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)

        return (mean_ret - risk_free_rate) / std_ret

    def calc_pf_daily_rets(self, eval_weights: np.ndarray, model_name: str):
        """
        Calculates daily returns for the given portfolio weights for each given window.
        Portfolio Weights (n,) x Returns (T, n) = weighted returns.

        @param eval_weights np.ndarray
            Portfolio allocation weights for which weighted returns need to be calculated
        @param model_name str
            Name of the model which generated the portfolio allocation weights
        """
        self.eval_weights = eval_weights
        
        pf_daily_returns = []
        
        # Iterating over window samples
        for i in range(self.eval_weights.shape[0]):
            weights = self.eval_weights[i]  # Shape: (50,)
            returns = self.eval_returns[i]  # Shape: (50, 50) - time steps x assets
            
            # Calculate daily portfolio returns (dot product at each time step)
            daily_returns = np.dot(returns, weights)
            pf_daily_returns.append(daily_returns) # Shape: (50,)
            
        self.all_daily_returns[model_name] = np.array(pf_daily_returns)
    
    def _daily_rets_calcd_check(self):
        if not self.all_daily_returns:
            raise ValueError(
                'No daily returns calculated.',
                'Run calc_pf_daily_rets and calc_eq_wt_daily_rets first.'
            )

    def calc_eq_wt_daily_rets(self): 
        """
        Calculates daily returns for the Equal Weighted portfolio for each given window.
        """
        # For equal weight portfolio
        self.eq_weights = self._equal_weight_pf(self.eval_returns.shape[2])
        
        eq_wt_daily_returns = []
        
        for i in range(self.eval_returns.shape[0]):
            returns = self.eval_returns[i]  # Shape: (50, 50)
            daily_returns = np.dot(returns, self.eq_weights)
            eq_wt_daily_returns.append(daily_returns)  # Shape: (50,)

        self.all_daily_returns['Equal Weight'] = np.array(eq_wt_daily_returns)

    def calc_total_performance(self, metric: str) -> pd.DataFrame:
        """
        Calculate per-window performance of all portfolios (incl. Equal Weight)
        based on given metric. 

        @param metric str
            String name of the metric to be calculated. `returns` or `sharpe`

        @return Dict[str, List]
            Dictionary containing calculated performance metric for each validation window
        """
        self._daily_rets_calcd_check()
        
        total_perfomances = {}
        for model, all_rets in self.all_daily_returns.items():
            model_rets = []
            for i in range(all_rets.shape[0]):
                if metric == 'returns':
                    window_metric = self._cumulative_return(all_rets[i])
                elif metric == 'sharpe':
                    window_metric = self._basic_sharpe(all_rets[i])
                
                model_rets.append(round(window_metric, 4))
            
            total_perfomances[model] = model_rets
        
        return pd.DataFrame(total_perfomances)

    def plot_windowed_comparison(self, output_path: str, plot: bool=False):
        """
        Plots and saves windowed comparisons of daily returns for every portfolio

        @param output_path str
            File path to save plot
        @param plot bool
            Toggle to show image while running code. Default = False
        """
        # for pf_type, array in self.all_daily_returns.items():
        self._daily_rets_calcd_check()
                
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        n_windows = next(iter(self.all_daily_returns.values())).shape[0]
        n_cols = min(3, n_windows)
        n_rows = (n_windows + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each window
        for window_idx in range(n_windows):
            ax = axes[window_idx]
            
            for i, (pf_type, daily_returns) in enumerate(self.all_daily_returns.items()):
                ax.plot(
                    daily_returns[window_idx],
                    label=pf_type,
                    color=colors[i % len(colors)],
                    alpha=0.8
                )
            
            ax.set_title(f'Window {window_idx + 1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Daily Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_windows, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        # Save and optionally show
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if plot:
            plt.show()

        plt.close('all')

class CandidatesGrid:
    def __init__(
            self,
            model_lib: Dict[str, Dict[str, Type]],
            loss_lib: Dict[str, Dict[str, Dict[str, Callable]]],
            hparams_config: Dict[str, Dict[str, Any]],
            results_dir: str | Path,
            loss_mode: str = 'all',
            enable_diagnostics: bool = False
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
        self.results_dir = results_dir

        if loss_mode not in ['all', 'custom']:
            raise ValueError('Incorrect Loss Mode. Mode must be `all` or `custom`')
        else:
            self.loss_mode = loss_mode
        self.enable_diagnostics = enable_diagnostics
        
        self.all_alloc_weights: Dict[str, Dict[str, np.ndarray]] = {}
    
    def required_reshapes(self, train_data, returns_train, val_data, returns_val):
        # Implement different reshaping for different models if needed.
        # Move Reshaper instance from pipeline.py to here
        pass

    def _train_eval_helper(
            self,
            model_name: str,
            model_obj: Type,
            loss_name: str,
            loss_func: Callable,
            train_ds: WindowDataset,
            val_ds: WindowDataset,
            X_train_shape: torch.Size,
            y_train_shape: torch.Size
        ) -> np.ndarray:
        #### Hyperparamater searching can be done here ####

        if self.enable_diagnostics:
            print(f'\n[Before training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        trainer = Trainer(
            model=model_obj,
            optimizer=optim.AdamW,
            loss=loss_func,
            model_hparams=self.hparams_config[model_name]['model'],
            optimizer_hparams=self.hparams_config[model_name]['optimizer'],
            train_hparams=self.hparams_config[model_name]['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2]
        )
        trainer.train(train_ds)
        trainer.evaluate(val_ds)

        loss_plot_name = model_name + f'-{loss_name}' + ' Loss Curves'
        # Plot loss curves
        train_val_losses_plot(
            trainer.train_losses,
            trainer.val_losses,
            loss_plot_name,
            self.results_dir / 'plots' / (loss_plot_name + '.png')
        )

        alloc_weights = trainer.get_val_alloc_weights()
        # trainer.device_cleanup()
        del trainer

        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()
        
        return alloc_weights
    
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

    def train_eval_grid(
            self, train_ds: WindowDataset, val_ds: WindowDataset
        ):
        """Loops over Loss functions first with a nested loop for models"""
        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
        
        # Grid with custom loss functions
        if 'custom' in self.loss_lib:
            print('\nTraining all models with all custom loss functions...')
            custom_combos = self.loss_lib['custom']['__default__'] # Custom combos have no category

            # Loop over loss functions
            for loss_name, loss_func in custom_combos.items():
                self.all_alloc_weights.setdefault(loss_name, {})

                for category, models_dict in self.model_lib.items():
                    # Loop over models
                    for model_name, model_obj in models_dict.items():
                        print('\n', '-'*10, f' Training {model_name} with {loss_name} ', '-'*10)
                        try: 
                            
                            alloc_weights = self._train_eval_helper(
                                model_name,
                                model_obj, 
                                loss_name,
                                loss_func,
                                train_ds,
                                val_ds,
                                X_train_shape,
                                y_train_shape
                            )
                            self.all_alloc_weights[loss_name][model_name] = alloc_weights

                        except Exception as error:
                            print(
                                f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.', error
                            )
                            continue
        
        else:
            print('\nNo custom loss functions provided. Moving to objectives.')
        
        if self.loss_mode == 'custom':
            return self.all_alloc_weights

        else: # mode == 'all'
            # Grid with only objectives
            print('\nTraining all models with all objectives (only) as loss functions...')
            objectives = self.loss_lib['objectives']['__default__'] # objectives have no category
            
            # Loop over loss functions
            for loss_name, loss_func in objectives.items():
                self.all_alloc_weights.setdefault(loss_name, {})
                
                for category, models_dict in self.model_lib.items():
                    # Loop over models
                    for model_name, model_obj in models_dict.items():
                        print('\n', '-'*10, f' Training {model_name} with {loss_name} ', '-'*10)
                        
                        
                        try: 
                            
                            alloc_weights = self._train_eval_helper(
                                model_name,
                                model_obj, 
                                loss_name,
                                loss_func,
                                train_ds,
                                val_ds,
                                X_train_shape,
                                y_train_shape
                            )
                            self.all_alloc_weights[loss_name][model_name] = alloc_weights

                        except Exception as error:
                            print(
                                f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                                error
                            )
                            continue
                
            return self.all_alloc_weights
    
    def _search_model(self, model_name: str) -> Type | None:
        """Search for required model"""
        for _, models_dict in self.model_lib.items():
            if model_name in models_dict:
                return models_dict[model_name]
        return None

    def train_eval_one_model(
            self, model_name: str, train_ds: WindowDataset, val_ds: WindowDataset
        ):

        if len(self.all_alloc_weights) != 0:
            print(self.all_alloc_weights)
            raise RuntimeError('Allocation weights already predicted. Create new instance of this class.')
        
        # Search for model
        model_obj = self._search_model(model_name)
        
        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
        
        # Grid with custom loss functions
        if 'custom' in self.loss_lib:
            print('\nTraining all models with all custom loss functions...')
            custom_combos = self.loss_lib['custom']['__default__'] # Custom combos have no category

            # Loop over loss functions
            for loss_name, loss_func in custom_combos.items():
                self.all_alloc_weights.setdefault(loss_name, {})

                print('\n', '-'*10, f' Training {model_name} with {loss_name} ', '-'*10)
                try:        
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_obj, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue

        else:
            print('\nNo custom loss functions provided. Moving to objectives.')
        
        if self.loss_mode == 'custom':
            return self.all_alloc_weights

        else: # mode == 'all'
            # Grid with only objectives
            print('\nTraining all models with all objectives (only) as loss functions...')
            objectives = self.loss_lib['objectives']['__default__'] # objectives have no category
            
            # Loop over loss functions
            for loss_name, loss_func in objectives.items():
                self.all_alloc_weights.setdefault(loss_name, {})
                try:        
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_obj, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue
            
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
            self, loss_name: str, train_ds: WindowDataset, val_ds: WindowDataset
        ):
        if len(self.all_alloc_weights) != 0:
            raise RuntimeError(
                'Allocation weights already predicted. Create new instance of this class.'
            )
    
        loss_func = self._search_loss_func(loss_name)

        self.all_alloc_weights.setdefault(loss_name, {})

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes() 
        
        for category, models_dict in self.model_lib.items():
            # Loop over models
            for model_name, model_obj in models_dict.items():
                print('\n', '-'*10, f' Training {model_name} with {loss_name} ', '-'*10)
                try: 
                    
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_obj, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue

        return self.all_alloc_weights