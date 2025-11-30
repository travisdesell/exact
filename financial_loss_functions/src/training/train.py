import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data_processing.dataset import WindowDataset

if torch.mps.is_available():
    DEVICE = torch.device('mps')
    print('Using mps for GPU acceleration.')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using cuda for GPU acceleration.')
else:
    DEVICE = torch.device('cpu')
    print('No GPU acceleration. Using CPU.')


class Trainer:
    """
    Class to train provided models with provided hyperparameters.
    """
    def __init__(
            self, model, optimizer, loss, hparams: Dict, in_size: int, num_stocks: int
        ):
        """
        Initialize Trainer instance to train given model.

        Parameters
        ----------
        model: torch.nn.Module
            Pytorch neural network object to be trained and evaluated
        optimizer: torch.optim
            Pytorch optimization object to be used to loss optimization
        loss
            Custom loss function
        hparams: Dict
            Dictionary containing hyperparameters required for training
        in_size: int
            Size of input window
        num_stocks: int
            Number of stocks, i.e, number of output nodes 
        """
        self.device = DEVICE
        self.hparams = hparams
        print('Training hyperparameters:\n', self.hparams)
        
        self.model = model(
            input_size = in_size,       # 300
            hidden_size = self.hparams['hidden_size'],
            num_layers = self.hparams['num_layers'],
            num_stocks = num_stocks,        # 50
            dropout_rate = self.hparams['dropout']
        ).to(self.device)
        
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams.get('weight_decay', 1e-5)
        )
        self.loss = loss
        
        self.train_losses = []
        self.val_losses = []
        self.avg_train_loss = None
        self.avg_val_loss = None

        self.val_alloc_weights = []
    
    def train(self, train_ds: WindowDataset):
        """
        Train inistalized model using a train data split.

        Parameters
        ----------
        train_ds: WindowDataset
            Training data split converted to windowed dataset tensors
        """
        start_time = time.time()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.hparams['train_batch_size'],
            shuffle=False
        )

        for epoch in range(self.hparams['epochs']):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
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
        
        Parameters
        ----------
        val_ds: WindowDataset
            Validation data split converted to windowed dataset tensors
        """
        start_time = time.time()
        val_loader = DataLoader(
            val_ds,
            batch_size=self.hparams['val_batch_size'],
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
        """Getter for allocation weights as numpy array"""
        if self.val_alloc_weights:
            wt_array = []
            for w in self.val_alloc_weights:
                wt_array.append(w.numpy())
            return np.vstack(wt_array)
        else:
            print('Model must be trained and validated.')
            return None
        
def train_val_losses_plot(
    train_losses: List[float],
    val_losses: List[float],
    title: str,
    output_path: str,
    plot: bool = False,
    sharey: bool = False,          # set True to use same y-axis for easier comparison
    figsize: tuple = (12, 4)
):
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

    plt.close()


class Evaluator:
    """
    Class to evaulate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks.
    """
    def __init__(self, eval_returns: np.ndarray):
        """
        Initialize Evaluator instance to evaulate and compare all generated weights.

        Parameters
        ----------
        eval_returns: np.ndarray
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
    def _equal_weight_pf(num_tickers) -> np.array:
        """Calculates simple equal weights for a portfolio"""
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
        
        Parameters
        ----------
        returns_arr: np.array (n,)
            array of discrete returns for each time step
        risk_free_rate: float
            Risk free rate for window used for returns. Default = 0.0
        """
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)

        return (mean_ret - risk_free_rate) / std_ret

    def calc_pf_daily_rets(self, eval_weights: np.ndarray, model_name: str):
        """
        Calculates daily returns for the given portfolio weights for each given window.
        Portfolio Weights (n,) x Returns (T, n) = weighted returns.

        Parameters
        ----------
        eval_weights: np.ndarray
            Portfolio allocation weights for which weighted returns need to be calculated
        model_name: str
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

        Parameters
        ----------
        metric: str
            String name of the metric to be calculated. `returns` or `sharpe`

        Returns
        -------
        all_returns: Dict[str, List]
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

        Parameters
        ----------
        output_path: str
            File path to save plot
        plot: bool
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