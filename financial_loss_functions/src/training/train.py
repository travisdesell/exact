import time
import torch
import numpy as np
from typing import List
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
    def __init__(
            self, model, optimizer, loss, hparams, in_size: int, num_stocks: int
        ):
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
        "Getter for allocation weights as numpy array"
        wt_array = []
        for w in self.val_alloc_weights:
            wt_array.append(w.numpy())
        return np.vstack(wt_array)
        
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
    def __init__(self, eval_returns: np.ndarray):
        # Returns by window
        self.eval_returns = eval_returns

        # Different Weights
        self.eval_weights = None
        self.eq_weights = None
        
        # Returns for each window
        self.all_daily_returns = {} # Add all returns for every window
        self.all_total_returns = {}

    def _equal_weight_pf(self, num_tickers) -> np.array:
        return np.full((num_tickers), 1/num_tickers)

    def calc_pf_daily_rets(self, eval_weights: np.ndarray, model_name: str):
        self.eval_weights = eval_weights
        
        pf_daily_returns = []
        pf_total_returns = []
        
        # Iterating over window samples
        for i in range(self.eval_weights.shape[0]):
            weights = self.eval_weights[i]  # Shape: (50,)
            returns = self.eval_returns[i]  # Shape: (50, 50) - time steps x assets
            
            # Calculate daily portfolio returns (dot product at each time step)
            daily_returns = np.dot(returns, weights)
            pf_daily_returns.append(daily_returns) # Shape: (50,)
            
            # Calculate total return for the entire window
            # Compounded returns
            total_return = np.prod(1 + daily_returns) - 1
            pf_total_returns.append(round(total_return, 4))
        
        self.all_daily_returns[model_name] = np.array(pf_daily_returns)
        self.all_total_returns[model_name] = pf_total_returns
    
    def calc_eq_wt_daily_rets(self): 
        # For equal weight portfolio
        self.eq_weights = self._equal_weight_pf(self.eval_returns.shape[2])
        
        eq_wt_daily_returns = []
        eq_wt_total_returns = []
        
        for i in range(self.eval_returns.shape[0]):
            returns = self.eval_returns[i]  # Shape: (50, 50)
            daily_returns = np.dot(returns, self.eq_weights)
            eq_wt_daily_returns.append(daily_returns)  # Shape: (50,)

            # Calculate total return for the entire window
            total_return = np.prod(1 + daily_returns) - 1
            eq_wt_total_returns.append(round(total_return, 4))

        self.all_daily_returns['Equal Weight'] = np.array(eq_wt_daily_returns)
        self.all_total_returns['Equal Weight'] = eq_wt_total_returns

    def plot_windowed_comparison(self, output_path: str, plot: bool=False):
        # for pf_type, array in self.all_daily_returns.items():
        if not self.all_daily_returns:
            print(
                'No daily returns calculated. Run calc_pf_daily_rets and calc_eq_wt_daily_rets first.'
            )
            return
        
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
