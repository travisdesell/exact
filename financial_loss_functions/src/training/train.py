import time
import torch
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
    def __init__(self, model, optimizer, loss, hparams, in_size: int, num_stocks: int):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

    def eval(self, val_ds: WindowDataset):
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
    
    def get_train_loss(self) -> float:
        if self.avg_train_loss:
            return self.avg_train_loss
        else:
            raise ValueError('Model not trained yet.')
    
    def get_val_loss(self) -> float:
        if self.avg_val_loss:
            return self.avg_val_loss
        else:
            raise ValueError('Model not evaluated yet.')
        
    def get_train_losses(self) -> List[float]:
        if len(self.train_losses) != 0:
            return self.train_losses
        else:
            raise ValueError('Model not trained yet.')
    
    def get_val_losses(self) -> List[float]:
        if len(self.val_losses) != 0:
            return self.val_losses
        else:
            raise ValueError('Model not evaluated yet.')
        
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
    ax1.plot(train_losses, marker='o', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Loss')
    ax1.grid(True)

    # Right: validation loss
    ax2.plot(val_losses, marker='o', linestyle='-')
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

    plt.close(fig)