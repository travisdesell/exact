import torch
from torch.utils.data import DataLoader
from src.data_processing.dataset import WindowDataset


class Trainer:
    def __init__(self, model, optimizer, loss, hparams, in_size: int, num_stocks: int):
        if torch.mps.is_available():
            self.device = torch.device('mps')
            print('Using mps for GPU acceleration.')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using cuda for GPU acceleration.')
        else:
            print('No GPU acceleration. Using CPU.')
            self.device = torch.device('cpu')
        
        self.hparams = hparams
        self.model = model(
            input_size = in_size,       # 300
            hidden_size = self.hparams['hidden_size'],
            num_layers = self.hparams['num_layers'],
            num_stocks = num_stocks,        # 50
            dropout_rate = self.hparams['dropout']
        ).to(self.device)
        
        self.optimizer = optimizer(self.model.parameters(), lr=self.hparams['lr'])
        self.loss = loss

        self.avg_train_loss = None
        self.avg_val_loss = None
    
    def train(self, train_ds: WindowDataset):
        train_loader = DataLoader(
            train_ds,
            batch_size=self.hparams['train_batch_size'],
            shuffle=False
        )

        for epoch in range(self.hparams['epochs']):
            self.model.train()
            total_loss = 0.0
            
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                weights = self.model(xb)              # (B, N)
                loss = self.loss(weights, yb)  # Finance-based loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            print(f'Epoch {epoch} | Train Loss: {loss:.4f}')

            self.avg_train_loss = total_loss / len(train_loader)

        print(f'Average Train Loss: {self.avg_train_loss:.4f}')

    def eval(self, val_ds: WindowDataset):
        val_loader = DataLoader(
            val_ds,
            batch_size=self.hparams['val_batch_size'],
            shuffle=False
        )

        # --- validation ---
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                weights = self.model(xb)
                val_loss = self.loss(weights, yb)
                val_losses.append(val_loss.item())
            self.avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'Average Val Loss: {self.avg_val_loss:.4f}')
    
    def get_train_loss(self):
        if self.avg_train_loss:
            return self.avg_train_loss
        else:
            raise ValueError('Model not trained yet.')
    
    def get_val_loss(self):
        if self.avg_val_loss:
            return self.avg_val_loss
        else:
            raise ValueError('Model not evaluated yet.')