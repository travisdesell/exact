import torch
from torch.utils.data import DataLoader
from src.models.lstm import FlattenedLSTM
from src.training.loss_functions import sharpe_loss
from src.data_processing.dataset import WindowDataset

if torch.mps.is_available():
    device = torch.device('mps')
    print('Using mps for GPU acceleration.')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using cuda for GPU acceleration.')
else:
    print('No GPU acceleration. Using CPU.')
    device = torch.device('cpu')


def train_lstm_base(X_train, y_train, X_val, y_val):
    b, t, nxf = X_train.shape
    _, t_out, n = y_train.shape

    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model = FlattenedLSTM(
        input_size=nxf,       # 300
        hidden_size=16,
        num_layers=2,
        num_stocks=n        # 50
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(64):
        model.train()
        total_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            weights = model(xb)              # (B, N)
            loss = sharpe_loss(weights, yb)  # Sharpe-based loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        
        print(f'Epoch {epoch} | Train Loss: {loss:.4f}')
        
        avg_train_loss = total_loss / len(train_loader)

    # --- validation ---
    model.eval()
    with torch.no_grad():
        val_losses = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            weights = model(xb)
            val_loss = sharpe_loss(weights, yb)
            val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        # print(weights)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")