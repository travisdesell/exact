import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Number of samples
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return one sample
        return self.X[idx], self.y[idx]