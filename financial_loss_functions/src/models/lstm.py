import torch
import torch.nn as nn

class FlattenedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_stocks):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        # x: (B, T, E)
        out, _ = self.lstm(x)      # (B, T, hidden)
        last = out[:, -1, :]       # (B, hidden)
        logits = self.fc(last)     # (B, N)
        weights = torch.softmax(logits, dim=-1)
        return weights