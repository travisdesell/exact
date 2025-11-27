import torch
import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers, num_stocks, dropout_rate=0.2
        ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        # x: (B, T, E)
        out, _ = self.lstm(x)      # (B, T, hidden)
        last = out[:, -1, :]      # (B, hidden)
        last = torch.relu(last)
        logits = self.fc(last)     # (B, N)
        weights = torch.softmax(logits, dim=-1)
        return weights

class AttentionLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_stocks, dropout_rate=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        # Attention layer components
        self.attention = nn.Linear(hidden_size, 1)  # Simple attention to compute scores over time steps
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        # x: (B, T, E)
        out, _ = self.lstm(x)  # (B, T, hidden)
        
        # Attention: Compute scores for each time step
        attn_scores = self.attention(out).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)
        
        # Weighted sum of hidden states (context vector)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)  # (B, hidden)
        
        # Apply ReLU activation
        context = torch.relu(context)
        
        logits = self.fc(context)  # (B, N)
        weights = torch.softmax(logits, dim=-1)
        return weights