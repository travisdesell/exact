import torch
import numpy as np
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
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        # x: (B, T, E)
        out, _ = self.lstm(x)      # (B, T, hidden)
        last = out[:, -1, :]      # (B, hidden)
        last = torch.relu(last)
        last = self.dropout(last)
        
        logits = self.fc(last)     # (B, N)
        # Strong equal-weight prior that never goes away
        equal_prior = torch.full_like(
            logits,
            fill_value=np.log(1.0 / logits.shape[-1]),
            device=logits.device
        )
        logits = logits + equal_prior
        weights = torch.softmax(logits, dim=-1)
        return weights

class SimpleAttentionLSTM(nn.Module):
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
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
    
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        # x: (B, T, E)
        out, _ = self.lstm(x)  # (B, T, hidden)
        out = torch.relu(out)
        out = self.dropout(out)
        
        attn_out, _ = self.attn(out, out, out)  # (B, T, H)
        context = attn_out.mean(dim=1)
        context = self.dropout(context)
        
        logits = self.fc(context)  # (B, N)
        # Strong equal-weight prior that never goes away
        equal_prior = torch.full_like(
            logits,
            fill_value=np.log(1.0 / logits.shape[-1]),
            device=logits.device
        )
        logits = logits + equal_prior
        weights = torch.softmax(logits, dim=-1)
        return weights