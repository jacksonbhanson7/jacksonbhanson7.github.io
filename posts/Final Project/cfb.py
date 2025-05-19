import torch
import torch.nn as nn
    
class FootballNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),     # e.g. 256 if hidden_size=128
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),   # e.g. 64 if hidden_size=128
            nn.ReLU(),

            nn.Linear(hidden_size // 2, 2)              # output layer for [home_score, away_score]
)

    def forward(self, x):
        return self.network(x)


