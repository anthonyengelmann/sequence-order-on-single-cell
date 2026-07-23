import torch.nn as nn

class FNNBaseline(nn.Module):
    def __init__(self, n_genes, hidden_dim=256, dropout = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)