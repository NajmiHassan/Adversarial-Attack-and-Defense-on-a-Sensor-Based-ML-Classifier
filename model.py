import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=561, num_classes=6, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
