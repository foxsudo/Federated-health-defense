import torch.nn as nn

class HealthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # 2 classes : sain / malade
        )

    def forward(self, x):
        return self.net(x)
