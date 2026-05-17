import torch
import torch.nn as nn


class LogRegMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.linear(x)
