from __future__ import annotations

import torch
import torch.nn as nn


class SandboxLogReg(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(64 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SandboxMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SandboxCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_sandbox_model(model_type: str, num_classes: int) -> nn.Module:
    model_type = model_type.lower()

    if model_type == 'logreg':
        return SandboxLogReg(num_classes=num_classes)

    if model_type == 'mlp':
        return SandboxMLP(num_classes=num_classes)

    if model_type == 'cnn':
        return SandboxCNN(num_classes=num_classes)

    raise ValueError(f'Unknown sandbox model type: {model_type}')
