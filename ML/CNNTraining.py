from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------
# CNN model (MNIST) — simple & strong
# ---------------------------
class CNNMnist(nn.Module):
    """
    A compact MNIST CNN:
      (1,28,28) -> Conv(32) -> ReLU -> MaxPool
               -> Conv(64) -> ReLU -> MaxPool
               -> Flatten  -> FC(128) -> ReLU -> FC(10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 64*7*7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=1)


# ---------------------------
# Training utilities
# ---------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 8
    num_workers: int = 2
    seed: int = 42


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def get_mnist_loaders(
    batch_size: int,
    num_workers: int,
    device: torch.device,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, test_loader


def train_mnist_cnn(
    model: CNNMnist,
    config: TrainConfig,
    device: torch.device,
    data_dir: str = "data",
) -> Dict[str, float]:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        data_dir=data_dir,
    )

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    last_loss = 0.0
    for epoch in range(1, config.epochs + 1):
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        last_loss = running_loss / max(1, len(train_loader))
        train_acc = _accuracy(model, train_loader, device)
        test_acc = _accuracy(model, test_loader, device)
        print(f"Epoch {epoch}/{config.epochs}  loss={last_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return {"train_acc": float(train_acc), "test_acc": float(test_acc), "loss": float(last_loss)}


# ---------------------------
# Save / load
# ---------------------------
def save_checkpoint(model: CNNMnist, path: str, extra: Dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "extra": extra or {}},
        p,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[CNNMnist, Dict]:
    payload = torch.load(path, map_location=device)
    model = CNNMnist()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload.get("extra", {})


# ---------------------------
# Run training
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMnist()

    config = TrainConfig(
        epochs=8,        # try 8–12 for ~99% (depends on random seed)
        lr=1e-3,
        batch_size=128,
    )

    metrics = train_mnist_cnn(model, config, device=device)
    save_checkpoint(model, "models/cnn_mnist.pt", extra={"metrics": metrics, "config": config.__dict__})
    print("Saved:", "models/cnn_mnist.pt", metrics)