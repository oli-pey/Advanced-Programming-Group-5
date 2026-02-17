from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LogRegMNIST(nn.Module):
    """
    Multinomial logistic regression for MNIST:
    flatten(28*28) -> Linear(784, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, 1, 28, 28) or (N, 784)
        returns logits shape: (N, 10)
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)  # (N, 784)
        return self.linear(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns probabilities shape: (N, 10)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns predicted class indices shape: (N,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    lr: float = 0.1
    epochs: int = 5
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


def train_mnist_logreg(
    model: LogRegMNIST,
    config: TrainConfig,
    device: torch.device,
    data_dir: str = "data",
) -> Dict[str, float]:
    torch.manual_seed(config.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0..1]
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST normalization
    ])

    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

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

        train_acc = _accuracy(model, train_loader, device)
        test_acc = _accuracy(model, test_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return {"train_acc": train_acc, "test_acc": test_acc, "loss": avg_loss}


def save_checkpoint(model: LogRegMNIST, path: str, extra: Dict | None = None) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path_obj)


def load_checkpoint(path: str, device: torch.device) -> Tuple[LogRegMNIST, Dict]:
    payload = torch.load(path, map_location=device)
    model = LogRegMNIST()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("extra", {})


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogRegMNIST()
    metrics = train_mnist_logreg(model, TrainConfig(epochs=10, lr=0.1), device=device)
    save_checkpoint(model, "models/logreg_mnist.pt", extra={"metrics": metrics})
