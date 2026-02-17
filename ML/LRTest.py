from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn


# --- Model (same as before) ---
class LogRegMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)  # (N, 784)
        return self.linear(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=1)


def load_checkpoint(path: str, device: torch.device) -> LogRegMNIST:
    payload = torch.load(path, map_location=device)
    model = LogRegMNIST()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model


# --- Preprocessing ---
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def preprocess_image_to_mnist_tensor(image_path: str, invert: bool = True) -> torch.Tensor:
    """
    Converts an image file to an MNIST-like tensor of shape (1, 1, 28, 28),
    normalized using MNIST mean/std.

    invert=True is often needed because MNIST digits are typically light (white)
    on dark (black) background. Many user images are black digit on white paper.
    """
    img = Image.open(image_path).convert("L")  # grayscale

    # Optional invert to match MNIST style (white digit on black background)
    if invert:
        img = ImageOps.invert(img)

    # Resize to 28x28
    img = img.resize((28, 28), resample=Image.Resampling.BILINEAR)

    # Convert to [0..1] float tensor
    arr = np.array(img, dtype=np.float32) / 255.0  # (28, 28)

    # Normalize like training
    arr = (arr - MNIST_MEAN) / MNIST_STD

    # Tensor shape: (1, 1, 28, 28)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return x


def predict_digit(model_path: str, image_path: str, invert: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path, device)

    x = preprocess_image_to_mnist_tensor(image_path, invert=invert).to(device)

    probs = model.predict_proba(x)[0].detach().cpu()
    pred = int(torch.argmax(probs).item())

    top3 = torch.topk(probs, k=3)
    print(f"Predicted digit: {pred}")
    print("Top-3 probabilities:")
    for p, idx in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {idx}: {p:.4f}")


if __name__ == "__main__":
    MODEL_PATH = "models/logreg_mnist.pt"
    DIGITS_DIR = Path("data/my_digits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(MODEL_PATH, device)

    print(f"Using model: {MODEL_PATH}")
    print(f"Scanning folder: {DIGITS_DIR}")
    print("-" * 40)

    for image_path in sorted(DIGITS_DIR.glob("*.png")):
        print(f"\nFile: {image_path.name}")

        x = preprocess_image_to_mnist_tensor(
            str(image_path),
            invert=False  # your images are white on black
        ).to(device)

        probs = model.predict_proba(x)[0].detach().cpu()
        pred = int(torch.argmax(probs).item())

        top3 = torch.topk(probs, k=3)

        print(f"Predicted digit: {pred}")
        print("Top-3 probabilities:")
        for p, idx in zip(top3.values.tolist(), top3.indices.tolist()):
            print(f"  {idx}: {p:.4f}")