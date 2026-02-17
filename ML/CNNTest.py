from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn


# --- CNN Model ---
class CNNMnist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
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


def load_checkpoint(path: str, device: torch.device) -> CNNMnist:
    payload = torch.load(path, map_location=device)
    model = CNNMnist()
    model.load_state_dict(payload["model_state_dict"])  # now matches
    model.to(device).eval()
    return model


# --- Preprocessing ---
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def preprocess_image_to_mnist_tensor(image_path: str, invert: bool = False) -> torch.Tensor:
    img = Image.open(image_path).convert("L")

    # Only invert if your input is black digit on white background
    if invert:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), resample=Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)


def predict_digit(model_path: str, image_path: str, invert: bool = False) -> None:
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
    MODEL_PATH = "models/cnn_mnist.pt"
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
    # Your image is "white digit on black", so invert should be False

