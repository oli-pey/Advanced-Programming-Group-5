import io
import numpy as np
from PIL import Image, ImageOps
import torch

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def preprocess_png_bytes(image_bytes: bytes, invert: bool = True) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    if invert:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), resample=Image.Resampling.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
