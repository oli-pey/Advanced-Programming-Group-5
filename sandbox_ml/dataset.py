from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from DB.database import SandboxSample, SandboxClass


SANDBOX_IMAGE_SIZE = 64

sandbox_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((SANDBOX_IMAGE_SIZE, SANDBOX_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class SandboxImageDataset(Dataset):
    def __init__(
        self,
        samples: list[SandboxSample],
        class_to_index: dict[int, int],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.class_to_index = class_to_index
        self.transform = transform or sandbox_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = Path(sample.image_path)

        if not image_path.exists():
            raise FileNotFoundError(f'Image file not found: {image_path}')

        image = Image.open(image_path).convert('L')
        x = self.transform(image)
        y = self.class_to_index[sample.class_id]
        return x, torch.tensor(y, dtype=torch.long)


def build_class_maps(classes: list[SandboxClass]) -> tuple[dict[int, int], dict[int, str]]:
    sorted_classes = sorted(classes, key=lambda c: c.name)

    class_to_index = {c.id: idx for idx, c in enumerate(sorted_classes)}
    index_to_label = {idx: c.name for idx, c in enumerate(sorted_classes)}

    return class_to_index, index_to_label
