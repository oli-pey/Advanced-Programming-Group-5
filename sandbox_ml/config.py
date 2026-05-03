from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    val_split: float = 0.2
    min_classes: int = 2
    min_samples_per_class: int = 5
    seed: int = 42
