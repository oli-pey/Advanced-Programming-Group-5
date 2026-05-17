from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
from uuid import uuid4

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from DB.database import (
    SandboxClass,
    SandboxDataset,
    SandboxSample,
    SandboxTrainedModel,
    SandboxTrainingJob,
    SessionLocal,
)
from sandbox_ml.config import TrainingConfig
from sandbox_ml.dataset import SandboxImageDataset, build_class_maps
from sandbox_ml.models import create_sandbox_model


MODEL_STORAGE_DIR = Path('storage') / 'sandbox_models'


class SandboxTrainingError(Exception):
    pass


def _validate_dataset(
    classes: list[SandboxClass],
    samples: list[SandboxSample],
    config: TrainingConfig,
) -> None:
    if len(classes) < config.min_classes:
        raise SandboxTrainingError(
            f'At least {config.min_classes} classes are required for training.'
        )

    counts = Counter(sample.class_id for sample in samples)

    missing = [
        sandbox_class.name
        for sandbox_class in classes
        if counts.get(sandbox_class.id, 0) < config.min_samples_per_class
    ]

    if missing:
        raise SandboxTrainingError(
            f'Each class needs at least {config.min_samples_per_class} samples. '
            f'Not enough samples for: {", ".join(missing)}'
        )


def _split_indices(
    total: int,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    val_size = max(1, int(total * val_split))
    train_size = total - val_size

    if train_size < 1:
        raise SandboxTrainingError(
            'Not enough samples to create a train/validation split.'
        )

    return indices[:train_size], indices[train_size:]


def _accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += int((preds == y).sum().item())
            total += int(y.numel())

    return correct / total if total else 0.0


def _make_model_path(owner_user_id: int, model_type: str) -> Path:
    model_dir = MODEL_STORAGE_DIR / f'user_{owner_user_id}'
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f'model_{uuid4().hex}_{model_type}.pt'


def train_sandbox_model(
    dataset_id: int,
    owner_user_id: int,
    model_type: str,
    model_name: str | None = None,
    config: TrainingConfig | None = None,
) -> SandboxTrainedModel:
    config = config or TrainingConfig()
    model_type = model_type.lower()

    db = SessionLocal()
    job = None

    try:
        dataset = (
            db.query(SandboxDataset)
            .filter(
                SandboxDataset.id == dataset_id,
                SandboxDataset.owner_user_id == owner_user_id,
            )
            .first()
        )

        if not dataset:
            raise SandboxTrainingError('Dataset not found.')

        classes = list(dataset.classes)
        samples = list(dataset.samples)

        _validate_dataset(
            classes=classes,
            samples=samples,
            config=config,
        )

        job = SandboxTrainingJob(
            dataset_id=dataset.id,
            owner_user_id=owner_user_id,
            model_type=model_type,
            status='running',
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        class_to_index, index_to_label = build_class_maps(classes)
        torch_dataset = SandboxImageDataset(
            samples=samples,
            class_to_index=class_to_index,
        )

        train_indices, val_indices = _split_indices(
            total=len(torch_dataset),
            val_split=config.val_split,
            seed=config.seed,
        )

        train_loader = DataLoader(
            Subset(torch_dataset, train_indices),
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(torch_dataset, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_sandbox_model(
            model_type=model_type,
            num_classes=len(classes),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )

        last_loss = 0.0

        for _epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())

            last_loss = running_loss / max(1, len(train_loader))

        train_accuracy = _accuracy(
            model=model,
            loader=train_loader,
            device=device,
        )
        val_accuracy = _accuracy(
            model=model,
            loader=val_loader,
            device=device,
        )

        checkpoint_path = _make_model_path(
            owner_user_id=owner_user_id,
            model_type=model_type,
        )

        payload = {
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'num_classes': len(classes),
            'class_index': {
                str(index): label
                for index, label in index_to_label.items()
            },
            'config': config.__dict__,
            'metrics': {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'loss': last_loss,
            },
        }
        torch.save(payload, checkpoint_path)

        job.status = 'done'
        job.train_accuracy = train_accuracy
        job.val_accuracy = val_accuracy
        job.finished_at = datetime.utcnow()

        trained_model = SandboxTrainedModel(
            dataset_id=dataset.id,
            training_job_id=job.id,
            owner_user_id=owner_user_id,
            name=model_name or f'{dataset.name}_{model_type}',
            model_type=model_type,
            checkpoint_path=checkpoint_path.as_posix(),
            class_index_json=json.dumps(payload['class_index']),
            metrics_json=json.dumps(payload['metrics']),
            is_shared=False,
            is_promoted_to_main_ui=False,
            created_at=datetime.utcnow(),
        )

        db.add(trained_model)
        db.commit()
        db.refresh(trained_model)

        return trained_model

    except Exception as exc:
        if job is not None:
            job.status = 'failed'
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
        raise

    finally:
        db.close()
