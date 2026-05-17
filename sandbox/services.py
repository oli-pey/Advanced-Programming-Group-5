from __future__ import annotations

from io import BytesIO
from PIL import Image
from sqlalchemy.orm import Session

from DB.database import SandboxDataset, SandboxClass, SandboxSample


class SandboxError(Exception):
    pass


VALID_SOURCE_TYPES = {'uploaded', 'drawn'}


def normalize_label(value: str) -> str:
    return ' '.join(value.strip().split()).lower()


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = ' '.join(value.strip().split())
    return cleaned or None


def list_datasets_for_user(
        db: Session, owner_user_id: int
) -> list[SandboxDataset]:
    return (
        db.query(SandboxDataset)
        .filter(SandboxDataset.owner_user_id == owner_user_id)
        .order_by(SandboxDataset.created_at.desc())
        .all()
    )


def get_dataset_for_user(
        db: Session,
        dataset_id: int,
        owner_user_id: int
) -> SandboxDataset | None:
    return (
        db.query(SandboxDataset)
        .filter(
            SandboxDataset.id == dataset_id,
            SandboxDataset.owner_user_id == owner_user_id
        )
        .first()
    )


def create_dataset(
    db: Session,
    owner_user_id: int,
    name: str,
    description: str | None = None,
    is_shared: bool = False,
) -> SandboxDataset:
    normalized_name = normalize_label(name)

    if not normalized_name:
        raise SandboxError('Dataset name cannot be empty.')

    existing = (
        db.query(SandboxDataset)
        .filter(SandboxDataset.owner_user_id == owner_user_id)
        .all()
    )

    for dataset in existing:
        if normalize_label(dataset.name) == normalized_name:
            raise SandboxError('Dataset name already exists for this user.')

    dataset = SandboxDataset(
        owner_user_id=owner_user_id,
        name=normalized_name,
        description=clean_text(description),
        is_shared=is_shared,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def delete_dataset(db: Session, dataset: SandboxDataset) -> None:
    db.delete(dataset)
    db.commit()


def create_class(
    db: Session,
    dataset: SandboxDataset,
    name: str,
    description: str | None = None,
) -> SandboxClass:
    normalized_name = normalize_label(name)

    if not normalized_name:
        raise SandboxError('Class name cannot be empty.')

    existing_classes = (
        db.query(SandboxClass)
        .filter(SandboxClass.dataset_id == dataset.id)
        .all()
    )

    for existing in existing_classes:
        if normalize_label(existing.name) == normalized_name:
            raise SandboxError('Class name already exists in this dataset.')

    sandbox_class = SandboxClass(
        dataset_id=dataset.id,
        name=normalized_name,
        description=clean_text(description),
    )

    db.add(sandbox_class)
    db.commit()
    db.refresh(sandbox_class)
    return sandbox_class


def delete_class(db: Session, sandbox_class: SandboxClass) -> None:
    db.delete(sandbox_class)
    db.commit()


def _validate_image_content(content: bytes) -> None:
    try:
        img = Image.open(BytesIO(content))
        img.verify()
    except Exception as exc:
        raise SandboxError('Uploaded file is not a valid image.') from exc


def create_sample(
    db: Session,
    dataset: SandboxDataset,
    class_id: int,
    filename: str,
    content: bytes,
    source_type: str = 'uploaded',
    user_note: str | None = None,
) -> SandboxSample:
    if source_type not in VALID_SOURCE_TYPES:
        raise SandboxError("source_type must be 'uploaded' or 'drawn'.")

    sandbox_class = (
        db.query(SandboxClass)
        .filter(
            SandboxClass.id == class_id,
            SandboxClass.dataset_id == dataset.id,
        )
        .first()
    )

    if not sandbox_class:
        raise SandboxError('Selected class does not belong to this dataset.')

    _validate_image_content(content)

    sample = SandboxSample(
        dataset_id=dataset.id,
        class_id=class_id,
        image_data=content,
        image_filename=filename,
        image_mime_type='image/png',
        source_type=source_type,
        user_note=clean_text(user_note),
    )

    db.add(sample)
    db.commit()
    db.refresh(sample)

    return sample


def delete_sample(db: Session, sample: SandboxSample) -> None:
    db.delete(sample)
    db.commit()
