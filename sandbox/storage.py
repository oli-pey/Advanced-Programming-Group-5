from pathlib import Path
from uuid import uuid4

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
BASE_STORAGE_DIR = Path('storage') / 'sandbox'


def validate_image_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f'Unsupported image extension: {suffix}')


def ensure_sample_dir(user_id: int, dataset_id: int) -> Path:
    path = (BASE_STORAGE_DIR /
            f'user_{user_id}' /
            f'dataset_{dataset_id}' /
            'samples'
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sample_file(
        user_id: int,
        dataset_id: int,
        filename: str,
        content: bytes
) -> str:
    if not filename:
        raise ValueError('Filename is required.')
    if not content:
        raise ValueError('File content is empty.')

    validate_image_filename(filename)

    sample_dir = ensure_sample_dir(user_id=user_id, dataset_id=dataset_id)
    suffix = Path(filename).suffix.lower()
    target = sample_dir / f'sample_{uuid4().hex}{suffix}'

    with target.open('wb') as f:
        f.write(content)

    return target.as_posix()


def delete_file_if_exists(file_path: str) -> None:
    path = Path(file_path)
    if path.exists() and path.is_file():
        path.unlink()
