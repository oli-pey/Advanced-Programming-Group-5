from datetime import datetime
import base64
import hashlib
import os
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Text,
    LargeBinary,
    DateTime,
    String,
    Boolean,
    ForeignKey,
    Float,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, mapped_column
from sqlalchemy.orm.attributes import Mapped

DATABASE_URL = "sqlite:///./mydata.db"
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(salt + digest).decode("utf-8")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    entries: Mapped[list["PredictionEntry"]] = relationship("PredictionEntry", back_populates="user")


class PredictionEntry(Base):
    __tablename__ = "input_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    original_image: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    downsized_image: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    prediction: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    probability: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="entries")

# Add this snippet to DB/database.py if you lost the sandbox table classes.
# Make sure these imports exist:
# from datetime import datetime
# from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
# from sqlalchemy.orm import Mapped, mapped_column, relationship

class SandboxDataset(Base):
    __tablename__ = 'sandbox_datasets'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    owner = relationship('User')
    classes = relationship('SandboxClass', back_populates='dataset', cascade='all, delete-orphan')
    samples = relationship('SandboxSample', back_populates='dataset', cascade='all, delete-orphan')


class SandboxClass(Base):
    __tablename__ = 'sandbox_classes'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('sandbox_datasets.id'), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship('SandboxDataset', back_populates='classes')
    samples = relationship('SandboxSample', back_populates='sandbox_class', cascade='all, delete-orphan')


class SandboxSample(Base):
    __tablename__ = 'sandbox_samples'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('sandbox_datasets.id'), nullable=False)
    class_id: Mapped[int] = mapped_column(ForeignKey('sandbox_classes.id'), nullable=False)
    image_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    image_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship('SandboxDataset', back_populates='samples')
    sandbox_class = relationship('SandboxClass', back_populates='samples')

class SandboxTrainingJob(Base):
    __tablename__ = 'sandbox_training_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('sandbox_datasets.id'), nullable=False)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)

    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default='queued', nullable=False)

    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)

    train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    dataset = relationship('SandboxDataset')
    owner = relationship('User')


class SandboxTrainedModel(Base):
    __tablename__ = 'sandbox_trained_models'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('sandbox_datasets.id'), nullable=False)
    training_job_id: Mapped[int] = mapped_column(ForeignKey('sandbox_training_jobs.id'), nullable=False)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)

    name: Mapped[str] = mapped_column(String(150), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(500), nullable=False)

    class_index_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)

    is_shared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_promoted_to_main_ui: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship('SandboxDataset')
    training_job = relationship('SandboxTrainingJob')
    owner = relationship('User')


def _seed_default_admin_user() -> None:
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            return

        admin_user = User(
            username=DEFAULT_ADMIN_USERNAME,
            password_hash=_hash_password(DEFAULT_ADMIN_PASSWORD),
            is_admin=True,
        )
        db.add(admin_user)
        db.commit()
    finally:
        db.close()


database_path = Path(engine.url.database or "mydata.db")
database_was_missing = not database_path.exists()

Base.metadata.create_all(bind=engine)

if database_was_missing:
    _seed_default_admin_user()
