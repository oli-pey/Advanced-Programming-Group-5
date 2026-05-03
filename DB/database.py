from datetime import datetime

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
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, mapped_column
from sqlalchemy.orm.attributes import Mapped

DATABASE_URL = "sqlite:///./mydata.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    entries = relationship("PredictionEntry", back_populates="user")


class PredictionEntry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    original_image = Column(LargeBinary, nullable=False)
    downsized_image = Column(LargeBinary, nullable=False)
    prediction = Column(Text, nullable=False)
    model_name = Column(String(50), nullable=True)
    probability = Column(Text, nullable=True)     
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="entries")

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
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship('SandboxDataset', back_populates='samples')
    sandbox_class = relationship('SandboxClass', back_populates='samples')



Base.metadata.create_all(bind=engine)
