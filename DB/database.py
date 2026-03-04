from sqlalchemy import create_engine, Column, Integer, Text, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./mydata.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionEntry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    original_image = Column(LargeBinary, nullable=False)  # Full-scale PNG
    downsized_image = Column(LargeBinary, nullable=False) # 28x28 PNG
    prediction = Column(Text)                             # Predicted digit
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)