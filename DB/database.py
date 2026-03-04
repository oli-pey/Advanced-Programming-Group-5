from sqlalchemy import create_engine, Column, Integer, Text, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pydantic import BaseModel, ConfigDict

# --- DATABASE SETUP ---
DATABASE_URL = "sqlite:///./mydata.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLALCHEMY MODEL (Object Orientation) ---
class PredictionEntry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    image = Column(LargeBinary, nullable=False)  # Stores the 28x28 processed bytes
    text = Column(Text)                          # Stores the predicted digit
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the tables
Base.metadata.create_all(bind=engine)

# --- PYDANTIC SCHEMA (Data Validation) ---
class PredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    text: str
    # Image is handled as bytes during processing