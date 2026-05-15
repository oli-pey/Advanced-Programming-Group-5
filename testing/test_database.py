import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from DB.database import Base, User, PredictionEntry, SandboxDataset, SandboxClass

@pytest.fixture
def db_session():
    """Setup in-memory DB for every test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_tc007_default_admin_seeding(db_session):
    """Verify admin record creation."""
    from DB.database import _seed_default_admin_user
    _seed_default_admin_user(db_session)
    
    admin = db_session.query(User).filter_by(is_admin=True).first()
    assert admin is not None
    assert admin.username == "admin"

def test_tc008_prediction_persistence(db_session):
    """Verify prediction record can be retrieved."""
    entry = PredictionEntry(digit=5, model_name="cnn", confidence=0.98)
    db_session.add(entry)
    db_session.commit()
    
    retrieved = db_session.query(PredictionEntry).first()
    assert retrieved.digit == 5

def test_tc009_dataset_cascade_deletion(db_session):
    """Verify deleting a dataset removes its linked classes."""
    dataset = SandboxDataset(name="TestSet")
    db_session.add(dataset)
    db_session.commit()
    
    cls = SandboxClass(dataset_id=dataset.id, label="label1")
    db_session.add(cls)
    db_session.commit()
    
    db_session.delete(dataset)
    db_session.commit()
    
    assert db_session.query(SandboxClass).count() == 0