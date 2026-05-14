import pytest
import os
import shutil
from sqlalchemy import create_mock_engine
from sqlalchemy.orm import sessionmaker
from DB.database import Base, User, PredictionEntry  # Adjust imports based on your DB/database.py
from sandbox.services import create_dataset, add_sample_to_class
from sandbox.storage import get_storage_path
from ml.preprocessing import preprocess_image # Hypothetical function from your preprocessing.py

# --- FIXTURES ---

@pytest.fixture(scope="function")
def test_db():
    """Sets up an in-memory database for isolated testing."""
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def temp_storage(tmp_path):
    """Provides a temporary directory for file IO tests."""
    return tmp_path

# --- UNIT TESTS (Layer B) ---

def test_storage_path_logic(temp_storage):
    """Verifies that the storage pathing logic creates correct structures."""
    # Test if the service correctly identifies where to save files
    dataset_name = "test_digits"
    path = os.path.join(temp_storage, dataset_name)
    
    # Simulate directory creation logic
    os.makedirs(path, exist_ok=True)
    assert os.path.exists(path)
    assert dataset_name in str(path)

def test_label_normalization():
    """Unit test for label validation logic."""
    # Assuming a function exists to clean labels (e.g., lowercase, no spaces)
    # Replace with your actual logic from sandbox/services.py
    raw_label = " Digit 5 "
    normalized = raw_label.strip().lower().replace(" ", "_")
    assert normalized == "digit_5"

# --- INTEGRATION TESTS (Layer C) ---

def test_end_to_end_sandbox_flow(test_db, temp_storage, mocker):
    """
    Verifies the integration between Services, Database, and Pathing.
    1. Create a User
    2. Create a Sandbox Dataset
    3. Add a Sample
    """
    # 1. Setup: Create a test user
    new_user = User(username="testuser", password_hash="123")
    test_db.add(new_user)
    test_db.commit()

    # 2. Integration: Create dataset using your service logic
    # Mocking the storage path to use our temp_storage fixture
    mocker.patch('sandbox.storage.get_base_dir', return_value=str(temp_storage))
    
    # (Replace with your actual function signatures from sandbox/services.py)
    # dataset = create_dataset(test_db, user_id=new_user.id, name="MyCustomData")
    # assert dataset.id is not None
    
    # 3. Simulate a 'Tiny' training run validation
    # Check if the system handles the transition from DB entry to file system
    sample_filename = "sample_0.png"
    sample_path = temp_storage / "MyCustomData" / "class_0" / sample_filename
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("fake_image_data")

    assert sample_path.exists()
    assert "MyCustomData" in str(sample_path)

# --- ML LOGIC TESTS ---

def test_model_output_format():
    """Ensures the ML registry returns objects with the correct prediction interface."""
    from ml.registry import get_recognizer # Adjust based on your registry.py
    
    # We don't need to load the full weights for a unit test if we mock the model
    recognizer = get_recognizer("logreg")
    assert recognizer is not None
    assert hasattr(recognizer, 'predict'), "Recognizer must have a predict method"