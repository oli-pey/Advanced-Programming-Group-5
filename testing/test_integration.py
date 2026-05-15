import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from web.index import LandingPage
from DB.database import PredictionEntry, SessionLocal, User
from ml.registry import get_recognizer


def test_tc010_end_to_end_prediction_flow(monkeypatch):
    """
    TC_010: Verify end-to-end prediction flow from drawing to database.
    Tests: Draw a digit, process it, and verify DB persistence.
    """
    # Create a simple test image (28x28 black-on-white digit)
    test_img = Image.new('L', (28, 28), color=255)  # White background
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    png_bytes = img_buffer.getvalue()

    # Create landing page
    page = LandingPage()
    page.selected_model = "logreg"

    # Mock the interactive image
    page.ii = Mock()
    page.ii.content = '<path d="M 10 10 L 20 20"/>'

    # Setup mock storage
    mock_user = Mock()
    mock_user.get.return_value = 1  # Return user_id = 1

    mock_storage = Mock()
    mock_storage.user = mock_user

    # Mock SVG to PNG conversion
    with patch('web.index.svg2rlg') as mock_svg2rlg, \
         patch('web.index.renderPM.drawToString') as mock_render, \
         patch('web.index.get_recognizer') as mock_get_recognizer, \
         patch('web.index.SessionLocal') as mock_session_class, \
         patch('web.index.ui.notify') as mock_notify:

        # Monkeypatch the storage to avoid NiceGUI initialization
        monkeypatch.setattr('web.index.app.storage', mock_storage)

        # Setup mock recognizer
        mock_recognizer = Mock()
        mock_result = Mock()
        mock_result.predicted_digit = 3
        mock_result.model_name = "logreg"
        mock_result.confidence = 0.95
        mock_recognizer.predict_from_png_bytes.return_value = mock_result
        mock_get_recognizer.return_value = mock_recognizer

        # Setup mock database session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.close = Mock()

        # Setup SVG to PNG conversion
        mock_svg2rlg.return_value = Mock()
        mock_render.return_value = png_bytes

        # Run the prediction
        import asyncio
        asyncio.run(page.process_drawing())

        # Verify database add was called
        mock_session.add.assert_called_once()
        entry_added = mock_session.add.call_args[0][0]
        assert isinstance(entry_added, PredictionEntry)
        assert entry_added.user_id == 1
        assert entry_added.prediction == "3"
        assert entry_added.model_name == "logreg"

        # Verify notification was shown
        mock_notify.assert_called()
        call_args = mock_notify.call_args[0][0]
        assert "3" in call_args  # Prediction digit in message
        assert "LOGREG" in call_args  # Model name in message

        # Verify canvas was cleared
        assert page.path == []


def test_tc012_sandbox_sample_upload():
    """
    TC_012: Verify that uploaded sample images are stored as LargeBinary
    in the sandbox_samples table and retrievable.
    """
    from DB.database import SandboxSample, SandboxClass, SandboxDataset

    # Create test image
    test_img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    png_bytes = img_buffer.getvalue()

    # Create in-memory test to verify image can be stored and retrieved
    db = SessionLocal()
    try:
        # Clean up any test data
        db.query(SandboxSample).filter(
            SandboxSample.user_note == "test_upload"
        ).delete()

        # Create a test dataset and class
        test_dataset = SandboxDataset(
            owner_user_id=1,
            name="Test Dataset",
            is_shared=False
        )
        db.add(test_dataset)
        db.flush()

        test_class = SandboxClass(
            dataset_id=test_dataset.id,
            name="Test Class"
        )
        db.add(test_class)
        db.flush()

        # Create and store a sample
        test_sample = SandboxSample(
            dataset_id=test_dataset.id,
            class_id=test_class.id,
            source_type="uploaded",
            image_filename="test_digit.png",
            image_data=png_bytes,
            image_mime_type="image/png",
            user_note="test_upload"
        )
        db.add(test_sample)
        db.commit()

        # Retrieve and verify
        retrieved = db.query(SandboxSample).filter(
            SandboxSample.user_note == "test_upload"
        ).first()

        assert retrieved is not None, "Sample not stored in database"
        assert retrieved.image_data == png_bytes, "Image data corrupted"
        assert retrieved.image_mime_type == "image/png"
        assert retrieved.source_type == "uploaded"
        assert retrieved.image_filename == "test_digit.png"

    finally:
        # Cleanup
        db.query(SandboxSample).filter(
            SandboxSample.user_note == "test_upload"
        ).delete()
        db.query(SandboxClass).filter(
            SandboxClass.name == "Test Class"
        ).delete()
        db.query(SandboxDataset).filter(
            SandboxDataset.name == "Test Dataset"
        ).delete()
        db.commit()
        db.close()
