from ml.registry import get_recognizer
from ml.recognizers.logreg_recognizer import LogRegRecognizer

def test_tc011_model_registry_integration():
    """Verify 'logreg' selection returns the LogReg instance."""
    recognizer = get_recognizer("logreg")
    assert isinstance(recognizer, LogRegRecognizer)
    assert hasattr(recognizer, 'predict')