from abc import ABC, abstractmethod
from .result import PredictionResult

class DigitRecognizer(ABC):
    @abstractmethod
    def predict_from_png_bytes(self, image_bytes: bytes) -> PredictionResult:
        raise NotImplementedError
