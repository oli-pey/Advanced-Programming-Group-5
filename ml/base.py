from abc import ABC, abstractmethod
from .result import PredictionResult


class DigitRecognizer(ABC):
    def predict(self, image_bytes: bytes) -> PredictionResult:
        return self.predict_from_png_bytes(image_bytes)

    @abstractmethod
    def predict_from_png_bytes(self, image_bytes: bytes) -> PredictionResult:
        raise NotImplementedError