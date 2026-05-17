from dataclasses import dataclass
from typing import Dict


@dataclass
class PredictionResult:
    predicted_digit: int
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    model_version: str
