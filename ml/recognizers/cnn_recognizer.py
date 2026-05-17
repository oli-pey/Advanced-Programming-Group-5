import torch
from ml.base import DigitRecognizer
from ml.preprocessing import preprocess_png_bytes
from ml.result import PredictionResult
from ml.models.cnn_model import CNNMnist


class CNNRecognizer(DigitRecognizer):
    def __init__(self, model_path: str):
        self.model = CNNMnist()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        payload = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()

    def predict_from_png_bytes(self, image_bytes: bytes) -> PredictionResult:
        x = preprocess_png_bytes(image_bytes, invert=True).to(self.device)

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

        return PredictionResult(
            predicted_digit=pred,
            confidence=conf,
            probabilities={str(i): float(probs[i].item()) for i in range(10)},
            model_name="cnn",
            model_version="v1",
        )
