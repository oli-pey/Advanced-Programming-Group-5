from __future__ import annotations

from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from sandbox_ml.models import create_sandbox_model


sandbox_predict_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class SandboxModelRecognizer:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        payload = torch.load(checkpoint_path, map_location=self.device)

        self.model_type = payload['model_type']
        self.num_classes = int(payload['num_classes'])
        self.class_index = payload['class_index']

        self.model = create_sandbox_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
        )
        self.model.load_state_dict(payload['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _predict_tensor(self, x: torch.Tensor) -> dict:
        x = x.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

        pred_index = int(torch.argmax(probs).item())
        pred_label = self.class_index[str(pred_index)]
        confidence = float(probs[pred_index].item())

        return {
            'predicted_label': pred_label,
            'confidence': confidence,
            'probabilities': {
                self.class_index[str(i)]: float(probs[i].item())
                for i in range(self.num_classes)
            },
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
        }

    def predict_from_image_path(self, image_path: str) -> dict:
        image = Image.open(Path(image_path)).convert('L')
        x = sandbox_predict_transform(image)
        return self._predict_tensor(x)

    def predict_from_image_bytes(self, image_bytes: bytes) -> dict:
        image = Image.open(BytesIO(image_bytes)).convert('L')
        x = sandbox_predict_transform(image)
        return self._predict_tensor(x)
