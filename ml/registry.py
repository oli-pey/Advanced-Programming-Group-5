from ml.recognizers.cnn_recognizer import CNNRecognizer
from ml.recognizers.logreg_recognizer import LogRegRecognizer

def get_recognizer(model_name: str):
    if model_name == "cnn":
        return CNNRecognizer("ml/models/cnn_mnist.pt")
    elif model_name == "logreg":
        return LogRegRecognizer("ml/models/logreg_mnist.pt")
    else:
        raise ValueError("Unknown model")
