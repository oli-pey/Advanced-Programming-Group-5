from ml.registry import get_recognizer

with open("test_digits/my_digit7.png", "rb") as f:
    image_bytes = f.read()

recognizer = get_recognizer("cnn")
result = recognizer.predict_from_png_bytes(image_bytes)

print(result)
