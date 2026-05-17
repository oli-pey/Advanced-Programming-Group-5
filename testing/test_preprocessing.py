import pytest
from ml.preprocessing import preprocess_png_bytes


def test_tc001_image_resizing_logic():
    """Verify image is resized to exactly (1, 1, 28, 28)."""
    # Create a dummy 500x500 white PNG in memory
    from PIL import Image
    import io

    img = Image.new('RGB', (500, 500), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')

    tensor = preprocess_png_bytes(img_byte_arr.getvalue())
    assert tensor.shape == (1, 1, 28, 28)


def test_tc002_pixel_normalization():
    """Verify pixels use MNIST mean (0.1307) and std (0.3081)."""
    # Test a single white pixel normalization
    from PIL import Image
    import io

    img = Image.new('L', (28, 28), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')

    tensor = preprocess_png_bytes(img_byte_arr.getvalue())
    # Standard MNIST normalization: (1.0 - 0.1307) / 0.3081 ≈ 2.8215
    assert tensor.max().item() == pytest.approx(2.8215, rel=1e-3)


def test_tc003_color_inversion():
    """Verify black-on-white is inverted to MNIST-style white-on-black."""
    from PIL import Image
    import io

    # Input: Black digit (0) on White background (255)
    img = Image.new('L', (28, 28), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')

    # Process with invert=True
    tensor = preprocess_png_bytes(img_byte_arr.getvalue(), invert=True)
    # Background should now be low (black) and digit high (white)
    assert tensor.min().item() < 0  # Normalized black is negative
