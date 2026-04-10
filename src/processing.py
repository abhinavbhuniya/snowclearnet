"""
SnowClearNet – Image Processing Pipeline
=========================================
Pre-processing, inference, and post-processing utilities.
Uses PIL / numpy only (no torch dependency).
"""

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 128  # Model expects 128×128 inputs


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize a PIL Image to (IMG_SIZE, IMG_SIZE) and convert to a
    float32 numpy array of shape (1, 3, H, W) in [0, 1].
    """
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr.transpose(2, 0, 1)                        # (3, H, W)
    return arr[np.newaxis, ...]                          # (1, 3, H, W)


def tensor_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a (3, H, W) or (1, 3, H, W) float32 array in [0, 1]
    back to a PIL Image.
    """
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def process_image(model, image: Image.Image):
    """
    Run the full snow-removal pipeline on a single PIL image.

    Parameters
    ----------
    model : SnowClearModel   – from src.model
    image : PIL.Image.Image  – input snowy image

    Returns
    -------
    output_pil : PIL.Image.Image   – Restored image
    psnr_val   : float             – PSNR (input vs output)
    ssim_val   : float             – SSIM (input vs output)
    """
    # Pre-process
    input_np = preprocess(image)            # (1, 3, 128, 128) float32

    # Inference
    output_np = model.predict(input_np)     # (1, 3, 128, 128) float32

    # Convert to PIL
    output_pil = tensor_to_pil(output_np)

    # Compute quality metrics (compare at model resolution)
    in_hwc = input_np[0].transpose(1, 2, 0)    # (128, 128, 3)
    out_hwc = output_np[0].transpose(1, 2, 0)  # (128, 128, 3)

    psnr_val = float(psnr(in_hwc, out_hwc, data_range=1.0))
    ssim_val = float(ssim(in_hwc, out_hwc, data_range=1.0, channel_axis=2))

    return output_pil, round(psnr_val, 2), round(ssim_val, 4)
