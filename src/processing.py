"""
SnowClearNet – Image Processing Pipeline (Improved)
==================================================
Pre-processing, inference, and post-processing utilities.
Includes blending + enhancement for better visual output.
"""

import numpy as np
from PIL import Image
import cv2
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
    """

    # -----------------------------------------------------------------------
    # Pre-process
    # -----------------------------------------------------------------------
    input_np = preprocess(image)            # (1, 3, 128, 128)

    # -----------------------------------------------------------------------
    # Model inference
    # -----------------------------------------------------------------------
    output_np = model.predict(input_np)     # (1, 3, 128, 128)

    # -----------------------------------------------------------------------
    # 🔥 FIX: Blend model output with input (prevents gray output)
    # -----------------------------------------------------------------------
    blended = 0.6 * input_np + 0.4 * output_np
    blended = np.clip(blended, 0, 1)

    # -----------------------------------------------------------------------
    # 🔥 Enhancement (contrast + brightness)
    # -----------------------------------------------------------------------
    # --- Advanced enhancement pipeline ---

    img_uint8 = (blended[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    # 1. Denoising (removes snow-like noise)
    denoised = cv2.fastNlMeansDenoisingColored(img_uint8, None, 10, 10, 7, 21)

    # 2. CLAHE contrast improvement
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 3. Sharpening
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])

    enhanced = cv2.filter2D(contrast, -1, kernel)

    # Convert back to float for metrics
    enhanced_float = enhanced.astype(np.float32) / 255.0
    enhanced_float = enhanced_float.transpose(2, 0, 1)[np.newaxis, ...]

    # -----------------------------------------------------------------------
    # Convert to PIL
    # -----------------------------------------------------------------------
    output_pil = Image.fromarray(enhanced)

    # -----------------------------------------------------------------------
    # Metrics (compare input vs enhanced output)
    # -----------------------------------------------------------------------
    in_hwc = input_np[0].transpose(1, 2, 0)
    out_hwc = enhanced_float[0].transpose(1, 2, 0)

    psnr_val = float(psnr(in_hwc, out_hwc, data_range=1.0))
    ssim_val = float(ssim(in_hwc, out_hwc, data_range=1.0, channel_axis=2))

    return output_pil, round(psnr_val, 2), round(ssim_val, 4)