"""
SnowClearNet – Image Processing Pipeline
=========================================
Pre-processing, inference, and post-processing utilities.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

IMG_SIZE = 128  # Model expects 128×128 inputs

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),            # → [0, 1] float32, (C, H, W)
])


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float tensor in [0, 1] back to a PIL Image."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_image(model, image: Image.Image, device: torch.device):
    """
    Run the full snow-removal pipeline on a single PIL image.

    Returns
    -------
    output_pil : PIL.Image.Image   – Restored image
    psnr_val   : float             – PSNR between input and output
    ssim_val   : float             – SSIM between input and output
    """
    # Pre-process
    input_tensor = preprocess(image).unsqueeze(0).to(device)   # (1, 3, 128, 128)

    # Forward pass
    output_tensor = model(input_tensor).squeeze(0)             # (3, 128, 128)

    # Convert to PIL
    output_pil = tensor_to_pil(output_tensor)

    # Compute quality metrics (input vs output at model resolution)
    input_np = preprocess(image).permute(1, 2, 0).numpy()      # (128, 128, 3)
    output_np = output_tensor.cpu().permute(1, 2, 0).numpy()

    psnr_val = float(psnr(input_np, output_np, data_range=1.0))
    ssim_val = float(ssim(input_np, output_np, data_range=1.0, channel_axis=2))

    return output_pil, round(psnr_val, 2), round(ssim_val, 4)
