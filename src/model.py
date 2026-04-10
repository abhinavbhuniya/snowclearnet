"""
SnowClearNet – ONNX Runtime Model Loader
==========================================
Loads the DATSRF model in ONNX format for lightweight serverless inference.
Falls back to a classical image-enhancement pipeline when no model is available.

To convert your PyTorch model to ONNX (run locally, not on Vercel):

    import torch
    from src.model_arch import DATSRF

    model = DATSRF()
    model.load_state_dict(torch.load("datsrf_model.pth", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 128, 128)
    torch.onnx.export(model, dummy, "datsrf_model.onnx",
                       input_names=["input"], output_names=["output"],
                       dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
"""

import os
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class SnowClearModel:
    """
    Wrapper that provides a unified .predict(input_np) → output_np interface
    using either ONNX Runtime or a classical fallback.
    """

    def __init__(self, model_path: str):
        self.session = None
        self.using_onnx = False

        if ORT_AVAILABLE and os.path.isfile(model_path):
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
                self.using_onnx = True
                print(f"[✓] Loaded ONNX model from {model_path}")
            except Exception as e:
                print(f"[!] Could not load ONNX model ({e}) – using enhancement fallback.")
        else:
            reason = "onnxruntime not installed" if not ORT_AVAILABLE else f"'{model_path}' not found"
            print(f"[!] {reason} – using classical enhancement fallback (demo mode).")

    def predict(self, input_np: np.ndarray) -> np.ndarray:
        """
        Run inference on a (1, 3, H, W) float32 array in [0, 1].
        Returns a (1, 3, H, W) float32 array in [0, 1].
        """
        if self.using_onnx and self.session is not None:
            outputs = self.session.run(None, {self.input_name: input_np})
            return np.clip(outputs[0], 0, 1).astype(np.float32)
        else:
            return self._enhance_fallback(input_np)

    # ------------------------------------------------------------------
    # Classical snow-removal simulation (used when no model is available)
    # ------------------------------------------------------------------
    @staticmethod
    def _enhance_fallback(img: np.ndarray) -> np.ndarray:
        """
        Apply aggressive classical image processing to simulate snow removal.
        Produces a clearly visible before/after difference.

        Pipeline:
        1. Heavy snow-pixel suppression (detect & darken bright regions)
        2. Strong contrast stretch per channel
        3. Warm colour correction (reduce blue cast from snow)
        4. De-haze via dark channel prior
        5. Saturation boost
        6. Sharpening pass

        Input / output: (1, 3, H, W) float32 in [0, 1].
        """
        x = img[0].copy()  # (3, H, W)

        # --- 1. Aggressive snow suppression ---
        # Snow pixels are bright and low-saturation
        brightness = x.mean(axis=0)  # (H, W)
        ch_min = x.min(axis=0)
        ch_max = x.max(axis=0)
        saturation = (ch_max - ch_min) / (ch_max + 1e-6)

        # Snow = bright + low saturation
        snow_mask = np.clip((brightness - 0.55) / 0.30, 0, 1) * np.clip((1.0 - saturation * 2), 0, 1)
        suppression = 1.0 - 0.60 * snow_mask  # heavy darkening of snow areas
        x = x * suppression[np.newaxis, :, :]

        # --- 2. Strong per-channel contrast stretch ---
        for c in range(3):
            lo, hi = np.percentile(x[c], [1, 99])
            if hi - lo > 0.01:
                x[c] = (x[c] - lo) / (hi - lo)
        x = np.clip(x, 0, 1)

        # --- 3. Warm colour correction (remove blue snow cast) ---
        x[0] = x[0] * 1.12          # red  +12%
        x[1] = x[1] * 1.05          # green +5%
        x[2] = x[2] * 0.80          # blue -20%

        # --- 4. Stronger de-haze via dark channel prior ---
        dark_ch = x.min(axis=0)
        atm = np.percentile(dark_ch, 99.5)
        t = 1.0 - 0.85 * (dark_ch / (atm + 1e-6))
        t = np.clip(t, 0.20, 1.0)
        for c in range(3):
            x[c] = (x[c] - atm * (1 - t)) / (t + 1e-6)
        x = np.clip(x, 0, 1)

        # --- 5. Saturation boost ---
        # Convert to HSV-like: boost saturation channel
        gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        sat_factor = 1.40
        for c in range(3):
            x[c] = gray + sat_factor * (x[c] - gray)
        x = np.clip(x, 0, 1)

        # --- 6. Sharpening (unsharp mask via simple kernel) ---
        from scipy.ndimage import uniform_filter
        for c in range(3):
            blurred = uniform_filter(x[c], size=3)
            x[c] = x[c] + 0.5 * (x[c] - blurred)  # sharpen

        x = np.clip(x, 0, 1).astype(np.float32)
        return x[np.newaxis, ...]  # (1, 3, H, W)


def load_model(weight_path: str) -> SnowClearModel:
    """Convenience loader – returns a SnowClearModel instance."""
    return SnowClearModel(weight_path)
