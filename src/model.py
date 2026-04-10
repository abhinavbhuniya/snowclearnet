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
        Apply classical image processing to simulate snow removal:
        1. Reduce bright-white pixel intensity (snow mask)
        2. Local contrast enhancement (CLAHE-like via numpy)
        3. Slight warm colour shift
        4. De-haze via dark channel prior approximation

        Input / output: (1, 3, H, W) float32 in [0, 1].
        """
        x = img[0].copy()  # (3, H, W)

        # --- 1. Snow suppression: dampen very bright pixels ---
        brightness = x.mean(axis=0)  # (H, W)
        snow_mask = np.clip((brightness - 0.65) / 0.35, 0, 1)  # 1 where bright
        suppression = 1.0 - 0.35 * snow_mask  # reduce intensity
        x = x * suppression[np.newaxis, :, :]

        # --- 2. Per-channel contrast stretch ---
        for c in range(3):
            lo, hi = np.percentile(x[c], [2, 98])
            if hi - lo > 0.01:
                x[c] = (x[c] - lo) / (hi - lo)

        # --- 3. Warm colour correction (less blue, more red/green) ---
        x[0] = x[0] * 1.05          # red  +5%
        x[1] = x[1] * 1.02          # green +2%
        x[2] = x[2] * 0.90          # blue  -10%

        # --- 4. Simple de-haze (subtract atmospheric light estimate) ---
        dark_ch = x.min(axis=0)
        atm = np.percentile(dark_ch, 99)
        t = 1.0 - 0.6 * (dark_ch / (atm + 1e-6))
        t = np.clip(t, 0.25, 1.0)
        for c in range(3):
            x[c] = (x[c] - atm * (1 - t)) / (t + 1e-6)

        x = np.clip(x, 0, 1).astype(np.float32)
        return x[np.newaxis, ...]  # (1, 3, H, W)


def load_model(weight_path: str) -> SnowClearModel:
    """Convenience loader – returns a SnowClearModel instance."""
    return SnowClearModel(weight_path)
