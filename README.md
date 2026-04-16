---
title: SnowClearNet
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: mit
---

# SnowClearNet

SnowClearNet restores snowy road-scene images for smart city traffic monitoring.

The app uses the existing ONNX model loader in `src/model.py` and the unchanged
image processing pipeline in `src/processing.py`. It runs on CPU only and reports
PSNR and SSIM for each processed image.
