"""
SnowClearNet Gradio app for Hugging Face Spaces.

The existing ONNX model loader and image processing pipeline are reused
unchanged. The model is loaded once at startup and runs on CPU.
"""

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.model import load_model
from src.processing import IMG_SIZE, process_image


def ensure_localhost_bypasses_proxy() -> None:
    """Keep Gradio's startup health check from going through system proxies."""
    hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        entries = [entry.strip() for entry in current.split(",") if entry.strip()]
        for host in hosts:
            if host not in entries:
                entries.append(host)
        os.environ[key] = ",".join(entries)


# Keep execution CPU-only. src.model also pins ONNX Runtime to CPUExecutionProvider.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ensure_localhost_bypasses_proxy()

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "datsrf_model.onnx"

# Loaded once globally so each request reuses the same ONNX Runtime session.
MODEL = load_model(str(MODEL_PATH))


def render_metrics(psnr_value: float | None = None, ssim_value: float | None = None) -> str:
    """Build the metric pills shown under the output image."""
    psnr_text = "--" if psnr_value is None else f"{psnr_value:.2f}"
    ssim_text = "--" if ssim_value is None else f"{ssim_value:.4f}"
    state_text = "Ready" if psnr_value is None else "Complete"
    return f"""
    <div class="metrics-shell">
        <div class="metric-card">
            <span>PSNR</span>
            <strong>{psnr_text}</strong>
            <small>dB</small>
        </div>
        <div class="metric-card">
            <span>SSIM</span>
            <strong>{ssim_text}</strong>
            <small>score</small>
        </div>
        <div class="metric-card metric-card-accent">
            <span>Status</span>
            <strong>{state_text}</strong>
            <small>CPU</small>
        </div>
    </div>
    """


def restore_image(image: Image.Image):
    """Run SnowClearNet on one uploaded image and return previews plus metrics."""
    if image is None:
        raise gr.Error("Please upload a PNG or JPG image first.")

    input_image = image.convert("RGB")
    output_image, psnr_value, ssim_value = process_image(MODEL, input_image)
    return output_image, render_metrics(psnr_value, ssim_value)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root,
.dark {
    --body-background-fill: #080b0f;
    --background-fill-primary: transparent;
    --background-fill-secondary: rgba(13, 18, 22, 0.94);
    --block-background-fill: rgba(13, 18, 22, 0.88);
    --block-border-color: rgba(159, 176, 190, 0.18);
    --block-border-width: 1px;
    --block-radius: 8px;
    --button-primary-background-fill: linear-gradient(135deg, #18b58f, #35a9c9);
    --button-primary-background-fill-hover: linear-gradient(135deg, #28cba4, #54bfd9);
    --button-primary-text-color: #ffffff;
    --input-background-fill: rgba(5, 8, 11, 0.66);
    --input-border-color: rgba(159, 176, 190, 0.18);
    --body-text-color: #edf7f6;
    --body-text-color-subdued: #99a8a8;
    --link-text-color: #65d8cb;
    --font: 'Inter', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

body,
.gradio-container {
    background:
        linear-gradient(135deg, rgba(24, 181, 143, 0.14) 0%, rgba(24, 181, 143, 0) 32%),
        linear-gradient(215deg, rgba(216, 91, 83, 0.12) 0%, rgba(216, 91, 83, 0) 36%),
        linear-gradient(180deg, #080b0f 0%, #101510 48%, #080b0f 100%) !important;
    color: #edf7f6 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 26px 22px 24px !important;
}

footer,
.api-docs,
.built-with {
    display: none !important;
}

.main,
.wrap,
.contain {
    background: transparent !important;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 54px;
    padding: 10px 0 22px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 8px;
    border: 1px solid rgba(101, 216, 203, 0.32);
    background: linear-gradient(135deg, rgba(24, 181, 143, 0.25), rgba(53, 169, 201, 0.14));
    color: #adfff1;
    font-weight: 800;
}

.brand-name {
    color: #f7fffd;
    font-size: 1rem;
    font-weight: 800;
    line-height: 1.1;
}

.brand-sub {
    color: #7c8a89;
    font-size: 0.78rem;
    margin-top: 2px;
}

.topbar-status {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.status-chip {
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.18);
    background: rgba(8, 11, 15, 0.46);
    color: #b9c7c5;
    font-size: 0.78rem;
    font-weight: 700;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
    gap: 22px;
    align-items: stretch;
    margin-bottom: 22px;
}

.hero-main {
    padding: 34px;
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.16);
    background:
        linear-gradient(135deg, rgba(237, 247, 246, 0.06), rgba(237, 247, 246, 0)),
        rgba(10, 14, 17, 0.56);
    box-shadow: 0 20px 70px rgba(0, 0, 0, 0.34);
}

.eyebrow {
    display: inline-flex;
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid rgba(101, 216, 203, 0.28);
    background: rgba(24, 181, 143, 0.10);
    color: #90f3e5;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.app-title {
    margin: 20px 0 14px;
    max-width: 760px;
    color: #f8fffd;
    font-size: 4rem;
    line-height: 0.98;
    font-weight: 800;
    letter-spacing: 0;
}

.app-subtitle {
    max-width: 690px;
    margin: 0;
    color: #a6b6b4;
    font-size: 1rem;
    line-height: 1.7;
}

.hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}

.hero-pill {
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.16);
    background: rgba(8, 11, 15, 0.42);
    color: #c9d6d4;
    padding: 10px 12px;
    font-size: 0.86rem;
    font-weight: 700;
}

.hero-side {
    display: grid;
    gap: 12px;
}

.info-card {
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.16);
    background: rgba(10, 14, 17, 0.56);
    padding: 18px;
}

.info-card span {
    display: block;
    color: #7f908e;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
}

.info-card strong {
    display: block;
    margin-top: 8px;
    color: #f5fffd;
    font-size: 1.3rem;
}

.info-card small {
    display: block;
    margin-top: 6px;
    color: #93a4a1;
    line-height: 1.5;
}

.console-header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 16px;
    margin: 6px 0 14px;
}

.console-title {
    margin: 0;
    color: #f6fffd;
    font-size: 1.35rem;
    font-weight: 800;
}

.console-copy {
    margin: 4px 0 0;
    color: #859593;
    font-size: 0.9rem;
}

.console-kicker {
    color: #65d8cb;
    font-size: 0.8rem;
    font-weight: 800;
}

.workspace {
    gap: 22px !important;
    align-items: stretch !important;
}

.glass-panel {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(159, 176, 190, 0.18) !important;
    border-radius: 8px !important;
    background:
        linear-gradient(145deg, rgba(24, 181, 143, 0.10), rgba(216, 91, 83, 0.04)),
        rgba(13, 18, 22, 0.92) !important;
    box-shadow: 0 26px 80px rgba(0, 0, 0, 0.38) !important;
    padding: 20px !important;
}

.panel-title {
    margin: 0 0 4px;
    color: #f7fffd;
    font-size: 1.05rem;
    font-weight: 800;
}

.panel-copy {
    margin: 0 0 16px;
    color: #8f9f9d;
    font-size: 0.88rem;
}

.upload-panel .image-container,
.result-panel .image-container {
    min-height: 330px !important;
    border-radius: 8px !important;
    border: 1px dashed rgba(159, 176, 190, 0.30) !important;
    background:
        linear-gradient(135deg, rgba(237, 247, 246, 0.04), rgba(237, 247, 246, 0)),
        rgba(5, 8, 11, 0.58) !important;
}

.result-panel .image-container {
    border-style: solid !important;
}

.upload-panel .image-container:hover {
    border-color: rgba(101, 216, 203, 0.55) !important;
}

.upload-panel label,
.result-panel label {
    color: #d7e2e0 !important;
    font-weight: 700 !important;
}

.upload-panel .wrap,
.result-panel .wrap,
.upload-panel .block,
.result-panel .block {
    border-radius: 8px !important;
}

.upload-panel button,
.result-panel button {
    border-radius: 8px !important;
}

.process-btn {
    margin-top: 18px !important;
}

.process-btn button {
    width: 100% !important;
    min-height: 48px !important;
    border-radius: 8px !important;
    border: 0 !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
    box-shadow: 0 12px 34px rgba(24, 181, 143, 0.24) !important;
}

.process-btn button:hover {
    transform: translateY(-1px);
}

.metrics-box {
    margin-top: 18px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.metrics-shell {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.metric-card {
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.18);
    background: rgba(5, 8, 11, 0.48);
    padding: 14px 12px;
    min-height: 88px;
}

.metric-card span {
    display: block;
    color: #7f908e;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-card strong {
    display: block;
    margin-top: 8px;
    color: #79eadc;
    font-family: var(--font-mono);
    font-size: 1.24rem;
    line-height: 1;
}

.metric-card small {
    display: block;
    margin-top: 8px;
    color: #7f908e;
    font-size: 0.76rem;
    font-weight: 700;
}

.metric-card-accent {
    background: rgba(24, 181, 143, 0.12);
    border-color: rgba(101, 216, 203, 0.28);
}

.usage-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.usage-step {
    border-radius: 8px;
    border: 1px solid rgba(159, 176, 190, 0.14);
    background: rgba(8, 11, 15, 0.42);
    padding: 14px;
    color: #a9b8b5;
    font-size: 0.84rem;
    line-height: 1.45;
}

.usage-step strong {
    display: block;
    margin-bottom: 4px;
    color: #f4fffd;
}

.app-footer {
    margin-top: 26px;
    padding-top: 22px;
    border-top: 1px solid rgba(159, 176, 190, 0.14);
    color: #758582;
    text-align: center;
    font-size: 0.84rem;
}

.app-footer strong {
    color: #65d8cb;
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }

    .app-title {
        font-size: 3rem;
    }

    .topbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .topbar-status {
        justify-content: flex-start;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 18px 12px !important;
    }

    .hero-main {
        padding: 22px;
    }

    .app-title {
        font-size: 2.45rem;
    }

    .glass-panel {
        padding: 16px !important;
    }

    .upload-panel .image-container,
    .result-panel .image-container {
        min-height: 270px !important;
    }

    .metrics-shell,
    .usage-strip {
        grid-template-columns: 1fr;
    }
}
"""


with gr.Blocks(title="SnowClearNet") as demo:
    gr.HTML(
        """
        <section class="topbar">
            <div class="brand-lockup">
                <div class="brand-mark">SC</div>
                <div>
                    <div class="brand-name">SnowClearNet</div>
                    <div class="brand-sub">Smart city road restoration</div>
                </div>
            </div>
            <div class="topbar-status">
                <span class="status-chip">CPU only</span>
                <span class="status-chip">ONNX Runtime</span>
                <span class="status-chip">128 x 128 model input</span>
            </div>
        </section>

        <section class="hero-grid">
            <div class="hero-main">
                <div class="eyebrow">DATSRF image restoration</div>
                <h1 class="app-title">Clearer roads from snowy frames.</h1>
                <p class="app-subtitle">
                    Upload a snowy traffic image and restore visibility for road-scene monitoring.
                    The pipeline runs locally on CPU and reports PSNR and SSIM after processing.
                </p>
                <div class="hero-actions">
                    <span class="hero-pill">PNG and JPG</span>
                    <span class="hero-pill">No GPU required</span>
                    <span class="hero-pill">Before-to-after workflow</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="info-card">
                    <span>Model</span>
                    <strong>DATSRF</strong>
                    <small>Dual-attention restoration architecture exported to ONNX.</small>
                </div>
                <div class="info-card">
                    <span>Output</span>
                    <strong>PSNR / SSIM</strong>
                    <small>Restoration quality metrics generated for each processed image.</small>
                </div>
            </div>
        </section>

        <section class="console-header">
            <div>
                <h2 class="console-title">Restoration console</h2>
                <p class="console-copy">Choose a frame, process it, and inspect the restored result.</p>
            </div>
            <div class="console-kicker">SnowClearNet pipeline</div>
        </section>
        """
    )

    with gr.Row(elem_classes=["workspace"]):
        with gr.Column(scale=1, elem_classes=["glass-panel", "upload-panel"]):
            gr.HTML(
                """
                <h2 class="panel-title">Upload image</h2>
                <p class="panel-copy">Drop in a snowy traffic or road scene.</p>
                """
            )
            input_image = gr.Image(
                type="pil",
                label="Input image",
                image_mode="RGB",
                height=390,
                sources=["upload", "clipboard"],
            )
            run_button = gr.Button(
                "Process Image",
                variant="primary",
                elem_classes=["process-btn"],
            )

        with gr.Column(scale=1, elem_classes=["glass-panel", "result-panel"]):
            gr.HTML(
                """
                <h2 class="panel-title">Restoration result</h2>
                <p class="panel-copy">Recovered visibility and quality metrics appear here.</p>
                """
            )
            output_image = gr.Image(
                type="pil",
                label="Output image",
                height=390,
            )
            metrics_output = gr.HTML(
                render_metrics(),
                elem_classes=["metrics-box"],
            )

    gr.HTML(
        """
        <section class="usage-strip">
            <div class="usage-step">
                <strong>1. Upload</strong>
                Add a snowy image from the camera feed or dataset.
            </div>
            <div class="usage-step">
                <strong>2. Restore</strong>
                Run the CPU ONNX model through the existing pipeline.
            </div>
            <div class="usage-step">
                <strong>3. Compare</strong>
                Review the output with PSNR and SSIM.
            </div>
        </section>
        """
    )

    gr.HTML(
        """
        <div class="app-footer">
            <strong>SnowClearNet</strong> for Smart City Traffic Monitoring &middot; ONNX Runtime &middot; Gradio
        </div>
        """
    )

    run_button.click(
        fn=restore_image,
        inputs=input_image,
        outputs=[output_image, metrics_output],
    )


if __name__ == "__main__":
    launch_kwargs = {
        "css": CSS,
        "show_error": True,
    }
    server_name = os.getenv("GRADIO_SERVER_NAME")
    if server_name:
        launch_kwargs["server_name"] = server_name

    demo.queue().launch(**launch_kwargs)
