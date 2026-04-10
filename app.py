"""
SnowClearNet – Flask Application
=================================
Main entry point. Handles uploads, inference, and Supabase persistence.
Deployed on Vercel as a serverless function.
"""

import os
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

from src.model import load_model
from src.processing import process_image
from src import database as db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "snowclearnet-dev-key")

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH", "16"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Directories – /tmp is writable on Vercel serverless
IS_VERCEL = os.getenv("VERCEL", False)
UPLOAD_FOLDER = "/tmp/uploads" if IS_VERCEL else os.path.join(app.root_path, "uploads")
OUTPUT_FOLDER = "/tmp/outputs" if IS_VERCEL else os.path.join(app.root_path, "static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(app.root_path, "datsrf_model.onnx")
model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(original: str) -> str:
    ext = original.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex[:12]}.{ext}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Accept upload → run inference → persist metadata → return JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload PNG or JPG."}), 400

    # Save original
    safe_name = secure_filename(file.filename)
    upload_name = unique_filename(safe_name)
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
    file.save(upload_path)

    try:
        img = Image.open(upload_path).convert("RGB")
    except Exception:
        os.remove(upload_path)
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    # Inference
    try:
        output_pil, psnr_val, ssim_val = process_image(model, img)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # Save output
    output_name = f"out_{upload_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    output_pil.save(output_path)

    # Save resized input for side-by-side comparison
    input_static_name = f"in_{upload_name}"
    input_static_path = os.path.join(OUTPUT_FOLDER, input_static_name)
    img.resize((128, 128)).save(input_static_path)

    # Persist to Supabase (non-fatal)
    try:
        db.save_record(upload_name, output_name, psnr_val, ssim_val)
    except Exception as e:
        print(f"[!] Supabase write failed: {e}")

    # On Vercel we serve from /tmp via a dedicated route
    if IS_VERCEL:
        input_url = f"/img/{input_static_name}"
        output_url = f"/img/{output_name}"
    else:
        input_url = f"/static/outputs/{input_static_name}"
        output_url = f"/static/outputs/{output_name}"

    return jsonify({
        "input_url": input_url,
        "output_url": output_url,
        "psnr": psnr_val,
        "ssim": ssim_val,
    })


@app.route("/img/<path:filename>")
def serve_tmp_image(filename):
    """Serve images from /tmp on Vercel (static dir is read-only)."""
    from flask import send_from_directory
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/history")
def history():
    try:
        records = db.get_all_records(limit=20)
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": f"File exceeds the {MAX_MB} MB upload limit."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "An internal server error occurred."}), 500

# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
