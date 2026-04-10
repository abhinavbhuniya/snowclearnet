"""
SnowClearNet – Flask Application
=================================
Main entry point for the web server.
Handles file uploads, model inference, and Supabase persistence.
"""

import os
import uuid

import torch
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

load_dotenv()  # Load .env variables

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "snowclearnet-dev-key")

# Upload constraints
MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH", 16))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Directories
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
OUTPUT_FOLDER = os.path.join(app.root_path, "static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Model initialisation (runs once at startup)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(app.root_path, "datsrf_model.pth")
model = load_model(MODEL_PATH, DEVICE)
print(f"[✓] Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(original: str) -> str:
    """Generate a collision-free filename while preserving the extension."""
    ext = original.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex[:12]}.{ext}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Accept an uploaded image, run snow-removal inference, persist
    metadata to Supabase, and return the results as JSON.
    """
    # --- validate upload ---
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload PNG or JPG."}), 400

    # --- save original ---
    safe_name = secure_filename(file.filename)
    upload_name = unique_filename(safe_name)
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
    file.save(upload_path)

    try:
        img = Image.open(upload_path).convert("RGB")
    except Exception:
        os.remove(upload_path)
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    # --- inference ---
    try:
        output_pil, psnr_val, ssim_val = process_image(model, img, DEVICE)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # --- save output ---
    output_name = f"out_{upload_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    output_pil.save(output_path)

    # --- also copy upload to static so the frontend can display it ---
    input_static_name = f"in_{upload_name}"
    input_static_path = os.path.join(OUTPUT_FOLDER, input_static_name)
    # Save the resized input at model resolution for fair side-by-side comparison
    img.resize((128, 128)).save(input_static_path)

    # --- persist to Supabase ---
    try:
        db.save_record(upload_name, output_name, psnr_val, ssim_val)
    except Exception as e:
        # Non-fatal: log but don't fail the request
        print(f"[!] Supabase write failed: {e}")

    # --- respond ---
    return jsonify({
        "input_url": f"/static/outputs/{input_static_name}",
        "output_url": f"/static/outputs/{output_name}",
        "psnr": psnr_val,
        "ssim": ssim_val,
    })


@app.route("/history")
def history():
    """Return recent processing records from Supabase."""
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
