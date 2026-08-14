"""
Semiconductor Image Restoration -- Web Demo

Single-page Flask app: upload a degraded image (.npy or standard image
format), the trained DistributionMixtureRestorationNet restores it.
Displays three panels: Uploaded (input) / Noise Removed (computed
difference) / Restored (output).

Run with:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import os
import io
import base64
import traceback

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, render_template, request

from models.restoration_net import DistributionMixtureRestorationNet

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_PATH = os.path.join("..", "results", "checkpoints", "DistributionMixtureRestorationNet.pth")
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"npy", "png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# Site palette, reused for the "noise removed" heatmap so it visually
# belongs to the page rather than looking like a generic matplotlib default.
SITE_CMAP = LinearSegmentedColormap.from_list(
    "site_theme", ["#FAF7F1", "#C7A97A", "#B4915B", "#16283F"]
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ============================================================
# Model loading (once, at startup)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_meta = {}
model_load_error = None

def load_model():
    global model, model_meta, model_load_error
    try:
        m = DistributionMixtureRestorationNet(base_ch=32, n_components=3,
                                                n_lr_blocks=4, n_hr_blocks=2, use_film=True)
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m = m.to(device).eval()
        model = m
        model_meta = {
            "epoch": ckpt.get("epoch", "unknown"),
            "psnr": ckpt.get("best_val_psnr", None),
            "ssim": ckpt.get("best_val_ssim", None),
            "params": sum(p.numel() for p in m.parameters()),
            "device": device,
        }
        model_load_error = None
    except Exception as e:
        model_load_error = str(e)

load_model()


# ============================================================
# Helpers
# ============================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_uploaded_array(file_storage, filename):
    """Returns a float32 numpy array, values roughly in [0,1] (not forced -- .npy
    inputs may legitimately exceed [0,1], matching the real degraded-image
    statistics this model was trained on)."""
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "npy":
        arr = np.load(io.BytesIO(file_storage.read())).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]  # collapse an accidental channel dim
    else:
        img = Image.open(file_storage.stream).convert("L")  # grayscale
        img = img.resize((128, 128), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
    if arr.shape != (128, 128):
        # resize any other-sized .npy to the model's expected input size
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        img = img.resize((128, 128), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
    return arr


def array_to_base64_png(arr, cmap=None, vmin=0.0, vmax=1.0):
    fig = plt.figure(figsize=(3.2, 3.2), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap=cmap or "gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@torch.no_grad()
def restore(degraded_np):
    x = torch.from_numpy(degraded_np).unsqueeze(0).unsqueeze(0).to(device)
    restored, mix_weights, beta, scale = model(x)
    restored_np = restored[0, 0].clamp(0, 1).cpu().numpy()

    upsampled_input = torch.nn.functional.interpolate(
        x, size=restored_np.shape, mode="bilinear", align_corners=False
    )[0, 0].clamp(0, 1).cpu().numpy()
    diff_np = np.abs(restored_np - upsampled_input)

    return restored_np, diff_np, beta.cpu().numpy().tolist()


# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_meta=model_meta,
                            model_load_error=model_load_error, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", model_meta=model_meta,
                                model_load_error=model_load_error, result=None,
                                error="The model failed to load -- see server log. Nothing was processed.")

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("index.html", model_meta=model_meta,
                                model_load_error=model_load_error, result=None,
                                error="No file was selected. Choose a .npy file or an image (PNG/JPG) to restore.")

    if not allowed_file(file.filename):
        return render_template("index.html", model_meta=model_meta,
                                model_load_error=model_load_error, result=None,
                                error=f"'{file.filename}' is not a supported file type. "
                                      f"Upload a .npy array or a PNG/JPG/BMP/TIFF image.")

    try:
        degraded_np = load_uploaded_array(file, file.filename)
        restored_np, diff_np, beta = restore(degraded_np)

        result = {
            "filename": file.filename,
            "uploaded_b64": array_to_base64_png(degraded_np, vmin=float(degraded_np.min()), vmax=float(degraded_np.max())),
            "diff_b64": array_to_base64_png(diff_np, cmap=SITE_CMAP, vmin=0.0, vmax=max(diff_np.max(), 1e-6)),
            "restored_b64": array_to_base64_png(restored_np),
            "input_shape": f"{degraded_np.shape[0]} x {degraded_np.shape[1]}",
            "output_shape": f"{restored_np.shape[0]} x {restored_np.shape[1]}",
            "input_range": f"[{degraded_np.min():.3f}, {degraded_np.max():.3f}]",
            "mean_change": f"{diff_np.mean():.4f}",
        }
        return render_template("index.html", model_meta=model_meta,
                                model_load_error=model_load_error, result=result)
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", model_meta=model_meta,
                                model_load_error=model_load_error, result=None,
                                error=f"Could not process '{file.filename}': {e}")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
