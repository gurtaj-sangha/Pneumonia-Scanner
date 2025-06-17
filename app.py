# app.py
import os, uuid
from pathlib import Path
from glob import glob

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from bpmer import PneumoniaScanner

def newest_model() -> Path:
    files = sorted(glob("models/*/final.h5"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No final.h5 found under models/")
    return Path(files[-1])

MODEL_PATH = newest_model()           
print("Loaded model:", MODEL_PATH)

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static")
scanner = PneumoniaScanner(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "xray_image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["xray_image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    fname = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    img_path = UPLOAD_DIR / fname
    file.save(img_path)

    idx, probs = scanner.predict(img_path)
    vis_name = f"vis_{fname}"
    scanner.visualize_prediction(img_path, UPLOAD_DIR / vis_name)

    return jsonify({
        "prediction": scanner.CLASS_NAMES[idx],
        "probs": {k: round(float(p), 4)
                  for k, p in zip(scanner.CLASS_NAMES, probs)},
        "image_url": f"/static/uploads/{fname}",
        "vis_url": f"/static/uploads/{vis_name}",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
