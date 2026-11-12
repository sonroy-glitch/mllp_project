import os
import time

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from model_loader import IMG_SIZE, ModelNotFoundError, load_model, preprocess
from sports_metadata import DEFAULT_CLASS_NAMES, describe

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024
TOP_K = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL = None
CLASS_NAMES = DEFAULT_CLASS_NAMES
MODEL_FILE = None
LOAD_ERROR = None


def initialise():
    global MODEL, CLASS_NAMES, MODEL_FILE, LOAD_ERROR
    try:
        model, classes, filename = load_model()
        MODEL = model
        MODEL_FILE = filename
        if classes:
            CLASS_NAMES = classes
        LOAD_ERROR = None
        app.logger.info("Loaded %s with %d classes", filename, len(CLASS_NAMES))
    except ModelNotFoundError as exc:
        LOAD_ERROR = str(exc)
        app.logger.warning(LOAD_ERROR)
    except Exception as exc:
        LOAD_ERROR = f"Failed to load the model: {exc}"
        app.logger.error(LOAD_ERROR)


def softmax(values):
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "model_file": MODEL_FILE,
            "num_classes": len(CLASS_NAMES),
            "input_size": IMG_SIZE,
            "error": LOAD_ERROR,
        }
    )


@app.get("/classes")
def classes():
    return jsonify({"classes": CLASS_NAMES, "count": len(CLASS_NAMES)})


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify({"error": LOAD_ERROR or "The model is not loaded."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image was supplied in the 'image' field."}), 400

    upload = request.files["image"]
    if not upload.filename:
        return jsonify({"error": "The uploaded file has no name."}), 400

    extension = os.path.splitext(upload.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Use "
                    + ", ".join(sorted(ALLOWED_EXTENSIONS))
                    + "."
                }
            ),
            400,
        )

    try:
        tensor = preprocess(upload.stream)
    except Exception:
        return jsonify({"error": "The uploaded file could not be read as an image."}), 400

    started = time.perf_counter()
    try:
        raw = MODEL.predict(tensor, verbose=0)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    probabilities = np.asarray(raw).reshape(-1).astype(np.float64)
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        probabilities = softmax(probabilities)

    if len(probabilities) != len(CLASS_NAMES):
        return (
            jsonify(
                {
                    "error": "The model outputs "
                    f"{len(probabilities)} classes but {len(CLASS_NAMES)} labels are "
                    "configured. Add a matching class_names.json to the model folder."
                }
            ),
            500,
        )

    order = np.argsort(probabilities)[::-1][:TOP_K]
    top_k = [
        {"sport": CLASS_NAMES[index], "confidence": float(probabilities[index])}
        for index in order
    ]

    best = top_k[0]
    info = describe(best["sport"])

    return jsonify(
        {
            "sport": best["sport"],
            "confidence": best["confidence"],
            "category": info["category"],
            "origin": info["origin"],
            "top_k": top_k,
            "description": info["description"],
            "meta": {
                "model": MODEL_FILE,
                "backbone": "EfficientNetB0 (ImageNet, frozen)",
                "input_size": IMG_SIZE,
                "num_classes": len(CLASS_NAMES),
                "inference_ms": elapsed_ms,
            },
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "The image exceeds the 5 MB limit."}), 413


initialise()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
