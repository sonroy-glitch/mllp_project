import json
import os
import pickle

import numpy as np
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
IMG_SIZE = 224
CANDIDATE_FILES = [
    "sports_model.pkl",
    "model.pkl",
    "sports_model.keras",
    "model.keras",
    "sports_model.h5",
    "model.h5",
]


class ModelNotFoundError(RuntimeError):
    pass


def _locate():
    for name in CANDIDATE_FILES:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return path

    if os.path.isdir(MODEL_DIR):
        for name in sorted(os.listdir(MODEL_DIR)):
            if name.endswith((".pkl", ".keras", ".h5")):
                return os.path.join(MODEL_DIR, name)

    raise ModelNotFoundError(
        "No model file found. Place sports_model.pkl (or .keras / .h5) in "
        f"{MODEL_DIR}."
    )


def _unwrap(payload):
    if isinstance(payload, dict):
        model = (
            payload.get("model")
            or payload.get("classifier")
            or payload.get("estimator")
        )
        classes = (
            payload.get("class_names")
            or payload.get("classes")
            or payload.get("labels")
        )
        if model is None:
            raise ModelNotFoundError(
                "The pickle file does not contain a 'model' key."
            )
        return model, list(classes) if classes is not None else None

    return payload, None


def _read_class_names():
    path = os.path.join(MODEL_DIR, "class_names.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return [data[key] for key in sorted(data, key=lambda k: int(k))]
    return list(data)


def load_model():
    path = _locate()

    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        model, classes = _unwrap(payload)
    else:
        import tensorflow as tf

        model = tf.keras.models.load_model(path)
        classes = None

    if classes is None:
        classes = _read_class_names()

    return model, classes, os.path.basename(path)


def preprocess(stream):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    image = Image.open(stream).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(preprocess_input(array), axis=0)
