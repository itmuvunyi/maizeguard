"""Prediction: load the saved model once and classify single images."""
import threading
import time
from pathlib import Path

from src.model import MODEL_PATH
from src.preprocessing import CLASS_NAMES, bytes_to_batch

ADVICE = {
    "Blight": "Northern leaf blight: long grey-brown lesions. Remove badly infected leaves, "
              "rotate crops and plant blight-tolerant improved seed next season.",
    "Common_Rust": "Common rust: small reddish-brown pustules. Usually manageable; use "
                   "rust-resistant certified seed and monitor spread.",
    "Gray_Leaf_Spot": "Gray leaf spot: rectangular grey lesions between veins. Improve "
                      "airflow, rotate crops and choose resistant hybrid seed.",
    "Healthy": "No disease detected. Keep scouting the field every week.",
}

_lock = threading.Lock()
_model, _mtime = None, None


def get_model(model_path=MODEL_PATH):
    """Return the model, reloading it automatically if the file changed (after retraining)."""
    global _model, _mtime
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")
    mtime = model_path.stat().st_mtime
    if _model is None or mtime != _mtime:
        with _lock:
            if _model is None or mtime != _mtime:
                from tensorflow import keras
                _model = keras.models.load_model(str(model_path), compile=False)
                _mtime = mtime
    return _model


def model_version(model_path=MODEL_PATH):
    p = Path(model_path)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None


def predict_image(data, model_path=MODEL_PATH):
    """Classify one image given as raw bytes."""
    model = get_model(model_path)
    probs = model(bytes_to_batch(data), training=False).numpy()[0]
    best = int(probs.argmax())
    label = CLASS_NAMES[best]
    return {"label": label, "confidence": float(probs[best]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "advice": ADVICE[label]}


if __name__ == "__main__":
    import sys
    print(predict_image(Path(sys.argv[1]).read_bytes()))
