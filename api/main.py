"""MaizeGuard REST API (FastAPI).

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""
import json
import os
import socket
import tempfile
import threading
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src import database as db
from src.insights import INSIGHTS_PATH
from src.model import METRICS_PATH, MODEL_PATH, retrain_model
from src.prediction import get_model, model_version, predict_image
from src.preprocessing import CLASS_NAMES, clean_image_bytes, normalise_label, read_zip

START_TIME = time.time()
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))   # automatic trigger
MIN_IMAGES = int(os.getenv("RETRAIN_MIN_IMAGES", "4"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))

app = FastAPI(title="MaizeGuard API", version="1.0",
              description="Maize leaf disease classification with upload + retraining.")


@app.on_event("startup")
def startup():
    db.init_db()
    try:
        get_model()                              # warm up so the first request is fast
    except Exception as exc:                     # API still starts so /health can report it
        print("Model not loaded:", exc)


# ------------------------------------------------------------------ monitoring
@app.get("/health")
def health():
    """Model up-time and status."""
    uptime = time.time() - START_TIME
    return {"status": "ok" if MODEL_PATH.exists() else "model_missing",
            "uptime_seconds": round(uptime, 1),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(START_TIME)),
            "model_version": model_version(),
            "container": socket.gethostname(),
            "classes": CLASS_NAMES}


@app.get("/metrics")
def metrics():
    """Evaluation of the model currently in production (test-set metrics)."""
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet. Train the model first.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/insights")
def insights():
    if not INSIGHTS_PATH.exists():
        raise HTTPException(404, "No insights file. Run: python -m src.insights")
    return json.loads(INSIGHTS_PATH.read_text())


@app.get("/stats")
def stats():
    """Live usage: predictions served, average latency, uploads waiting for retraining."""
    return {"predictions": db.prediction_stats(), "pending_uploads": db.pending_counts(),
            "retrain_threshold": RETRAIN_THRESHOLD}


# ------------------------------------------------------------------ prediction
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Predict the class of ONE maize leaf image."""
    try:
        data = clean_image_bytes(file.file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    t0 = time.perf_counter()
    try:
        result = predict_image(data)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["container"] = socket.gethostname()
    db.log_prediction(result["label"], result["confidence"], result["latency_ms"])
    return result


# ---------------------------------------------------------- upload + retraining
@app.post("/upload")
def upload(files: List[UploadFile] = File(...), label: str = Form(None)):
    """Bulk upload for retraining.

    Send many images plus a `label`, or a .zip laid out as <class>/<image>.
    Each image is validated, cleaned and saved to the database.
    """
    form_label = normalise_label(label) if label else None
    if label and not form_label:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, rejected = 0, []
    for f in files:
        raw = f.file.read()
        if (f.filename or "").lower().endswith(".zip"):
            try:
                items = list(read_zip(raw))
            except Exception:
                rejected.append({"file": f.filename, "reason": "not a valid zip"})
                continue
        else:
            items = [(f.filename, None, raw)]
        for name, folder_label, data in items:
            lab = folder_label or form_label
            if not lab:
                rejected.append({"file": name, "reason": "no class label"})
                continue
            try:
                db.add_upload(name, lab, clean_image_bytes(data))
                saved += 1
            except ValueError as exc:
                rejected.append({"file": name, "reason": str(exc)})

    pending = db.pending_counts()
    auto = None
    if sum(pending.values()) >= RETRAIN_THRESHOLD:       # automatic retraining trigger
        auto = _start_retraining("auto: upload threshold reached")
    return {"saved": saved, "rejected": rejected, "pending_uploads": pending,
            "auto_retrain_started": bool(auto)}


def _run_retraining(run_id):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ids, paths, labels = db.export_pending(tmp)          # database -> image files
            result = retrain_model(paths, labels, epochs=RETRAIN_EPOCHS)
        db.mark_used(ids, run_id)
        msg = ("New model promoted to production." if result["promoted"] else
               "New model was worse on the test set, so the old model was kept.")
        db.finish_run(run_id, "completed", msg, len(ids), result["accuracy_before"],
                      result["accuracy_after"], result["promoted"])
    except Exception as exc:
        db.finish_run(run_id, "failed", f"{type(exc).__name__}: {exc}")


def _start_retraining(trigger):
    if not MODEL_PATH.exists():
        return None
    run_id = db.start_run(trigger)
    if run_id:
        threading.Thread(target=_run_retraining, args=(run_id,), daemon=True).start()
    return run_id


@app.post("/retrain")
def retrain():
    """Manual trigger: fine-tune the current model on all uploaded, not-yet-used images."""
    n = sum(db.pending_counts().values())
    if n < MIN_IMAGES:
        raise HTTPException(400, f"Upload at least {MIN_IMAGES} new images first (have {n}).")
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No base model found. Train the model first.")
    run_id = _start_retraining("manual")
    if not run_id:
        raise HTTPException(409, "A retraining run is already in progress.")
    return {"run_id": run_id, "status": "running", "images": n}


@app.get("/retrain/status")
def retrain_status():
    return {"runs": db.list_runs()}
