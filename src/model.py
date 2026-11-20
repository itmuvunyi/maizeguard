"""Model creation, training, evaluation and retraining.

Train from the command line:   python -m src.model --epochs 10
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from src.preprocessing import (BASE_DIR, CLASS_NAMES, IMG_SIZE, TEST_DIR, TRAIN_DIR,
                               list_images, make_dataset, sample_per_class, split_train_val)

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / "maize_leaf_model.h5"
METRICS_PATH = MODEL_DIR / "metrics.json"


# ------------------------------------------------------------------- creation
def build_model(learning_rate=1e-3, dropout=0.3, l2=1e-4):
    """MobileNetV2 (pre-trained on ImageNet, frozen) + a small regularised head."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                          include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_minus1_1")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs, outputs, name="maize_leaf_classifier")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])


def unfreeze_top_layers(model, n_layers=30):
    """Make the last n layers of the MobileNetV2 base trainable (fine-tuning).

    BatchNormalization layers stay frozen so their statistics are not destroyed.
    """
    from tensorflow import keras
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def get_callbacks(monitor="val_loss"):
    from tensorflow import keras
    return [keras.callbacks.EarlyStopping(monitor=monitor, patience=3,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.3, patience=2)]


# ----------------------------------------------------------------- evaluation
def evaluate_model(model, paths, labels):
    """Accuracy, loss, precision, recall, F1 + per-class report + confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, log_loss,
                                 precision_recall_fscore_support)
    idx = list(range(len(CLASS_NAMES)))
    probs = model.predict(make_dataset(paths, labels), verbose=0)
    y_true, y_pred = np.array(labels), probs.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0)
    return {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, probs, labels=idx)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "per_class": classification_report(y_true, y_pred, labels=idx,
                                           target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
        "class_names": CLASS_NAMES,
    }


def save_metrics(test_metrics, history=None, note="initial training"):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
               "test": test_metrics, "history": history or {}}
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _history(*histories):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


# ------------------------------------------------------------------- training
def train_model(train_dir=TRAIN_DIR, test_dir=TEST_DIR, epochs=10, fine_tune_epochs=5):
    """Full offline training: frozen base first, then fine-tune the top layers."""
    paths, labels = list_images(train_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {train_dir}")
    tr_p, tr_l, va_p, va_l = split_train_val(paths, labels)
    train_ds, val_ds = make_dataset(tr_p, tr_l, training=True), make_dataset(va_p, va_l)

    model = build_model()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=get_callbacks())

    unfreeze_top_layers(model)
    compile_model(model, 1e-5)                  # small learning rate for fine-tuning
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                   callbacks=get_callbacks())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    te_p, te_l = list_images(test_dir)
    metrics = evaluate_model(model, te_p, te_l) if te_p else {}
    save_metrics(metrics, _history(h1, h2))
    return model, metrics


# ----------------------------------------------------------------- retraining
def retrain_model(new_paths, new_labels, model_path=MODEL_PATH, train_dir=TRAIN_DIR,
                  test_dir=TEST_DIR, epochs=5, replay_per_class=40, tolerance=0.02):
    """Fine-tune the EXISTING saved model on newly uploaded images.

    * The current model file is loaded and used as the pre-trained starting point.
    * A small sample of the original training data is mixed in ("replay") so the
      model does not forget what it already knows.
    * The model is evaluated on the test set before and after. The new version
      replaces the old one only if accuracy did not drop by more than `tolerance`.
    """
    from tensorflow import keras
    model_path = Path(model_path)
    model = keras.models.load_model(str(model_path), compile=False)

    te_p, te_l = list_images(test_dir)
    before = evaluate_model(model, te_p, te_l) if te_p else None

    old_p, old_l = sample_per_class(*list_images(train_dir), replay_per_class)
    paths, labels = list(new_paths) + old_p, list(new_labels) + old_l
    tr_p, tr_l, va_p, va_l = split_train_val(paths, labels)

    unfreeze_top_layers(model, n_layers=20)
    compile_model(model, 1e-5)
    monitor = "val_loss" if va_p else "loss"
    hist = model.fit(make_dataset(tr_p, tr_l, training=True),
                     validation_data=make_dataset(va_p, va_l) if va_p else None,
                     epochs=epochs, callbacks=get_callbacks(monitor), verbose=2)

    after = evaluate_model(model, te_p, te_l) if te_p else None
    promoted = before is None or after["accuracy"] >= before["accuracy"] - tolerance
    if promoted:
        archive = model_path.parent / "archive"
        archive.mkdir(exist_ok=True)
        shutil.copy2(model_path, archive / f"model_{time.strftime('%Y%m%d_%H%M%S')}.h5")
        tmp = model_path.with_name("tmp_" + model_path.name)
        model.save(str(tmp))
        os.replace(tmp, model_path)             # atomic swap: API never sees half a file
        if after:
            save_metrics(after, _history(hist), note="retrained on uploaded data")
    return {
        "promoted": bool(promoted),
        "new_images": len(new_paths), "replay_images": len(old_p),
        "epochs_run": len(hist.history["loss"]),
        "accuracy_before": before["accuracy"] if before else None,
        "accuracy_after": after["accuracy"] if after else None,
        "f1_before": before["f1"] if before else None,
        "f1_after": after["f1"] if after else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    args = ap.parse_args()
    _, m = train_model(epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs)
    print({k: v for k, v in m.items() if k in ("accuracy", "loss", "precision", "recall", "f1")})
