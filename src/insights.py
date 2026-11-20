"""Dataset insights used by the notebook and the dashboard.

Computes three image features per class:
  * class balance   - how many images each class has
  * colour          - share of green in the leaf (green / (red+green+blue))
  * brightness and contrast (texture) - mean and spread of grey levels

Run:  python -m src.insights      (writes models/data_insights.json)
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import BASE_DIR, CLASS_NAMES, SEED, TRAIN_DIR, list_images

INSIGHTS_PATH = BASE_DIR / "models" / "data_insights.json"


def image_features(path):
    arr = np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.float32)
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    grey = arr.mean(axis=2)
    return {"red": float(r), "green": float(g), "blue": float(b),
            "green_ratio": float(g / (r + g + b + 1e-6)),
            "brightness": float(grey.mean()), "contrast": float(grey.std())}


def compute_insights(data_dir=TRAIN_DIR, max_per_class=200, save=True):
    paths, labels = list_images(data_dir)
    rng = random.Random(SEED)
    out = {"class_counts": {}, "features": {}}
    for idx, name in enumerate(CLASS_NAMES):
        group = [p for p, l in zip(paths, labels) if l == idx]
        out["class_counts"][name] = len(group)
        rng.shuffle(group)
        feats = [image_features(p) for p in group[:max_per_class]]
        if feats:
            out["features"][name] = {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}
    if save:
        INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        INSIGHTS_PATH.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(compute_insights(), indent=2))
