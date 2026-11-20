"""Split the downloaded Kaggle dataset into data/train and data/test (80 / 20).

Usage:  python scripts/prepare_data.py <folder that contains the class folders>
The script searches the given folder for the four class folders, skips corrupt
images and copies the rest.
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image

from src.preprocessing import (CLASS_NAMES, IMAGE_EXTS, SEED, TEST_DIR, TRAIN_DIR,
                               normalise_label)


def main(source, test_fraction=0.2):
    source = Path(source)
    found = {}
    for d in [source, *source.rglob("*")]:
        if d.is_dir() and normalise_label(d.name) and normalise_label(d.name) not in found:
            found[normalise_label(d.name)] = d
    missing = [c for c in CLASS_NAMES if c not in found]
    if missing:
        sys.exit(f"Could not find folders for: {missing} under {source}")

    rng = random.Random(SEED)
    for cls, folder in found.items():
        files = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                Image.open(f).verify()
                files.append(f)
            except Exception:
                print("skipping corrupt file:", f.name)
        rng.shuffle(files)
        n_test = int(len(files) * test_fraction)
        for subset, group in ((TEST_DIR, files[:n_test]), (TRAIN_DIR, files[n_test:])):
            (subset / cls).mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, subset / cls / f.name)
        print(f"{cls:15s} train={len(files) - n_test:5d}  test={n_test:5d}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
