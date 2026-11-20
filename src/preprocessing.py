"""Data acquisition + preprocessing for the MaizeGuard pipeline.

Every image, whether it comes from the training folders, a single prediction
request or a bulk retraining upload, goes through the SAME steps:
    1. validate  (is it really an image?)
    2. decode    (bytes -> RGB pixels)
    3. resize    (224 x 224, the input size of MobileNetV2)
    4. scale     (done inside the model, so the API can never forget it)
"""
import io
import os
import random
import zipfile
from pathlib import Path

from PIL import Image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(os.getenv("TRAIN_DIR", BASE_DIR / "data" / "train"))
TEST_DIR = Path(os.getenv("TEST_DIR", BASE_DIR / "data" / "test"))


# ---------------------------------------------------------------- acquisition
def list_images(data_dir):
    """Return (paths, labels) for a folder laid out as data_dir/<class>/<image>."""
    paths, labels = [], []
    data_dir = Path(data_dir)
    for idx, name in enumerate(CLASS_NAMES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                paths.append(str(f))
                labels.append(idx)
    return paths, labels


def normalise_label(name):
    """Map 'common rust', 'Common-Rust', 'common_rust' ... to the official class name."""
    key = "".join(c for c in str(name).lower() if c.isalnum())
    for cls in CLASS_NAMES:
        if key == "".join(c for c in cls.lower() if c.isalnum()):
            return cls
    return None


# ----------------------------------------------------------------- validation
def clean_image_bytes(data, max_side=512):
    """Validate raw bytes and return them re-encoded as a clean RGB JPEG.

    Raises ValueError if the bytes are not a readable image. Used on every
    uploaded file before it is stored in the database.
    """
    try:
        Image.open(io.BytesIO(data)).verify()          # detects corrupt files
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"not a valid image: {exc}") from exc
    if min(img.size) < 32:
        raise ValueError("image is too small (minimum 32x32 pixels)")
    img.thumbnail((max_side, max_side))                # keep the database small
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def read_zip(data):
    """Yield (filename, folder_label_or_None, bytes) for each image in a zip.

    A zip laid out as <class>/<image> labels every image from its folder name.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            p = Path(info.filename)
            if info.is_dir() or p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("."):
                continue
            label = next((normalise_label(part) for part in reversed(p.parts[:-1])
                          if normalise_label(part)), None)
            yield p.name, label, zf.read(info)


# ------------------------------------------------------- tensor preprocessing
def decode_image(raw):
    """bytes tensor -> float32 tensor (224, 224, 3) with values 0-255."""
    import tensorflow as tf
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, IMG_SIZE)


def augment(img):
    """Random flips / rotation / brightness / contrast. Training only."""
    import tensorflow as tf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=25.0)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 255.0)


def make_dataset(paths, labels, training=False, batch_size=BATCH_SIZE):
    """Build a tf.data pipeline: read -> decode -> resize -> (augment) -> batch."""
    import tensorflow as tf
    n = len(CLASS_NAMES)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_image(tf.io.read_file(p)), tf.one_hot(y, n)),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_train_val(paths, labels, val_fraction=0.2):
    """Shuffle and split into train / validation lists (stratified when possible)."""
    from collections import Counter
    from sklearn.model_selection import train_test_split
    if len(paths) < 10:                                # too few to hold any out
        return paths, labels, [], []
    counts = Counter(labels)
    n_val = int(round(len(paths) * val_fraction))
    stratify = labels if min(counts.values()) >= 2 and n_val >= len(counts) else None
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=val_fraction, random_state=SEED, stratify=stratify)
    return tr_p, tr_l, va_p, va_l


def sample_per_class(paths, labels, per_class):
    """Pick up to `per_class` random images from each class (used as replay data)."""
    rng = random.Random(SEED)
    out_p, out_l = [], []
    for idx in range(len(CLASS_NAMES)):
        group = [p for p, l in zip(paths, labels) if l == idx]
        rng.shuffle(group)
        out_p += group[:per_class]
        out_l += [idx] * len(group[:per_class])
    return out_p, out_l


def bytes_to_batch(data):
    """Raw image bytes -> model input batch of shape (1, 224, 224, 3)."""
    import tensorflow as tf
    return tf.expand_dims(decode_image(tf.constant(data)), 0)
