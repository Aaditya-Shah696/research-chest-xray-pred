"""Resize once to 224x224 uint8 grayscale and write sharded TFRecords."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from .labels import LABELS, N_LABELS

IMAGE_SIZE = 224
SHARD_SIZE = 512


def index_images(data_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(data_dir):
        # Never index our own output if someone points data_dir at the repo.
        dirnames[:] = [name for name in dirnames if name not in {"tfrecords", ".venv", "__pycache__"}]
        for filename in filenames:
            lower = filename.lower()
            if lower.endswith((".png", ".jpg", ".jpeg")):
                found[filename] = Path(dirpath) / filename
    return found


def load_uint8(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        resized = gray.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.uint8)
    if array.shape != (size, size):
        raise ValueError(f"{path} resized to {array.shape}, expected {(size, size)}")
    return array


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _float_feature(values: list[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def example_from_row(row: pd.Series, image: np.ndarray) -> tf.train.Example:
    labels = [float(row[label]) for label in LABELS]
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": _bytes_feature(image.tobytes()),
                "image_index": _bytes_feature(str(row["Image Index"]).encode("utf-8")),
                "patient_id": _int_feature(int(row["Patient ID"])),
                "labels": _float_feature(labels),
            }
        )
    )


FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_index": tf.io.FixedLenFeature([], tf.string),
    "patient_id": tf.io.FixedLenFeature([], tf.int64),
    "labels": tf.io.FixedLenFeature([N_LABELS], tf.float32),
}


def write_split(frame: pd.DataFrame, image_index: dict[str, Path], out_dir: Path, split: str) -> int:
    rows = frame[frame["split"] == split].sort_values("Image Index")
    if rows.empty:
        raise ValueError(f"No rows to write for split {split}")
    dest = out_dir / "tfrecords" / split
    dest.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, int(np.ceil(len(rows) / SHARD_SIZE)))
    written = 0
    for shard in range(n_shards):
        chunk = rows.iloc[shard * SHARD_SIZE : (shard + 1) * SHARD_SIZE]
        path = dest / f"shard-{shard:05d}-of-{n_shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for _, row in chunk.iterrows():
                image = load_uint8(image_index[row["Image Index"]])
                writer.write(example_from_row(row, image).SerializeToString())
                written += 1
                if written % 2000 == 0:
                    print(f"  {split}: wrote {written} images")
    print(f"  {split}: {written} images in {n_shards} shard(s)")
    return written


def verify_roundtrip(out_dir: Path, labels: pd.DataFrame) -> None:
    """Read one stored example and check the shape, dtype, and label width."""
    shards = sorted((out_dir / "tfrecords").rglob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No TFRecords under {out_dir}")
    raw = next(iter(tf.data.TFRecordDataset([str(shards[0])])))
    parsed = tf.io.parse_single_example(raw, FEATURE_SPEC)
    image = tf.io.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE))
    if tuple(image.shape) != (IMAGE_SIZE, IMAGE_SIZE):
        raise AssertionError(f"Roundtrip image shape {image.shape}")
    if int(parsed["labels"].shape[0]) != N_LABELS:
        raise AssertionError("Roundtrip label vector has the wrong length")
    image_index = parsed["image_index"].numpy().decode("utf-8")
    match = labels[labels["Image Index"] == image_index]
    if match.empty:
        raise AssertionError(f"{image_index} is in a TFRecord but not in labels.csv")
    stored = parsed["labels"].numpy()
    expected = match.iloc[0][list(LABELS)].to_numpy(dtype=np.float32)
    if not np.allclose(stored, expected):
        raise AssertionError(f"Label mismatch for {image_index}")
    print(f"Roundtrip OK: {image_index} shape={tuple(image.shape)} dtype=uint8 labels={N_LABELS}")


def write_metadata(out_dir: Path, frame: pd.DataFrame, meta: dict) -> None:
    ordered = frame.copy()
    ordered["split"] = pd.Categorical(ordered["split"], ["train", "val", "test"], ordered=True)
    ordered = ordered.sort_values(["split", "Image Index"])
    columns = ["Image Index", "Patient ID", "split", *LABELS]
    ordered[columns].to_csv(out_dir / "labels.csv", index=False)
    (out_dir / "label_order.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
