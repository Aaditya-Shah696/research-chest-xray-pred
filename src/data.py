"""tf.data pipeline: uint8 grayscale, 3 channels, per-model preprocess_input, light augmentation."""

from __future__ import annotations

from pathlib import Path

import keras
import tensorflow as tf

from .records import FEATURE_SPEC, IMAGE_SIZE

# Importing applications does not build a network. Each preprocess_input is the
# one shipped with that backbone. Do not share a single rescale across models.
from keras.applications.densenet import preprocess_input as densenet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess

PREPROCESS = {
    "EfficientNetB0": efficientnet_preprocess,
    "DenseNet201": densenet_preprocess,
    "ResNet50": resnet_preprocess,
    "InceptionV3": inception_preprocess,
    "Xception": xception_preprocess,
}


def list_shards(preprocessed_dir: Path, split: str) -> list[str]:
    files = sorted((preprocessed_dir / "tfrecords" / split).glob("*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecords for {split} in {preprocessed_dir}")
    return [str(path) for path in files]


def parse_example(raw: tf.Tensor) -> dict:
    parsed = tf.io.parse_single_example(raw, FEATURE_SPEC)
    image = tf.io.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 1))
    image = tf.cast(image, tf.float32)
    image = tf.repeat(image, repeats=3, axis=-1)
    return {
        "image": image,
        "labels": parsed["labels"],
        "image_index": parsed["image_index"],
        "patient_id": parsed["patient_id"],
    }


def make_augmenter(seed: int) -> keras.Sequential:
    # 0.02 of a turn is about 7 degrees. Zoom is plus or minus 5%.
    # Layers expect a batch, so they run after batch() in the pipeline.
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(
                0.02,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="bilinear",
                seed=seed,
            ),
            keras.layers.RandomZoom(
                0.05,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="bilinear",
                seed=seed,
            ),
        ],
        name="augment",
    )


def _dataset(files: list[str]) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.with_options(options)


def train_dataset(
    preprocessed_dir: Path,
    backbone: str,
    batch_size: int,
    seed: int,
    augmenter: keras.Sequential,
    repeat: bool = False,
) -> tf.data.Dataset:
    preprocess = PREPROCESS[backbone]

    def _map(batch: dict):
        image = augmenter(batch["image"], training=True)
        image = preprocess(image)
        return image, batch["labels"]

    ds = _dataset(list_shards(preprocessed_dir, "train"))
    ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds


def xy_dataset(preprocessed_dir: Path, split: str, backbone: str, batch_size: int, repeat: bool = False) -> tf.data.Dataset:
    """Validation (image, labels) with preprocessing and no augmentation."""
    preprocess = PREPROCESS[backbone]

    def _map(batch: dict):
        return preprocess(batch["image"]), batch["labels"]

    ds = _dataset(list_shards(preprocessed_dir, split))
    ds = ds.batch(batch_size)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds


def predict_dataset(preprocessed_dir: Path, split: str, backbone: str, batch_size: int) -> tf.data.Dataset:
    """Yields image, image_index, patient_id, labels. No shuffle."""
    preprocess = PREPROCESS[backbone]

    def _map(batch: dict):
        return (
            preprocess(batch["image"]),
            batch["image_index"],
            batch["patient_id"],
            batch["labels"],
        )

    ds = _dataset(list_shards(preprocessed_dir, split))
    ds = ds.batch(batch_size)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def raw_image_dataset(preprocessed_dir: Path, split: str) -> tf.data.Dataset:
    """Uint8 grayscale images for Grad-CAM panels, before preprocessing."""

    def _map(example: dict):
        gray = tf.cast(example["image"][:, :, :1], tf.uint8)
        return gray, example["image_index"]

    ds = _dataset(list_shards(preprocessed_dir, split))
    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(32)


def preprocess_means(sample: tf.Tensor | None = None) -> dict[str, float]:
    """Mean pixel after each preprocess_input. The means must not all match."""
    if sample is None:
        sample = tf.ones((1, 16, 16, 3), tf.float32) * 128.0
    means = {}
    for name, fn in PREPROCESS.items():
        out = fn(sample)
        means[name] = float(tf.reduce_mean(out).numpy())
    rounded = {round(value, 3) for value in means.values()}
    if len(rounded) < 3:
        raise AssertionError(
            "preprocess_input collapsed to one shared transform. "
            f"Means: {means}"
        )
    return means
