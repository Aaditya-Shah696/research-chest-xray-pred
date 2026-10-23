"""Two-stage training with checkpoints, resume, and a per-session time budget."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .data import make_augmenter, predict_dataset, train_dataset, xy_dataset
from .labels import LABELS, N_LABELS, SMALL_DATASET_IMAGES
from .metrics import macro_auroc
from .model import (
    CUSTOM_OBJECTS,
    DEFAULT_BATCH,
    WeightedBinaryCrossentropy,
    align_batch_size,
    assert_stage1,
    assert_stage2,
    build_model,
    class_weights,
    compile_model,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top,
)
from .paths import find_preprocessed_dir, resolve_out_dir
from .plan import infer_phase, should_start_epoch
from .preprocess import load_label_order

HISTORY_FIELDS = ["stage", "epoch", "loss", "val_loss", "val_auroc", "lr", "seconds"]
DEFAULT_BUDGET_SEC = 4.5 * 3600
PATIENCE = 2
MIN_DELTA = 1e-4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_history(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _latest_epoch_file(directory: Path, stage: int) -> tuple[Path | None, int]:
    files = sorted(directory.glob(f"stage{stage}_epoch_*.keras"))
    if not files:
        return None, 0
    last = files[-1]
    return last, int(last.stem.split("_")[-1])


def _read_state(directory: Path | None) -> dict:
    if directory is None:
        return {}
    path = directory / "state.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_resume_dir(backbone: str, writable: Path) -> Path | None:
    if writable.name != backbone:
        return None
    if (_latest_epoch_file(writable, 1)[1] > 0) or (_latest_epoch_file(writable, 2)[1] > 0):
        return writable
    if (writable / "state.json").is_file():
        return writable
    return None


def _preds_on_disk(folder: Path, backbone: str) -> bool:
    return (folder / f"preds_val_{backbone}.csv").is_file() and (
        folder / f"preds_test_{backbone}.csv"
    ).is_file()


def _best_checkpoint(read_dir: Path, writable: Path) -> Path | None:
    for folder in (writable, read_dir):
        for name in ("stage2_best.keras", "stage1_best.keras"):
            path = folder / name
            if path.is_file():
                return path
        for stage in (2, 1):
            path, epoch = _latest_epoch_file(folder, stage)
            if path is not None and epoch > 0:
                return path
    return None


def _current_lr(optimizer) -> float:
    learning_rate = optimizer.learning_rate
    if callable(learning_rate):
        learning_rate = learning_rate(optimizer.iterations)
    return float(tf.convert_to_tensor(learning_rate).numpy())


def _predict_batch(model: keras.Model, images: tf.Tensor) -> np.ndarray:
    """Infer once on this full batch, without predict_on_batch replica gathering."""
    predictions = np.asarray(model(images, training=False))
    expected = int(tf.shape(images)[0])
    if predictions.shape != (expected, N_LABELS):
        raise RuntimeError(
            f"Prediction shape {predictions.shape} does not match "
            f"batch shape ({expected}, {N_LABELS})."
        )
    return predictions


def _dataset_auroc(model: keras.Model, dataset: tf.data.Dataset, steps: int) -> float:
    truths = []
    probs = []
    for step, (images, labels) in enumerate(dataset):
        if step >= steps:
            break
        truths.append(np.asarray(labels))
        probs.append(_predict_batch(model, images))
    if not truths:
        return float("nan")
    return macro_auroc(np.concatenate(truths), np.concatenate(probs))


def _decode_index(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_prediction_csvs(
    model: keras.Model,
    preprocessed_dir: Path,
    backbone: str,
    out_dir: Path,
    batch_size: int,
) -> None:
    for split in ("val", "test"):
        dataset = predict_dataset(preprocessed_dir, split, backbone, batch_size)
        # One pass keeps each probability next to its Image Index. model.predict
        # on a finite tf.data pipeline also warns when the iterator ends.
        prob_batches = []
        indexes: list[str] = []
        for images, index, _patient, _labels in dataset:
            prob_batches.append(_predict_batch(model, images))
            indexes.extend(_decode_index(value) for value in index.numpy())
        if not prob_batches:
            raise RuntimeError(f"No {split} images to score.")
        probabilities = np.concatenate(prob_batches, axis=0)
        if len(indexes) != len(probabilities):
            raise RuntimeError(
                f"{split} prediction length {len(probabilities)} does not match "
                f"{len(indexes)} image indexes."
            )
        frame = pd.DataFrame(probabilities, columns=list(LABELS))
        frame.insert(0, "Image Index", indexes)
        if list(frame.columns) != ["Image Index", *LABELS]:
            raise AssertionError("Prediction columns left the fixed label order.")
        path = out_dir / f"preds_{split}_{backbone}.csv"
        frame.to_csv(path, index=False, float_format="%.6f")
        print(f"Wrote {path} ({len(frame)} rows)")


def _make_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not set memory growth on {gpu}: {exc}")
    if len(gpus) >= 2:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy on {len(gpus)} GPUs.")
            return strategy, len(gpus)
        except Exception as exc:
            print(f"MirroredStrategy failed ({exc}). Using one device.")
    if gpus:
        print(f"Using one GPU: {gpus[0]}.")
    else:
        print("No GPU visible. Training on CPU.")
    return tf.distribute.get_strategy(), max(len(gpus), 1 if gpus else 1)


def _enable_precision(use_mixed: bool, n_gpus: int) -> None:
    force = os.environ.get("CXR_MIXED_PRECISION", "") == "1"
    if use_mixed and (n_gpus > 0 and tf.config.list_physical_devices("GPU") or force):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16. The sigmoid head stays float32.")
    else:
        keras.mixed_precision.set_global_policy("float32")
        if use_mixed:
            print("Mixed precision stays off because no GPU is visible. It turns on when a GPU is visible.")


def _load_model(path: Path, compile_model_flag: bool) -> keras.Model:
    print(f"Loading {path}")
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=compile_model_flag)


def _fit_stage(
    model: keras.Model,
    stage: int,
    start_epoch: int,
    target_epochs: int,
    train_ds_fn,
    val_ds_fn,
    train_steps: int,
    val_steps: int,
    ckpt: Path,
    state: dict,
    history_path: Path,
    budget_sec: float,
    started_at: float,
    patience: int,
) -> str:
    """Train until the epoch target, early stopping, or the time budget.

    Returns 'complete', 'early', or 'time'.
    """
    best_key = f"stage{stage}_best_auroc"
    best_score = float(state.get(best_key, -1.0))
    wait_key = f"stage{stage}_wait"
    wait = int(state.get(wait_key, 0)) if start_epoch else 0
    last_epoch_sec: float | None = None
    epoch = start_epoch
    outcome = "complete"
    while epoch < target_epochs:
        elapsed = time.time() - started_at
        if not should_start_epoch(elapsed, last_epoch_sec, budget_sec):
            print(
                f"Time budget reached after {elapsed / 3600:.2f} h "
                f"(limit {budget_sec / 3600:.2f} h). Stopping before the next epoch."
            )
            state["time_stopped_stage"] = stage
            outcome = "time"
            break
        epoch_started = time.time()
        history = model.fit(
            train_ds_fn(),
            validation_data=val_ds_fn(),
            epochs=epoch + 1,
            initial_epoch=epoch,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            shuffle=False,
            verbose=1,
        )
        score = _dataset_auroc(model, val_ds_fn(), val_steps)
        epoch_sec = time.time() - epoch_started
        epoch_path = ckpt / f"stage{stage}_epoch_{epoch + 1:02d}.keras"
        model.save(epoch_path)
        improved = np.isfinite(score) and score > best_score + MIN_DELTA
        if improved:
            best_score = float(score)
            best_path = ckpt / f"stage{stage}_best.keras"
            model.save(best_path)
            state[best_key] = best_score
            wait = 0
            print(f"Stage {stage} epoch {epoch + 1}: val AUROC {score:.4f} (best).")
        else:
            wait += 1
            shown = "nan" if not np.isfinite(score) else f"{score:.4f}"
            print(f"Stage {stage} epoch {epoch + 1}: val AUROC {shown}. Wait {wait}/{patience}.")
        state[f"stage{stage}_epoch"] = epoch + 1
        state[wait_key] = wait
        _write_json(ckpt / "state.json", state)
        _append_history(
            history_path,
            {
                "stage": stage,
                "epoch": epoch + 1,
                "loss": history.history["loss"][-1],
                "val_loss": history.history["val_loss"][-1],
                "val_auroc": score,
                "lr": _current_lr(model.optimizer),
                "seconds": round(epoch_sec, 1),
            },
        )
        epoch += 1
        last_epoch_sec = epoch_sec
        if stage == 2 and wait >= patience:
            print(f"Early stopping on validation AUROC (patience {patience}).")
            state["early_stopped_stage"] = 2
            outcome = "early"
            break
    if stage == 1 and outcome == "complete":
        state["stage1_complete"] = True
    _write_json(ckpt / "state.json", state)
    return outcome


def run_training(
    backbone: str,
    seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    weights: str | None,
    batch_size: int | None = None,
    out_dir: str | Path | None = None,
    preprocessed_dir: str | Path | None = None,
    mixed_precision: bool = True,
) -> dict:
    keras.utils.set_random_seed(seed)
    out_dir = resolve_out_dir(out_dir)
    preprocessed_dir = find_preprocessed_dir(preprocessed_dir)
    meta = load_label_order(preprocessed_dir)
    labels = pd.read_csv(preprocessed_dir / "labels.csv")
    n_images = int(meta.get("n_images", len(labels)))
    if n_images < SMALL_DATASET_IMAGES:
        raise RuntimeError(
            f"The preprocessed set has only {n_images} images. Refusing a full training run. "
            "Point the notebook at the NIH preprocess output."
        )

    budget_sec = float(os.environ.get("CXR_TIME_BUDGET_SEC", str(DEFAULT_BUDGET_SEC)))
    patience = PATIENCE
    if batch_size is None:
        batch_size = DEFAULT_BATCH[backbone]

    print(f"TensorFlow {tf.__version__}  Keras {keras.__version__}")
    print(f"BACKBONE={backbone} SEED={seed} WEIGHTS={weights}")
    print(f"Stage 1 epochs={stage1_epochs}  Stage 2 epochs={stage2_epochs}  patience={patience}")
    print(f"Preprocessed data: {preprocessed_dir}")
    print(f"Output: {out_dir}")
    print(f"Time budget: {budget_sec / 3600:.2f} h")

    train_rows = labels[labels["split"] == "train"]
    pos_weight, neg_weight = class_weights(train_rows[list(LABELS)].to_numpy())
    print("Positive weights (train split, clipped):")
    for label, weight in zip(LABELS, pos_weight):
        print(f"  {label:22s} {weight:.2f}")

    strategy, _device_count = _make_strategy()
    _enable_precision(mixed_precision, len(tf.config.list_physical_devices("GPU")))
    batch_size = align_batch_size(batch_size, strategy.num_replicas_in_sync)
    print(f"Global batch size: {batch_size}")

    writable = out_dir / "checkpoints" / backbone
    writable.mkdir(parents=True, exist_ok=True)
    read_dir = _find_resume_dir(backbone, writable)
    state = _read_state(read_dir)
    if state.get("backbone") not in (None, backbone):
        raise RuntimeError(
            f"Checkpoint backbone is {state.get('backbone')}, not {backbone}. "
            "Use a separate output folder per run."
        )
    state.setdefault("backbone", backbone)
    state.setdefault("labels", list(LABELS))
    if state.get("batch_size") not in (None, batch_size):
        print(
            f"Warning: this session uses batch size {batch_size}, "
            f"the checkpoint used {state.get('batch_size')}."
        )
    state["batch_size"] = batch_size
    if state.get("labels") != list(LABELS):
        raise RuntimeError("Checkpoint label order does not match the code. Do not resume this folder.")

    s1_epoch = _latest_epoch_file(read_dir, 1)[1] if read_dir else 0
    s2_epoch = _latest_epoch_file(read_dir, 2)[1] if read_dir else 0
    # Epoch counters in state are a fallback when the filename parse fails.
    s1_epoch = max(s1_epoch, int(state.get("stage1_epoch", 0) if read_dir else 0))
    s2_epoch = max(s2_epoch, int(state.get("stage2_epoch", 0) if read_dir else 0))
    preds_exist = _preds_on_disk(out_dir, backbone) or (
        read_dir is not None and read_dir != writable and _preds_on_disk(read_dir.parent.parent, backbone)
    )
    # Predictions from a previous run sit next to that run's files,
    # often beside checkpoints/ rather than inside it.
    if read_dir is not None:
        for candidate in (read_dir, read_dir.parent, read_dir.parent.parent):
            if _preds_on_disk(candidate, backbone):
                preds_exist = True
                break

    phase = infer_phase(
        state,
        stage1_epoch=s1_epoch,
        stage2_epoch=s2_epoch,
        stage1_target=stage1_epochs,
        stage2_target=stage2_epochs,
        preds_exist=preds_exist,
    )
    print(f"Resume phase: {phase} (stage1 epoch {s1_epoch}/{stage1_epochs}, stage2 epoch {s2_epoch}/{stage2_epochs})")

    if phase == "done":
        print("Training already finished and prediction files are on disk. Nothing to do.")
        return {"phase": phase, "out_dir": str(out_dir), "backbone": backbone}

    history_path = out_dir / f"history_{backbone}.csv"
    loss = WeightedBinaryCrossentropy(pos_weight, neg_weight)
    n_train = int(len(train_rows))
    steps = max(1, math.ceil(n_train / batch_size))
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=max(steps * stage2_epochs, 1),
        alpha=0.01,
    )
    augmenter = make_augmenter(seed)

    n_val = int((labels["split"] == "val").sum())
    train_steps = max(1, math.ceil(n_train / batch_size))
    val_steps = max(1, math.ceil(n_val / batch_size))

    def train_ds_fn():
        # repeat() plus an explicit step count keeps Keras from treating the end
        # of a finite dataset as an interrupted epoch.
        return train_dataset(preprocessed_dir, backbone, batch_size, seed, augmenter, repeat=True)

    def val_ds_fn():
        return xy_dataset(preprocessed_dir, "val", backbone, batch_size, repeat=True)

    started_at = time.time()

    def fresh_stage1() -> keras.Model:
        try:
            model = build_model(backbone, weights)
        except Exception as exc:
            if weights == "imagenet":
                raise RuntimeError(
                    "Could not load ImageNet weights. Turn Internet on for this session so the weights can download."
                ) from exc
            raise
        freeze_backbone(model)
        compile_model(model, keras.optimizers.Adam(learning_rate=1e-3), loss)
        assert_stage1(model)
        print(f"Stage 1 trainable parameters: {trainable_parameter_count(model)}")
        return model

    def enter_stage2(model: keras.Model) -> keras.Model:
        unfreeze_top(model, fraction=0.30)
        compile_model(model, keras.optimizers.Adam(learning_rate=schedule), loss)
        assert_stage2(model)
        print(f"Stage 2 trainable parameters: {trainable_parameter_count(model)}")
        state["stage2_wait"] = 0
        return model

    with strategy.scope():
        if phase == "predict":
            best = _best_checkpoint(read_dir or writable, writable)
            if best is None:
                raise FileNotFoundError(f"No checkpoint to predict from in {writable}")
            model = _load_model(best, compile_model_flag=False)
        elif phase == "stage2" and s2_epoch > 0:
            latest, _ = _latest_epoch_file(read_dir or writable, 2)
            model = _load_model(latest, compile_model_flag=True)
            assert_stage2(model)
            print(f"Resuming stage 2 at epoch {s2_epoch}. Trainable parameters: {trainable_parameter_count(model)}")
        elif phase == "stage2":
            best = _best_checkpoint(read_dir or writable, writable)
            if best is None:
                model = fresh_stage1()
            else:
                model = _load_model(best, compile_model_flag=False)
            model = enter_stage2(model)
        elif s1_epoch > 0:
            latest, _ = _latest_epoch_file(read_dir or writable, 1)
            model = _load_model(latest, compile_model_flag=True)
            assert_stage1(model)
            print(f"Resuming stage 1 at epoch {s1_epoch}. Trainable parameters: {trainable_parameter_count(model)}")
        else:
            if history_path.exists():
                history_path.unlink()
            model = fresh_stage1()

        if phase == "stage1":
            outcome = _fit_stage(
                model,
                stage=1,
                start_epoch=s1_epoch,
                target_epochs=stage1_epochs,
                train_ds_fn=train_ds_fn,
                val_ds_fn=val_ds_fn,
                train_steps=train_steps,
                val_steps=val_steps,
                ckpt=writable,
                state=state,
                history_path=history_path,
                budget_sec=budget_sec,
                started_at=started_at,
                patience=patience,
            )
            if outcome == "time":
                phase = "predict"
            else:
                phase = "stage2"
                best = writable / "stage1_best.keras"
                if best.is_file():
                    model = _load_model(best, compile_model_flag=False)
                model = enter_stage2(model)
                s2_epoch = 0

        if phase == "stage2":
            outcome = _fit_stage(
                model,
                stage=2,
                start_epoch=s2_epoch,
                target_epochs=stage2_epochs,
                train_ds_fn=train_ds_fn,
                val_ds_fn=val_ds_fn,
                train_steps=train_steps,
                val_steps=val_steps,
                ckpt=writable,
                state=state,
                history_path=history_path,
                budget_sec=budget_sec,
                started_at=started_at,
                patience=patience,
            )
            del outcome
            phase = "predict"

    if phase == "predict":
        best = _best_checkpoint(read_dir or writable, writable)
        if best is None:
            raise FileNotFoundError("Training stopped but no checkpoint was written.")
        # Reload outside the strategy scope's last model so prediction uses the best epoch.
        model = _load_model(best, compile_model_flag=False)
        final_path = out_dir / f"final_{backbone}.keras"
        model.save(final_path)
        print(f"Wrote {final_path}")
        write_prediction_csvs(model, preprocessed_dir, backbone, out_dir, batch_size)
        state["preds_written"] = True
        _write_json(writable / "state.json", state)

    if "float32" not in str(model.output.dtype):
        raise AssertionError(f"Saved model output dtype is {model.output.dtype}")
    print("Training pass finished.")
    return {"phase": "done", "out_dir": str(out_dir), "backbone": backbone, "n_labels": N_LABELS}
