"""Align prediction files, ensemble them, and write metrics plus figures."""

from __future__ import annotations

import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_curve

from .data import PREPROCESS, raw_image_dataset
from .labels import FOCUS_LABELS, LABELS
from .metrics import (
    binary_counts,
    patient_bootstrap_auroc,
    patient_bootstrap_macro,
    safe_auroc,
    select_focus_cases,
    sensitivity_at_specificity,
    youden_threshold,
)
from .model import CUSTOM_OBJECTS, find_last_conv
from .paths import find_prediction_csvs, find_preprocessed_dir, resolve_out_dir, shallow_find
from .preprocess import load_label_order

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#000000"]
MODEL_ORDER = ["EfficientNetB0", "DenseNet201", "ResNet50", "InceptionV3", "Xception", "ensemble"]
EXPECTED_COLUMNS = ["Image Index", *LABELS]


def prediction_columns_ok(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path.name} is not in the fixed label order, so it cannot be averaged.\n"
            f"file:     {columns}\n"
            f"expected: {EXPECTED_COLUMNS}"
        )


def load_split_predictions(kind: str, extra_roots: list[Path] | None = None) -> dict[str, pd.DataFrame]:
    paths = find_prediction_csvs(kind, extra_roots=extra_roots)
    if not paths:
        raise FileNotFoundError(
            f"No preds_{kind}_<backbone>.csv files found. "
            "Run 01_train.ipynb first, or set OUTPUT_DIR to the directory that contains the prediction CSVs."
        )
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        # preds_val_EfficientNetB0 -> EfficientNetB0
        backbone = path.stem.split("_", 2)[-1]
        frame = pd.read_csv(path)
        prediction_columns_ok(frame, path)
        if frame["Image Index"].duplicated().any():
            raise ValueError(f"Duplicate Image Index in {path}")
        frame["Image Index"] = frame["Image Index"].astype(str)
        frame = frame.sort_values("Image Index").reset_index(drop=True)
        if backbone in frames:
            raise ValueError(f"Two prediction files for {backbone} ({kind}). Keep one.")
        frames[backbone] = frame
        print(f"Loaded {path} ({len(frame)} rows)")
    return align_frames(frames)


def align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("No prediction frames to align.")
    names = list(frames)
    reference_name = names[0]
    reference = frames[reference_name]["Image Index"]
    for name, frame in frames.items():
        prediction_columns_ok(frame, Path(name))
        if not frame["Image Index"].equals(reference):
            raise AssertionError(
                f"{name} does not have the same Image Index values, in the same order, as "
                f"{reference_name}. Refusing to average mismatched rows."
            )
    return frames


def average_ensemble(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stack = np.stack([frame[list(LABELS)].to_numpy(dtype=np.float64) for frame in frames.values()], axis=0)
    mean = stack.mean(axis=0)
    ensemble = pd.DataFrame(mean, columns=list(LABELS))
    ensemble.insert(0, "Image Index", next(iter(frames.values()))["Image Index"].to_numpy())
    return ensemble


def _truth(preprocessed_dir: Path, split: str) -> pd.DataFrame:
    labels = pd.read_csv(preprocessed_dir / "labels.csv")
    labels["Image Index"] = labels["Image Index"].astype(str)
    truth = labels[labels["split"] == split][["Image Index", "Patient ID", *LABELS]].copy()
    return truth.sort_values("Image Index").reset_index(drop=True)


def _align_truth(truth: pd.DataFrame, prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = prediction.sort_values("Image Index").reset_index(drop=True)
    truth = truth.sort_values("Image Index").reset_index(drop=True)
    if not truth["Image Index"].equals(prediction["Image Index"]):
        missing = sorted(set(truth["Image Index"]) ^ set(prediction["Image Index"]))[:5]
        raise AssertionError(
            "Prediction rows and labels.csv do not describe the same images. "
            f"Examples from the symmetric difference: {missing}"
        )
    return truth, prediction


def _metric_row(model: str, label: str, y_true, y_score, patient_ids, threshold, n_bootstrap, seed) -> dict:
    point, low, high, n_defined = patient_bootstrap_auroc(
        y_true, y_score, patient_ids, n_bootstrap=n_bootstrap, seed=seed
    )
    pred = (np.asarray(y_score) >= threshold).astype(int)
    operating = binary_counts(y_true, pred)
    at_half = binary_counts(y_true, (np.asarray(y_score) >= 0.5).astype(int))
    return {
        "model": model,
        "label": label,
        "auroc": point,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "auroc_boot_n": n_defined,
        "threshold": threshold,
        "sensitivity": operating["sensitivity"],
        "specificity": operating["specificity"],
        "accuracy": operating["accuracy"],
        "f1": operating["f1"],
        "sensitivity_at_90_specificity": sensitivity_at_specificity(y_true, y_score, 0.90),
        "sensitivity_at_0.5": at_half["sensitivity"],
    }


def _ordered_names(frames: dict[str, pd.DataFrame]) -> list[str]:
    names = [name for name in MODEL_ORDER if name in frames]
    names.extend(name for name in frames if name not in names and name != "ensemble")
    return names


def save_then_show(fig: plt.Figure, path: Path) -> None:
    """Write the file before show() so the saved figure is not blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if float(pixels.std()) < 0.5:
        raise RuntimeError(f"Figure looks blank: {path}")
    # Agg (headless batch runs) has nothing to display. Inline
    # notebooks still show the figure because savefig already ran.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
        plt.show()
    plt.close(fig)
    print(f"Wrote {path}")


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.2), facecolor="white")
    ax.set_facecolor("white")
    for index, (name, fpr, tpr, auroc) in enumerate(curves):
        width = 2.6 if name == "ensemble" else 1.8
        label = f"{name} ({auroc:.3f})" if np.isfinite(auroc) else f"{name} (undefined)"
        ax.plot(fpr, tpr, color=COLORS[index % len(COLORS)], lw=width, label=label)
    ax.plot([0, 1], [0, 1], color="0.75", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_then_show(fig, path)


def grad_cam(model: keras.Model, images: tf.Tensor, class_index: int) -> tuple[np.ndarray, str]:
    conv = find_last_conv(model)
    grad_model = keras.Model(model.inputs, [conv.output, model.output])
    images = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(images, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError(f"No Grad-CAM gradient for layer {conv.name}.")
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)
    cam_min = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
    cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = tf.image.resize(cam[..., None], (224, 224))[..., 0]
    return cam.numpy(), conv.name


def _overlay(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = plt.get_cmap("jet")(np.clip(cam, 0, 1))[..., :3]
    base = np.repeat(gray.astype(np.float32)[..., None] / 255.0, 3, axis=-1)
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)


def _load_gray(preprocessed_dir: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    dataset = raw_image_dataset(preprocessed_dir, "test")
    for grays, indexes in dataset:
        for gray, index in zip(grays.numpy(), indexes.numpy()):
            name = index.decode("utf-8") if isinstance(index, bytes) else str(index)
            if name in wanted:
                found[name] = np.squeeze(gray)
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        sample = sorted(missing)[:5]
        raise FileNotFoundError(f"Test TFRecords are missing {len(missing)} Grad-CAM images, for example {sample}")
    return found


def _find_final_weights(backbone: str, out_dir: Path) -> Path:
    direct = out_dir / f"final_{backbone}.keras"
    if direct.is_file():
        return direct
    for match in shallow_find(out_dir, f"final_{backbone}.keras", max_depth=3):
        return match
    for match in shallow_find(out_dir, "stage2_best.keras", max_depth=4):
        if backbone in str(match):
            return match
    for match in shallow_find(out_dir, "stage1_best.keras", max_depth=4):
        if backbone in str(match):
            return match
    raise FileNotFoundError(
        f"No weights for {backbone}. Expected final_{backbone}.keras in OUTPUT_DIR."
    )


def _numbers_markdown(metrics: pd.DataFrame) -> str:
    lines = []
    lines.append("| Model | Mass AUROC | Mass sensitivity | Nodule AUROC | Nodule sensitivity |")
    lines.append("| --- | --- | --- | --- | --- |")
    models = [name for name in MODEL_ORDER if name in set(metrics["model"])]
    for model in models:
        cells = [model]
        for label in FOCUS_LABELS:
            row = metrics[(metrics["model"] == model) & (metrics["label"] == label)]
            if row.empty or not np.isfinite(row.iloc[0]["auroc"]):
                cells.extend(["", ""])
                continue
            auroc = float(row.iloc[0]["auroc"])
            low = row.iloc[0]["auroc_ci_low"]
            high = row.iloc[0]["auroc_ci_high"]
            sens = row.iloc[0]["sensitivity"]
            if np.isfinite(low) and np.isfinite(high):
                cells.append(f"{auroc:.3f} ({low:.3f}, {high:.3f})")
            else:
                cells.append(f"{auroc:.3f}")
            cells.append("" if not np.isfinite(sens) else f"{float(sens):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Sensitivity is at the per-label Youden threshold chosen on the validation split. "
        "The interval is a 95% patient-level bootstrap."
    )
    return "\n".join(lines) + "\n"


def run_evaluation(
    n_bootstrap: int = 1000,
    seed: int = 42,
    out_dir: str | Path | None = None,
    preprocessed_dir: str | Path | None = None,
) -> dict:
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    out_dir = resolve_out_dir(out_dir)
    preprocessed_dir = find_preprocessed_dir(preprocessed_dir)
    load_label_order(preprocessed_dir)
    print(f"Preprocessed data: {preprocessed_dir}")
    print(f"Output: {out_dir}")

    val_preds = load_split_predictions("val", extra_roots=[out_dir])
    test_preds = load_split_predictions("test", extra_roots=[out_dir])
    if set(val_preds) != set(test_preds):
        raise AssertionError(f"Val models {sorted(val_preds)} and test models {sorted(test_preds)} differ.")
    missing = [name for name in MODEL_ORDER if name not in val_preds and name != "ensemble"]
    if missing:
        print("Backbones without prediction files (ensemble skips them): " + ", ".join(missing))

    val_preds["ensemble"] = average_ensemble({name: val_preds[name] for name in val_preds})
    test_preds["ensemble"] = average_ensemble({name: test_preds[name] for name in test_preds if name != "ensemble"})
    align_frames({name: val_preds[name] for name in val_preds})
    align_frames({name: test_preds[name] for name in test_preds})

    for kind, frame in (("val", val_preds["ensemble"]), ("test", test_preds["ensemble"])):
        path = out_dir / f"preds_{kind}_ensemble.csv"
        frame.to_csv(path, index=False, float_format="%.6f")
        print(f"Wrote {path}")

    val_truth = _truth(preprocessed_dir, "val")
    test_truth = _truth(preprocessed_dir, "test")
    names = _ordered_names(val_preds)

    rows = []
    thresholds: dict[str, dict[str, float]] = {}
    for name in names:
        truth_val, pred_val = _align_truth(val_truth, val_preds[name])
        truth_test, pred_test = _align_truth(test_truth, test_preds[name])
        model_thresholds = {}
        for label in LABELS:
            model_thresholds[label] = youden_threshold(
                truth_val[label].to_numpy(), pred_val[label].to_numpy()
            )
        thresholds[name] = model_thresholds
        y_true = truth_test[list(LABELS)].to_numpy()
        y_score = pred_test[list(LABELS)].to_numpy()
        patients = truth_test["Patient ID"].to_numpy()
        for index, label in enumerate(LABELS):
            rows.append(
                _metric_row(
                    name,
                    label,
                    y_true[:, index],
                    y_score[:, index],
                    patients,
                    model_thresholds[label],
                    n_bootstrap,
                    seed,
                )
            )
        macro, low, high, n_defined = patient_bootstrap_macro(
            y_true, y_score, patients, n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "model": name,
                "label": "MACRO",
                "auroc": macro,
                "auroc_ci_low": low,
                "auroc_ci_high": high,
                "auroc_boot_n": n_defined,
                "threshold": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "accuracy": float("nan"),
                "f1": float("nan"),
                "sensitivity_at_90_specificity": float("nan"),
                "sensitivity_at_0.5": float("nan"),
            }
        )

    metrics = pd.DataFrame(rows)
    results = out_dir / "results"
    figures = out_dir / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics_path = results / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    threshold_rows = [
        {"model": model, "label": label, "threshold": value}
        for model, mapping in thresholds.items()
        for label, value in mapping.items()
    ]
    pd.DataFrame(threshold_rows).to_csv(results / "thresholds.csv", index=False)

    focus = metrics[metrics["label"].isin(FOCUS_LABELS)].copy()
    focus_path = results / "focus_mass_nodule.csv"
    focus.to_csv(focus_path, index=False)
    print("Mass and Nodule (test):")
    print(focus.to_string(index=False))

    (results / "numbers.md").write_text(_numbers_markdown(metrics), encoding="utf-8")

    for label in FOCUS_LABELS:
        curves = []
        y_true = test_truth[label].to_numpy()
        for name in names:
            _truth_aligned, pred = _align_truth(test_truth, test_preds[name])
            y_score = pred[label].to_numpy()
            if np.unique(y_true).size < 2:
                fpr = np.array([0.0, 1.0])
                tpr = np.array([0.0, 1.0])
                auroc = float("nan")
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auroc = safe_auroc(y_true, y_score)
            curves.append((name, fpr, tpr, auroc))
        plot_roc(curves, f"{label} on the test split", figures / f"roc_{label.lower()}.png")

    # Best single network by validation AUROC on Mass. The ensemble has no Grad-CAM.
    single = [name for name in names if name != "ensemble"]
    mass_scores = {}
    for name in single:
        truth_val, pred_val = _align_truth(val_truth, val_preds[name])
        mass_scores[name] = safe_auroc(truth_val["Mass"].to_numpy(), pred_val["Mass"].to_numpy())
    best = sorted(single, key=lambda name: (-(mass_scores[name] if np.isfinite(mass_scores[name]) else -1), name))[0]
    best_score = mass_scores[best]
    best_text = f"{best_score:.3f}" if np.isfinite(best_score) else "undefined"
    print(f"Grad-CAM model: {best} (validation Mass AUROC {best_text}).")
    (results / "gradcam_model.json").write_text(
        json.dumps(
            {
                "backbone": best,
                "val_mass_auroc": {
                    name: (float(score) if np.isfinite(score) else None) for name, score in mass_scores.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    truth_test, pred_test = _align_truth(test_truth, test_preds[best])
    cases = select_focus_cases(
        truth_test["Image Index"].to_numpy(),
        truth_test["Mass"].to_numpy(),
        pred_test["Mass"].to_numpy(),
        thresholds[best]["Mass"],
        k=4,
    )
    if len(cases) < 1:
        raise RuntimeError("No test images available for Grad-CAM.")
    weights_path = _find_final_weights(best, out_dir)
    print(f"Loading weights for Grad-CAM: {weights_path}")
    model = keras.models.load_model(weights_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    grays = _load_gray(preprocessed_dir, {case["image_index"] for case in cases})
    preprocess = PREPROCESS[best]
    batch = []
    for case in cases:
        gray = grays[case["image_index"]].astype(np.float32)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        batch.append(rgb)
    batch_tensor = preprocess(tf.constant(np.stack(batch), dtype=tf.float32))
    cams, conv_name = grad_cam(model, batch_tensor, LABELS.index("Mass"))
    print(f"Grad-CAM layer: {conv_name}")

    columns = 4
    n_cases = len(cases)
    n_rows = int(np.ceil(n_cases / columns))
    fig, axes = plt.subplots(n_rows, columns, figsize=(12, 3.1 * n_rows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for axis, case, cam in zip(axes, cases, cams):
        axis.set_facecolor("white")
        axis.imshow(_overlay(grays[case["image_index"]], cam))
        axis.set_title(f"{case['kind']}\n{case['image_index']}", fontsize=8)
        axis.axis("off")
    for axis in axes[n_cases:]:
        axis.axis("off")
        axis.set_facecolor("white")
    fig.suptitle(f"Mass Grad-CAM, {best}", fontsize=12)
    fig.tight_layout()
    save_then_show(fig, figures / "gradcam_mass.png")

    summary = {
        "best_backbone_for_mass": best,
        "val_mass_auroc": mass_scores,
        "n_bootstrap": n_bootstrap,
        "models": names,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"metrics": metrics, "focus": focus, "best": best, "out_dir": str(out_dir)}
