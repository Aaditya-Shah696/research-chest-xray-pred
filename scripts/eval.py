"""Per-label test metrics for the five ChestX-ray14 models and their mean.

Reads the NIH Data_Entry_2017 label file and one test CSV plus one validation
CSV per model. Writes:

- results/metrics_all.csv
- results/mass_nodule_summary.csv
- results/figures/roc_mass_nodule.png

Data_Entry_2017.csv is not stored in this repository (112,120 rows). The
prediction CSVs from the reported runs are under data/<model>/.

Example:
    python scripts/eval.py --labels /path/to/Data_Entry_2017.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]

# Report name, directory name, prediction filename stem.
# Order matches results/metrics_all.csv.
MODELS = (
    ("EfficientNetB0", "efficientnetb0", "EfficientNetB0"),
    ("ResNet50", "resnet50", "ResNet50"),
    ("InceptionV3", "inceptionv3", "InceptionV3"),
    ("DenseNet201", "densenet201", "DenseNet201"),
    ("Xception", "xception", "Xception"),
)

DEFAULT_DIRS = {
    folder: ROOT / "data" / folder
    for folder in ("efficientnetb0", "resnet50", "inceptionv3", "densenet201", "xception")
}

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
FINDINGS = [label for label in LABELS if label != "No Finding"]
ENSEMBLE = "Ensemble (mean of 5)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to NIH Data_Entry_2017.csv (not included in this repository).",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=None,
        help=(
            "Directory whose subfolders are efficientnetb0, resnet50, "
            "inceptionv3, densenet201, and xception. "
            "Default: the prediction folders already in this repository."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="Where to write the tables and the ROC figure (default: results/).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def model_dir(folder: str, pred_root: Path | None) -> Path:
    if pred_root is None:
        return DEFAULT_DIRS[folder]
    return pred_root / folder


def load_split(directory: Path, stem: str, split: str) -> pd.DataFrame:
    path = directory / f"preds_{split}_{stem}.csv"
    frame = pd.read_csv(path).set_index("Image Index")
    if list(frame.columns) != LABELS:
        raise AssertionError(f"{path} columns {list(frame.columns)} != {LABELS}")
    expected = 25596 if split == "test" else 8536
    if len(frame) != expected or not frame.index.is_unique:
        raise AssertionError(f"{path} has {len(frame)} rows; expected {expected} unique images")
    return frame


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if 0 < y_true.sum() < len(y_true):
        return float(roc_auc_score(y_true, scores))
    return float("nan")


def operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float, float, float]:
    predicted = scores >= threshold
    true_pos = int((predicted & (y_true == 1)).sum())
    false_pos = int((predicted & (y_true == 0)).sum())
    false_neg = int((~predicted & (y_true == 1)).sum())
    true_neg = int((~predicted & (y_true == 0)).sum())
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    accuracy = (true_pos + true_neg) / len(y_true)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    ppv = true_pos / max(true_pos + false_pos, 1)
    return sensitivity, specificity, accuracy, f1, ppv


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    if len(labels) != 112120:
        raise AssertionError(f"{args.labels} has {len(labels)} rows; expected 112120")
    labels = labels.set_index("Image Index")
    ground_truth = pd.DataFrame(
        {
            label: labels["Finding Labels"].str.split("|").apply(lambda parts, label=label: label in parts).astype(int)
            for label in LABELS
        },
        index=labels.index,
    )

    predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, folder, stem in MODELS:
        directory = model_dir(folder, args.pred_root)
        predictions[name] = (
            load_split(directory, stem, "test"),
            load_split(directory, stem, "val"),
        )

    reference = predictions["EfficientNetB0"]
    for name in predictions:
        if set(predictions[name][0].index) != set(reference[0].index):
            raise AssertionError(f"{name} test ids do not match EfficientNetB0")
        if set(predictions[name][1].index) != set(reference[1].index):
            raise AssertionError(f"{name} validation ids do not match EfficientNetB0")
    test_index = reference[0].index
    val_index = reference[1].index
    predictions = {
        name: (test.loc[test_index, LABELS], val.loc[val_index, LABELS])
        for name, (test, val) in predictions.items()
    }
    member_names = [name for name, _, _ in MODELS]
    predictions[ENSEMBLE] = (
        sum(predictions[name][0] for name in member_names) / 5,
        sum(predictions[name][1] for name in member_names) / 5,
    )

    y_test = ground_truth.loc[test_index]
    y_val = ground_truth.loc[val_index]
    print("test pos Mass", int(y_test.Mass.sum()), "Nodule", int(y_test.Nodule.sum()))
    test_patients = labels.loc[test_index, "Patient ID"].values
    val_patients = labels.loc[val_index, "Patient ID"].values
    print(
        "patient overlap test/val",
        len(set(test_patients) & set(val_patients)),
        "test pts",
        len(set(test_patients)),
    )

    rng = np.random.default_rng(args.seed)
    unique_patients = np.unique(test_patients)
    patient_rows = {patient: np.where(test_patients == patient)[0] for patient in unique_patients}
    bootstraps = [
        np.concatenate([patient_rows[patient] for patient in rng.choice(unique_patients, len(unique_patients))])
        for _ in range(args.bootstrap)
    ]

    rows = []
    boot_scores: dict[tuple[str, str], np.ndarray] = {}
    for name, (test_scores, val_scores) in predictions.items():
        for label in LABELS:
            y_true = y_test[label].values
            scores = test_scores[label].values
            resampled = np.array([auc(y_true[draw], scores[draw]) for draw in bootstraps])
            boot_scores[(name, label)] = resampled
            rows.append(
                dict(
                    model=name,
                    label=label,
                    test_auroc=auc(y_true, scores),
                    ci_low=float(np.nanpercentile(resampled, 2.5)),
                    ci_high=float(np.nanpercentile(resampled, 97.5)),
                    val_auroc=auc(y_val[label].values, val_scores[label].values),
                    test_pos=int(y_true.sum()),
                )
            )
    metrics = pd.DataFrame(rows)
    summary_rows = []
    for name, group in metrics.groupby("model", sort=False):
        findings = group[group.label != "No Finding"]
        summary_rows.append(
            dict(
                model=name,
                mean14=findings.test_auroc.mean(),
                mean15=group.test_auroc.mean(),
                val14=findings.val_auroc.mean(),
                val15=group.val_auroc.mean(),
            )
        )
    summary = pd.DataFrame(summary_rows).set_index("model")
    print(summary)

    focus_rows = []
    for name, (test_scores, val_scores) in predictions.items():
        for label in ("Mass", "Nodule"):
            y_val_label = y_val[label].values
            val_label_scores = val_scores[label].values
            false_pos_rate, true_pos_rate, thresholds = roc_curve(y_val_label, val_label_scores)
            youden = thresholds[int(np.argmax(true_pos_rate - false_pos_rate))]
            at_most_10 = np.where(false_pos_rate <= 0.10)[0]
            specificity_90 = thresholds[at_most_10[-1]]
            y_true = y_test[label].values
            scores = test_scores[label].values
            sensitivity, specificity, accuracy, f1, ppv = operating_point(y_true, scores, youden)
            sensitivity_90, specificity_90_test, _, _, _ = operating_point(y_true, scores, specificity_90)
            record = metrics[(metrics.model == name) & (metrics.label == label)].iloc[0]
            focus_rows.append(
                dict(
                    model=name,
                    label=label,
                    test_auroc=record.test_auroc,
                    ci_low=record.ci_low,
                    ci_high=record.ci_high,
                    youden_thr_val=youden,
                    sensitivity=sensitivity,
                    specificity=specificity,
                    accuracy=accuracy,
                    f1=f1,
                    ppv=ppv,
                    prevalence=float(y_true.mean()),
                    thr_spec90_val=specificity_90,
                    sens_at_spec90=sensitivity_90,
                    spec_at_spec90thr=specificity_90_test,
                )
            )
    focus = pd.DataFrame(focus_rows)

    best_single = summary.drop(ENSEMBLE).mean14.idxmax()
    print("best", best_single)
    for label in ("Mass", "Nodule"):
        difference = boot_scores[(ENSEMBLE, label)] - boot_scores[(best_single, label)]
        observed = (
            metrics[(metrics.model == ENSEMBLE) & (metrics.label == label)].test_auroc.iloc[0]
            - metrics[(metrics.model == best_single) & (metrics.label == label)].test_auroc.iloc[0]
        )
        print(
            "PAIRED",
            label,
            round(float(observed), 4),
            np.percentile(difference, [2.5, 97.5]).round(4),
            "p(d<=0)",
            float((difference <= 0).mean()),
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    for axis, label in zip(axes, ("Mass", "Nodule")):
        for name, (test_scores, _) in predictions.items():
            false_pos_rate, true_pos_rate, _ = roc_curve(y_test[label], test_scores[label])
            axis.plot(
                false_pos_rate,
                true_pos_rate,
                lw=2.5 if "Ens" in name else 1.2,
                label=f"{name} ({auc(y_test[label].values, test_scores[label].values):.3f})",
            )
        axis.plot([0, 1], [0, 1], "k--", lw=0.7)
        axis.set_title(f"{label} ROC, official test (n=25,596)")
        axis.set_xlabel("1 - specificity")
        axis.set_ylabel("Sensitivity")
        axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()

    out_dir = args.out_dir
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_all.csv"
    focus_path = out_dir / "mass_nodule_summary.csv"
    figure_path = figure_dir / "roc_mass_nodule.png"
    metrics.to_csv(metrics_path, index=False)
    focus.to_csv(focus_path, index=False)
    figure.savefig(figure_path, dpi=200, facecolor="white")
    print(focus.round(4).to_string())
    print("wrote", metrics_path)
    print("wrote", focus_path)
    print("wrote", figure_path)


if __name__ == "__main__":
    main()
