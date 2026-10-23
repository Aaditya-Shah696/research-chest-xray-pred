"""Validation search over ensembles of the five ChestX-ray14 models.

Compares every non-empty subset with a simple mean, rank averaging, weights
proportional to validation AUROC, and a 0.1 weight grid. The reported ensemble
in the README is the simple five-model mean. This script also records which
configuration won on validation.

Reads the NIH Data_Entry_2017 label file (not in this repository) and the
per-model prediction CSVs under data/. Writes:

- results/ensemble_search.csv
- results/ensemble_winners.csv
- results/figures/roc_best_ensemble.png

Model order in the weight vectors is ResNet50, EfficientNetB0, Xception,
InceptionV3, DenseNet201. That order is the one stored in
results/ensemble_search.csv.

Example:
    python scripts/search.py --labels /path/to/Data_Entry_2017.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]

# Name, directory name, prediction filename stem. Order is part of the result file.
MODELS = (
    ("ResNet50", "resnet50", "ResNet50"),
    ("EfficientNetB0", "efficientnetb0", "EfficientNetB0"),
    ("Xception", "xception", "Xception"),
    ("InceptionV3", "inceptionv3", "InceptionV3"),
    ("DenseNet201", "densenet201", "DenseNet201"),
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
NAMES = [name for name, _, _ in MODELS]


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
        help="Where to write the search tables and the ROC figure (default: results/).",
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
        raise AssertionError(f"{path} columns do not match the NIH label order")
    return frame


def labels_matrix(entry: pd.DataFrame, index: pd.Index) -> np.ndarray:
    findings = entry.loc[index, "Finding Labels"]
    return np.stack(
        [findings.str.split("|").apply(lambda parts, label=label: label in parts).values for label in FINDINGS],
        1,
    ).astype(bool)


def rank_scores(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.stack([rankdata(values[model, :, label]) / values.shape[1] for label in range(14)], 1)
            for model in range(5)
        ]
    )


def aucs(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC for each column. scores and y_true are (n, 14)."""
    out = []
    for label in range(scores.shape[1]):
        ranks = rankdata(scores[:, label])
        positive = y_true[:, label]
        n_pos = positive.sum()
        n_neg = len(positive) - n_pos
        out.append((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return np.array(out)


def combine(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, values, 1)


def main() -> None:
    args = parse_args()
    entry = pd.read_csv(args.labels).set_index("Image Index")
    raw = {}
    for name, folder, stem in MODELS:
        directory = model_dir(folder, args.pred_root)
        raw[name] = (load_split(directory, stem, "test"), load_split(directory, stem, "val"))

    test_index = raw["ResNet50"][0].index
    val_index = raw["ResNet50"][1].index
    for name in NAMES:
        if set(raw[name][0].index) != set(test_index) or set(raw[name][1].index) != set(val_index):
            raise AssertionError(f"{name} image ids do not match ResNet50")
        if list(raw[name][0].columns) != LABELS:
            raise AssertionError(f"{name} columns do not match the NIH label order")

    test_scores = np.stack([raw[name][0].loc[test_index, FINDINGS].values for name in NAMES])
    val_scores = np.stack([raw[name][1].loc[val_index, FINDINGS].values for name in NAMES])
    y_test = labels_matrix(entry, test_index)
    y_val = labels_matrix(entry, val_index)
    test_ranks = rank_scores(test_scores)
    val_ranks = rank_scores(val_scores)
    mass_index = FINDINGS.index("Mass")
    nodule_index = FINDINGS.index("Nodule")

    grid = [np.array(choice) / 10 for choice in itertools.product(range(11), repeat=5) if sum(choice) == 10]
    grid_val = [aucs(combine(val_scores, weights), y_val) for weights in grid]
    single_val = np.array([aucs(val_scores[model], y_val).mean() for model in range(5)])

    rows: list[dict] = []

    def add(name: str, subset: tuple[int, ...], method: str, weights: np.ndarray, values: np.ndarray, test_values: np.ndarray) -> None:
        val_auc = aucs(combine(values, weights), y_val)
        test_auc = aucs(combine(test_values, weights), y_test)
        rows.append(
            dict(
                config=name,
                subset="+".join(NAMES[index] for index in subset),
                method=method,
                weights=" ".join(f"{weight:.2f}" for weight in weights),
                val_mean14=val_auc.mean(),
                val_mass=val_auc[mass_index],
                val_nodule=val_auc[nodule_index],
                test_mean14=test_auc.mean(),
                test_mass=test_auc[mass_index],
                test_nodule=test_auc[nodule_index],
            )
        )

    for size in range(1, 6):
        for subset in itertools.combinations(range(5), size):
            joined = "+".join(NAMES[index] for index in subset)
            weights = np.zeros(5)
            weights[list(subset)] = 1 / size
            add(f"mean[{joined}]", subset, "mean", weights, val_scores, test_scores)
            if size == 1:
                continue
            add(f"rank[{joined}]", subset, "rank", weights, val_ranks, test_ranks)
            auc_weights = np.zeros(5)
            auc_weights[list(subset)] = single_val[list(subset)]
            auc_weights /= auc_weights.sum()
            add(f"aucw[{joined}]", subset, "valauc_weighted", auc_weights, val_scores, test_scores)
            eligible = [
                index
                for index, weights in enumerate(grid)
                if all(weights[model] == 0 or model in subset for model in range(5))
                and all(weights[model] > 0 for model in subset)
            ]
            if eligible:
                best = max(eligible, key=lambda index: grid_val[index].mean())
                add(f"grid[{joined}]", subset, "grid_mean14", grid[best], val_scores, test_scores)

    for label, label_index in (("Mass", mass_index), ("Nodule", nodule_index)):
        best = max(range(len(grid)), key=lambda index: grid_val[index][label_index])
        subset = tuple(index for index in range(5) if grid[best][index] > 0)
        add(
            f"grid_{label}[{'+'.join(NAMES[index] for index in subset)}]",
            subset,
            f"grid_{label}_only",
            grid[best],
            val_scores,
            test_scores,
        )

    search = pd.DataFrame(rows)
    print(len(search), "configs")
    print(search.sort_values("val_mean14", ascending=False).head(10).round(4).to_string())
    winner = search.loc[search.val_mean14.idxmax()]
    best_mass = search.loc[search.val_mass.idxmax()]
    best_nodule = search.loc[search.val_nodule.idxmax()]
    print("WIN", winner.config, "| MASS", best_mass.config, "| NOD", best_nodule.config)

    def scores_for(row: pd.Series, split: str) -> np.ndarray:
        weights = np.array([float(value) for value in row.weights.split()])
        if split == "t":
            values = test_ranks if row.method == "rank" else test_scores
        else:
            values = val_ranks if row.method == "rank" else val_scores
        return combine(values, weights)

    reference = {
        "ResNet50": search[search.config == "mean[ResNet50]"].iloc[0],
        "5-model mean": search[search.config == "mean[ResNet50+EfficientNetB0+Xception+InceptionV3+DenseNet201]"].iloc[0],
    }
    evaluated = {
        "Winner (val mean14): " + winner.config: winner,
        "Val-best Mass: " + best_mass.config: best_mass,
        "Val-best Nodule: " + best_nodule.config: best_nodule,
        **reference,
    }

    test_patients = entry.loc[test_index, "Patient ID"].values
    rng = np.random.default_rng(args.seed)
    unique_patients = np.unique(test_patients)
    patient_rows = {patient: np.where(test_patients == patient)[0] for patient in unique_patients}
    bootstraps = [
        np.concatenate([patient_rows[patient] for patient in rng.choice(unique_patients, len(unique_patients))])
        for _ in range(args.bootstrap)
    ]
    test_by_name = {name: scores_for(row, "t") for name, row in evaluated.items()}
    boot_by_name = {
        name: np.array([aucs(scores[draw], y_test[draw]) for draw in bootstraps])
        for name, scores in test_by_name.items()
    }

    winner_rows = []
    resnet_scores = test_by_name["ResNet50"]
    for name, row in evaluated.items():
        scores = test_by_name[name]
        val_pred = scores_for(row, "v")
        test_auc = aucs(scores, y_test)
        resampled = boot_by_name[name]
        difference = resampled - boot_by_name["ResNet50"]
        observed = test_auc - aucs(resnet_scores, y_test)
        false_pos_rate, true_pos_rate, thresholds = roc_curve(y_val[:, mass_index], val_pred[:, mass_index])
        threshold = thresholds[int(np.argmax(true_pos_rate - false_pos_rate))]
        predicted = scores[:, mass_index] >= threshold
        y_mass = y_test[:, mass_index]

        def interval(values: np.ndarray) -> str:
            return f"[{np.percentile(values, 2.5):.3f}, {np.percentile(values, 97.5):.3f}]"

        winner_rows.append(
            dict(
                config=name,
                val_mean14=row.val_mean14,
                test_mean14=test_auc.mean(),
                mean_ci=interval(resampled.mean(1)),
                mass=test_auc[mass_index],
                mass_ci=interval(resampled[:, mass_index]),
                nodule=test_auc[nodule_index],
                nodule_ci=interval(resampled[:, nodule_index]),
                d_mean=observed.mean(),
                d_mean_ci=interval(difference.mean(1)),
                d_mass=observed[mass_index],
                d_mass_ci=interval(difference[:, mass_index]),
                d_nod=observed[nodule_index],
                d_nod_ci=interval(difference[:, nodule_index]),
                mass_sens=(predicted & y_mass).sum() / y_mass.sum(),
                mass_spec=(~predicted & ~y_mass).sum() / (~y_mass).sum(),
            )
        )
    winners = pd.DataFrame(winner_rows)
    print(winners.round(4).to_string())

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    plotted = [list(evaluated)[0], "ResNet50", "5-model mean"]
    for axis, (label, label_index) in zip(axes, (("Mass", mass_index), ("Nodule", nodule_index))):
        for name in plotted:
            false_pos_rate, true_pos_rate, _ = roc_curve(y_test[:, label_index], test_by_name[name][:, label_index])
            score = aucs(test_by_name[name], y_test)[label_index]
            axis.plot(false_pos_rate, true_pos_rate, label=f"{name.split(': ')[-1]} ({score:.3f})")
        axis.plot([0, 1], [0, 1], "k--", lw=0.7)
        axis.set_title(f"{label}, official test")
        axis.set_xlabel("1 - specificity")
        axis.set_ylabel("Sensitivity")
        axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()

    out_dir = args.out_dir
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    search_path = out_dir / "ensemble_search.csv"
    winners_path = out_dir / "ensemble_winners.csv"
    figure_path = figure_dir / "roc_best_ensemble.png"
    search.to_csv(search_path, index=False)
    winners.to_csv(winners_path, index=False)
    figure.savefig(figure_path, dpi=200, facecolor="white")
    print("wrote", search_path)
    print("wrote", winners_path)
    print("wrote", figure_path)


if __name__ == "__main__":
    main()
