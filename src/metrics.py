"""Thresholds, operating-point stats, and patient-level bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold on the validation set that maximizes sensitivity + specificity - 1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return 0.5
    _fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1.0 - _fpr
    j = tpr + specificity - 1.0
    best = float(thresholds[int(np.argmax(j))])
    if not np.isfinite(best):
        best = 1.0
    return float(np.clip(best, 0.0, 1.0))


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """Highest sensitivity among ROC points whose specificity is at least the target."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr
    eligible = specificity >= target_specificity
    if not np.any(eligible):
        return float("nan")
    return float(np.max(tpr[eligible]))


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    f1_den = 2 * tp + fp + fn
    f1 = (2 * tp / f1_den) if f1_den else 0.0
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def patient_bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    """Point AUROC on the full set, plus a 95% patient-level percentile interval.

    Each draw resamples patients with replacement and keeps every image from
    those patients. Draws with only one class are skipped.
    Returns point, ci_low, ci_high, n_defined_draws.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    patient_ids = np.asarray(patient_ids)
    point = safe_auroc(y_true, y_score)
    uniq, inverse = np.unique(patient_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n_patients = len(uniq)
    for _ in range(n_bootstrap):
        draw = rng.integers(0, n_patients, size=n_patients)
        idx = np.concatenate([groups[i] for i in draw])
        score = safe_auroc(y_true[idx], y_score[idx])
        if np.isfinite(score):
            scores.append(score)
    if len(scores) < 2:
        return point, float("nan"), float("nan"), len(scores)
    low, high = np.percentile(np.asarray(scores), [2.5, 97.5])
    return point, float(low), float(high), len(scores)


def choose_thresholds(val_true: pd.DataFrame, val_pred: pd.DataFrame) -> dict[str, float]:
    merged = val_true.merge(val_pred, on="Image Index", suffixes=("_true", "_pred"))
    thresholds: dict[str, float] = {}
    for label in LABELS:
        y_col = f"{label}_true" if f"{label}_true" in merged.columns else label
        # After a merge of two frames that both have label columns, suffixes apply.
        if f"{label}_true" not in merged.columns:
            raise KeyError(f"Missing ground truth column for {label}")
        thresholds[label] = youden_threshold(
            merged[f"{label}_true"].to_numpy(),
            merged[f"{label}_pred"].to_numpy(),
        )
    return thresholds


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = [safe_auroc(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    finite = [score for score in scores if np.isfinite(score)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def patient_bootstrap_macro(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    """Same patient resample as the per-label interval, applied to the macro AUROC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    point = macro_auroc(y_true, y_score)
    uniq, inverse = np.unique(patient_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n_patients = len(uniq)
    for _ in range(n_bootstrap):
        draw = rng.integers(0, n_patients, size=n_patients)
        idx = np.concatenate([groups[i] for i in draw])
        score = macro_auroc(y_true[idx], y_score[idx])
        if np.isfinite(score):
            scores.append(score)
    if len(scores) < 2:
        return point, float("nan"), float("nan"), len(scores)
    low, high = np.percentile(np.asarray(scores), [2.5, 97.5])
    return point, float(low), float(high), len(scores)


def select_focus_cases(
    image_index: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    k: int = 4,
) -> list[dict]:
    """Pick k true positives and k false positives for a Grad-CAM figure.

    When a split does not contain k of either kind, the remaining panels are the
    highest-scoring positives or negatives and are labeled as such.
    """
    image_index = np.asarray(image_index)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    predicted = y_score >= threshold

    def ranked(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        return idx[np.argsort(-y_score[idx])]

    chosen: list[dict] = []
    used: set[int] = set()

    def take(indices: np.ndarray, kind: str, need: int) -> int:
        got = 0
        for idx in indices:
            idx = int(idx)
            if idx in used:
                continue
            chosen.append(
                {
                    "image_index": str(image_index[idx]),
                    "kind": kind,
                    "score": float(y_score[idx]),
                    "y": int(y_true[idx]),
                }
            )
            used.add(idx)
            got += 1
            if got >= need:
                break
        return got

    n_tp = take(ranked((y_true == 1) & predicted), "true positive", k)
    if n_tp < k:
        take(ranked(y_true == 1), "highest positive", k - n_tp)
    n_fp = take(ranked((y_true == 0) & predicted), "false positive", k)
    if n_fp < k:
        take(ranked(y_true == 0), "highest negative", k - n_fp)
    return chosen
