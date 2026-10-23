"""Fixed label vocabulary for NIH ChestX-ray14.

The order below is sorted once and written into every file this project
produces, so column i is the same finding in every model.
"""

from __future__ import annotations

FINDINGS = (
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
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)

# 14 findings plus "No Finding", sorted so the order cannot drift.
LABELS: tuple[str, ...] = tuple(sorted((*FINDINGS, "No Finding")))
LABEL_SET = set(LABELS)
FOCUS_LABELS = ("Mass", "Nodule")
N_LABELS = len(LABELS)

# A full NIH run is far larger than this. Below it, training is refused so a
# tiny folder is not mistaken for a ChestX-ray14 result.
SMALL_DATASET_IMAGES = 500


def encode_finding_labels(text: str) -> list[float]:
    """Map one NIH 'Finding Labels' cell to the fixed column order."""
    parts = [part.strip() for part in str(text).split("|") if part.strip()]
    unknown = [part for part in parts if part not in LABEL_SET]
    if unknown:
        known = ", ".join(LABELS)
        raise ValueError(f"Unknown finding label(s) {unknown}. Expected one of: {known}")
    present = set(parts)
    return [1.0 if label in present else 0.0 for label in LABELS]
