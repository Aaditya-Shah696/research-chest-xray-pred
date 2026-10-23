"""Patient-level train, validation, and test assignment.

The test set is NIH's official test_list.txt. train_val_list.txt is then split
by patient with GroupShuffleSplit. Images from the same patient stay together,
and the three patient sets are required to be disjoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .labels import LABELS, encode_finding_labels


def read_id_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_data_entry(data_dir: Path) -> pd.DataFrame:
    matches = sorted(data_dir.glob("Data_Entry*.csv"))
    if not matches:
        raise FileNotFoundError(f"No Data_Entry*.csv in {data_dir}")
    frame = pd.read_csv(matches[0])
    frame.columns = [str(column).strip() for column in frame.columns]
    required = ["Image Index", "Finding Labels", "Patient ID"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{matches[0].name} is missing {missing}. Columns: {list(frame.columns)}")
    frame = frame.copy()
    frame["Image Index"] = frame["Image Index"].astype(str).str.strip()
    frame["Patient ID"] = pd.to_numeric(frame["Patient ID"], errors="raise").astype(np.int64)
    frame["Finding Labels"] = frame["Finding Labels"].astype(str)
    if frame["Image Index"].duplicated().any():
        dupes = frame.loc[frame["Image Index"].duplicated(), "Image Index"].head(5).tolist()
        raise ValueError(f"Duplicate Image Index values, for example {dupes}")
    return frame


def assign_splits(
    frame: pd.DataFrame,
    train_val_ids: set[str],
    test_ids: set[str],
    seed: int = 42,
    val_fraction: float = 0.1,
) -> pd.DataFrame:
    """Attach a split column and the fixed label columns. Does not touch image files."""
    overlap = train_val_ids & test_ids
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"Official lists share {len(overlap)} image ids, for example {sample}")

    known = set(frame["Image Index"])
    in_either = train_val_ids | test_ids
    missing_rows = len(known - in_either)
    if missing_rows:
        print(f"Images in the CSV but in neither official list (excluded): {missing_rows}")

    listed = frame[frame["Image Index"].isin(in_either)].copy()
    missing_files_in_lists = len(in_either - known)
    if missing_files_in_lists:
        print(f"Ids in the official lists but not in the CSV (excluded): {missing_files_in_lists}")

    listed["official_split"] = np.where(listed["Image Index"].isin(test_ids), "test", "train_val")
    test = listed[listed["official_split"] == "test"].copy()
    train_val = listed[listed["official_split"] == "train_val"].copy()
    if test.empty or train_val.empty:
        raise ValueError(
            f"Need both official splits. train_val images={len(train_val)} test images={len(test)}"
        )

    train_val = train_val.reset_index(drop=True)
    n_patients = train_val["Patient ID"].nunique()
    if n_patients < 2:
        raise ValueError("Need at least 2 train_val patients to hold out a validation set.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(train_val, groups=train_val["Patient ID"].to_numpy())
    )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if len(set(train_idx.tolist()) & set(val_idx.tolist())):
        raise AssertionError("GroupShuffleSplit returned overlapping rows.")
    if len(train_idx) + len(val_idx) != len(train_val):
        raise AssertionError("GroupShuffleSplit did not cover every train_val row.")

    split_name = np.empty(len(train_val), dtype=object)
    split_name[train_idx] = "train"
    split_name[val_idx] = "val"
    train_val["split"] = split_name
    test["split"] = "test"

    combined = pd.concat([train_val, test], ignore_index=True)
    combined = _attach_label_columns(combined)
    assert_no_patient_overlap(combined)

    patient_counts = combined.groupby("split", observed=False)["Patient ID"].nunique()
    image_counts = combined.groupby("split", observed=False).size()
    print("Images per split:", {key: int(image_counts.get(key, 0)) for key in ("train", "val", "test")})
    print("Patients per split:", {key: int(patient_counts.get(key, 0)) for key in ("train", "val", "test")})
    val_patients = int(patient_counts.get("val", 0))
    train_val_patients = int(train_val["Patient ID"].nunique())
    print(
        f"Validation patients: {val_patients} of {train_val_patients} "
        f"({val_patients / train_val_patients:.1%} of train_val patients)."
    )
    return combined


def _attach_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame["Finding Labels"].map(encode_finding_labels)
    matrix = np.vstack(encoded.to_numpy()).astype(np.float32)
    out = frame.copy()
    for index, label in enumerate(LABELS):
        out[label] = matrix[:, index]
    disease_cols = [label for label in LABELS if label != "No Finding"]
    cooccur = int(((out["No Finding"] == 1) & (out[disease_cols].sum(axis=1) > 0)).sum())
    if cooccur:
        print(f"Rows where No Finding is combined with another label: {cooccur}")
    return out


def assert_no_patient_overlap(frame: pd.DataFrame) -> None:
    sets = {
        split: set(group["Patient ID"].tolist())
        for split, group in frame.groupby("split", observed=False)
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in sets or right not in sets:
            raise AssertionError(f"Missing split {left!r} or {right!r}. Have {sorted(sets)}")
        overlap = sets[left] & sets[right]
        if overlap:
            sample = sorted(overlap)[:5]
            raise AssertionError(f"Patient overlap between {left} and {right}: {sample}")


def positive_count_table(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["split"] = pd.Categorical(ordered["split"], ["train", "val", "test"], ordered=True)
    table = ordered.groupby("split", observed=False)[list(LABELS)].sum().astype(int)
    return table.reindex(["train", "val", "test"])
