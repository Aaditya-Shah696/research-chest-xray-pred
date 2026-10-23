"""Build the patient-level split and the grayscale TFRecords."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .labels import LABELS, SMALL_DATASET_IMAGES
from .paths import resolve_data_dir, resolve_out_dir
from .records import IMAGE_SIZE, index_images, verify_roundtrip, write_metadata, write_split
from .splits import (
    assert_no_patient_overlap,
    assign_splits,
    load_data_entry,
    positive_count_table,
    read_id_list,
)


def run_preprocess(
    seed: int = 42,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    val_fraction: float = 0.1,
) -> dict:
    data_dir = resolve_data_dir(data_dir)
    out_dir = resolve_out_dir(out_dir)
    print(f"DATA_DIR: {data_dir}")
    print(f"OUTPUT_DIR: {out_dir}")

    entry = load_data_entry(data_dir)
    train_val_path = data_dir / "train_val_list.txt"
    test_path = data_dir / "test_list.txt"
    if not train_val_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Need train_val_list.txt and test_list.txt in {data_dir}. "
            "Those are the official NIH splits."
        )
    train_val_ids = set(read_id_list(train_val_path))
    test_ids = set(read_id_list(test_path))
    listed = entry[entry["Image Index"].isin(train_val_ids | test_ids)]
    print(f"CSV rows: {len(entry)}. Rows on an official list: {len(listed)}.")

    if len(listed) < SMALL_DATASET_IMAGES:
        raise RuntimeError(
            f"Only {len(listed)} images are available. Refusing a full preprocess on a tiny "
            "folder. Set DATA_DIR to the NIH release."
        )

    frame = assign_splits(
        entry,
        train_val_ids,
        test_ids,
        seed=seed,
        val_fraction=val_fraction,
    )
    assert_no_patient_overlap(frame)

    print("Indexing image files...")
    images = index_images(data_dir)
    print(f"Found {len(images)} image files.")
    missing = [image_id for image_id in frame["Image Index"] if image_id not in images]
    if missing:
        sample = missing[:10]
        raise FileNotFoundError(
            f"{len(missing)} listed images are not on disk, for example {sample}. "
            "Check that the NIH image folders are inside DATA_DIR."
        )

    for split in ("train", "val", "test"):
        write_split(frame, images, out_dir, split)

    meta = {
        "labels": list(LABELS),
        "image_size": IMAGE_SIZE,
        "seed": seed,
        "val_fraction": val_fraction,
        "n_images": int(len(frame)),
    }
    write_metadata(out_dir, frame, meta)
    verify_roundtrip(out_dir, pd.read_csv(out_dir / "labels.csv"))

    counts = positive_count_table(frame)
    print("Positive counts (rows are labels, columns are splits):")
    print(counts.T.to_string())
    print("Fixed label order:")
    for index, label in enumerate(LABELS):
        print(f"  {index:2d}  {label}")
    print(f"Wrote {out_dir / 'labels.csv'}")
    print(f"Wrote TFRecords under {out_dir / 'tfrecords'}")
    return {"out_dir": str(out_dir), "counts": counts, "meta": meta, "n_images": int(len(frame))}


def load_label_order(preprocessed_dir: Path) -> dict:
    path = Path(preprocessed_dir) / "label_order.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("labels") != list(LABELS):
        raise RuntimeError(
            "label_order.json does not match the code. The ensemble cannot mix these files.\n"
            f"file: {meta.get('labels')}\ncode: {list(LABELS)}"
        )
    return meta
