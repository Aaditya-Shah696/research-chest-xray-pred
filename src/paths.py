"""Resolve training inputs and outputs, and locate committed prediction CSVs."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def repo_root() -> Path:
    """Directory that contains this project (the parent of src/)."""
    root = Path(__file__).resolve().parents[1]
    if (root / "src" / "__init__.py").is_file():
        return root
    raise FileNotFoundError(
        "Could not find the project root. Expected src/__init__.py next to the data and notebooks directories."
    )


def add_src_to_path() -> Path:
    import sys

    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def default_data_dir() -> Path:
    override = _env_path("DATA_DIR")
    if override is not None:
        return override
    return repo_root() / "data" / "nih"


def default_out_dir() -> Path:
    override = _env_path("OUTPUT_DIR")
    if override is not None:
        return override
    return repo_root() / "data" / "output"


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    path = Path(explicit) if explicit else default_data_dir()
    if (path / "Data_Entry_2017.csv").is_file() or list(path.glob("Data_Entry*.csv")):
        return path
    raise FileNotFoundError(
        "NIH files not found. Set DATA_DIR to the directory that contains Data_Entry_2017.csv."
    )


def resolve_out_dir(explicit: str | Path | None = None) -> Path:
    path = Path(explicit) if explicit else default_out_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_preprocessed_dir(explicit: str | Path | None = None) -> Path:
    """Directory that contains label_order.json and labels.csv."""
    if explicit:
        path = Path(explicit)
        if (path / "label_order.json").is_file():
            return path
        raise FileNotFoundError(f"No label_order.json in {path}")

    root = default_out_dir()
    found: list[Path] = []
    if root.exists():
        for match in shallow_find(root, "label_order.json", max_depth=3):
            found.append(match.parent)
    if not found:
        raise FileNotFoundError(
            "Preprocessed dataset not found (label_order.json). "
            "Run 00_preprocess.ipynb. Set OUTPUT_DIR if that output is not in data/output."
        )
    # Prefer the copy with the most shards.
    found.sort(key=lambda path: (len(list((path / "tfrecords").rglob("*.tfrecord"))), str(path)))
    return found[-1]


def shallow_find(root: Path, pattern: str, max_depth: int = 3) -> list[Path]:
    """Find files without walking the NIH image tree."""
    matches: list[Path] = []
    if not root.is_dir():
        return matches
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        parts = current.parts
        if any(part == "images" or part.startswith("images_") for part in parts):
            dirnames[:] = []
            continue
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = max_depth + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                matches.append(current / filename)
    return matches


def find_prediction_csvs(kind: str, extra_roots: list[Path] | None = None) -> list[Path]:
    """kind is 'val' or 'test'."""
    pattern = f"preds_{kind}_*.csv"
    roots = [default_out_dir()]
    try:
        roots.append(repo_root() / "data")
    except FileNotFoundError:
        roots.append(Path("data"))
    if extra_roots:
        roots.extend(extra_roots)
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for match in shallow_find(root, pattern, max_depth=3):
            # Skip the ensemble file here; it is an output, not an input model.
            if match.name.startswith(f"preds_{kind}_ensemble"):
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(match)
    return sorted(found)
