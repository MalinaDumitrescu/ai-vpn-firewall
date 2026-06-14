"""
Shared utilities for clean-pipeline data-leakage validation.

Every helper here is read-only with respect to the pipeline outputs; the
write-side helpers live in `preprocessing_metadata.py` and
`threshold_provenance.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import hashlib
import numpy as np
import pandas as pd


# Metadata columns that must NOT be used as model features.
METADATA_COLS: frozenset = frozenset(
    {"flow_id", "capture_id", "dataset", "label", "source_file", "app", "split"}
)

# Forbidden substrings in model feature column names (case-insensitive).
# `label` is forbidden as a model FEATURE column (it is the target).
FORBIDDEN_FEATURE_SUBSTRINGS: tuple = (
    "capture_id",
    "dataset",
    "source_file",
    "split",
    "file_path",
    "filename",
    "label_source",
    "session_name",
    "pcap",
    "target",
    "label",
)


# ────────────────────────────────────────────────────────────────────────
# Artifact location
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleanArtifactPaths:
    """Resolved absolute paths to clean-pipeline artifacts."""
    repo_root: Path
    features_parquet: Path
    splits_dir: Path
    split_manifest: Path
    train_captures_txt: Path
    val_captures_txt: Path
    test_captures_txt: Path
    evaluation_report: Path                 # may not exist on a fresh checkout
    deployment_recommendation: Path         # may not exist on a fresh checkout
    preprocessing_metadata: Path
    threshold_provenance: Path

    def assert_minimum_present(self) -> None:
        """Loudly fail if the pipeline has not been run yet."""
        missing = []
        for label, p in [
            ("features.parquet", self.features_parquet),
            ("clean_split_manifest.json", self.split_manifest),
            ("clean_train_captures.txt", self.train_captures_txt),
            ("clean_val_captures.txt", self.val_captures_txt),
            ("clean_test_captures.txt", self.test_captures_txt),
        ]:
            if not p.exists():
                missing.append(f"  - {label}: {p}")
        if missing:
            raise FileNotFoundError(
                "Clean-pipeline artifacts missing. Run "
                "`python -m src.clean_pipeline.run_pipeline` first.\n"
                + "\n".join(missing)
            )


def locate_clean_artifacts(repo_root: Path | str | None = None) -> CleanArtifactPaths:
    """Resolve the standard clean-pipeline artifact locations.

    Assumes repo layout from `configs/clean_pipeline.yaml`:
      artifacts/clean_pipeline/features.parquet
      artifacts/clean_pipeline/models/evaluation_report.json
      artifacts/clean_pipeline/eval_v3/clean_deployment_recommendation.json
      data/splits/clean_{train,val,test}_captures.txt
      data/splits/clean_split_manifest.json
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root).resolve()
    art = repo_root / "artifacts" / "clean_pipeline"
    splits = repo_root / "data" / "splits"
    return CleanArtifactPaths(
        repo_root=repo_root,
        features_parquet=art / "features.parquet",
        splits_dir=splits,
        split_manifest=splits / "clean_split_manifest.json",
        train_captures_txt=splits / "clean_train_captures.txt",
        val_captures_txt=splits / "clean_val_captures.txt",
        test_captures_txt=splits / "clean_test_captures.txt",
        evaluation_report=art / "models" / "evaluation_report.json",
        deployment_recommendation=art / "eval_v3" / "clean_deployment_recommendation.json",
        preprocessing_metadata=art / "preprocessing_metadata.json",
        threshold_provenance=art / "eval_v3" / "policy_threshold_provenance.json",
    )


# ────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────

def _read_capture_list(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Capture list not found: {path}")
    captures = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                captures.add(s)
    return captures


def load_split_capture_sets(paths: CleanArtifactPaths) -> Dict[str, Set[str]]:
    """Return {'train': {...}, 'val': {...}, 'test': {...}} from txt files."""
    return {
        "train": _read_capture_list(paths.train_captures_txt),
        "val":   _read_capture_list(paths.val_captures_txt),
        "test":  _read_capture_list(paths.test_captures_txt),
    }


def load_features_dataframe(paths: CleanArtifactPaths) -> pd.DataFrame:
    """Load the clean-pipeline features parquet with the `split` column."""
    if not paths.features_parquet.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {paths.features_parquet}"
        )
    df = pd.read_parquet(paths.features_parquet)
    required = {"flow_id", "capture_id", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"features.parquet is missing required columns: {sorted(missing)}"
        )
    return df


def model_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the model-feature columns (all columns except metadata)."""
    return [c for c in df.columns if c not in METADATA_COLS]


# ────────────────────────────────────────────────────────────────────────
# Row-level hashing
# ────────────────────────────────────────────────────────────────────────

def hash_feature_rows(
    df: pd.DataFrame, feature_cols: Iterable[str], precision: int = 9
) -> pd.Series:
    """
    Compute a stable SHA1 hash per row over the given feature columns.

    Floats are rounded to `precision` digits before hashing so that trivial
    binary-noise differences do not mask true duplicates.
    """
    sub = df[list(feature_cols)].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].round(precision)

    # numpy.tobytes is ~10x faster than a per-row hash; we mix per-column
    # to keep determinism across pandas versions.
    arrs = [sub[c].to_numpy() for c in sub.columns]
    n = len(sub)
    hashes = np.empty(n, dtype=object)
    for i in range(n):
        h = hashlib.sha1()
        for a in arrs:
            v = a[i]
            h.update(repr(v).encode("utf-8"))
            h.update(b"|")
        hashes[i] = h.hexdigest()
    return pd.Series(hashes, index=sub.index, name="row_hash")


# ────────────────────────────────────────────────────────────────────────
# Overlap matrix
# ────────────────────────────────────────────────────────────────────────

def capture_overlap_matrix(
    splits: Dict[str, Set[str]], order: tuple = ("train", "val", "test")
) -> pd.DataFrame:
    """Pairwise capture-overlap count between splits (symmetric, 0 off-diag is good)."""
    mat = pd.DataFrame(0, index=list(order), columns=list(order), dtype=int)
    for a in order:
        for b in order:
            if a == b:
                mat.loc[a, b] = len(splits[a])
            else:
                mat.loc[a, b] = len(splits[a].intersection(splits[b]))
    return mat
