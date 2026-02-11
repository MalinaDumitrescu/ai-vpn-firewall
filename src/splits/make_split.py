from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import json
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    min_train_per_class: int
    min_val_per_class: int
    min_test_per_class: int

    train_list_path: Path
    val_list_path: Path
    test_list_path: Path
    manifest_path: Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_splits_cfg(repo_root: Path, cfg_path: Path) -> SplitConfig:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    seed = int(cfg.get("seed", 42))

    ratios = cfg.get("ratios") or {}
    train_ratio = float(ratios.get("train", 0.70))
    val_ratio = float(ratios.get("val", 0.15))
    test_ratio = float(ratios.get("test", 0.15))

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("ratios.train + ratios.val + ratios.test must sum to 1.0")

    mins = cfg.get("min_captures_per_class") or {}
    min_train = int(mins.get("train", 10))
    min_val = int(mins.get("val", 5))
    min_test = int(mins.get("test", 5))

    outputs = cfg.get("outputs") or {}
    train_list_path = (repo_root / str(outputs.get("train_list", "data/splits/vnat_train_captures.txt"))).resolve()
    val_list_path = (repo_root / str(outputs.get("val_list", "data/splits/vnat_val_captures.txt"))).resolve()
    test_list_path = (repo_root / str(outputs.get("test_list", "data/splits/vnat_test_captures.txt"))).resolve()
    manifest_path = (repo_root / str(outputs.get("manifest", "data/splits/vnat_split_manifest.json"))).resolve()

    return SplitConfig(
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_train_per_class=min_train,
        min_val_per_class=min_val,
        min_test_per_class=min_test,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
        manifest_path=manifest_path,
    )


def _target_counts(n: int, train_r: float, val_r: float) -> Tuple[int, int, int]:
    """
    Convert ratios to integer capture counts that sum exactly to n.
    """
    train = int(round(n * train_r))
    val = int(round(n * val_r))
    test = n - train - val

    # fix rare rounding weirdness
    if test < 0:
        test = 0
        val = n - train
    if train + val + test != n:
        test = n - train - val
    return train, val, test


def _greedy_size_balanced_assign(
    captures: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> Dict[str, List[str]]:
    """
    captures: columns ['capture_id', 'n_flows'] for one class
    Strategy:
      - shuffle within same size bucket using seed
      - sort by n_flows desc
      - assign each capture to split that still needs captures and currently has lowest total flows
    """
    rng = np.random.default_rng(seed)

    caps = captures[["capture_id", "n_flows"]].copy()
    caps = caps.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    caps = caps.sort_values("n_flows", ascending=False).reset_index(drop=True)

    need = {"train": n_train, "val": n_val, "test": n_test}
    total_flows = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}

    for _, row in caps.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        eligible = [s for s in ("train", "val", "test") if need[s] > 0]
        if not eligible:
            eligible = ["train", "val", "test"]

        min_flow = min(total_flows[s] for s in eligible)
        candidates = [s for s in eligible if total_flows[s] == min_flow]

        chosen = candidates[int(rng.integers(0, len(candidates)))]

        out[chosen].append(cid)
        total_flows[chosen] += w
        need[chosen] -= 1

    assert len(out["train"]) == n_train
    assert len(out["val"]) == n_val
    assert len(out["test"]) == n_test
    return out


def make_vnat_capture_split(
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, List[str]]:
    cfg = _load_splits_cfg(repo_root, splits_yaml)

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    if "capture_id" not in df.columns or "label" not in df.columns:
        raise ValueError("flows.parquet must contain capture_id and label columns")

    cap = (
        df.groupby("capture_id")
        .agg(label=("label", "first"), n_flows=("label", "size"))
        .reset_index()
    )

    check = df.groupby("capture_id")["label"].nunique()
    mixed = int((check > 1).sum())
    if mixed:
        raise ValueError(f"Found {mixed} captures with mixed labels. Fix labeling before splitting.")

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for y in [0, 1]:
        cap_y = cap[cap["label"] == y].copy()
        n = len(cap_y)
        if n == 0:
            raise ValueError(f"No captures found for label={y}")

        n_train, n_val, n_test = _target_counts(n, cfg.train_ratio, cfg.val_ratio)

        if n_train < cfg.min_train_per_class or n_val < cfg.min_val_per_class or n_test < cfg.min_test_per_class:
            raise ValueError(
                f"Not enough captures for label={y} with requested ratios. "
                f"Got train/val/test={n_train}/{n_val}/{n_test}, "
                f"mins={cfg.min_train_per_class}/{cfg.min_val_per_class}/{cfg.min_test_per_class}"
            )

        assigned = _greedy_size_balanced_assign(
            captures=cap_y,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=cfg.seed + int(y) * 1000,
        )

        splits["train"].extend(assigned["train"])
        splits["val"].extend(assigned["val"])
        splits["test"].extend(assigned["test"])

    rng = np.random.default_rng(cfg.seed)
    for k in ["train", "val", "test"]:
        rng.shuffle(splits[k])

    return splits


def write_split_files(
    splits: Dict[str, List[str]],
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, object]:
    cfg = _load_splits_cfg(repo_root, splits_yaml)
    cfg.train_list_path.parent.mkdir(parents=True, exist_ok=True)

    train_text = "\n".join(splits["train"]) + "\n"
    val_text = "\n".join(splits["val"]) + "\n"
    test_text = "\n".join(splits["test"]) + "\n"

    cfg.train_list_path.write_text(train_text, encoding="utf-8")
    cfg.val_list_path.write_text(val_text, encoding="utf-8")
    cfg.test_list_path.write_text(test_text, encoding="utf-8")

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    cap = df.groupby("capture_id").agg(label=("label", "first"), n_flows=("label", "size")).reset_index()

    cap_map = {str(r["capture_id"]): (int(r["label"]), int(r["n_flows"])) for _, r in cap.iterrows()}

    def split_stats(capture_ids: List[str]) -> Dict[str, object]:
        labels = [cap_map[c][0] for c in capture_ids]
        flows = [cap_map[c][1] for c in capture_ids]
        return {
            "n_captures": int(len(capture_ids)),
            "n_flows": int(sum(flows)),
            "captures_by_label": {0: int(sum(l == 0 for l in labels)), 1: int(sum(l == 1 for l in labels))},
            "flows_by_label": {
                0: int(sum(f for l, f in zip(labels, flows) if l == 0)),
                1: int(sum(f for l, f in zip(labels, flows) if l == 1)),
            },
        }

    manifest = {
        "dataset": "vnat",
        "flows_parquet": str(flows_parquet.resolve()),
        "flows_parquet_sha256": _sha256_file(flows_parquet),
        "splits_yaml": str(splits_yaml.resolve()),
        "splits_yaml_sha256": _sha256_file(splits_yaml),
        "seed": cfg.seed,
        "ratios": {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio},
        "paths": {
            "train_list": str(cfg.train_list_path),
            "val_list": str(cfg.val_list_path),
            "test_list": str(cfg.test_list_path),
        },
        "files_sha256": {
            "train_list": _sha256_file(cfg.train_list_path),
            "val_list": _sha256_file(cfg.val_list_path),
            "test_list": _sha256_file(cfg.test_list_path),
        },
        "split_stats": {
            "train": split_stats(splits["train"]),
            "val": split_stats(splits["val"]),
            "test": split_stats(splits["test"]),
        },
    }

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    # Minimal runnable entrypoint (nice for quick checks)
    from src.utils.paths import load_paths

    paths = load_paths()
    flows_path = paths.data_processed / "vnat" / "flows.parquet"
    splits_yaml = paths.configs_dir / "splits.yaml"

    splits = make_vnat_capture_split(flows_path, splits_yaml, repo_root=paths.repo_root)
    manifest = write_split_files(splits, flows_path, splits_yaml, repo_root=paths.repo_root)

    print(json.dumps(manifest["split_stats"], indent=2))
