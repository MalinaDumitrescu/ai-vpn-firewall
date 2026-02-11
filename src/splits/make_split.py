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

    cap_train_r: float
    cap_val_r: float
    cap_test_r: float

    flow_train_r: float
    flow_val_r: float
    flow_test_r: float

    min_train_per_class: int
    min_val_per_class: int
    min_test_per_class: int

    giants_top_k_per_class: int

    abs_flow_ratio_tol: float

    train_list_path: Path
    val_list_path: Path
    test_list_path: Path
    manifest_path: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cfg(repo_root: Path, cfg_path: Path) -> SplitConfig:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    seed = int(cfg.get("seed", 42))

    cr = cfg.get("capture_ratios") or {}
    cap_train_r = float(cr.get("train", 0.70))
    cap_val_r = float(cr.get("val", 0.15))
    cap_test_r = float(cr.get("test", 0.15))
    if not np.isclose(cap_train_r + cap_val_r + cap_test_r, 1.0):
        raise ValueError("capture_ratios must sum to 1.0")

    fr = cfg.get("flow_ratios") or {}
    flow_train_r = float(fr.get("train", 0.70))
    flow_val_r = float(fr.get("val", 0.15))
    flow_test_r = float(fr.get("test", 0.15))
    if not np.isclose(flow_train_r + flow_val_r + flow_test_r, 1.0):
        raise ValueError("flow_ratios must sum to 1.0")

    mins = cfg.get("min_captures_per_class") or {}
    min_train = int(mins.get("train", 10))
    min_val = int(mins.get("val", 5))
    min_test = int(mins.get("test", 5))

    giants = cfg.get("giants") or {}
    top_k = int(giants.get("top_k_per_class", 2))

    tol = cfg.get("tolerance") or {}
    abs_flow_ratio_tol = float(tol.get("abs_flow_ratio", 0.05))

    outputs = cfg.get("outputs") or {}
    train_list_path = (repo_root / str(outputs.get("train_list", "data/splits/vnat_train_captures.txt"))).resolve()
    val_list_path = (repo_root / str(outputs.get("val_list", "data/splits/vnat_val_captures.txt"))).resolve()
    test_list_path = (repo_root / str(outputs.get("test_list", "data/splits/vnat_test_captures.txt"))).resolve()
    manifest_path = (repo_root / str(outputs.get("manifest", "data/splits/vnat_split_manifest.json"))).resolve()

    return SplitConfig(
        seed=seed,
        cap_train_r=cap_train_r,
        cap_val_r=cap_val_r,
        cap_test_r=cap_test_r,
        flow_train_r=flow_train_r,
        flow_val_r=flow_val_r,
        flow_test_r=flow_test_r,
        min_train_per_class=min_train,
        min_val_per_class=min_val,
        min_test_per_class=min_test,
        giants_top_k_per_class=top_k,
        abs_flow_ratio_tol=abs_flow_ratio_tol,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
        manifest_path=manifest_path,
    )


def _cap_targets(n_caps: int, train_r: float, val_r: float) -> Tuple[int, int, int]:
    train = int(round(n_caps * train_r))
    val = int(round(n_caps * val_r))
    test = n_caps - train - val
    # safety for rounding
    if test < 0:
        test = 0
        val = n_caps - train
    if train + val + test != n_caps:
        test = n_caps - train - val
    return train, val, test


def _flow_targets(total_flows: int, train_r: float, val_r: float) -> Dict[str, int]:
    train = int(round(total_flows * train_r))
    val = int(round(total_flows * val_r))
    test = total_flows - train - val
    if test < 0:
        test = 0
        val = total_flows - train
    return {"train": train, "val": val, "test": test}


def _assign_giants_with_cap_limits(
    giants: pd.DataFrame,
    cap_need: Dict[str, int],
    flow_targets: Dict[str, int],
    seed: int,
) -> Dict[str, List[str]]:
    """
    Place giants first, respecting capture needs.
    IMPORTANT: we choose the split that minimizes TOTAL error across all splits,
    not only the chosen split's local error.
    """
    rng = np.random.default_rng(seed)
    g = giants.sort_values("n_flows", ascending=False).reset_index(drop=True)

    out = {"train": [], "val": [], "test": []}
    flows = {"train": 0, "val": 0, "test": 0}

    def global_score(chosen: str, w: int) -> int:
        total = 0
        for s in ("train", "val", "test"):
            after = flows[s] + (w if s == chosen else 0)
            total += abs(after - flow_targets[s])
        return total

    for _, row in g.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        eligible = [s for s in ("train", "val", "test") if cap_need[s] > 0]
        if not eligible:
            eligible = ["train", "val", "test"]

        best = min(global_score(s, w) for s in eligible)
        cand = [s for s in eligible if global_score(s, w) == best]
        chosen = cand[int(rng.integers(0, len(cand)))]

        out[chosen].append(cid)
        flows[chosen] += w
        if cap_need[chosen] > 0:
            cap_need[chosen] -= 1

    return out

def _assign_remaining_greedy(
    remaining: pd.DataFrame,
    cap_need: Dict[str, int],
    flow_targets: Dict[str, int],
    already_flows: Dict[str, int],
    seed: int,
) -> Dict[str, List[str]]:
    """
    Greedy assignment with HARD capture limits.
    IMPORTANT: choose the split that minimizes TOTAL error across all splits.
    """
    rng = np.random.default_rng(seed)

    rem = remaining[["capture_id", "n_flows"]].copy()
    rem = rem.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rem = rem.sort_values("n_flows", ascending=False).reset_index(drop=True)

    out = {"train": [], "val": [], "test": []}
    flows = dict(already_flows)

    def global_score(chosen: str, w: int) -> int:
        total = 0
        for s in ("train", "val", "test"):
            after = flows[s] + (w if s == chosen else 0)
            total += abs(after - flow_targets[s])
        return total

    for _, row in rem.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        eligible = [s for s in ("train", "val", "test") if cap_need[s] > 0]
        if not eligible:
            eligible = ["train", "val", "test"]

        best = min(global_score(s, w) for s in eligible)
        cand = [s for s in eligible if global_score(s, w) == best]
        chosen = cand[int(rng.integers(0, len(cand)))]

        out[chosen].append(cid)
        flows[chosen] += w
        cap_need[chosen] -= 1

    return out

def make_vnat_capture_split(
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, List[str]]:
    cfg = _load_cfg(repo_root, splits_yaml)

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    if "capture_id" not in df.columns or "label" not in df.columns:
        raise ValueError("flows.parquet must contain capture_id and label columns")

    # capture table
    cap = (
        df.groupby("capture_id")
        .agg(label=("label", "first"), n_flows=("label", "size"))
        .reset_index()
    )

    # ensure constant label per capture
    check = df.groupby("capture_id")["label"].nunique()
    mixed = int((check > 1).sum())
    if mixed:
        raise ValueError(f"Found {mixed} captures with mixed labels. Fix labeling before splitting.")

    final = {"train": [], "val": [], "test": []}

    for y in [0, 1]:
        cap_y = cap[cap["label"] == y].copy().reset_index(drop=True)
        n_caps = len(cap_y)
        if n_caps == 0:
            raise ValueError(f"No captures found for label={y}")

        # capture count targets for this class
        n_train, n_val, n_test = _cap_targets(n_caps, cfg.cap_train_r, cfg.cap_val_r)

        # enforce minimum captures per class per split
        if n_train < cfg.min_train_per_class or n_val < cfg.min_val_per_class or n_test < cfg.min_test_per_class:
            raise ValueError(
                f"Not enough captures for label={y} with requested ratios. "
                f"Got train/val/test={n_train}/{n_val}/{n_test}, "
                f"mins={cfg.min_train_per_class}/{cfg.min_val_per_class}/{cfg.min_test_per_class}"
            )

        cap_need = {"train": n_train, "val": n_val, "test": n_test}

        # flow targets (within this class)
        total_flows_y = int(cap_y["n_flows"].sum())
        flow_t = _flow_targets(total_flows_y, cfg.flow_train_r, cfg.flow_val_r)

        # pick giants
        cap_y_sorted = cap_y.sort_values("n_flows", ascending=False).reset_index(drop=True)
        k = min(cfg.giants_top_k_per_class, len(cap_y_sorted))
        giants = cap_y_sorted.iloc[:k].copy()
        giant_ids = set(giants["capture_id"].astype(str).tolist())

        # assign giants first with cap limits + flow objective
        g_assign = _assign_giants_with_cap_limits(
            giants=giants,
            cap_need=cap_need,
            flow_targets=flow_t,
            seed=cfg.seed + y * 1000,
        )

        # current flows from giants
        cap_map = cap_y.set_index("capture_id")["n_flows"].to_dict()
        g_flows = {
            "train": int(sum(cap_map[c] for c in g_assign["train"])),
            "val": int(sum(cap_map[c] for c in g_assign["val"])),
            "test": int(sum(cap_map[c] for c in g_assign["test"])),
        }

        # remaining captures
        rem = cap_y[~cap_y["capture_id"].astype(str).isin(giant_ids)].copy()

        # assign remaining with hard cap limits
        rem_assign = _assign_remaining_greedy(
            remaining=rem,
            cap_need=cap_need,
            flow_targets=flow_t,
            already_flows=g_flows,
            seed=cfg.seed + y * 1000 + 77,
        )

        # merge for this class
        for split in ("train", "val", "test"):
            final[split].extend(g_assign[split])
            final[split].extend(rem_assign[split])

    # deterministic shuffle
    rng = np.random.default_rng(cfg.seed)
    for split in ("train", "val", "test"):
        rng.shuffle(final[split])

    return final


def write_split_files(
    splits: Dict[str, List[str]],
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, object]:
    cfg = _load_cfg(repo_root, splits_yaml)
    cfg.train_list_path.parent.mkdir(parents=True, exist_ok=True)

    cfg.train_list_path.write_text("\n".join(splits["train"]) + "\n", encoding="utf-8")
    cfg.val_list_path.write_text("\n".join(splits["val"]) + "\n", encoding="utf-8")
    cfg.test_list_path.write_text("\n".join(splits["test"]) + "\n", encoding="utf-8")

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    cap = df.groupby("capture_id").agg(label=("label", "first"), n_flows=("label", "size")).reset_index()
    cap_map = {str(r["capture_id"]): (int(r["label"]), int(r["n_flows"])) for _, r in cap.iterrows()}

    def split_stats(ids: List[str]) -> Dict[str, object]:
        labels = [cap_map[c][0] for c in ids]
        flows = [cap_map[c][1] for c in ids]
        return {
            "n_captures": int(len(ids)),
            "n_flows": int(sum(flows)),
            "captures_by_label": {"0": int(sum(l == 0 for l in labels)), "1": int(sum(l == 1 for l in labels))},
            "flows_by_label": {
                "0": int(sum(f for l, f in zip(labels, flows) if l == 0)),
                "1": int(sum(f for l, f in zip(labels, flows) if l == 1)),
            },
        }

    manifest = {
        "dataset": "vnat",
        "flows_parquet": str(flows_parquet.resolve()),
        "flows_parquet_sha256": _sha256_file(flows_parquet),
        "splits_yaml": str(splits_yaml.resolve()),
        "splits_yaml_sha256": _sha256_file(splits_yaml),
        "seed": cfg.seed,
        "capture_ratios": {"train": cfg.cap_train_r, "val": cfg.cap_val_r, "test": cfg.cap_test_r},
        "flow_ratios": {"train": cfg.flow_train_r, "val": cfg.flow_val_r, "test": cfg.flow_test_r},
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
    from src.utils.paths import load_paths

    paths = load_paths()
    flows_path = paths.data_processed / "vnat" / "flows.parquet"
    splits_yaml = paths.configs_dir / "splits.yaml"

    splits = make_vnat_capture_split(flows_path, splits_yaml, repo_root=paths.repo_root)
    manifest = write_split_files(splits, flows_path, splits_yaml, repo_root=paths.repo_root)

    print(json.dumps(manifest["split_stats"], indent=2))
