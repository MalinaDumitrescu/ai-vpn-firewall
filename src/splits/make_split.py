from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import json
import numpy as np
import pandas as pd
import yaml


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_r: float
    val_r: float
    test_r: float

    min_train_per_class: int
    min_val_per_class: int
    min_test_per_class: int

    giants_top_k_per_class: int
    giants_percentile: float

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

    fr = cfg.get("flow_ratios") or {}
    train_r = float(fr.get("train", 0.70))
    val_r = float(fr.get("val", 0.15))
    test_r = float(fr.get("test", 0.15))

    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("flow_ratios.train + flow_ratios.val + flow_ratios.test must sum to 1.0")

    mins = cfg.get("min_captures_per_class") or {}
    min_train = int(mins.get("train", 10))
    min_val = int(mins.get("val", 5))
    min_test = int(mins.get("test", 5))

    giants = cfg.get("giants") or {}
    top_k = int(giants.get("top_k_per_class", 2))
    perc = float(giants.get("percentile", 95))

    tol = cfg.get("tolerance") or {}
    abs_flow_ratio_tol = float(tol.get("abs_flow_ratio", 0.03))

    outputs = cfg.get("outputs") or {}
    train_list_path = (repo_root / str(outputs.get("train_list", "data/splits/vnat_train_captures.txt"))).resolve()
    val_list_path = (repo_root / str(outputs.get("val_list", "data/splits/vnat_val_captures.txt"))).resolve()
    test_list_path = (repo_root / str(outputs.get("test_list", "data/splits/vnat_test_captures.txt"))).resolve()
    manifest_path = (repo_root / str(outputs.get("manifest", "data/splits/vnat_split_manifest.json"))).resolve()

    return SplitConfig(
        seed=seed,
        train_r=train_r,
        val_r=val_r,
        test_r=test_r,
        min_train_per_class=min_train,
        min_val_per_class=min_val,
        min_test_per_class=min_test,
        giants_top_k_per_class=top_k,
        giants_percentile=perc,
        abs_flow_ratio_tol=abs_flow_ratio_tol,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
        manifest_path=manifest_path,
    )


# ---------------------------
# Core split logic
# ---------------------------

def _pick_giants_per_class(cap_y: pd.DataFrame, top_k: int, percentile: float) -> pd.DataFrame:
    """
    cap_y: columns capture_id, label, n_flows
    Returns a DataFrame subset of "giant" captures for this class.
    """
    cap_y = cap_y.sort_values("n_flows", ascending=False).reset_index(drop=True)

    if top_k > 0:
        k = min(top_k, len(cap_y))
        return cap_y.iloc[:k].copy()

    # percentile-based fallback
    thr = np.percentile(cap_y["n_flows"].to_numpy(), percentile)
    giants = cap_y[cap_y["n_flows"] >= thr].copy()
    if len(giants) == 0:
        giants = cap_y.iloc[:1].copy()
    return giants


def _targets(total_flows: int, train_r: float, val_r: float, test_r: float) -> Dict[str, int]:
    train = int(round(total_flows * train_r))
    val = int(round(total_flows * val_r))
    test = total_flows - train - val
    if test < 0:
        test = 0
        val = total_flows - train
    return {"train": train, "val": val, "test": test}


def _assign_giants_round_robin(
    giants: pd.DataFrame,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Distribute giants across splits to avoid domination.
    Simple but effective: shuffle giants, then round-robin train->val->test->train...
    """
    rng = np.random.default_rng(seed)
    g = giants.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    order = ["train", "val", "test"]
    out = {"train": [], "val": [], "test": []}
    for i, row in g.iterrows():
        out[order[i % 3]].append(str(row["capture_id"]))
    return out


def _greedy_mass_target_assign(
    remaining: pd.DataFrame,
    targets: Dict[str, int],
    already_flows: Dict[str, int],
    min_caps_needed: Dict[str, int],
    seed: int,
) -> Dict[str, List[str]]:
    """
    Assign remaining captures to minimize deviation from flow targets,
    while ensuring we can still satisfy min capture counts.

    remaining: columns capture_id, n_flows
    targets: desired total flows per split
    already_flows: current flows per split (from giants)
    min_caps_needed: how many captures still required per split (per class)
    """
    rng = np.random.default_rng(seed)

    rem = remaining[["capture_id", "n_flows"]].copy()
    rem = rem.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rem = rem.sort_values("n_flows", ascending=False).reset_index(drop=True)

    out = {"train": [], "val": [], "test": []}
    flows = dict(already_flows)
    caps_left = dict(min_caps_needed)

    for _, row in rem.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        # eligible splits: any, but prefer splits still needing captures (to satisfy min caps)
        preferred = [s for s in ("train", "val", "test") if caps_left[s] > 0]
        eligible = preferred if preferred else ["train", "val", "test"]

        # choose split that minimizes absolute error to target after adding w
        def score(s: str) -> float:
            after = flows[s] + w
            return abs(after - targets[s])

        best = min(score(s) for s in eligible)
        candidates = [s for s in eligible if score(s) == best]
        chosen = candidates[int(rng.integers(0, len(candidates)))]

        out[chosen].append(cid)
        flows[chosen] += w
        if caps_left[chosen] > 0:
            caps_left[chosen] -= 1

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

    # split per class (stratified)
    for y in [0, 1]:
        cap_y = cap[cap["label"] == y].copy().reset_index(drop=True)
        n_caps = len(cap_y)
        if n_caps == 0:
            raise ValueError(f"No captures found for label={y}")

        # Minimum capture counts check
        if n_caps < (cfg.min_train_per_class + cfg.min_val_per_class + cfg.min_test_per_class):
            raise ValueError(
                f"Not enough captures for label={y}: {n_caps} "
                f"(need at least {cfg.min_train_per_class + cfg.min_val_per_class + cfg.min_test_per_class})"
            )

        # Decide giants
        giants = _pick_giants_per_class(
            cap_y, top_k=cfg.giants_top_k_per_class, percentile=cfg.giants_percentile
        )
        giant_ids = set(giants["capture_id"].astype(str).tolist())

        # Assign giants round-robin (prevents domination)
        g_assign = _assign_giants_round_robin(giants, seed=cfg.seed + y * 1000)

        # flows contributed by giants so far
        cap_map = cap_y.set_index("capture_id")["n_flows"].to_dict()
        g_flows = {
            "train": int(sum(cap_map[c] for c in g_assign["train"])),
            "val": int(sum(cap_map[c] for c in g_assign["val"])),
            "test": int(sum(cap_map[c] for c in g_assign["test"])),
        }

        # Remaining captures
        rem = cap_y[~cap_y["capture_id"].astype(str).isin(giant_ids)].copy()

        # Targets by flow mass (within this class)
        total_flows_y = int(cap_y["n_flows"].sum())
        targets = _targets(total_flows_y, cfg.train_r, cfg.val_r, cfg.test_r)

        # ensure we can still satisfy min captures per split for this class
        # giants already gave some captures:
        g_caps = {k: len(v) for k, v in g_assign.items()}
        min_caps_needed = {
            "train": max(cfg.min_train_per_class - g_caps["train"], 0),
            "val": max(cfg.min_val_per_class - g_caps["val"], 0),
            "test": max(cfg.min_test_per_class - g_caps["test"], 0),
        }

        rem_assign = _greedy_mass_target_assign(
            remaining=rem,
            targets=targets,
            already_flows=g_flows,
            min_caps_needed=min_caps_needed,
            seed=cfg.seed + y * 1000 + 77,
        )

        # Merge class split lists
        for split in ("train", "val", "test"):
            final[split].extend(g_assign[split])
            final[split].extend(rem_assign[split])

    # deterministic shuffle of final lists
    rng = np.random.default_rng(cfg.seed)
    for k in ("train", "val", "test"):
        rng.shuffle(final[k])

    return final


# ---------------------------
# Writing + manifest
# ---------------------------

def write_split_files(
    splits: Dict[str, List[str]],
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, object]:
    cfg = _load_cfg(repo_root, splits_yaml)
    cfg.train_list_path.parent.mkdir(parents=True, exist_ok=True)

    # write lists
    cfg.train_list_path.write_text("\n".join(splits["train"]) + "\n", encoding="utf-8")
    cfg.val_list_path.write_text("\n".join(splits["val"]) + "\n", encoding="utf-8")
    cfg.test_list_path.write_text("\n".join(splits["test"]) + "\n", encoding="utf-8")

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    cap = df.groupby("capture_id").agg(label=("label", "first"), n_flows=("label", "size")).reset_index()
    cap_map = {str(r["capture_id"]): (int(r["label"]), int(r["n_flows"])) for _, r in cap.iterrows()}

    def split_stats(capture_ids: List[str]) -> Dict[str, object]:
        labels = [cap_map[c][0] for c in capture_ids]
        flows = [cap_map[c][1] for c in capture_ids]
        return {
            "n_captures": int(len(capture_ids)),
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
        "flow_ratios": {"train": cfg.train_r, "val": cfg.val_r, "test": cfg.test_r},
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
