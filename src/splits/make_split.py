from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import json

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Config
# ============================================================

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


# ============================================================
# Helpers: targets
# ============================================================

def _cap_targets(n_caps: int, train_r: float, val_r: float) -> Tuple[int, int, int]:
    train = int(round(n_caps * train_r))
    val = int(round(n_caps * val_r))
    test = n_caps - train - val

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


# ============================================================
# Assignment: per-class, capture-limited, flow-guided
# ============================================================

def _assign_giants_with_cap_limits(
    giants: pd.DataFrame,  # must contain capture_id, n_flows (weight)
    cap_need: Dict[str, int],
    flow_targets: Dict[str, int],
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    g = giants.sort_values("n_flows", ascending=False).reset_index(drop=True)

    out = {"train": [], "val": [], "test": []}
    flows = {"train": 0, "val": 0, "test": 0}

    def global_score(chosen: str, w: int) -> int:
        return sum(
            abs((flows[s] + (w if s == chosen else 0)) - flow_targets[s])
            for s in ("train", "val", "test")
        )

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
        cap_need[chosen] -= 1

    return out


def _assign_remaining_greedy(
    remaining: pd.DataFrame,  # must contain capture_id, n_flows (weight)
    cap_need: Dict[str, int],
    flow_targets: Dict[str, int],
    already_flows: Dict[str, int],
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)

    rem = remaining[["capture_id", "n_flows"]].copy()
    rem = rem.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rem = rem.sort_values("n_flows", ascending=False).reset_index(drop=True)

    out = {"train": [], "val": [], "test": []}
    flows = dict(already_flows)

    def global_score(chosen: str, w: int) -> int:
        return sum(
            abs((flows[s] + (w if s == chosen else 0)) - flow_targets[s])
            for s in ("train", "val", "test")
        )

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


# ============================================================
# Rebalance (SWAPS ONLY) to satisfy guardrails
# ============================================================

def _rebalance_with_swaps_only(
    splits: Dict[str, List[str]],
    cap_map: Dict[str, Tuple[int, int]],  # cid -> (label, weight)
    *,
    min_val_vpn_flows: int,
    min_test_vpn_flows: int,
    min_val_total_flows: int,
    min_test_total_flows: int,
    min_val_vpn_caps: int,
    min_test_vpn_caps: int,
    keep_giants_in_train: bool,
    giant_flow_threshold: int,
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)

    def is_vpn(cid: str) -> bool:
        return cap_map[cid][0] == 1

    def is_nonvpn(cid: str) -> bool:
        return cap_map[cid][0] == 0

    def flows(cid: str) -> int:
        return cap_map[cid][1]

    def is_giant(cid: str) -> bool:
        return flows(cid) >= giant_flow_threshold

    def vpn_cap_count(ids: List[str]) -> int:
        return sum(1 for c in ids if is_vpn(c))

    def stats(ids: List[str]) -> Dict[str, int]:
        total = sum(flows(c) for c in ids)
        vpn = sum(flows(c) for c in ids if is_vpn(c))
        return {"total": total, "vpn": vpn}

    def candidates(ids: List[str], pred, *, desc: bool, allow_giants: bool) -> List[str]:
        out = [c for c in ids if pred(c)]
        if not allow_giants:
            out = [c for c in out if not is_giant(c)]
        out.sort(key=lambda c: flows(c), reverse=desc)
        return out

    def do_swap(a_split: str, b_split: str, a_cid: str, b_cid: str) -> None:
        splits[a_split].remove(a_cid)
        splits[b_split].remove(b_cid)
        splits[a_split].append(b_cid)
        splits[b_split].append(a_cid)

    # --- 1) Min VPN CAPTURES in val/test ---
    for target, min_caps in (("val", min_val_vpn_caps), ("test", min_test_vpn_caps)):
        if min_caps <= 0:
            continue

        for _ in range(2000):
            if vpn_cap_count(splits[target]) >= min_caps:
                break

            vpn_train = candidates(splits["train"], is_vpn, desc=True, allow_giants=True)
            if not vpn_train:
                break

            nonvpn_target = candidates(splits[target], is_nonvpn, desc=True, allow_giants=False)
            if not nonvpn_target:
                break

            a = vpn_train[0]
            b = nonvpn_target[0]

            if keep_giants_in_train and is_giant(a):
                vpn_train = [c for c in vpn_train if not is_giant(c)]
                if not vpn_train:
                    break
                a = vpn_train[0]

            do_swap("train", target, a, b)

    # --- 2) Min VPN FLOWS in val/test ---
    for target, min_vpn in (("val", min_val_vpn_flows), ("test", min_test_vpn_flows)):
        if min_vpn <= 0:
            continue

        for _ in range(4000):
            if stats(splits[target])["vpn"] >= min_vpn:
                break

            vpn_train = candidates(splits["train"], is_vpn, desc=True, allow_giants=True)
            nonvpn_target = candidates(splits[target], is_nonvpn, desc=True, allow_giants=False)
            if not vpn_train or not nonvpn_target:
                break

            a = vpn_train[0]
            b = nonvpn_target[0]

            if keep_giants_in_train and is_giant(a):
                vpn_train = [c for c in vpn_train if not is_giant(c)]
                if not vpn_train:
                    break
                a = vpn_train[0]

            do_swap("train", target, a, b)

    # --- 3) Min TOTAL FLOWS in val/test ---
    for target, min_total in (("val", min_val_total_flows), ("test", min_test_total_flows)):
        if min_total <= 0:
            continue

        for _ in range(8000):
            if stats(splits[target])["total"] >= min_total:
                break

            nonvpn_train = candidates(
                splits["train"],
                is_nonvpn,
                desc=True,
                allow_giants=not keep_giants_in_train,
            )
            if not nonvpn_train:
                break

            small_target = candidates(splits[target], lambda _: True, desc=False, allow_giants=False)
            if not small_target:
                break

            a = nonvpn_train[0]
            b = small_target[0]

            if keep_giants_in_train and is_giant(a):
                nonvpn_train = [c for c in nonvpn_train if not is_giant(c)]
                if not nonvpn_train:
                    break
                a = nonvpn_train[0]

            do_swap("train", target, a, b)

    rng.shuffle(splits["train"])
    rng.shuffle(splits["val"])
    rng.shuffle(splits["test"])
    return splits


def _assert_split_sizes_unchanged(before: Dict[str, int], after: Dict[str, List[str]]) -> None:
    for k in ("train", "val", "test"):
        if len(after[k]) != before[k]:
            raise RuntimeError(
                f"BUG: rebalance changed split size for {k}: {before[k]} -> {len(after[k])}. "
                f"Rebalance must use swaps only."
            )


# ============================================================
# Public API
# ============================================================

def make_vnat_capture_split(
    flows_parquet: Path,
    splits_yaml: Path,
    repo_root: Path,
) -> Dict[str, List[str]]:
    """
    IMPORTANT:
    - Split is CAPTURE-level stratified (label 0/1).
    - "flow mass" objective and guardrails use TRAINABLE mass: n_trainable = sum(min_packets_ok).
    """
    cfg = _load_cfg(repo_root, splits_yaml)

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label", "min_packets_ok"])
    if "capture_id" not in df.columns or "label" not in df.columns:
        raise ValueError("flows.parquet must contain capture_id and label columns")
    if "min_packets_ok" not in df.columns:
        raise ValueError("flows.parquet must contain min_packets_ok for trainable-mass splitting")

    df["capture_id"] = df["capture_id"].astype(str)

    cap = (
        df.groupby("capture_id")
        .agg(
            label=("label", "first"),
            n_flows=("label", "size"),
            n_trainable=("min_packets_ok", "sum"),
        )
        .reset_index()
    )
    cap["n_trainable"] = cap["n_trainable"].astype(int)

    # constant label per capture
    check = df.groupby("capture_id")["label"].nunique()
    mixed = int((check > 1).sum())
    if mixed:
        raise ValueError(f"Found {mixed} captures with mixed labels. Fix labeling before splitting.")

    FLOW_MASS_COL = "n_trainable"  # <-- the main fix

    final: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    # stratify by label at CAPTURE level
    for y in (0, 1):
        cap_y = cap[cap["label"] == y].copy().reset_index(drop=True)
        n_caps = len(cap_y)
        if n_caps == 0:
            raise ValueError(f"No captures found for label={y}")

        n_train, n_val, n_test = _cap_targets(n_caps, cfg.cap_train_r, cfg.cap_val_r)

        if (
            n_train < cfg.min_train_per_class
            or n_val < cfg.min_val_per_class
            or n_test < cfg.min_test_per_class
        ):
            raise ValueError(
                f"Not enough captures for label={y} with requested ratios. "
                f"Got train/val/test={n_train}/{n_val}/{n_test}, "
                f"mins={cfg.min_train_per_class}/{cfg.min_val_per_class}/{cfg.min_test_per_class}"
            )

        cap_need = {"train": n_train, "val": n_val, "test": n_test}

        # flow targets computed on TRAINABLE mass
        total_mass_y = int(cap_y[FLOW_MASS_COL].sum())
        flow_t = _flow_targets(total_mass_y, cfg.flow_train_r, cfg.flow_val_r)

        # giants based on TRAINABLE mass
        cap_y_sorted = cap_y.sort_values(FLOW_MASS_COL, ascending=False).reset_index(drop=True)
        k = min(cfg.giants_top_k_per_class, len(cap_y_sorted))
        giants = cap_y_sorted.iloc[:k].copy()
        giant_ids = set(giants["capture_id"].astype(str).tolist())

        # rename weight col to n_flows for existing helpers
        giants_for_assign = giants[["capture_id", FLOW_MASS_COL]].rename(columns={FLOW_MASS_COL: "n_flows"})

        g_assign = _assign_giants_with_cap_limits(
            giants=giants_for_assign,
            cap_need=cap_need,
            flow_targets=flow_t,
            seed=cfg.seed + y * 1000,
        )

        cap_mass_map = cap_y.set_index("capture_id")[FLOW_MASS_COL].to_dict()
        g_flows = {
            "train": int(sum(cap_mass_map[c] for c in g_assign["train"])),
            "val": int(sum(cap_mass_map[c] for c in g_assign["val"])),
            "test": int(sum(cap_mass_map[c] for c in g_assign["test"])),
        }

        rem = cap_y[~cap_y["capture_id"].astype(str).isin(giant_ids)].copy()
        rem_for_assign = rem[["capture_id", FLOW_MASS_COL]].rename(columns={FLOW_MASS_COL: "n_flows"})

        rem_assign = _assign_remaining_greedy(
            remaining=rem_for_assign,
            cap_need=cap_need,
            flow_targets=flow_t,
            already_flows=g_flows,
            seed=cfg.seed + y * 1000 + 77,
        )

        for split in ("train", "val", "test"):
            final[split].extend(g_assign[split])
            final[split].extend(rem_assign[split])

    # ---- Guardrails from YAML (SWAPS ONLY) ----
    cfg_raw = yaml.safe_load(Path(splits_yaml).read_text(encoding="utf-8")) or {}

    vpn_min = cfg_raw.get("vpn_flow_min") or {}
    min_val_vpn_flows = int(vpn_min.get("val", 0))
    min_test_vpn_flows = int(vpn_min.get("test", 0))

    tot_min = cfg_raw.get("min_total_flows") or {}
    val_frac = float(tot_min.get("val_frac", 0.0))
    test_frac = float(tot_min.get("test_frac", 0.0))
    val_abs = int(tot_min.get("val_abs", 0))
    test_abs = int(tot_min.get("test_abs", 0))

    guard = cfg_raw.get("guardrails") or {}
    keep_giants_in_train = bool(guard.get("keep_giants_in_train", True))
    giant_flow_threshold = int(guard.get("giant_flow_threshold", 5000))
    min_vpn_caps = guard.get("min_vpn_captures") or {}
    min_val_vpn_caps = int(min_vpn_caps.get("val", cfg.min_val_per_class))
    min_test_vpn_caps = int(min_vpn_caps.get("test", cfg.min_test_per_class))

    # totals computed on TRAINABLE mass
    total_all = int(cap[FLOW_MASS_COL].sum())
    min_val_total_flows = max(val_abs, int(round(val_frac * total_all)))
    min_test_total_flows = max(test_abs, int(round(test_frac * total_all)))

    # cap_map_all stores TRAINABLE mass as the "flow" weight
    cap_map_all = {
        str(r["capture_id"]): (int(r["label"]), int(r[FLOW_MASS_COL]))
        for _, r in cap.iterrows()
    }

    before_sizes = {k: len(final[k]) for k in ("train", "val", "test")}

    if (
        min_val_vpn_flows > 0
        or min_test_vpn_flows > 0
        or min_val_total_flows > 0
        or min_test_total_flows > 0
        or min_val_vpn_caps > 0
        or min_test_vpn_caps > 0
    ):
        final = _rebalance_with_swaps_only(
            final,
            cap_map_all,
            min_val_vpn_flows=min_val_vpn_flows,
            min_test_vpn_flows=min_test_vpn_flows,
            min_val_total_flows=min_val_total_flows,
            min_test_total_flows=min_test_total_flows,
            min_val_vpn_caps=min_val_vpn_caps,
            min_test_vpn_caps=min_test_vpn_caps,
            keep_giants_in_train=keep_giants_in_train,
            giant_flow_threshold=giant_flow_threshold,
            seed=cfg.seed + 999,
        )

    _assert_split_sizes_unchanged(before_sizes, final)

    # ---- Final policy assertions ----
    if keep_giants_in_train:
        def _is_giant_cap(cid: str) -> bool:
            return cap_map_all[cid][1] >= giant_flow_threshold

        giants_in_val = [c for c in final["val"] if _is_giant_cap(c)]
        giants_in_test = [c for c in final["test"] if _is_giant_cap(c)]

        if giants_in_val or giants_in_test:
            raise RuntimeError(
                "Policy violation: keep_giants_in_train=True but found giant captures in val/test. "
                f"giants_in_val={giants_in_val}, giants_in_test={giants_in_test}"
            )

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
    """
    Writes split lists and a manifest that reports BOTH:
      - raw flow counts
      - trainable flow counts (min_packets_ok==True)
    """
    cfg = _load_cfg(repo_root, splits_yaml)
    cfg.train_list_path.parent.mkdir(parents=True, exist_ok=True)

    cfg.train_list_path.write_text("\n".join(splits["train"]) + "\n", encoding="utf-8")
    cfg.val_list_path.write_text("\n".join(splits["val"]) + "\n", encoding="utf-8")
    cfg.test_list_path.write_text("\n".join(splits["test"]) + "\n", encoding="utf-8")

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label", "min_packets_ok"])
    df["capture_id"] = df["capture_id"].astype(str)

    cap = (
        df.groupby("capture_id")
        .agg(
            label=("label", "first"),
            n_flows=("label", "size"),
            n_trainable=("min_packets_ok", "sum"),
        )
        .reset_index()
    )
    cap["n_trainable"] = cap["n_trainable"].astype(int)

    # cap_map: (label, raw_flows, trainable_flows)
    cap_map = {
        str(r["capture_id"]): (int(r["label"]), int(r["n_flows"]), int(r["n_trainable"]))
        for _, r in cap.iterrows()
    }

    def split_stats(ids: List[str]) -> Dict[str, object]:
        labels = [cap_map[c][0] for c in ids]
        raw = [cap_map[c][1] for c in ids]
        trainable = [cap_map[c][2] for c in ids]

        def by_label(weights: List[int]) -> Dict[str, int]:
            return {
                "0": int(sum(w for l, w in zip(labels, weights) if l == 0)),
                "1": int(sum(w for l, w in zip(labels, weights) if l == 1)),
            }

        return {
            "n_captures": int(len(ids)),
            "raw_flows": int(sum(raw)),
            "trainable_flows": int(sum(trainable)),
            "captures_by_label": {
                "0": int(sum(l == 0 for l in labels)),
                "1": int(sum(l == 1 for l in labels)),
            },
            "raw_flows_by_label": by_label(raw),
            "trainable_flows_by_label": by_label(trainable),
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

    # --- Trainable flow ratio diagnostics (this is what you actually care about) ---
    total_trainable = (
        manifest["split_stats"]["train"]["trainable_flows"]
        + manifest["split_stats"]["val"]["trainable_flows"]
        + manifest["split_stats"]["test"]["trainable_flows"]
    )
    achieved = {k: manifest["split_stats"][k]["trainable_flows"] / max(total_trainable, 1) for k in ("train", "val", "test")}
    target = {"train": cfg.flow_train_r, "val": cfg.flow_val_r, "test": cfg.flow_test_r}
    abs_err = {k: abs(achieved[k] - target[k]) for k in achieved}

    manifest["trainable_flow_ratio_check"] = {
        "target": target,
        "achieved": achieved,
        "abs_error": abs_err,
        "tolerance": cfg.abs_flow_ratio_tol,
        "within_tolerance": {k: abs_err[k] <= cfg.abs_flow_ratio_tol for k in abs_err},
    }

    # keep the old raw ratio too (for transparency)
    total_raw = (
        manifest["split_stats"]["train"]["raw_flows"]
        + manifest["split_stats"]["val"]["raw_flows"]
        + manifest["split_stats"]["test"]["raw_flows"]
    )
    achieved_raw = {k: manifest["split_stats"][k]["raw_flows"] / max(total_raw, 1) for k in ("train", "val", "test")}
    abs_err_raw = {k: abs(achieved_raw[k] - target[k]) for k in achieved_raw}
    manifest["raw_flow_ratio_check"] = {
        "target": target,
        "achieved": achieved_raw,
        "abs_error": abs_err_raw,
        "tolerance": cfg.abs_flow_ratio_tol,
        "within_tolerance": {k: abs_err_raw[k] <= cfg.abs_flow_ratio_tol for k in abs_err_raw},
    }

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    from src.utils.paths import load_paths

    paths = load_paths()
    flows_path = paths.data_processed / "vnat" / "flows.parquet"
    splits_yaml_path = paths.configs_dir / "splits.yaml"

    splits = make_vnat_capture_split(flows_path, splits_yaml_path, repo_root=paths.repo_root)
    manifest = write_split_files(splits, flows_path, splits_yaml_path, repo_root=paths.repo_root)

    print(json.dumps(manifest["split_stats"], indent=2))
