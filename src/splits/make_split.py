from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

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


def _as_int(v, *, path: str) -> int:
    """
    Strict int parsing so config typos like '50j' fail loudly with a helpful message.
    """
    try:
        if isinstance(v, bool):
            raise ValueError("bool is not int")
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid integer at '{path}': {v!r}") from e


def _as_float(v, *, path: str) -> float:
    try:
        if isinstance(v, bool):
            raise ValueError("bool is not float")
        return float(v)
    except Exception as e:
        raise ValueError(f"Invalid float at '{path}': {v!r}") from e


def _load_cfg(repo_root: Path, cfg_path: Path) -> SplitConfig:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    seed = _as_int(cfg.get("seed", 42), path="seed")

    cr = cfg.get("capture_ratios") or {}
    cap_train_r = _as_float(cr.get("train", 0.70), path="capture_ratios.train")
    cap_val_r = _as_float(cr.get("val", 0.15), path="capture_ratios.val")
    cap_test_r = _as_float(cr.get("test", 0.15), path="capture_ratios.test")
    if not np.isclose(cap_train_r + cap_val_r + cap_test_r, 1.0):
        raise ValueError("capture_ratios must sum to 1.0")

    fr = cfg.get("flow_ratios") or {}
    flow_train_r = _as_float(fr.get("train", 0.70), path="flow_ratios.train")
    flow_val_r = _as_float(fr.get("val", 0.15), path="flow_ratios.val")
    flow_test_r = _as_float(fr.get("test", 0.15), path="flow_ratios.test")
    if not np.isclose(flow_train_r + flow_val_r + flow_test_r, 1.0):
        raise ValueError("flow_ratios must sum to 1.0")

    mins = cfg.get("min_captures_per_class") or {}
    min_train = _as_int(mins.get("train", 10), path="min_captures_per_class.train")
    min_val = _as_int(mins.get("val", 5), path="min_captures_per_class.val")
    min_test = _as_int(mins.get("test", 5), path="min_captures_per_class.test")

    giants = cfg.get("giants") or {}
    top_k = _as_int(giants.get("top_k_per_class", 2), path="giants.top_k_per_class")

    tol = cfg.get("tolerance") or {}
    abs_flow_ratio_tol = _as_float(tol.get("abs_flow_ratio", 0.05), path="tolerance.abs_flow_ratio")

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


def _flow_targets(total_mass: int, train_r: float, val_r: float) -> Dict[str, int]:
    train = int(round(total_mass * train_r))
    val = int(round(total_mass * val_r))
    test = total_mass - train - val

    if test < 0:
        test = 0
        val = total_mass - train

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
#
# IMPORTANT:
#   - "total flows" guardrails can use RAW flows (optional legacy)
#   - vpn_trainable_flow_min uses TRAINABLE flows (sum per-split over VPN captures)
#   - GIANTS: defined by TRAINABLE flows per capture (matches your YAML comment)
#   - vpn-heavy: TRAINABLE flows per capture
#   - vpn_trainable_flow_min_train: min VPN trainable flows in TRAIN
#   - max_vpn_fraction_trainable: cap VPN fraction in val/test by trainable mass
#
# Fixes in this version:
#   1) iterative rebalance until stable (prevents later steps undoing earlier guardrails)
#   2) when boosting total trainable/raw in val/test, prefer swapping OUT nonVPN first
# ============================================================

def _rebalance_with_swaps_only(
    splits: Dict[str, List[str]],
    cap_map: Dict[str, Dict[str, int]],  # cid -> {"label":0/1, "raw":int, "trainable":int}
    *,
    min_train_vpn_trainable: int,
    min_val_vpn_trainable: int,
    min_test_vpn_trainable: int,
    min_val_total_raw: int,
    min_test_total_raw: int,
    min_val_total_trainable: int,
    min_test_total_trainable: int,
    min_val_vpn_caps: int,
    min_test_vpn_caps: int,
    min_val_vpn_heavy_caps: int,
    min_test_vpn_heavy_caps: int,
    vpn_heavy_trainable_threshold: int,
    max_val_vpn_fraction_trainable: float,
    max_test_vpn_fraction_trainable: float,
    keep_giants_in_train: bool,
    giant_trainable_threshold: int,
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)

    def label(cid: str) -> int:
        return int(cap_map[cid]["label"])

    def is_vpn(cid: str) -> bool:
        return label(cid) == 1

    def is_nonvpn(cid: str) -> bool:
        return label(cid) == 0

    def raw(cid: str) -> int:
        return int(cap_map[cid]["raw"])

    def trainable(cid: str) -> int:
        return int(cap_map[cid]["trainable"])

    # GIANT defined by TRAINABLE flows per capture
    def is_giant(cid: str) -> bool:
        return trainable(cid) >= giant_trainable_threshold

    def is_vpn_heavy(cid: str) -> bool:
        return is_vpn(cid) and trainable(cid) >= vpn_heavy_trainable_threshold

    def vpn_cap_count(ids: List[str]) -> int:
        return sum(1 for c in ids if is_vpn(c))

    def vpn_heavy_count(ids: List[str]) -> int:
        return sum(1 for c in ids if is_vpn_heavy(c))

    def stats(ids: List[str]) -> Dict[str, int]:
        total_raw = sum(raw(c) for c in ids)
        total_trainable = sum(trainable(c) for c in ids)
        vpn_trainable = sum(trainable(c) for c in ids if is_vpn(c))
        return {"total_raw": total_raw, "total_trainable": total_trainable, "vpn_trainable": vpn_trainable}

    def vpn_fraction_trainable(ids: List[str]) -> float:
        st = stats(ids)
        denom = max(int(st["total_trainable"]), 1)
        return float(st["vpn_trainable"]) / float(denom)

    def candidates(ids: List[str], pred, *, desc: bool, allow_giants: bool, key_fn) -> List[str]:
        out = [c for c in ids if pred(c)]
        if not allow_giants:
            out = [c for c in out if not is_giant(c)]
        out.sort(key=key_fn, reverse=desc)
        return out

    def do_swap(a_split: str, b_split: str, a_cid: str, b_cid: str) -> None:
        splits[a_split].remove(a_cid)
        splits[b_split].remove(b_cid)
        splits[a_split].append(b_cid)
        splits[b_split].append(a_cid)

    def signature() -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        # order-insensitive stable signature
        return (
            tuple(sorted(splits["train"])),
            tuple(sorted(splits["val"])),
            tuple(sorted(splits["test"])),
        )

    def one_pass() -> None:
        # --- 0) HARD POLICY: evict GIANTS from val/test back into train (SWAPS ONLY) ---
        if keep_giants_in_train and giant_trainable_threshold > 0:

            def evict_from(split_name: str) -> None:
                guard_iter = 0
                while True:
                    guard_iter += 1
                    if guard_iter > 20000:
                        raise RuntimeError(
                            f"Could not evict giants from {split_name} after many swaps. Check thresholds."
                        )

                    giants_here = sorted(
                        [c for c in splits[split_name] if is_giant(c)],
                        key=trainable,
                        reverse=True,
                    )
                    if not giants_here:
                        break

                    g = giants_here[0]
                    y = label(g)

                    train_candidates = [c for c in splits["train"] if label(c) == y and not is_giant(c)]
                    if not train_candidates:
                        raise RuntimeError(
                            f"Impossible policy: keep_giants_in_train=True but train has no non-giant "
                            f"candidates to swap with for label={y}. "
                            f"Try increasing guardrails.giant_flow_threshold or set keep_giants_in_train=false."
                        )

                    swap_in = min(train_candidates, key=trainable)
                    do_swap("train", split_name, swap_in, g)

            evict_from("val")
            evict_from("test")

        # --- 1) Min VPN CAPTURES in val/test ---
        for target, min_caps in (("val", min_val_vpn_caps), ("test", min_test_vpn_caps)):
            if min_caps <= 0:
                continue

            for _ in range(4000):
                if vpn_cap_count(splits[target]) >= min_caps:
                    break

                vpn_train = candidates(
                    splits["train"], is_vpn, desc=True, allow_giants=True, key_fn=trainable
                )
                nonvpn_target = candidates(
                    splits[target], is_nonvpn, desc=True, allow_giants=False, key_fn=raw
                )
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

        # --- 1.5) Min VPN HEAVY CAPTURES in val/test ---
        for target, min_heavy in (("val", min_val_vpn_heavy_caps), ("test", min_test_vpn_heavy_caps)):
            if min_heavy <= 0 or vpn_heavy_trainable_threshold <= 0:
                continue

            for _ in range(6000):
                if vpn_heavy_count(splits[target]) >= min_heavy:
                    break

                heavy_train = candidates(
                    splits["train"], is_vpn_heavy, desc=True, allow_giants=True, key_fn=trainable
                )
                nonvpn_target = candidates(
                    splits[target], is_nonvpn, desc=True, allow_giants=False, key_fn=raw
                )
                if not heavy_train or not nonvpn_target:
                    break

                a = heavy_train[0]
                b = nonvpn_target[0]

                if keep_giants_in_train and is_giant(a):
                    heavy_train = [c for c in heavy_train if not is_giant(c)]
                    if not heavy_train:
                        break
                    a = heavy_train[0]

                do_swap("train", target, a, b)

        # --- 2) Min VPN TRAINABLE FLOWS in val/test ---
        for target, min_vpn_tr in (("val", min_val_vpn_trainable), ("test", min_test_vpn_trainable)):
            if min_vpn_tr <= 0:
                continue

            for _ in range(8000):
                if stats(splits[target])["vpn_trainable"] >= min_vpn_tr:
                    break

                vpn_train = candidates(
                    splits["train"], is_vpn, desc=True, allow_giants=True, key_fn=trainable
                )
                nonvpn_target = candidates(
                    splits[target], is_nonvpn, desc=True, allow_giants=False, key_fn=raw
                )
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

        # --- 3) Min TOTAL TRAINABLE FLOWS in val/test ---
        for target, min_total_tr in (("val", min_val_total_trainable), ("test", min_test_total_trainable)):
            if min_total_tr <= 0:
                continue

            for _ in range(12000):
                if stats(splits[target])["total_trainable"] >= min_total_tr:
                    break

                nonvpn_train = candidates(
                    splits["train"],
                    is_nonvpn,
                    desc=True,
                    allow_giants=not keep_giants_in_train,
                    key_fn=trainable,
                )
                if not nonvpn_train:
                    break

                # Prefer swapping OUT nonVPN from target first to preserve VPN signal.
                small_target = candidates(
                    splits[target],
                    is_nonvpn,
                    desc=False,
                    allow_giants=False,
                    key_fn=trainable,
                )
                if not small_target:
                    small_target = candidates(
                        splits[target],
                        lambda _: True,
                        desc=False,
                        allow_giants=False,
                        key_fn=trainable,
                    )
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

        # --- 4) Min TOTAL RAW FLOWS in val/test (optional legacy) ---
        for target, min_total_raw in (("val", min_val_total_raw), ("test", min_test_total_raw)):
            if min_total_raw <= 0:
                continue

            for _ in range(12000):
                if stats(splits[target])["total_raw"] >= min_total_raw:
                    break

                nonvpn_train = candidates(
                    splits["train"],
                    is_nonvpn,
                    desc=True,
                    allow_giants=not keep_giants_in_train,
                    key_fn=raw,
                )
                if not nonvpn_train:
                    break

                # Prefer swapping OUT nonVPN from target first to preserve VPN signal.
                small_target = candidates(
                    splits[target],
                    is_nonvpn,
                    desc=False,
                    allow_giants=False,
                    key_fn=raw,
                )
                if not small_target:
                    small_target = candidates(
                        splits[target],
                        lambda _: True,
                        desc=False,
                        allow_giants=False,
                        key_fn=raw,
                    )
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

        # --- 5) Min VPN TRAINABLE FLOWS in TRAIN ---
        if min_train_vpn_trainable > 0:
            for _ in range(12000):
                if stats(splits["train"])["vpn_trainable"] >= min_train_vpn_trainable:
                    break

                src = "val" if stats(splits["val"])["vpn_trainable"] >= stats(splits["test"])["vpn_trainable"] else "test"

                vpn_src = candidates(
                    splits[src],
                    is_vpn,
                    desc=True,
                    allow_giants=False,
                    key_fn=trainable,
                )
                nonvpn_train = candidates(
                    splits["train"],
                    is_nonvpn,
                    desc=False,
                    allow_giants=not keep_giants_in_train,
                    key_fn=trainable,
                )

                if not vpn_src or not nonvpn_train:
                    break

                a = vpn_src[0]
                b = nonvpn_train[0]

                if keep_giants_in_train and is_giant(b):
                    nonvpn_train = [c for c in nonvpn_train if not is_giant(c)]
                    if not nonvpn_train:
                        break
                    b = nonvpn_train[0]

                do_swap(src, "train", a, b)

        # --- 6) Max VPN FRACTION by trainable mass in val/test ---
        def enforce_max_vpn_fraction(target: str, max_frac: float) -> None:
            if max_frac <= 0.0 or max_frac >= 1.0:
                return

            for _ in range(20000):
                frac = vpn_fraction_trainable(splits[target])
                if frac <= max_frac:
                    break

                vpn_target = candidates(
                    splits[target],
                    is_vpn,
                    desc=True,
                    allow_giants=False,
                    key_fn=trainable,
                )
                if not vpn_target:
                    break

                nonvpn_train = candidates(
                    splits["train"],
                    is_nonvpn,
                    desc=True,
                    allow_giants=not keep_giants_in_train,
                    key_fn=trainable,
                )
                if not nonvpn_train:
                    break

                a = vpn_target[0]
                b = nonvpn_train[0]

                if keep_giants_in_train and is_giant(b):
                    nonvpn_train = [c for c in nonvpn_train if not is_giant(c)]
                    if not nonvpn_train:
                        break
                    b = nonvpn_train[0]

                do_swap(target, "train", a, b)

        enforce_max_vpn_fraction("val", max_val_vpn_fraction_trainable)
        enforce_max_vpn_fraction("test", max_test_vpn_fraction_trainable)

    # Iterate passes until the assignment stabilizes (prevents step 5/6 undoing step 2/3).
    max_rounds = 50
    last_sig = None
    for _ in range(max_rounds):
        cur = signature()
        if last_sig is not None and cur == last_sig:
            break
        last_sig = cur
        one_pass()

    # Shuffle final order (does not change membership)
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
      - Split is CAPTURE-level stratified by label 0/1.
      - The flow objective uses TRAINABLE mass (sum(min_packets_ok)).
      - Guardrails are enforced with SWAPS ONLY and use:
          * total RAW size: raw flow counts (optional)
          * learning signal: TRAINABLE flow counts
          * GIANTS: TRAINABLE flows per capture (matches splits.yaml comment)
          * VPN-heavy: TRAINABLE flows per capture
          * vpn_trainable_flow_min_train
          * max_vpn_fraction_trainable
    """
    cfg = _load_cfg(repo_root, splits_yaml)

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label", "min_packets_ok"])
    for col in ("capture_id", "label", "min_packets_ok"):
        if col not in df.columns:
            raise ValueError(f"flows.parquet must contain {col} column")

    df["capture_id"] = df["capture_id"].astype(str)

    cap = (
        df.groupby("capture_id")
        .agg(
            label=("label", "first"),
            n_flows=("label", "size"),               # RAW
            n_trainable=("min_packets_ok", "sum"),   # TRAINABLE
        )
        .reset_index()
    )
    cap["n_trainable"] = cap["n_trainable"].astype(int)
    cap["n_flows"] = cap["n_flows"].astype(int)

    check = df.groupby("capture_id")["label"].nunique()
    mixed = int((check > 1).sum())
    if mixed:
        raise ValueError(f"Found {mixed} captures with mixed labels. Fix labeling before splitting.")

    FLOW_MASS_COL = "n_trainable"

    final: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

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

        total_mass_y = int(cap_y[FLOW_MASS_COL].sum())
        flow_t = _flow_targets(total_mass_y, cfg.flow_train_r, cfg.flow_val_r)

        cap_y_sorted = cap_y.sort_values(FLOW_MASS_COL, ascending=False).reset_index(drop=True)
        k = min(cfg.giants_top_k_per_class, len(cap_y_sorted))
        giants = cap_y_sorted.iloc[:k].copy()
        giant_ids = set(giants["capture_id"].astype(str).tolist())

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

    # min VPN trainable in TRAIN
    vpn_min_train = cfg_raw.get("vpn_trainable_flow_min_train") or {}
    min_train_vpn_trainable = _as_int(vpn_min_train.get("train", 0), path="vpn_trainable_flow_min_train.train")

    vpn_min = cfg_raw.get("vpn_trainable_flow_min") or {}
    min_val_vpn_trainable = _as_int(vpn_min.get("val", 0), path="vpn_trainable_flow_min.val")
    min_test_vpn_trainable = _as_int(vpn_min.get("test", 0), path="vpn_trainable_flow_min.test")

    tot_min = cfg_raw.get("min_total_flows") or {}
    val_frac = _as_float(tot_min.get("val_frac", 0.0), path="min_total_flows.val_frac")
    test_frac = _as_float(tot_min.get("test_frac", 0.0), path="min_total_flows.test_frac")
    val_abs = _as_int(tot_min.get("val_abs", 0), path="min_total_flows.val_abs")
    test_abs = _as_int(tot_min.get("test_abs", 0), path="min_total_flows.test_abs")

    tot_trainable = cfg_raw.get("min_total_trainable_flows") or {}
    min_val_total_trainable = _as_int(tot_trainable.get("val_abs", 0), path="min_total_trainable_flows.val_abs")
    min_test_total_trainable = _as_int(tot_trainable.get("test_abs", 0), path="min_total_trainable_flows.test_abs")

    guard = cfg_raw.get("guardrails") or {}
    keep_giants_in_train = bool(guard.get("keep_giants_in_train", True))
    giant_trainable_threshold = _as_int(guard.get("giant_flow_threshold", 0), path="guardrails.giant_flow_threshold")

    min_vpn_caps = guard.get("min_vpn_captures") or {}
    min_val_vpn_caps = _as_int(min_vpn_caps.get("val", cfg.min_val_per_class), path="guardrails.min_vpn_captures.val")
    min_test_vpn_caps = _as_int(min_vpn_caps.get("test", cfg.min_test_per_class), path="guardrails.min_vpn_captures.test")

    min_vpn_heavy = guard.get("min_vpn_heavy_captures") or {}
    min_val_vpn_heavy_caps = _as_int(min_vpn_heavy.get("val", 0), path="guardrails.min_vpn_heavy_captures.val")
    min_test_vpn_heavy_caps = _as_int(min_vpn_heavy.get("test", 0), path="guardrails.min_vpn_heavy_captures.test")
    vpn_heavy_trainable_threshold = _as_int(
        guard.get("vpn_heavy_trainable_threshold", 0),
        path="guardrails.vpn_heavy_trainable_threshold",
    )

    max_vpn_frac = cfg_raw.get("max_vpn_fraction_trainable") or {}
    max_val_vpn_fraction_trainable = _as_float(max_vpn_frac.get("val", 0.0), path="max_vpn_fraction_trainable.val")
    max_test_vpn_fraction_trainable = _as_float(max_vpn_frac.get("test", 0.0), path="max_vpn_fraction_trainable.test")

    total_raw_all = int(cap["n_flows"].sum())
    min_val_total_raw = max(val_abs, int(round(val_frac * total_raw_all)))
    min_test_total_raw = max(test_abs, int(round(test_frac * total_raw_all)))

    cap_map_both: Dict[str, Dict[str, int]] = {
        str(r["capture_id"]): {
            "label": int(r["label"]),
            "raw": int(r["n_flows"]),
            "trainable": int(r["n_trainable"]),
        }
        for _, r in cap.iterrows()
    }

    total_vpn_caps = sum(1 for v in cap_map_both.values() if v["label"] == 1)
    total_vpn_trainable = sum(v["trainable"] for v in cap_map_both.values() if v["label"] == 1)

    # feasibility checks
    if min_val_vpn_caps + min_test_vpn_caps > total_vpn_caps:
        raise ValueError(
            f"Impossible guardrail: min_vpn_captures(val)+min_vpn_captures(test)={min_val_vpn_caps}+{min_test_vpn_caps} "
            f"exceeds total VPN captures available={total_vpn_caps}."
        )

    if min_train_vpn_trainable < 0:
        raise ValueError("vpn_trainable_flow_min_train.train must be >= 0")

    if min_val_vpn_trainable + min_test_vpn_trainable > total_vpn_trainable:
        raise ValueError(
            f"Impossible guardrail: vpn_trainable_flow_min(val)+vpn_trainable_flow_min(test)={min_val_vpn_trainable}+{min_test_vpn_trainable} "
            f"exceeds total VPN trainable flows available={total_vpn_trainable}. Lower vpn_trainable_flow_min."
        )

    if min_train_vpn_trainable > total_vpn_trainable:
        raise ValueError(
            f"Impossible guardrail: vpn_trainable_flow_min_train(train)={min_train_vpn_trainable} "
            f"exceeds total VPN trainable flows available={total_vpn_trainable}."
        )

    if (min_val_vpn_heavy_caps > 0 or min_test_vpn_heavy_caps > 0):
        if vpn_heavy_trainable_threshold <= 0:
            raise ValueError(
                "Invalid guardrail: min_vpn_heavy_captures is set but vpn_heavy_trainable_threshold <= 0."
            )
        total_vpn_heavy_caps = sum(
            1 for v in cap_map_both.values()
            if v["label"] == 1 and v["trainable"] >= vpn_heavy_trainable_threshold
        )
        if min_val_vpn_heavy_caps + min_test_vpn_heavy_caps > total_vpn_heavy_caps:
            raise ValueError(
                f"Impossible guardrail: min_vpn_heavy_captures(val)+min_vpn_heavy_captures(test)="
                f"{min_val_vpn_heavy_caps}+{min_test_vpn_heavy_caps} exceeds total VPN-heavy captures available="
                f"{total_vpn_heavy_caps} at threshold={vpn_heavy_trainable_threshold}."
            )

    for name, v in (("val", max_val_vpn_fraction_trainable), ("test", max_test_vpn_fraction_trainable)):
        if v != 0.0 and not (0.0 < v < 1.0):
            raise ValueError(f"max_vpn_fraction_trainable.{name} must be 0 or within (0,1). Got {v}.")

    before_sizes = {k: len(final[k]) for k in ("train", "val", "test")}

    need_rebalance = (
        keep_giants_in_train
        or min_train_vpn_trainable > 0
        or min_val_vpn_trainable > 0
        or min_test_vpn_trainable > 0
        or min_val_total_raw > 0
        or min_test_total_raw > 0
        or min_val_total_trainable > 0
        or min_test_total_trainable > 0
        or min_val_vpn_caps > 0
        or min_test_vpn_caps > 0
        or min_val_vpn_heavy_caps > 0
        or min_test_vpn_heavy_caps > 0
        or max_val_vpn_fraction_trainable > 0.0
        or max_test_vpn_fraction_trainable > 0.0
    )

    if need_rebalance:
        final = _rebalance_with_swaps_only(
            final,
            cap_map_both,
            min_train_vpn_trainable=min_train_vpn_trainable,
            min_val_vpn_trainable=min_val_vpn_trainable,
            min_test_vpn_trainable=min_test_vpn_trainable,
            min_val_total_raw=min_val_total_raw,
            min_test_total_raw=min_test_total_raw,
            min_val_total_trainable=min_val_total_trainable,
            min_test_total_trainable=min_test_total_trainable,
            min_val_vpn_caps=min_val_vpn_caps,
            min_test_vpn_caps=min_test_vpn_caps,
            min_val_vpn_heavy_caps=min_val_vpn_heavy_caps,
            min_test_vpn_heavy_caps=min_test_vpn_heavy_caps,
            vpn_heavy_trainable_threshold=vpn_heavy_trainable_threshold,
            max_val_vpn_fraction_trainable=max_val_vpn_fraction_trainable,
            max_test_vpn_fraction_trainable=max_test_vpn_fraction_trainable,
            keep_giants_in_train=keep_giants_in_train,
            giant_trainable_threshold=giant_trainable_threshold,
            seed=cfg.seed + 999,
        )

    _assert_split_sizes_unchanged(before_sizes, final)

    if keep_giants_in_train and giant_trainable_threshold > 0:
        giants_in_val = [c for c in final["val"] if cap_map_both[c]["trainable"] >= giant_trainable_threshold]
        giants_in_test = [c for c in final["test"] if cap_map_both[c]["trainable"] >= giant_trainable_threshold]
        if giants_in_val or giants_in_test:
            raise RuntimeError(
                "Policy violation: keep_giants_in_train=True but found TRAINABLE-giant captures in val/test. "
                f"giants_in_val={giants_in_val}, giants_in_test={giants_in_test}"
            )

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
    cap["n_flows"] = cap["n_flows"].astype(int)

    cap_map: Dict[str, Tuple[int, int, int]] = {
        str(r["capture_id"]): (int(r["label"]), int(r["n_flows"]), int(r["n_trainable"]))
        for _, r in cap.iterrows()
    }

    def split_stats(ids: List[str]) -> Dict[str, Any]:
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

    manifest: Dict[str, object] = {
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

    total_trainable = (
        manifest["split_stats"]["train"]["trainable_flows"]
        + manifest["split_stats"]["val"]["trainable_flows"]
        + manifest["split_stats"]["test"]["trainable_flows"]
    )
    achieved_trainable = {
        k: manifest["split_stats"][k]["trainable_flows"] / max(int(total_trainable), 1)
        for k in ("train", "val", "test")
    }
    target = {"train": cfg.flow_train_r, "val": cfg.flow_val_r, "test": cfg.flow_test_r}
    abs_err_trainable = {k: abs(achieved_trainable[k] - target[k]) for k in achieved_trainable}

    manifest["trainable_flow_ratio_check"] = {
        "target": target,
        "achieved": achieved_trainable,
        "abs_error": abs_err_trainable,
        "tolerance": cfg.abs_flow_ratio_tol,
        "within_tolerance": {k: abs_err_trainable[k] <= cfg.abs_flow_ratio_tol for k in abs_err_trainable},
    }

    total_raw = (
        manifest["split_stats"]["train"]["raw_flows"]
        + manifest["split_stats"]["val"]["raw_flows"]
        + manifest["split_stats"]["test"]["raw_flows"]
    )
    achieved_raw = {
        k: manifest["split_stats"][k]["raw_flows"] / max(int(total_raw), 1)
        for k in ("train", "val", "test")
    }
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
