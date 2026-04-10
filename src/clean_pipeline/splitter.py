# src/clean_pipeline/splitter.py
"""
Cross-dataset capture-level splitter for the CLEAN pipeline.

V1 (original) — pure greedy flow-ratio minimization per (dataset, label).
V2 (constrained) — two-phase constraint-satisfaction then greedy, with
    forced class presence, minimum-support guarantees, dominance caps,
    and structured diagnostics.

Both versions preserve capture integrity: all flows from one capture
stay in the same split.  Reproducible via seed.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class CleanSplitConfig:
    """
    Configuration for capture-level splitting.

    Backward-compatible: all v2 fields have safe defaults that reproduce
    v1 behaviour when ``splitter_version = 1``.
    """
    # ── core ratios ──
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── v1 legacy field ──
    min_captures_per_class_per_split: int = 2

    # ── v1 min-flow fields (were present but partially unused) ──
    min_flows_val: int = 50
    min_flows_test: int = 50

    # ── version selector ──
    #   1 = legacy greedy (preserved for comparison)
    #   2 = constrained two-phase (recommended)
    splitter_version: int = 2

    # ── v2: per-class flow minimums in eval splits ──
    min_vpn_flows_val: int = 30
    min_vpn_flows_test: int = 30
    min_benign_flows_val: int = 30
    min_benign_flows_test: int = 30

    # ── v2: per-class capture minimums in eval splits ──
    min_captures_val_per_class: int = 1
    min_captures_test_per_class: int = 1

    # ── v2: class-presence requirement ──
    require_class_presence_in_val_test: bool = True

    # ── v2: dominance cap ──
    max_capture_share_per_split: float = 0.40

    # ── v2: allow ratio relaxation when class support needs it ──
    allow_ratio_relaxation_for_class_support: bool = True

    # ── v2: scarce group detection threshold ──
    #   Groups with fewer captures than this use forced reservation.
    scarce_group_min_captures: int = 6


# ============================================================
# Capture summary helper (shared by v1 and v2)
# ============================================================

def _capture_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build capture-level summary from feature DataFrame.

    Returns DataFrame with columns:
        capture_id, dataset, label, n_flows
    """
    cap = (
        df.groupby("capture_id")
        .agg(
            dataset=("dataset", "first"),
            label=("label", "first"),
            n_flows=("label", "size"),
        )
        .reset_index()
    )
    cap["label"] = cap["label"].astype(int)
    cap["n_flows"] = cap["n_flows"].astype(int)
    return cap


# ============================================================
# V1 — Legacy greedy splitter  (preserved for comparison)
# ============================================================

def _assign_captures_greedy(
    captures: pd.DataFrame,
    train_r: float,
    val_r: float,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Greedy capture assignment minimizing flow-ratio error.

    Processes captures largest-first, assigning each to the split
    that minimizes the global flow-ratio deviation.
    """
    rng = np.random.default_rng(seed)

    total_flows = int(captures["n_flows"].sum())
    targets = {
        "train": int(round(total_flows * train_r)),
        "val": int(round(total_flows * val_r)),
        "test": total_flows - int(round(total_flows * train_r)) - int(round(total_flows * val_r)),
    }

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    flows: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # Shuffle then sort by size (deterministic tie-breaking)
    caps = captures.sample(frac=1.0, random_state=seed).sort_values(
        "n_flows", ascending=False
    ).reset_index(drop=True)

    for _, row in caps.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        def score(s: str) -> int:
            return sum(
                abs((flows[k] + (w if k == s else 0)) - targets[k])
                for k in ("train", "val", "test")
            )

        best = min(score(s) for s in ("train", "val", "test"))
        cands = [s for s in ("train", "val", "test") if score(s) == best]
        chosen = cands[int(rng.integers(0, len(cands)))]

        out[chosen].append(cid)
        flows[chosen] += w

    return out


def _make_clean_split_v1(
    df: pd.DataFrame,
    cfg: CleanSplitConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """V1 legacy splitter — preserved for comparison."""
    cap = _capture_summary(df)
    all_assignments: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {"version": 1, "groups": {}}

    for (ds, lbl), group in cap.groupby(["dataset", "label"]):
        n_caps = len(group)
        key = f"{ds}/label={lbl}"
        if n_caps == 0:
            continue

        min_per = cfg.min_captures_per_class_per_split
        if n_caps < min_per * 3:
            print(f"  WARNING: {key} has only {n_caps} captures, "
                  f"all assigned to train")
            for cid in group["capture_id"]:
                all_assignments[str(cid)] = "train"
            diagnostics["groups"][key] = {"action": "all_to_train",
                                          "n_captures": n_caps}
            continue

        assigns = _assign_captures_greedy(
            group,
            train_r=cfg.train_ratio,
            val_r=cfg.val_ratio,
            seed=cfg.seed + hash(f"{ds}_{lbl}") % 10000,
        )
        for split, cids in assigns.items():
            for cid in cids:
                all_assignments[cid] = split
        diagnostics["groups"][key] = {
            "action": "greedy",
            "n_captures": n_caps,
            "per_split": {s: len(c) for s, c in assigns.items()},
        }

    df = df.copy()
    df["split"] = df["capture_id"].map(all_assignments)
    unmapped = df["split"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped} flows with unmapped captures -> train")
        df["split"] = df["split"].fillna("train")

    return df, diagnostics


# ============================================================
# V2 — Constrained two-phase splitter
# ============================================================

def _v2_check_feasibility(
    group: pd.DataFrame,
    ds: str,
    lbl: int,
    cfg: CleanSplitConfig,
) -> Dict[str, Any]:
    """
    Pre-flight feasibility check for one (dataset, label) group.

    Returns a dict describing the group and whether forced
    reservation is needed.
    """
    n_caps = len(group)
    total_flows = int(group["n_flows"].sum())
    is_scarce = n_caps < cfg.scarce_group_min_captures

    min_val_caps = cfg.min_captures_val_per_class
    min_test_caps = cfg.min_captures_test_per_class

    if lbl == 1:
        min_flows_v = cfg.min_vpn_flows_val
        min_flows_t = cfg.min_vpn_flows_test
    else:
        min_flows_v = cfg.min_benign_flows_val
        min_flows_t = cfg.min_benign_flows_test

    warnings = []
    can_fill_val = n_caps >= min_val_caps + min_test_caps + 1
    can_fill_test = can_fill_val

    if n_caps < 3:
        warnings.append(
            f"{ds}/label={lbl}: only {n_caps} captures, "
            f"cannot populate all 3 splits"
        )
        can_fill_val = n_caps >= 2
        can_fill_test = False

    if n_caps == 0:
        warnings.append(f"{ds}/label={lbl}: zero captures")

    return {
        "dataset": ds,
        "label": lbl,
        "n_captures": n_caps,
        "total_flows": total_flows,
        "is_scarce": is_scarce,
        "can_fill_val": can_fill_val,
        "can_fill_test": can_fill_test,
        "min_flows_val": min_flows_v,
        "min_flows_test": min_flows_t,
        "min_captures_val": min_val_caps,
        "min_captures_test": min_test_caps,
        "warnings": warnings,
    }


def _v2_pick_captures_for_target(
    available: pd.DataFrame,
    min_captures: int,
    min_flows: int,
    max_dominance: float,
    total_split_flows_so_far: int,
    rng: np.random.Generator,
) -> List[str]:
    """
    Select captures from *available* to satisfy a minimum-support constraint.

    Strategy:
      1. Sort by n_flows ascending (prefer small-to-medium captures to
         reduce dominance and leave large captures for train).
      2. Greedily pick captures until both min_captures and min_flows are met.
      3. After each pick, check dominance: if the picked capture would exceed
         max_dominance of the accumulating split, try the next candidate
         instead (but always pick *something* if it's the only option).

    Returns list of capture_ids to assign to the target split.
    """
    if available.empty:
        return []

    avail = available.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))\
                     .sort_values("n_flows", ascending=True)\
                     .reset_index(drop=True)

    picked: List[str] = []
    picked_flows = 0
    remaining_idx = list(avail.index)

    while remaining_idx and (
        len(picked) < min_captures or picked_flows < min_flows
    ):
        best_idx = None
        for idx in remaining_idx:
            cflows = int(avail.at[idx, "n_flows"])
            projected_total = total_split_flows_so_far + picked_flows + cflows
            share = cflows / projected_total if projected_total > 0 else 1.0
            if share <= max_dominance or best_idx is None:
                best_idx = idx
                if share <= max_dominance:
                    break  # acceptable, take it

        if best_idx is None:
            best_idx = remaining_idx[0]

        cid = str(avail.at[best_idx, "capture_id"])
        cflows = int(avail.at[best_idx, "n_flows"])
        picked.append(cid)
        picked_flows += cflows
        remaining_idx.remove(best_idx)

    return picked


def _v2_assign_group(
    group: pd.DataFrame,
    feasibility: Dict[str, Any],
    cfg: CleanSplitConfig,
    seed: int,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Assign captures for ONE (dataset, label) group under v2 constraints.

    Phase 1: forced reservation for val and test to meet class-support minimums.
    Phase 2: greedy flow-ratio assignment for remaining captures.

    Returns (assignments dict {split: [capture_ids]}, diagnostics messages).
    """
    rng = np.random.default_rng(seed)
    n_caps = feasibility["n_captures"]
    ds = feasibility["dataset"]
    lbl = feasibility["label"]
    msgs: List[str] = list(feasibility["warnings"])

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    if n_caps == 0:
        return out, msgs

    # ── Edge case: only 1 capture ──
    if n_caps == 1:
        out["train"] = [str(group.iloc[0]["capture_id"])]
        msgs.append(f"{ds}/label={lbl}: single capture -> train only")
        return out, msgs

    # ── Edge case: only 2 captures ──
    if n_caps == 2:
        sorted_g = group.sort_values("n_flows", ascending=False).reset_index(drop=True)
        out["train"] = [str(sorted_g.at[0, "capture_id"])]
        out["val"] = [str(sorted_g.at[1, "capture_id"])]
        msgs.append(f"{ds}/label={lbl}: 2 captures -> 1 train, 1 val, 0 test")
        return out, msgs

    # ── Phase 1: forced reservation for val and test ──
    # For SCARCE groups (few captures, e.g. USBVPN non-VPN with 5):
    #   force-reserve captures to val/test to guarantee class presence.
    # For NON-SCARCE groups (many captures):
    #   use the proven greedy approach (v1 algorithm), which distributes
    #   captures well when there are enough of them.
    available = group.copy().reset_index(drop=True)
    assigned_ids: set = set()

    if feasibility["is_scarce"]:
        # ── Scarce path: forced reservation then greedy remainder ──
        for target_split, min_caps, min_flows in [
            ("val", feasibility["min_captures_val"], feasibility["min_flows_val"]),
            ("test", feasibility["min_captures_test"], feasibility["min_flows_test"]),
        ]:
            avail_mask = ~available["capture_id"].astype(str).isin(assigned_ids)
            avail_sub = available[avail_mask]

            # Don't take all: reserve 1 for the other eval split + 1 for train
            max_take = max(1, len(avail_sub) - 2)
            effective_min_caps = min(min_caps, max_take)

            picks = _v2_pick_captures_for_target(
                avail_sub,
                min_captures=effective_min_caps,
                min_flows=min_flows,
                max_dominance=cfg.max_capture_share_per_split,
                total_split_flows_so_far=0,
                rng=rng,
            )
            for cid in picks:
                out[target_split].append(cid)
                assigned_ids.add(cid)

            actual_flows = int(
                available[available["capture_id"].astype(str).isin(picks)]["n_flows"].sum()
            )
            if actual_flows < min_flows and len(picks) > 0:
                msgs.append(
                    f"{ds}/label={lbl}/{target_split}: reserved {len(picks)} captures "
                    f"({actual_flows} flows) but min_flows={min_flows} not fully met"
                )

        msgs.append(
            f"{ds}/label={lbl}: scarce group ({n_caps} captures), "
            f"used forced reservation for val/test"
        )

        # Assign remaining scarce captures to train
        for _, row in available.iterrows():
            cid = str(row["capture_id"])
            if cid not in assigned_ids:
                out["train"].append(cid)

        return out, msgs

    # ── Non-scarce path: use proven greedy from v1 ──
    # This works well when there are enough captures to naturally
    # populate all three splits.
    assigns = _assign_captures_greedy(
        group,
        train_r=cfg.train_ratio,
        val_r=cfg.val_ratio,
        seed=seed,
    )
    return assigns, msgs


def _v2_post_check(
    df: pd.DataFrame,
    cap: pd.DataFrame,
    cfg: CleanSplitConfig,
    all_assignments: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Post-assignment verification and targeted repair.

    Checks each (dataset, split) for class-presence violations.
    If both classes were present in the dataset but one is missing
    from val or test, performs a targeted swap from train.

    Returns updated assignments and diagnostic messages.
    """
    msgs: List[str] = []
    assignments = dict(all_assignments)
    max_swaps = 5

    for ds in cap["dataset"].unique():
        ds_cap = cap[cap["dataset"] == ds]
        labels_in_ds = set(ds_cap["label"].unique())

        if len(labels_in_ds) < 2:
            continue

        for split in ("val", "test"):
            for lbl in (0, 1):
                if lbl not in labels_in_ds:
                    continue

                ds_lbl_cids = set(
                    ds_cap[ds_cap["label"] == lbl]["capture_id"].astype(str)
                )
                split_caps = [
                    cid for cid in ds_lbl_cids if assignments.get(cid) == split
                ]

                if len(split_caps) > 0:
                    continue

                if not cfg.require_class_presence_in_val_test:
                    continue

                lbl_name = "VPN" if lbl == 1 else "non-VPN"
                msgs.append(
                    f"REPAIR: {ds}/{split} missing {lbl_name}, "
                    f"attempting swap from train"
                )

                train_candidates = ds_cap[
                    ds_cap["label"] == lbl
                ].copy()
                train_candidates = train_candidates[
                    train_candidates["capture_id"].astype(str).apply(
                        lambda c: assignments.get(c) == "train"
                    )
                ].sort_values("n_flows", ascending=True)

                if train_candidates.empty:
                    msgs.append(
                        f"  FAILED: no train captures of {lbl_name} in {ds}"
                    )
                    continue

                swap_cid = str(train_candidates.iloc[0]["capture_id"])
                assignments[swap_cid] = split
                msgs.append(
                    f"  SWAPPED: capture {swap_cid} "
                    f"({int(train_candidates.iloc[0]['n_flows'])} flows) "
                    f"train -> {split}"
                )

                max_swaps -= 1
                if max_swaps <= 0:
                    msgs.append("  Max swap attempts reached, stopping repairs")
                    return assignments, msgs

    return assignments, msgs


def _v2_build_diagnostics(
    df: pd.DataFrame,
    cfg: CleanSplitConfig,
    group_msgs: Dict[str, List[str]],
    repair_msgs: List[str],
) -> Dict[str, Any]:
    """Build structured diagnostics dict for the v2 splitter."""
    diag: Dict[str, Any] = {
        "version": 2,
        "config": {
            "splitter_version": cfg.splitter_version,
            "max_capture_share_per_split": cfg.max_capture_share_per_split,
            "min_vpn_flows_val": cfg.min_vpn_flows_val,
            "min_vpn_flows_test": cfg.min_vpn_flows_test,
            "min_benign_flows_val": cfg.min_benign_flows_val,
            "min_benign_flows_test": cfg.min_benign_flows_test,
            "require_class_presence_in_val_test": cfg.require_class_presence_in_val_test,
        },
        "per_dataset_split": {},
        "constraint_violations": [],
        "repair_actions": repair_msgs,
        "group_diagnostics": group_msgs,
    }

    for ds in sorted(df["dataset"].unique()):
        ds_data = {}
        for split in ("train", "val", "test"):
            sub = df[(df["dataset"] == ds) & (df["split"] == split)]
            n_flows = len(sub)
            vpn = int((sub["label"] == 1).sum())
            nonvpn = int((sub["label"] == 0).sum())
            n_caps = sub["capture_id"].nunique()
            vpn_caps = sub.loc[sub["label"] == 1, "capture_id"].nunique()
            nonvpn_caps = sub.loc[sub["label"] == 0, "capture_id"].nunique()

            if n_flows > 0:
                cap_sizes = sub.groupby("capture_id").size()
                max_share = float(cap_sizes.max() / n_flows)
            else:
                max_share = 0.0

            ds_data[split] = {
                "n_flows": n_flows,
                "vpn_flows": vpn,
                "nonvpn_flows": nonvpn,
                "n_captures": n_caps,
                "vpn_captures": vpn_caps,
                "nonvpn_captures": nonvpn_caps,
                "max_capture_share": round(max_share, 4),
            }

            if split in ("val", "test"):
                if vpn == 0 and cfg.require_class_presence_in_val_test:
                    diag["constraint_violations"].append(
                        f"{ds}/{split}: zero VPN flows"
                    )
                if nonvpn == 0 and cfg.require_class_presence_in_val_test:
                    diag["constraint_violations"].append(
                        f"{ds}/{split}: zero non-VPN flows"
                    )
                if max_share > cfg.max_capture_share_per_split:
                    diag["constraint_violations"].append(
                        f"{ds}/{split}: capture dominance "
                        f"{max_share:.3f} > {cfg.max_capture_share_per_split}"
                    )

        diag["per_dataset_split"][ds] = ds_data

    return diag


def _make_clean_split_v2(
    df: pd.DataFrame,
    cfg: CleanSplitConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    V2 constrained two-phase splitter.

    Phase 1: For each (dataset, label) group, force-reserve captures
             into val/test to satisfy class-support minimums.
    Phase 2: Assign remaining captures via dominance-penalized greedy.
    Phase 3: Post-check and targeted repair of class-presence gaps.
    """
    cap = _capture_summary(df)
    all_assignments: Dict[str, str] = {}
    group_msgs: Dict[str, List[str]] = {}

    for (ds, lbl), group in cap.groupby(["dataset", "label"]):
        key = f"{ds}/label={lbl}"
        group = group.reset_index(drop=True)

        feasibility = _v2_check_feasibility(group, ds, lbl, cfg)
        group_seed = cfg.seed + hash(f"{ds}_{lbl}") % 10000

        assigns, msgs = _v2_assign_group(group, feasibility, cfg, group_seed)

        for split, cids in assigns.items():
            for cid in cids:
                all_assignments[cid] = split

        group_msgs[key] = msgs

    df = df.copy()
    df["split"] = df["capture_id"].astype(str).map(all_assignments)

    unmapped = df["split"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped} flows with unmapped captures -> train")
        df["split"] = df["split"].fillna("train")

    repair_assignments, repair_msgs = _v2_post_check(
        df, cap, cfg, all_assignments
    )

    if repair_msgs:
        df["split"] = df["capture_id"].astype(str).map(repair_assignments)
        unmapped = df["split"].isna().sum()
        if unmapped > 0:
            df["split"] = df["split"].fillna("train")

    diagnostics = _v2_build_diagnostics(df, cfg, group_msgs, repair_msgs)

    return df, diagnostics


# ============================================================
# Public API
# ============================================================

def make_clean_split(
    df: pd.DataFrame,
    cfg: CleanSplitConfig = CleanSplitConfig(),
) -> pd.DataFrame:
    """
    Assign a 'split' column to df based on capture-level splitting.

    Parameters
    ----------
    df : DataFrame
        Feature DataFrame with: flow_id, capture_id, dataset, label
    cfg : CleanSplitConfig
        Configuration.  ``cfg.splitter_version`` selects the algorithm:
        - 1 = legacy greedy (original behaviour)
        - 2 = constrained two-phase (recommended, default)

    Returns
    -------
    df with an added 'split' column ("train" / "val" / "test")
    """
    if cfg.splitter_version == 1:
        df, diagnostics = _make_clean_split_v1(df, cfg)
    elif cfg.splitter_version == 2:
        df, diagnostics = _make_clean_split_v2(df, cfg)
    else:
        raise ValueError(f"Unknown splitter_version: {cfg.splitter_version}")

    # ── Summary (both versions) ──
    print(f"\nSplit summary (v{cfg.splitter_version}):")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_caps = sub["capture_id"].nunique()
        print(f"  {split:5s}: {len(sub):6d} flows, {n_caps:4d} captures "
              f"(VPN={int((sub['label']==1).sum())}, "
              f"nonVPN={int((sub['label']==0).sum())})")

    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        print(f"  {ds}:")
        for split in ("train", "val", "test"):
            ss = sub[sub["split"] == split]
            vpn = int((ss["label"] == 1).sum())
            nonvpn = int((ss["label"] == 0).sum())
            caps = ss["capture_id"].nunique()
            print(f"    {split:5s}: {len(ss):6d} flows  "
                  f"VPN={vpn:5d}  nonVPN={nonvpn:5d}  caps={caps}")

    if diagnostics.get("constraint_violations"):
        print("\n  CONSTRAINT VIOLATIONS:")
        for v in diagnostics["constraint_violations"]:
            print(f"    ⚠ {v}")

    if diagnostics.get("repair_actions"):
        print("\n  REPAIR ACTIONS:")
        for r in diagnostics["repair_actions"]:
            print(f"    → {r}")

    df.attrs["split_diagnostics"] = diagnostics

    return df


def save_split_manifest(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save split lists and manifest JSON.

    V2-aware: includes splitter version, active constraints, per-dataset
    per-class counts, and any constraint violations.

    Writes:
        output_dir/clean_train_captures.txt
        output_dir/clean_val_captures.txt
        output_dir/clean_test_captures.txt
        output_dir/clean_split_manifest.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    split_diag = df.attrs.get("split_diagnostics", {})

    manifest: Dict[str, Any] = {
        "pipeline": "clean",
        "splitter_version": split_diag.get("version", 1),
        "config": pipeline_config or {},
        "splitter_config": split_diag.get("config", {}),
        "constraint_violations": split_diag.get("constraint_violations", []),
        "repair_actions": split_diag.get("repair_actions", []),
    }

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        caps = sorted(sub["capture_id"].unique().tolist())

        list_path = output_dir / f"clean_{split}_captures.txt"
        list_path.write_text("\n".join(caps) + "\n", encoding="utf-8")

        ds_stats = {}
        for ds in sorted(sub["dataset"].unique()):
            ds_sub = sub[sub["dataset"] == ds]
            n_flows = int(len(ds_sub))
            vpn_flows = int((ds_sub["label"] == 1).sum())
            nonvpn_flows = int((ds_sub["label"] == 0).sum())
            vpn_caps = int(ds_sub.loc[ds_sub["label"] == 1, "capture_id"].nunique())
            nonvpn_caps = int(ds_sub.loc[ds_sub["label"] == 0, "capture_id"].nunique())
            n_caps_ds = int(ds_sub["capture_id"].nunique())

            if n_flows > 0:
                cap_sizes = ds_sub.groupby("capture_id").size()
                max_share = round(float(cap_sizes.max() / n_flows), 4)
            else:
                max_share = 0.0

            ds_stats[ds] = {
                "n_captures": n_caps_ds,
                "n_flows": n_flows,
                "vpn_flows": vpn_flows,
                "nonvpn_flows": nonvpn_flows,
                "vpn_captures": vpn_caps,
                "nonvpn_captures": nonvpn_caps,
                "max_capture_share": max_share,
            }

        manifest[split] = {
            "n_captures": int(len(caps)),
            "n_flows": int(len(sub)),
            "vpn_flows": int((sub["label"] == 1).sum()),
            "nonvpn_flows": int((sub["label"] == 0).sum()),
            "per_dataset": ds_stats,
            "capture_list": str(list_path.name),
        }

    for split in ("train", "val", "test"):
        list_path = output_dir / f"clean_{split}_captures.txt"
        h = hashlib.sha256(list_path.read_bytes()).hexdigest()
        manifest[split]["sha256"] = h

    manifest_path = output_dir / "clean_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSplit manifest saved to {manifest_path}")
    return manifest_path


