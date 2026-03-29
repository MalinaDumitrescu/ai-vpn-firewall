from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def make_iscx_capture_split(
    flows_parquet: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    min_vpn_flows_val: int = 200,
    min_vpn_flows_test: int = 200,
) -> Dict[str, List[str]]:
    """
    Creates stratified capture-group splits for the ISCX dataset.

    This function ensures that the validation and test sets contain a minimum
    number of VPN flows, which is crucial for reliable evaluation when VPN
    samples are rare.

    The splitting strategy is as follows:
    1.  Separate captures into VPN and non-VPN groups.
    2.  For VPN captures:
        a. Shuffle the captures.
        b. Greedily assign captures to the 'test' set until `min_vpn_flows_test` is met.
        c. Greedily assign captures to the 'val' set until `min_vpn_flows_val` is met.
        d. Assign all remaining VPN captures to the 'train' set.
    3.  For non-VPN captures:
        a. Split them into train, val, and test sets based on the provided
           fractions (`train_frac`, `val_frac`, `test_frac`).
    4.  Combine the respective VPN and non-VPN capture lists for the final splits.
    5.  Perform sanity checks to ensure no overlap and full coverage.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-9:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0. "
            f"Got {train_frac} + {val_frac} + {test_frac} = {train_frac + val_frac + test_frac}"
        )

    if not flows_parquet.exists():
        raise FileNotFoundError(f"Missing flows parquet: {flows_parquet}")

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    df["capture_id"] = df["capture_id"].astype(str)

    # Check that each capture has exactly one label
    nunique = df.groupby("capture_id")["label"].nunique()
    mixed = int((nunique > 1).sum())
    if mixed:
        examples = nunique[nunique > 1].index.tolist()[:10]
        raise ValueError(
            f"Found {mixed} captures with mixed labels. Examples: {examples}"
        )

    cap = (
        df.groupby("capture_id")
        .agg(
            label=("label", "first"),
            n_flows=("label", "size"),
        )
        .reset_index()
    )

    vpn_caps_df = cap.loc[cap["label"] == 1, ["capture_id", "n_flows"]].copy()
    non_caps_df = cap.loc[cap["label"] == 0, ["capture_id", "n_flows"]].copy()

    if vpn_caps_df.empty:
        raise ValueError("No VPN captures found in ISCX flows.")
    if non_caps_df.empty:
        raise ValueError("No non-VPN captures found in ISCX flows.")

    rng = np.random.default_rng(seed)

    # Shuffle captures
    vpn_caps_df = vpn_caps_df.sample(frac=1, random_state=rng)
    non_caps_list = non_caps_df["capture_id"].astype(str).tolist()
    rng.shuffle(non_caps_list)

    # --- VPN Capture Splitting (Flow-based) ---
    vpn_test = []
    vpn_val = []
    vpn_caps_list = list(vpn_caps_df.itertuples(index=False, name=None))

    # Greedily assign to test set to meet flow minimum
    test_flows = 0
    while test_flows < min_vpn_flows_test and vpn_caps_list:
        cap_id, n_flows = vpn_caps_list.pop(0)
        vpn_test.append(str(cap_id))
        test_flows += n_flows

    # Greedily assign to val set to meet flow minimum
    val_flows = 0
    while val_flows < min_vpn_flows_val and vpn_caps_list:
        cap_id, n_flows = vpn_caps_list.pop(0)
        vpn_val.append(str(cap_id))
        val_flows += n_flows

    # Assign the rest to train
    vpn_train = [str(cap_id) for cap_id, n_flows in vpn_caps_list]

    # --- Non-VPN Capture Splitting (Count-based) ---
    def split_list(items: List[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val

        if n_test < 0:
            n_test = 0
            n_train = n - n_val - n_test
        if n_train + n_val + n_test != n:
            n_train = n - n_val - n_test

        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return train, val, test

    non_train, non_val, non_test = split_list(non_caps_list)

    # --- Combine and Finalize ---
    train_ids = sorted(vpn_train + non_train)
    val_ids = sorted(vpn_val + non_val)
    test_ids = sorted(vpn_test + non_test)

    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)

    # Overlap checks
    if s_train & s_val:
        raise ValueError(f"Overlap train/val: {sorted(list(s_train & s_val))[:10]}")
    if s_train & s_test:
        raise ValueError(f"Overlap train/test: {sorted(list(s_train & s_test))[:10]}")
    if s_val & s_test:
        raise ValueError(f"Overlap val/test: {sorted(list(s_val & s_test))[:10]}")

    # Coverage check
    all_caps = set(cap["capture_id"].astype(str).tolist())
    all_assigned = s_train | s_val | s_test

    if all_assigned != all_caps:
        missing = sorted(list(all_caps - all_assigned))[:10]
        extra = sorted(list(all_assigned - all_caps))[:10]
        raise ValueError(
            f"Split coverage mismatch. Missing={missing}, Extra={extra}"
        )

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    return splits


def write_capture_lists(splits: Dict[str, List[str]], out_dir: Path, prefix: str) -> None:
    required = {"train", "val", "test"}
    missing = required - set(splits.keys())
    if missing:
        raise ValueError(f"Missing split keys: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ["train", "val", "test"]:
        p = out_dir / f"{prefix}_{k}_captures.txt"
        values = [str(x).strip() for x in splits[k] if str(x).strip()]
        p.write_text("\n".join(values) + "\n", encoding="utf-8")
