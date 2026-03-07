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
    min_vpn_captures_val: int = 3,
    min_vpn_captures_test: int = 3,
) -> Dict[str, List[str]]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-9:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0. "
            f"Got {train_frac} + {val_frac} + {test_frac} = {train_frac + val_frac + test_frac}"
        )

    if min_vpn_captures_val < 0 or min_vpn_captures_test < 0:
        raise ValueError("min_vpn_captures_val and min_vpn_captures_test must be >= 0")

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

    vpn_caps = cap.loc[cap["label"] == 1, "capture_id"].astype(str).tolist()
    non_caps = cap.loc[cap["label"] == 0, "capture_id"].astype(str).tolist()

    if not vpn_caps:
        raise ValueError("No VPN captures found in ISCX flows.")
    if not non_caps:
        raise ValueError("No non-VPN captures found in ISCX flows.")

    rng = np.random.default_rng(seed)
    rng.shuffle(vpn_caps)
    rng.shuffle(non_caps)

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

    vpn_train, vpn_val, vpn_test = split_list(vpn_caps)
    non_train, non_val, non_test = split_list(non_caps)

    # Guardrails: ensure enough VPN captures in val/test
    if len(vpn_val) < min_vpn_captures_val and len(vpn_train) > 0:
        take = min(min_vpn_captures_val - len(vpn_val), len(vpn_train))
        vpn_val += vpn_train[:take]
        vpn_train = vpn_train[take:]

    if len(vpn_test) < min_vpn_captures_test and len(vpn_train) > 0:
        take = min(min_vpn_captures_test - len(vpn_test), len(vpn_train))
        vpn_test += vpn_train[:take]
        vpn_train = vpn_train[take:]

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