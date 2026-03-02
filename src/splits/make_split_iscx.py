from __future__ import annotations
from pathlib import Path
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
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    df["capture_id"] = df["capture_id"].astype(str)

    # Group by capture_id and take the first label.
    # IMPORTANT: We must ensure we get ALL captures, even if they have mixed labels (though they shouldn't).
    # Using 'first' is fine if captures are pure.
    cap = (
        df.groupby("capture_id")
          .agg(label=("label","first"), n_flows=("label","size"))
          .reset_index()
    )

    # stratify by label at capture level
    # Convert to list to ensure we have a clean list of strings
    vpn_caps = cap.loc[cap.label == 1, "capture_id"].to_list()
    non_caps = cap.loc[cap.label == 0, "capture_id"].to_list()

    rng = np.random.default_rng(seed)
    rng.shuffle(vpn_caps)
    rng.shuffle(non_caps)

    def split_list(items):
        n = len(items)
        n_train = int(round(train_frac * n))
        n_val   = int(round(val_frac * n))
        # ensure sums to n
        n_test  = n - n_train - n_val
        
        # Handle edge cases where n is small
        if n_test < 0: n_test = 0
        if n_train + n_val + n_test != n:
             # Adjust train to make it sum up if rounding errors
             n_train = n - n_val - n_test

        train = items[:n_train]
        val   = items[n_train:n_train+n_val]
        test  = items[n_train+n_val:]
        return train, val, test

    vpn_train, vpn_val, vpn_test = split_list(vpn_caps)
    non_train, non_val, non_test = split_list(non_caps)

    # guardrails: ensure enough vpn captures in val/test
    if len(vpn_val) < min_vpn_captures_val and len(vpn_train) > 0:
        take = min(min_vpn_captures_val - len(vpn_val), len(vpn_train))
        vpn_val += vpn_train[:take]
        vpn_train = vpn_train[take:]

    if len(vpn_test) < min_vpn_captures_test and len(vpn_train) > 0:
        take = min(min_vpn_captures_test - len(vpn_test), len(vpn_train))
        vpn_test += vpn_train[:take]
        vpn_train = vpn_train[take:]

    splits = {
        "train": sorted(vpn_train + non_train),
        "val":   sorted(vpn_val + non_val),
        "test":  sorted(vpn_test + non_test),
    }
    return splits

def write_capture_lists(splits: dict, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ["train","val","test"]:
        p = out_dir / f"{prefix}_{k}_captures.txt"
        # Write lines, ensuring no empty lines and proper encoding
        p.write_text("\n".join(map(str, splits[k])) + "\n", encoding="utf-8")
