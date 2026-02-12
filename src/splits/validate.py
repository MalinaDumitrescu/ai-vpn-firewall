from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Optional

import json
import pandas as pd
import yaml


def _read_list(p: Path) -> List[str]:
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def validate_split_files(
    flows_parquet: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    *,
    splits_yaml: Optional[Path] = None,
) -> Dict[str, object]:
    train = _read_list(train_list)
    val = _read_list(val_list)
    test = _read_list(test_list)

    s_train, s_val, s_test = set(train), set(val), set(test)

    # --- Overlap checks ---
    if s_train & s_val:
        raise ValueError(f"Overlap train/val: {sorted(list(s_train & s_val))[:10]}")
    if s_train & s_test:
        raise ValueError(f"Overlap train/test: {sorted(list(s_train & s_test))[:10]}")
    if s_val & s_test:
        raise ValueError(f"Overlap val/test: {sorted(list(s_val & s_test))[:10]}")

    # --- Load flows ---
    df = pd.read_parquet(flows_parquet, columns=["capture_id", "label"])
    if "capture_id" not in df.columns or "label" not in df.columns:
        raise ValueError("flows.parquet must contain capture_id and label columns")

    df["capture_id"] = df["capture_id"].astype(str)

    # --- Mixed-label capture check ---
    nunique = df.groupby("capture_id")["label"].nunique()
    mixed = int((nunique > 1).sum())
    if mixed:
        examples = nunique[nunique > 1].index.tolist()[:10]
        raise ValueError(
            f"Found {mixed} captures with mixed labels. Examples: {examples}. "
            "Fix labeling before trusting split lists."
        )

    cap = (
        df.groupby("capture_id")
        .agg(label=("label", "first"), n_flows=("label", "size"))
        .reset_index()
    )
    all_caps = set(cap["capture_id"].tolist())

    listed = s_train | s_val | s_test
    missing = all_caps - listed
    extra = listed - all_caps

    if missing:
        raise ValueError(f"Some captures are missing from split lists. Examples: {sorted(list(missing))[:10]}")
    if extra:
        raise ValueError(f"Split lists contain unknown captures. Examples: {sorted(list(extra))[:10]}")

    cap_map = {str(r["capture_id"]): (int(r["label"]), int(r["n_flows"])) for _, r in cap.iterrows()}

    # --- Policy-aware check: no giants in val/test ---
    if splits_yaml is not None:
        raw = yaml.safe_load(Path(splits_yaml).read_text(encoding="utf-8")) or {}
        guard = raw.get("guardrails") or {}
        keep_giants_in_train = bool(guard.get("keep_giants_in_train", True))
        giant_flow_threshold = int(guard.get("giant_flow_threshold", 5000))

        if keep_giants_in_train:
            giants_in_val = [c for c in s_val if cap_map[c][1] >= giant_flow_threshold]
            giants_in_test = [c for c in s_test if cap_map[c][1] >= giant_flow_threshold]
            if giants_in_val or giants_in_test:
                raise ValueError(
                    "Policy violation: keep_giants_in_train=true but found giant captures in val/test. "
                    f"threshold={giant_flow_threshold}, "
                    f"giants_in_val={giants_in_val}, giants_in_test={giants_in_test}"
                )

    def stats(ids: Set[str]) -> Dict[str, object]:
        labels = [cap_map[c][0] for c in ids]
        flows = [cap_map[c][1] for c in ids]
        return {
            "n_captures": int(len(ids)),
            "n_flows": int(sum(flows)),
            "captures_by_label": {0: int(sum(l == 0 for l in labels)), 1: int(sum(l == 1 for l in labels))},
            "flows_by_label": {
                0: int(sum(f for l, f in zip(labels, flows) if l == 0)),
                1: int(sum(f for l, f in zip(labels, flows) if l == 1)),
            },
        }

    return {"train": stats(s_train), "val": stats(s_val), "test": stats(s_test)}


if __name__ == "__main__":
    from src.utils.paths import load_paths

    paths = load_paths()
    flows_parquet = paths.data_processed / "vnat" / "flows.parquet"

    train_list = paths.data_splits / "vnat_train_captures.txt"
    val_list = paths.data_splits / "vnat_val_captures.txt"
    test_list = paths.data_splits / "vnat_test_captures.txt"

    splits_yaml = paths.configs_dir / "splits.yaml"

    out = validate_split_files(flows_parquet, train_list, val_list, test_list, splits_yaml=splits_yaml)
    print(json.dumps(out, indent=2))
