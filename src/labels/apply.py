from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib

import pandas as pd

from src.labels.vnat import label_from_filename as vnat_label_from_filename
from src.labels.iscx import label_from_filename as iscx_label_from_filename
from src.labels.validate import validate_labels_df


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_id_col(df: pd.DataFrame) -> str:
    for c in ("file_names", "capture_id", "capture_name"):
        if c in df.columns:
            return c
    raise ValueError(f"Need one of file_names/capture_id/capture_name. Got: {list(df.columns)}")


def apply_labels_to_flows_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Adds deterministic label metadata columns based on naming:
      - label_derived (int 0/1)
      - label_name (vpn/nonvpn)
      - label_rule (audit string)
      - app_derived
      - vpn_type (optional, currently None)
      - label_match (bool, only if df already has 'label')
    """
    id_col = _pick_id_col(df)

    def _label_one(x: Any) -> Dict[str, Any]:
        s = str(x)
        if dataset == "vnat":
            lab = vnat_label_from_filename(s)
        elif dataset == "iscx":
            lab = iscx_label_from_filename(s)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        d = asdict(lab)
        # normalize names for column outputs
        return {
            "label_derived": int(d["label"]),
            "label_name": str(d["label_name"]),
            "label_rule": str(d["rule"]),
            "app_derived": str(d["app"]),
            "vpn_type": d.get("vpn_type", None),
        }

    out = df.copy()
    tmp = out[id_col].map(_label_one).apply(pd.Series)
    out = pd.concat([out, tmp], axis=1)

    if "label" in out.columns:
        out["label_match"] = (out["label"].astype("int64") == out["label_derived"].astype("int64"))

    return out


def apply_labels_to_flows_parquet(
    flows_parquet: Path,
    dataset: str,
    out_parquet: Optional[Path] = None,
    out_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Loads flows parquet, applies labels, validates consistency, writes outputs.
    Returns manifest dict (also written to json).
    """
    flows_parquet = Path(flows_parquet)
    if not flows_parquet.exists():
        raise FileNotFoundError(f"Missing flows parquet: {flows_parquet}")

    if out_parquet is None:
        out_parquet = flows_parquet.parent / "labeled_flows.parquet"
    if out_manifest is None:
        out_manifest = flows_parquet.parent / "labeled_flows_manifest.json"

    df = pd.read_parquet(flows_parquet)

    labeled = apply_labels_to_flows_df(df, dataset=dataset)

    # Hard validation: if df already has label, it must match derived
    # (your current pipelines already build labels, so this enforces no drift)
    _ = validate_labels_df(labeled, dataset=dataset, require_label_col=("label" in labeled.columns))

    labeled.to_parquet(out_parquet, index=False)

    # manifest
    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "input": {
            "flows_parquet": str(flows_parquet.resolve()),
            "flows_sha256": _sha256_file(flows_parquet),
            "rows": int(len(df)),
            "columns": list(df.columns),
        },
        "output": {
            "labeled_flows_parquet": str(out_parquet.resolve()),
            "labeled_flows_sha256": _sha256_file(out_parquet),
            "rows": int(len(labeled)),
            "columns": list(labeled.columns),
        },
        "derived": {
            "label_counts": labeled["label_derived"].value_counts().to_dict(),
            "apps_top20": labeled["app_derived"].value_counts().head(20).to_dict(),
            "rules": labeled["label_rule"].value_counts().to_dict(),
        },
    }

    if "label" in labeled.columns:
        manifest["consistency"] = {
            "label_match_rate": float(labeled["label_match"].mean()),
            "mismatches": int((~labeled["label_match"]).sum()),
        }

    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
