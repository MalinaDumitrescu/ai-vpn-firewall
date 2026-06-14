"""Run the audit notebook's logic directly to materialize the figure + summary."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean_pipeline.validation import (
    FORBIDDEN_FEATURE_SUBSTRINGS, METADATA_COLS,
    capture_overlap_matrix, ensure_policy_threshold_provenance,
    ensure_preprocessing_metadata, hash_feature_rows,
    load_features_dataframe, load_split_capture_sets,
    locate_clean_artifacts, model_feature_columns,
)
from src.clean_pipeline.validation.preprocessing_metadata import stats_match_train

paths = locate_clean_artifacts()
paths.assert_minimum_present()
print("Repo:", paths.repo_root)

# 1. Overlap matrix + figure
splits = load_split_capture_sets(paths)
mat = capture_overlap_matrix(splits)
print("Overlap matrix:\n", mat)

fig_dir = paths.repo_root / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)
off_diag = mat.values.astype(float).copy()
np.fill_diagonal(off_diag, np.nan)

fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=140)
vmax = max(1, int(np.nanmax(off_diag)) if np.any(np.isfinite(off_diag)) else 1)
im = ax.imshow(off_diag, cmap="Reds", vmin=0, vmax=vmax)
for i, _ in enumerate(mat.index):
    for j, _ in enumerate(mat.columns):
        v = int(mat.iloc[i, j])
        if i == j:
            ax.text(j, i, f"{v}\n(captures)", ha="center", va="center",
                    color="#444", fontsize=9)
        else:
            color = "white" if v > 0 else "#666"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=11)
ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
ax.set_title("Pairwise capture overlap between splits\n(off-diagonal must be 0)")
ax.set_xlabel("split B"); ax.set_ylabel("split A")
plt.colorbar(im, ax=ax, label="|A \u2229 B| (captures)")
plt.tight_layout()
out_png = fig_dir / "capture_overlap_matrix.png"
plt.savefig(out_png, bbox_inches="tight")
plt.close(fig)
print("Saved figure:", out_png)

# 2. Duplicate rows
df = load_features_dataframe(paths)
feat_cols = model_feature_columns(df)
row_hash = hash_feature_rows(df, feat_cols)
work = df[["split", "capture_id", "dataset"]].copy()
work["row_hash"] = row_hash.values
splits_per_hash = work.groupby("row_hash")["split"].nunique()
cross_hashes = splits_per_hash[splits_per_hash > 1].index
cross = work[work["row_hash"].isin(cross_hashes)]
print(f"Cross-split duplicate hashes: {len(cross_hashes)}; rows: {len(cross)}")

# 3. Threshold provenance
prov = ensure_policy_threshold_provenance(paths)
print("global_policy_fit_split:", prov.get("global_policy_fit_split"))

# 4. Preprocessing metadata
meta = ensure_preprocessing_metadata(paths)
mism = stats_match_train(meta, df)
print(f"meta fit_split={meta.fit_split}, mismatches={len(mism)}")

# 5. Forbidden substrings
violations = []
for col in feat_cols:
    cl = col.lower()
    for sub in FORBIDDEN_FEATURE_SUBSTRINGS:
        if sub in cl:
            violations.append({"column": col, "matched_substring": sub})
            break
print(f"Forbidden-substring violations: {len(violations)}")

# Summary
rows = [
    {"check_name": "no_capture_overlap_between_splits",
     "status": "PASS" if (mat.values.sum() - np.trace(mat.values)) == 0 else "FAIL",
     "details": f"off-diagonal sum = {int(mat.values.sum() - np.trace(mat.values))}",
     "counts": json.dumps({"train": int(mat.loc['train','train']),
                            "val":   int(mat.loc['val','val']),
                            "test":  int(mat.loc['test','test'])})},
    {"check_name": "no_exact_duplicate_feature_rows_across_splits",
     "status": "PASS" if len(cross_hashes) == 0 else "FAIL",
     "details": f"{len(cross_hashes)} hashes, {len(cross)} rows span >1 split",
     "counts": json.dumps({"n_rows": int(len(df)),
                            "n_cross_hashes": int(len(cross_hashes)),
                            "n_cross_rows": int(len(cross))})},
    {"check_name": "thresholds_are_validation_derived_only",
     "status": "PASS" if str(prov.get("global_policy_fit_split","")).lower() in {"val","validation"} else "FAIL",
     "details": f"global={prov.get('global_policy_fit_split')}, policies={list(prov['policies'].keys())}",
     "counts": json.dumps({"n_policies": len(prov["policies"])})},
    {"check_name": "scaler_is_fit_only_on_training_data",
     "status": "PASS" if (meta.fit_split == "train" and not mism) else "FAIL",
     "details": f"fit_split={meta.fit_split}, transformer={meta.transformer_type}, mismatches={len(mism)}",
     "counts": json.dumps({"n_features": len(meta.feature_columns)})},
    {"check_name": "no_metadata_columns_in_model_features",
     "status": "PASS" if (not violations and not set(feat_cols).intersection(METADATA_COLS)) else "FAIL",
     "details": f"{len(violations)} substring violations",
     "counts": json.dumps({"n_violations": len(violations)})},
]
summary_dir = paths.repo_root / "artifacts" / "validation"
summary_dir.mkdir(parents=True, exist_ok=True)
(summary_dir / "data_leakage_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_csv(summary_dir / "data_leakage_summary.csv", index=False)
print("Wrote summary to:", summary_dir)
print(pd.DataFrame(rows).to_string(index=False))
