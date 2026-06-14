"""
Generate and validate the demo CSV for unified_relative_shape_v2__lgbm.

Run from project root:
    python scripts/generate_demo_csv.py

Output:
    artifacts/unified_feature_contract_v2/runtime_export/demo_data/
        unified_model_demo_flows.csv
        unified_model_demo_manifest.json
        demo_csv_validation.md

Does NOT retrain, does NOT overwrite old demo CSVs.
All rows come from the test split of unified_flows.parquet.
"""
import json
import textwrap
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
BASE       = ROOT / "artifacts" / "unified_feature_contract_v2"
RE         = BASE / "runtime_export"
MODEL_DIR  = RE / "runtime_models" / "unified_relative_shape_v2__lgbm"
DEMO_DIR   = RE / "demo_data"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

DEMO_CSV      = DEMO_DIR / "unified_model_demo_flows.csv"
BENCH_CSV     = DEMO_DIR / "unified_model_benchmark_flows.csv"
MANIFEST_JSON = DEMO_DIR / "unified_model_demo_manifest.json"
VALIDATION_MD = DEMO_DIR / "demo_csv_validation.md"

MODEL_ID  = "unified_relative_shape_v2__lgbm"
FAMILY    = "unified_relative_shape_v2"
DATA_PATH = BASE / "data" / "unified_flows.parquet"

# ── Load model metadata ───────────────────────────────────────────────────
feat_data  = json.load(open(MODEL_DIR / "feature_order.json"))
FEATURES   = feat_data["features"]            # 12 features
thr        = json.load(open(MODEL_DIR / "thresholds.json"))
REVIEW_THR = thr["review_threshold"]
BLOCK_THR  = thr["block_threshold"]

print(f"[gen_demo] Model:     {MODEL_ID}")
print(f"[gen_demo] Features:  {len(FEATURES)} → {FEATURES}")
print(f"[gen_demo] Thresholds: review={REVIEW_THR:.4f}  block={BLOCK_THR:.4f}")

# ── Load model artifacts ──────────────────────────────────────────────────
clf = joblib.load(MODEL_DIR / "model.pkl")
iso = joblib.load(MODEL_DIR / "calibrator.pkl")

def score_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add vpn_score and decision columns."""
    X = df[FEATURES].values
    p_raw = clf.predict_proba(X)[:, 1]
    p_cal = iso.predict(p_raw)
    df = df.copy()
    df["vpn_score_raw"]  = p_raw
    df["vpn_score"]      = p_cal
    df["decision"] = [
        "SIMULATED_BLOCK" if p >= BLOCK_THR else
        ("FLAG_REVIEW"    if p >= REVIEW_THR else "PASS")
        for p in p_cal
    ]
    return df

# ── Load unified flows — TEST SPLIT ONLY ─────────────────────────────────
full = pd.read_parquet(DATA_PATH)
test = full[full["split"] == "test"].copy()
print(f"\n[gen_demo] Test split: {len(test):,} flows  "
      f"(VPN={int((test['label']==1).sum())}  "
      f"benign={int((test['label']==0).sum())})")
print(f"[gen_demo] Captures in test: {test['capture_id'].nunique()}")

# Verify no NaN/inf in required features
for col in FEATURES:
    n_nan = test[col].isna().sum()
    n_inf = np.isinf(test[col]).sum()
    if n_nan or n_inf:
        print(f"  WARNING: {col}  NaN={n_nan}  inf={n_inf} — filling")
        test[col] = test[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ── Identify per-capture label purity (should already be clean) ───────────
cap_labels = test.groupby("capture_id")["label"].nunique()
mixed = cap_labels[cap_labels > 1].index
if len(mixed):
    print(f"  WARNING: dropping {len(mixed)} mixed-label captures ({test[test['capture_id'].isin(mixed)].shape[0]} flows)")
    test = test[~test["capture_id"].isin(mixed)]

# ── Score all test flows ──────────────────────────────────────────────────
test_scored = score_df(test)

# ─────────────────────────────────────────────────────────────────────────
# DEMO CSV selection strategy
# ─────────────────────────────────────────────────────────────────────────
# Goals:
#   • At least 3 VPN captures per dataset that has VPN (USBVPN, VNAT)
#   • At least 3 benign captures per dataset (all 3)
#   • Include one "difficult" VPN capture (highest VPN flows scored PASS = FN)
#   • Include one review-band capture for realism
#   • Cap total rows at ~600 for a clean demo (not overwhelming)
#   • Prefer captures where model makes correct decisions (clean TP/TN)
# ─────────────────────────────────────────────────────────────────────────

METADATA_COLS = ["flow_id", "capture_id", "dataset", "label",
                 "q_packet_count", "q_min_packets_ok"]

# Per-capture summary
cap_summary = (
    test_scored.groupby(["capture_id", "dataset", "label"])
    .agg(
        n_flows=("flow_id", "count"),
        vpn_score_mean=("vpn_score", "mean"),
        vpn_score_max=("vpn_score", "max"),
        n_block=("decision", lambda x: (x == "SIMULATED_BLOCK").sum()),
        n_review=("decision", lambda x: (x == "FLAG_REVIEW").sum()),
        n_pass=("decision", lambda x: (x == "PASS").sum()),
    )
    .reset_index()
)
cap_summary["tp_rate"]  = np.where(
    cap_summary["label"] == 1,
    cap_summary["n_block"] / cap_summary["n_flows"].clip(1),
    0.0
)
cap_summary["tn_rate"]  = np.where(
    cap_summary["label"] == 0,
    cap_summary["n_pass"] / cap_summary["n_flows"].clip(1),
    0.0
)

# ── Select VPN captures ───────────────────────────────────────────────────
vpn_caps = cap_summary[cap_summary["label"] == 1].copy()
vpn_caps = vpn_caps[vpn_caps["n_flows"] >= 3]  # at least 3 flows

selected_captures = []

# Per dataset: best TP captures
for ds in ["usbvpn", "vnat", "iscx"]:
    subset = vpn_caps[vpn_caps["dataset"] == ds].sort_values("tp_rate", ascending=False)
    # Take top-3 TP captures
    for _, row in subset.head(3).iterrows():
        selected_captures.append(row["capture_id"])

# One difficult FN capture (VPN capture where many flows are misclassified)
fn_caps = vpn_caps[vpn_caps["n_flows"] >= 5].sort_values("tp_rate", ascending=True)
if len(fn_caps) > 0:
    fn_cap = fn_caps.iloc[0]["capture_id"]
    if fn_cap not in selected_captures:
        selected_captures.append(fn_cap)
        print(f"[gen_demo] Added difficult FN capture: {fn_cap}  tp_rate={fn_caps.iloc[0]['tp_rate']:.3f}")

# ── Select benign captures ────────────────────────────────────────────────
ben_caps = cap_summary[cap_summary["label"] == 0].copy()
ben_caps = ben_caps[ben_caps["n_flows"] >= 3]

for ds in ["usbvpn", "vnat", "iscx"]:
    subset = ben_caps[ben_caps["dataset"] == ds].sort_values("tn_rate", ascending=False)
    for _, row in subset.head(3).iterrows():
        selected_captures.append(row["capture_id"])

# One review-band capture (benign capture with some FLAG_REVIEW flows)
review_ben = (
    ben_caps[ben_caps["n_review"] > 0]
    .sort_values("n_review", ascending=False)
)
if len(review_ben) > 0:
    rev_cap = review_ben.iloc[0]["capture_id"]
    if rev_cap not in selected_captures:
        selected_captures.append(rev_cap)
        print(f"[gen_demo] Added review-band benign capture: {rev_cap}  n_review={int(review_ben.iloc[0]['n_review'])}")

# Deduplicate
selected_captures = list(dict.fromkeys(selected_captures))
print(f"\n[gen_demo] Selected {len(selected_captures)} captures")

# ── Build demo DataFrame ──────────────────────────────────────────────────
demo = test_scored[test_scored["capture_id"].isin(selected_captures)].copy()

# Cap rows at 600 by sampling captures if needed
MAX_ROWS = 600
if len(demo) > MAX_ROWS:
    # Sort captures by dataset+label for balanced trimming
    cap_order = demo.groupby("capture_id").first().sort_values(["dataset", "label"]).index.tolist()
    kept = []
    running = 0
    for cap in cap_order:
        cap_rows = demo[demo["capture_id"] == cap]
        if running + len(cap_rows) <= MAX_ROWS:
            kept.append(cap)
            running += len(cap_rows)
    selected_captures = kept
    demo = demo[demo["capture_id"].isin(selected_captures)].copy()
    print(f"[gen_demo] Trimmed to {len(demo)} rows across {len(selected_captures)} captures (MAX_ROWS={MAX_ROWS})")

# ── Build output columns ──────────────────────────────────────────────────
OUTPUT_COLS = (
    ["flow_id", "capture_id", "dataset", "label", "q_packet_count"]
    + FEATURES
    + ["vpn_score_raw", "vpn_score", "decision"]
)
# Keep only columns that exist
OUTPUT_COLS = [c for c in OUTPUT_COLS if c in demo.columns]
demo_out = demo[OUTPUT_COLS].reset_index(drop=True)

# Final NaN/inf check
for col in FEATURES:
    demo_out[col] = pd.to_numeric(demo_out[col], errors="coerce")
    bad = demo_out[col].isna() | np.isinf(demo_out[col])
    if bad.any():
        print(f"  WARNING: {col} has {bad.sum()} bad values — replacing with 0")
        demo_out.loc[bad, col] = 0.0

demo_out.to_csv(DEMO_CSV, index=False)
print(f"\n[gen_demo] Wrote demo CSV: {DEMO_CSV.relative_to(ROOT)}")
print(f"  Rows:     {len(demo_out)}")
print(f"  Captures: {demo_out['capture_id'].nunique()}")
print(f"  Datasets: {demo_out['dataset'].value_counts().to_dict()}")
print(f"  Labels:   {demo_out['label'].value_counts().to_dict()}")
print(f"  Actions:  {demo_out['decision'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────
# BENCHMARK CSV — all models sharing unified_relative_shape_v2 features
# ─────────────────────────────────────────────────────────────────────────
# Load all models in the family and score the same rows
models_dir = BASE / "models"
bench_rows = demo[["flow_id","capture_id","dataset","label"] + FEATURES].copy().reset_index(drop=True)
bench_model_cols = {}

for model_subdir in sorted(models_dir.iterdir()):
    if not model_subdir.is_dir():
        continue
    fo_path = model_subdir / "feature_order.json"
    if not fo_path.exists():
        continue
    try:
        fo = json.load(open(fo_path))
        model_feats = fo.get("features", [])
    except Exception:
        continue
    # Only include models whose features are a subset of what we have
    if not all(f in FEATURES for f in model_feats):
        continue
    try:
        m_clf = joblib.load(model_subdir / "model.pkl")
        X_bench = bench_rows[model_feats].values
        p_raw   = m_clf.predict_proba(X_bench)[:, 1]
        mid = model_subdir.name
        bench_model_cols[mid] = p_raw
        print(f"[bench]  Scored: {mid}")
    except Exception as e:
        print(f"[bench]  SKIP {model_subdir.name}: {e}")

if bench_model_cols:
    bench_out = bench_rows.copy()
    for mid, scores in bench_model_cols.items():
        bench_out[f"score__{mid}"] = scores
    bench_out.to_csv(BENCH_CSV, index=False)
    print(f"\n[gen_demo] Wrote benchmark CSV: {BENCH_CSV.relative_to(ROOT)}")
    print(f"  Models scored: {len(bench_model_cols)}")
    print(f"  Rows: {len(bench_out)}")
else:
    print("[gen_demo] No compatible benchmark models found.")

# ─────────────────────────────────────────────────────────────────────────
# MANIFEST JSON
# ─────────────────────────────────────────────────────────────────────────
label_dist   = demo_out["label"].value_counts().to_dict()
action_dist  = demo_out["decision"].value_counts().to_dict()
dataset_dist = demo_out["dataset"].value_counts().to_dict()

# TP/FP/TN/FN
vpn_rows = demo_out[demo_out["label"] == 1]
ben_rows = demo_out[demo_out["label"] == 0]
TP = int((vpn_rows["decision"] == "SIMULATED_BLOCK").sum())
FN = int(len(vpn_rows) - TP)
TN = int((ben_rows["decision"] == "PASS").sum())
FP = int(len(ben_rows) - TN)
FP_review = int((ben_rows["decision"] == "FLAG_REVIEW").sum())
TP_review = int((vpn_rows["decision"] == "FLAG_REVIEW").sum())

# Per-capture details
cap_details = (
    demo_out.groupby(["capture_id","dataset","label"])
    .agg(
        n_flows=("flow_id","count"),
        mean_score=("vpn_score","mean"),
        decisions=("decision", lambda x: Counter(x).most_common()),
    )
    .reset_index()
)
cap_list = cap_details.to_dict("records")
for row in cap_list:
    row["decisions"] = dict(row["decisions"])
    row["n_flows"] = int(row["n_flows"])
    row["mean_score"] = round(float(row["mean_score"]), 4)
    row["label"] = int(row["label"])

manifest = {
    "schema_version": "1.0",
    "created": "2026-05-30",
    "csv_filename": DEMO_CSV.name,
    "benchmark_csv_filename": BENCH_CSV.name if bench_model_cols else None,
    "selected_model_id": MODEL_ID,
    "feature_family": FAMILY,
    "feature_count": len(FEATURES),
    "features": FEATURES,
    "split_used": "test",
    "row_count": len(demo_out),
    "capture_count": int(demo_out["capture_id"].nunique()),
    "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
    "dataset_distribution": {str(k): int(v) for k, v in dataset_dist.items()},
    "expected_action_distribution": {str(k): int(v) for k, v in action_dist.items()},
    "expected_confusion": {
        "TP_SIMULATED_BLOCK": TP,
        "FN_PASS_or_REVIEW":  FN,
        "TN_PASS":            TN,
        "FP_FLAGGED_or_BLOCK": FP,
        "FP_FLAGGED_REVIEW":  FP_review,
        "TP_FLAGGED_REVIEW":  TP_review,
    },
    "thresholds": {
        "review_threshold": REVIEW_THR,
        "block_threshold":  BLOCK_THR,
        "policy": "PASS/FLAG_REVIEW/SIMULATED_BLOCK"
    },
    "selected_captures": cap_list,
    "metadata_columns": ["flow_id","capture_id","dataset","label","q_packet_count"],
    "model_input_columns": FEATURES,
    "output_columns": ["vpn_score_raw","vpn_score","decision"],
    "intended_use": "App demo/testing only. Simulation mode.",
    "warnings": [
        "SIMULATION ONLY: No packets are blocked or modified.",
        "NOT PRODUCTION-READY: Research prototype for academic evaluation only.",
        "All rows from test split only — no training data included.",
        "Feature values extracted from real captured traffic (USBVPN/ISCX/VNAT datasets).",
        "Do not use VPN label column as a model input — it is metadata only.",
        "Compare with legacy full_canonical__lgbm before replacing app default.",
        "Demo CSV is designed for offline testing; live traffic results may differ."
    ]
}

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\n[gen_demo] Wrote manifest: {MANIFEST_JSON.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────
# VALIDATION REPORT (Markdown)
# ─────────────────────────────────────────────────────────────────────────
def _fmt_bool(b): return "✅ PASS" if b else "❌ FAIL"

checks = {}

# 1. All required features present
checks["all_features_present"] = all(f in demo_out.columns for f in FEATURES)
missing_feats = [f for f in FEATURES if f not in demo_out.columns]

# 2. No NaN in feature columns
nan_counts = {f: int(demo_out[f].isna().sum()) for f in FEATURES}
checks["no_nan_in_features"] = all(v == 0 for v in nan_counts.values())

# 3. No inf in feature columns
inf_counts = {f: int(np.isinf(demo_out[f]).sum()) for f in FEATURES}
checks["no_inf_in_features"] = all(v == 0 for v in inf_counts.values())

# 4. All feature columns numeric
checks["all_features_numeric"] = all(
    pd.api.types.is_numeric_dtype(demo_out[f]) for f in FEATURES
)
non_numeric = [f for f in FEATURES if not pd.api.types.is_numeric_dtype(demo_out[f])]

# 5. No mixed-label captures
cap_lab_nu = demo_out.groupby("capture_id")["label"].nunique()
checks["no_mixed_label_captures"] = bool((cap_lab_nu == 1).all())
mixed_caps = cap_lab_nu[cap_lab_nu > 1].index.tolist()

# 6. Model can score the CSV
try:
    _X = demo_out[FEATURES].values
    _p = clf.predict_proba(_X)[:, 1]
    checks["model_inference_ok"] = True
except Exception as _e:
    checks["model_inference_ok"] = False
    print(f"  FAIL: model inference: {_e}")

# 7. Actions span all expected categories (PASS, FLAG_REVIEW, SIMULATED_BLOCK)
has_pass    = (demo_out["decision"] == "PASS").any()
has_review  = (demo_out["decision"] == "FLAG_REVIEW").any()
has_block   = (demo_out["decision"] == "SIMULATED_BLOCK").any()
checks["has_all_action_types"] = bool(has_pass and has_block)  # review optional

# 8. Both label classes present
checks["has_both_label_classes"] = len(demo_out["label"].unique()) == 2

# 9. Metadata columns not in FEATURES
meta_cols = ["flow_id","capture_id","dataset","label","q_packet_count"]
checks["metadata_not_in_features"] = not any(m in FEATURES for m in meta_cols)

overall_pass = all(checks.values())

def feat_nan_table():
    rows = [f"| `{f}` | {nan_counts[f]} | {inf_counts[f]} |" for f in FEATURES]
    return "\n".join(rows)

validation_report = textwrap.dedent(f"""\
    # Demo CSV Validation Report

    **CSV**: `{DEMO_CSV.name}`  
    **Model**: `{MODEL_ID}`  
    **Generated**: 2026-05-30  
    **Overall**: {"✅ ALL CHECKS PASSED" if overall_pass else "❌ SOME CHECKS FAILED"}

    ---

    ## Validation Checks

    | Check | Result |
    |-------|--------|
    | All required features present | {_fmt_bool(checks["all_features_present"])} |
    | No NaN in feature columns | {_fmt_bool(checks["no_nan_in_features"])} |
    | No inf in feature columns | {_fmt_bool(checks["no_inf_in_features"])} |
    | All feature columns numeric | {_fmt_bool(checks["all_features_numeric"])} |
    | No mixed-label captures | {_fmt_bool(checks["no_mixed_label_captures"])} |
    | Model inference succeeds | {_fmt_bool(checks["model_inference_ok"])} |
    | Actions span PASS + SIMULATED_BLOCK | {_fmt_bool(checks["has_all_action_types"])} |
    | FLAG_REVIEW actions present | {"✅ Yes" if has_review else "ℹ️ None (optional)"} |
    | Both label classes present | {_fmt_bool(checks["has_both_label_classes"])} |
    | Metadata columns not used as features | {_fmt_bool(checks["metadata_not_in_features"])} |

    ---

    ## CSV Structure

    | Property | Value |
    |----------|-------|
    | Total rows | {len(demo_out):,} |
    | Captures | {int(demo_out["capture_id"].nunique())} |
    | Datasets | {", ".join(f"{k}={v}" for k, v in dataset_dist.items())} |
    | VPN flows (label=1) | {int((demo_out["label"]==1).sum())} |
    | Benign flows (label=0) | {int((demo_out["label"]==0).sum())} |
    | Total columns | {len(demo_out.columns)} |
    | Feature columns | {len(FEATURES)} |
    | Metadata columns | flow_id, capture_id, dataset, label, q_packet_count |
    | Output columns | vpn_score_raw, vpn_score, decision |

    ---

    ## Expected Model Results

    | Action | Count |
    |--------|-------|
    | SIMULATED_BLOCK | {action_dist.get("SIMULATED_BLOCK", 0)} |
    | FLAG_REVIEW | {action_dist.get("FLAG_REVIEW", 0)} |
    | PASS | {action_dist.get("PASS", 0)} |

    ### Confusion (VPN flows)

    | Outcome | Count |
    |---------|-------|
    | TP — VPN correctly SIMULATED_BLOCK | {TP} |
    | FN — VPN incorrectly PASS or FLAG_REVIEW | {FN} |
    | TP_review — VPN in FLAG_REVIEW band | {TP_review} |

    ### Confusion (Benign flows)

    | Outcome | Count |
    |---------|-------|
    | TN — Benign correctly PASS | {TN} |
    | FP — Benign incorrectly flagged | {FP} |
    | FP_review — Benign in FLAG_REVIEW band | {FP_review} |

    ---

    ## Feature NaN / Inf Counts

    | Feature | NaN | Inf |
    |---------|-----|-----|
    {feat_nan_table()}

    ---

    ## Missing Features

    {"None — all required features present." if not missing_feats else "- " + chr(10).join(f"`{f}`" for f in missing_feats)}

    ---

    ## Non-numeric Features

    {"None — all feature columns are numeric." if not non_numeric else "- " + chr(10).join(f"`{f}`" for f in non_numeric)}

    ---

    ## Mixed-label Captures

    {"None detected." if not mixed_caps else "WARNING: " + ", ".join(mixed_caps)}

    ---

    ## Usage

    ```python
    import pandas as pd, joblib, json
    from pathlib import Path

    BASE = Path("runtime_export/runtime_models/{MODEL_ID}")
    clf  = joblib.load(BASE / "model.pkl")
    iso  = joblib.load(BASE / "calibrator.pkl")
    feat = json.load(open(BASE / "feature_order.json"))["features"]
    thr  = json.load(open(BASE / "thresholds.json"))

    df = pd.read_csv("runtime_export/demo_data/{DEMO_CSV.name}")
    X  = df[feat].values
    p_raw = clf.predict_proba(X)[:, 1]
    p_cal = iso.predict(p_raw)

    def action(p):
        if p >= thr["block_threshold"]:  return "SIMULATED_BLOCK"
        if p >= thr["review_threshold"]: return "FLAG_REVIEW"
        return "PASS"

    df["vpn_score"] = p_cal
    df["decision"]  = [action(p) for p in p_cal]
    print(df[["capture_id","label","vpn_score","decision"]].head(10))
    ```

    ---

    ## Warnings

    - ⚠️ **SIMULATION ONLY** — no packets are blocked or modified
    - ⚠️ **NOT PRODUCTION-READY** — research prototype only
    - All rows from test split; no training data included
    - `label` column is metadata only; do not use as model input
""")

VALIDATION_MD.write_text(validation_report, encoding="utf-8")
print(f"[gen_demo] Wrote validation report: {VALIDATION_MD.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
DEMO CSV GENERATION COMPLETE
{'='*60}
  Model:        {MODEL_ID}
  Rows:         {len(demo_out)}
  Captures:     {demo_out['capture_id'].nunique()}
  VPN flows:    {int((demo_out['label']==1).sum())}
  Benign flows: {int((demo_out['label']==0).sum())}
  Datasets:     {demo_out['dataset'].value_counts().to_dict()}
  Actions:      {demo_out['decision'].value_counts().to_dict()}
  TP (block):   {TP}   FN: {FN}   TN: {TN}   FP: {FP}
  All checks:   {"PASSED" if overall_pass else "FAILED"}
{'='*60}
""")

