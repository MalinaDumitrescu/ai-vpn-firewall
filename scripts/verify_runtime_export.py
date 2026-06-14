"""Verify runtime export for unified_relative_shape_v2__lgbm and validate demo CSV.

Outputs:
  thesis_exports/runtime_export_inventory.csv
  thesis_exports/runtime_export_inventory.md
  thesis_exports/unified_model_demo_flows_validation.md
  thesis_exports/unified_model_demo_flows.csv      (only if no existing demo CSV)

No retraining, no overwriting of model artifacts, no threshold changes.
"""
from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
EXP_DIR      = PROJECT_ROOT / "artifacts" / "unified_feature_contract_v2"
RT_DIR       = EXP_DIR / "runtime_export"
RT_MODEL_DIR = RT_DIR / "runtime_models" / "unified_relative_shape_v2__lgbm"
DEMO_DIR     = RT_DIR / "demo_data"
THESIS_DIR   = EXP_DIR / "thesis_exports"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_12 = [
    "sz_cv", "sz_iqr", "sz_qratio", "sz_median_to_mean",
    "sz_p25_median_ratio", "sz_p75_median_ratio", "sz_iqr_norm_median",
    "iat_cv", "iat_iqr",
    "direction_balance_bytes", "direction_balance_packets", "dispersion_symmetry",
]

# =================================================================
# 1. Runtime export inventory
# =================================================================
INVENTORY = [
    # (artifact name,            relative path,                                                       purpose)
    ("model_artifact",           "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/model.pkl",            "Serialised LightGBM model"),
    ("calibrator",               "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/calibrator.pkl",       "Isotonic regression calibrator"),
    ("feature_order",            "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/feature_order.json",   "Authoritative 12-feature input order"),
    ("thresholds",               "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/thresholds.json",      "review / block thresholds (do NOT modify)"),
    ("feature_family",           "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/feature_family.json",  "Feature-family JSON (unified_relative_shape_v2)"),
    ("feature_contract",         "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/feature_contract.json", "Canonical formulas + extractor version"),
    ("extractor_config",         "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/extractor_config.json", "Live extractor configuration"),
    ("model_card",               "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/model_card.md",        "Model card (purpose, limits, contact)"),
    ("preprocessing_scaler",     "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/scaler.pkl",            "Optional sklearn scaler (none expected; LGBM does not need scaling)"),
    ("preprocessing_pipeline",   "runtime_export/runtime_models/unified_relative_shape_v2__lgbm/preprocessing.pkl",     "Optional preprocessing pipeline (none expected)"),
    ("runtime_readme",           "runtime_export/RUNTIME_README.md",                                                    "Runtime README + warnings"),
    ("runtime_requirements",     "runtime_export/requirements_runtime.txt",                                             "Python deps to run model"),
    ("runtime_smoke_test_out",   "runtime_export/smoke_test_output.txt",                                                "Smoke test log"),
    ("runtime_smoke_test_script","runtime_export/scripts/smoke_test_unified_model.py",                                  "Smoke test script"),
    ("app_registry_json",        "runtime_export/app_model_registry/unified_firewall_candidate.json",                   "App registry candidate entry"),
    ("app_registry_csv",         "runtime_export/app_model_registry/model_registry.csv",                                "App registry CSV"),
    ("demo_csv",                 "runtime_export/demo_data/unified_model_demo_flows.csv",                               "Demo CSV for app inference (Phase: demo)"),
    ("demo_manifest",            "runtime_export/demo_data/unified_model_demo_manifest.json",                           "Expected demo results"),
    ("demo_validation",          "runtime_export/demo_data/demo_csv_validation.md",                                     "Existing demo-CSV validation report"),
    ("demo_benchmark_csv",       "runtime_export/demo_data/unified_model_benchmark_flows.csv",                          "Benchmark CSV (same rows scored by family models)"),
]

inv_rows = []
notes_for_missing = {
    "preprocessing_scaler":   "Not required: LightGBM accepts unscaled numeric inputs.",
    "preprocessing_pipeline": "Not required: feature_order.json + raw 12 columns are sufficient.",
}
_BASE = "artifacts/unified_feature_contract_v2/"
for name, rel, purpose in INVENTORY:
    full_rel = _BASE + rel
    p = PROJECT_ROOT / full_rel
    exists = p.exists()
    note = "" if exists else notes_for_missing.get(name, "NOT FOUND on disk")
    size = p.stat().st_size if exists and p.is_file() else 0
    inv_rows.append({
        "artifact": name,
        "path": full_rel,
        "exists": "yes" if exists else "no",
        "size_bytes": size,
        "purpose": purpose,
        "notes": note,
    })

inv_df = pd.DataFrame(inv_rows)
inv_csv = THESIS_DIR / "runtime_export_inventory.csv"
inv_md  = THESIS_DIR / "runtime_export_inventory.md"
inv_df.to_csv(inv_csv, index=False)

md_lines = [
    "# Runtime export inventory — unified_relative_shape_v2__lgbm",
    "",
    "_Generated by `scripts/verify_runtime_export.py`._  ",
    "_No artifact was modified; this report only inspects the export bundle._",
    "",
    "| artifact | path | exists | size_bytes | purpose | notes |",
    "|---|---|---|---:|---|---|",
]
for r in inv_rows:
    md_lines.append("| {artifact} | `{path}` | {exists} | {size_bytes} | {purpose} | {notes} |".format(**r))
md_lines.append("")
md_lines.append(f"**Required artifacts present:** "
                f"{sum(1 for r in inv_rows if r['exists']=='yes')} / {len(inv_rows)}.")
inv_md.write_text("\n".join(md_lines), encoding="utf-8")

print(f"Saved inventory CSV: {inv_csv.relative_to(PROJECT_ROOT)}")
print(f"Saved inventory MD : {inv_md.relative_to(PROJECT_ROOT)}")
print(f"Required artifacts present: "
      f"{sum(1 for r in inv_rows if r['exists']=='yes')} / {len(inv_rows)}")

# =================================================================
# 2. Verify load + feature_order + thresholds
# =================================================================
print("\n=== Load verification ===")
issues = []

# feature_order
fo_path = RT_MODEL_DIR / "feature_order.json"
fo_obj  = json.loads(fo_path.read_text(encoding="utf-8"))
rt_features = fo_obj["features"] if isinstance(fo_obj, dict) else fo_obj
fo_missing = [f for f in EXPECTED_12 if f not in rt_features]
fo_extra   = [f for f in rt_features  if f not in EXPECTED_12]
fo_order_ok = (rt_features == EXPECTED_12)
print(f"  feature_order count   : {len(rt_features)}")
print(f"  missing               : {fo_missing or 'none'}")
print(f"  extra                 : {fo_extra or 'none'}")
print(f"  exact order match     : {fo_order_ok}")
if fo_missing or fo_extra or not fo_order_ok:
    issues.append("feature_order mismatch")

# model + calibrator
try:
    model      = joblib.load(RT_MODEL_DIR / "model.pkl")
    calibrator = joblib.load(RT_MODEL_DIR / "calibrator.pkl")
    print(f"  model loaded          : {type(model).__name__}")
    print(f"  calibrator loaded     : {type(calibrator).__name__}")
except Exception as e:
    issues.append(f"could not load model/calibrator: {e}")
    print("  LOAD FAILED:", e)
    raise

# thresholds
thr_path = RT_MODEL_DIR / "thresholds.json"
thr = json.loads(thr_path.read_text(encoding="utf-8"))
print(f"  thresholds            : {thr}")

# =================================================================
# 3. Validate the existing demo CSV
# =================================================================
print("\n=== Demo CSV validation ===")
demo_existing = DEMO_DIR / "unified_model_demo_flows.csv"
val_report = THESIS_DIR / "unified_model_demo_flows_validation.md"

def _validate_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    checks = {}
    # required features
    miss = [f for f in EXPECTED_12 if f not in df.columns]
    extra_feat = [f for f in df.columns if f in EXPECTED_12]
    checks["all_required_features_present"] = (len(miss) == 0, miss)
    # numeric
    non_numeric = []
    for f in EXPECTED_12:
        if f in df.columns:
            try:
                pd.to_numeric(df[f], errors="raise")
            except Exception:
                non_numeric.append(f)
    checks["all_feature_columns_numeric"] = (len(non_numeric) == 0, non_numeric)
    # NaN/inf
    nan_cols = [f for f in EXPECTED_12 if f in df.columns and df[f].isna().any()]
    inf_cols = [f for f in EXPECTED_12 if f in df.columns and np.isinf(df[f].astype(float)).any()]
    checks["no_nan_in_features"] = (len(nan_cols) == 0, nan_cols)
    checks["no_inf_in_features"] = (len(inf_cols) == 0, inf_cols)
    # optional metadata
    optional = ["session_id", "flow_id", "capture_id", "dataset", "label"]
    present_meta = [c for c in optional if c in df.columns]
    checks["optional_metadata_present"] = (True, present_meta)
    # mixed-label captures
    mixed = []
    if "capture_id" in df.columns and "label" in df.columns:
        per = df.groupby("capture_id")["label"].nunique()
        mixed = sorted(per[per > 1].index.astype(str).tolist())
    checks["no_mixed_label_captures"] = (len(mixed) == 0, mixed)
    return df, checks

def _score(df: pd.DataFrame) -> dict:
    X  = df[EXPECTED_12].astype("float32").to_numpy()
    raw = model.predict_proba(X)[:, 1]
    # also calibrated, for reporting
    try:
        cal = np.clip(np.asarray(calibrator.predict(raw), dtype=float), 0.0, 1.0)
    except Exception:
        cal = None
    action = np.where(raw >= thr["block_threshold"], "SIMULATED_BLOCK",
              np.where(raw >= thr["review_threshold"], "FLAG_REVIEW", "PASS"))
    out = {
        "n_rows": int(len(df)),
        "PASS":               int((action == "PASS").sum()),
        "FLAG_REVIEW":        int((action == "FLAG_REVIEW").sum()),
        "SIMULATED_BLOCK":    int((action == "SIMULATED_BLOCK").sum()),
        "raw_mean":           float(raw.mean()),
        "raw_max":            float(raw.max()),
        "cal_mean":           float(cal.mean()) if cal is not None else None,
    }
    if "label" in df.columns:
        y = df["label"].astype(int).to_numpy()
        pred = (raw >= thr["block_threshold"]).astype(int)
        tp = int(((y==1)&(pred==1)).sum()); fn = int(((y==1)&(pred==0)).sum())
        tn = int(((y==0)&(pred==0)).sum()); fp = int(((y==0)&(pred==1)).sum())
        out.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "recall": tp/(tp+fn) if (tp+fn) else float("nan"),
                    "fpr":    fp/(fp+tn) if (fp+tn) else float("nan")})
    return out

if demo_existing.exists():
    df_demo, checks = _validate_csv(demo_existing)
    score = _score(df_demo)
    used_source = str(demo_existing.relative_to(PROJECT_ROOT))
    created_now = False
else:
    # Create one from the test split of unified_flows.parquet
    print("  no existing demo CSV — creating one from test split")
    src_parq = EXP_DIR / "data" / "unified_flows.parquet"
    full = pd.read_parquet(src_parq)
    test = full[full["split"] == "test"].copy()
    # pick a handful of full captures with both labels covered
    keep_cols = EXPECTED_12 + [c for c in ["dataset","capture_id","flow_id","session_id","label"] if c in test.columns]
    # pick the first 10 captures balanced by label
    by_cap = test.groupby("capture_id")["label"].mean()
    vpn_caps   = list(by_cap[by_cap > 0.9].index[:6])
    benign_caps= list(by_cap[by_cap < 0.1].index[:4])
    cap_keep = vpn_caps + benign_caps
    new = test[test["capture_id"].isin(cap_keep)][keep_cols].dropna(subset=EXPECTED_12).copy()
    new = new.replace([np.inf, -np.inf], np.nan).dropna(subset=EXPECTED_12)
    out_csv = THESIS_DIR / "unified_model_demo_flows.csv"
    new.to_csv(out_csv, index=False)
    df_demo, checks = _validate_csv(out_csv)
    score = _score(df_demo)
    used_source = str(out_csv.relative_to(PROJECT_ROOT))
    created_now = True

print(f"  source CSV            : {used_source}")
print(f"  rows                  : {score['n_rows']}")
for k, (ok, det) in checks.items():
    flag = "OK" if ok else "FAIL"
    print(f"  {k:<40} {flag}   ({det if not ok else det if isinstance(det,list) and det else ''})")
print(f"  action dist           : PASS={score['PASS']} REVIEW={score['FLAG_REVIEW']} BLOCK={score['SIMULATED_BLOCK']}")
if "TP" in score:
    print(f"  label-aware           : TP={score['TP']} FP={score['FP']} TN={score['TN']} FN={score['FN']}  "
          f"recall={score['recall']:.4f} fpr={score['fpr']:.4f}")

# --- Save validation MD ---
def _check_line(name, ok, detail):
    sym = "✅" if ok else "❌"
    if isinstance(detail, list):
        det_txt = "" if not detail else f" — {detail}"
    else:
        det_txt = f" — {detail}"
    return f"- {sym} **{name}**{det_txt}"

md = [
    "# Demo CSV validation — `unified_model_demo_flows.csv`",
    "",
    f"_Validated against runtime model `unified_relative_shape_v2__lgbm`._",
    f"_Source CSV: `{used_source}`._",
    f"_{'Created by this script from the test split of unified_flows.parquet.' if created_now else 'Pre-existing demo CSV (validated in place; not modified).'}_",
    "",
    "## Structural checks",
    "",
    _check_line("all 12 required features present",            checks["all_required_features_present"][0], checks["all_required_features_present"][1]),
    _check_line("all feature columns numeric",                 checks["all_feature_columns_numeric"][0],   checks["all_feature_columns_numeric"][1]),
    _check_line("no NaN in feature columns",                   checks["no_nan_in_features"][0],            checks["no_nan_in_features"][1]),
    _check_line("no ±Inf in feature columns",                  checks["no_inf_in_features"][0],            checks["no_inf_in_features"][1]),
    _check_line("optional metadata columns present",           checks["optional_metadata_present"][0],     checks["optional_metadata_present"][1]),
    _check_line("no mixed-label captures",                     checks["no_mixed_label_captures"][0],       checks["no_mixed_label_captures"][1]),
    "",
    "## Inference test (runtime bundle)",
    "",
    f"- **Rows scored:** {score['n_rows']:,}",
    f"- **Block threshold:** {thr['block_threshold']}",
    f"- **Review threshold:** {thr['review_threshold']}",
    f"- **Action distribution:** PASS = {score['PASS']:,}, FLAG_REVIEW = {score['FLAG_REVIEW']:,}, "
    f"SIMULATED_BLOCK = {score['SIMULATED_BLOCK']:,}",
    f"- **raw mean / max:** {score['raw_mean']:.4f} / {score['raw_max']:.4f}",
]
if "TP" in score:
    md += [
        f"- **Label-aware (block threshold):** TP = {score['TP']}, FP = {score['FP']}, "
        f"TN = {score['TN']}, FN = {score['FN']}",
        f"- **recall = {score['recall']:.4f}**, **FPR = {score['fpr']:.4f}** "
        f"_(figure-time numbers — do not cite as production metrics)_",
    ]
md += [
    "",
    "## Caveats (mandatory for the thesis)",
    "",
    "- This demo CSV is for **input-compatibility testing only**.",
    "- It **must not** be used to claim deployment performance.",
    "- The `label` column is metadata; it must **not** be passed as a model input.",
    "- The `capture_id`, `flow_id`, `session_id`, `dataset` columns are metadata; they must not be passed to the model.",
    "- Thresholds are read-only from `runtime_export/runtime_models/unified_relative_shape_v2__lgbm/thresholds.json`.",
]
val_report.write_text("\n".join(md), encoding="utf-8")
print(f"\nSaved validation report: {val_report.relative_to(PROJECT_ROOT)}")
print("Issues:", issues or "none")

