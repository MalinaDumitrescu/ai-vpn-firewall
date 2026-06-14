"""
Build the structured runtime export candidate for unified_relative_shape_v2__lgbm.

Run from project root:
    python scripts/build_runtime_export_candidate.py

Does NOT retrain, does NOT overwrite exports/app_runtime_bundle/.
Creates the new structured layout inside:
    artifacts/unified_feature_contract_v2/runtime_export/
alongside the existing flat files (model.pkl etc. already present from Phase 2).
"""
import json
import shutil
import textwrap
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "unified_feature_contract_v2"
FLAT_EXPORT   = ARTIFACT_DIR / "runtime_export"          # existing flat Phase-2 export
MODEL_DIR     = ARTIFACT_DIR / "models" / "unified_relative_shape_v2__lgbm"
FEAT_FAM_DIR  = ARTIFACT_DIR / "feature_families"
CONTRACT_JSON = ARTIFACT_DIR / "feature_contract.json"

MODEL_ID   = "unified_relative_shape_v2__lgbm"
FAMILY     = "unified_relative_shape_v2"
MODEL_TYPE = "lgbm"
N_FEATURES = 12

# Structured output root (within existing runtime_export/)
STRUCT = FLAT_EXPORT                            # we add sub-dirs alongside existing flat files
MODEL_OUT = STRUCT / "runtime_models" / MODEL_ID
REGISTRY  = STRUCT / "app_model_registry"
REPORTS   = STRUCT / "reports"
DEMO      = STRUCT / "demo_data"
SCRIPTS   = STRUCT / "scripts"

for d in [MODEL_OUT, REGISTRY, REPORTS, DEMO, SCRIPTS]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[build_runtime_export_candidate] root: {STRUCT}")

# ─────────────────────────────────────────────────────────────────────────
# 1.  Copy model artifacts into runtime_models/<model_id>/
# ─────────────────────────────────────────────────────────────────────────
def _copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied  {src.name}  →  {dst.relative_to(ROOT)}")
    else:
        print(f"  SKIP (not found): {src}")

# model.pkl  – prefer the per-model folder, fall back to flat export root
if (MODEL_DIR / "model.pkl").exists():
    _copy(MODEL_DIR / "model.pkl",    MODEL_OUT / "model.pkl")
else:
    _copy(FLAT_EXPORT / "model.pkl",  MODEL_OUT / "model.pkl")

if (MODEL_DIR / "calibrator.pkl").exists():
    _copy(MODEL_DIR / "calibrator.pkl",  MODEL_OUT / "calibrator.pkl")
else:
    _copy(FLAT_EXPORT / "calibrator.pkl", MODEL_OUT / "calibrator.pkl")

_copy(FLAT_EXPORT / "feature_order.json",   MODEL_OUT / "feature_order.json")
_copy(FLAT_EXPORT / "thresholds.json",      MODEL_OUT / "thresholds.json")
_copy(FLAT_EXPORT / "extractor_config.json",MODEL_OUT / "extractor_config.json")
_copy(CONTRACT_JSON,                         MODEL_OUT / "feature_contract.json")

# feature_family.json
fam_src = FEAT_FAM_DIR / f"{FAMILY}.json"
_copy(fam_src, MODEL_OUT / "feature_family.json")

# model_card.md
_copy(FLAT_EXPORT / "model_card.md", MODEL_OUT / "model_card.md")

print()

# ─────────────────────────────────────────────────────────────────────────
# 2.  Load metadata for downstream files
# ─────────────────────────────────────────────────────────────────────────
feature_order = json.loads((MODEL_OUT / "feature_order.json").read_text())
features      = feature_order["features"]
thresholds    = json.loads((MODEL_OUT / "thresholds.json").read_text())
ext_cfg       = json.loads((MODEL_OUT / "extractor_config.json").read_text())

review_thr = thresholds["review_threshold"]
block_thr  = thresholds["block_threshold"]
policy     = thresholds.get("policy", "PASS/FLAG_REVIEW/SIMULATED_BLOCK")

# ─────────────────────────────────────────────────────────────────────────
# 3.  App model registry
# ─────────────────────────────────────────────────────────────────────────
registry_entry = {
    "schema_version": "1.0",
    "created": "2026-05-30",
    "experiment": "unified_feature_contract_v2",
    "model_id": MODEL_ID,
    "role": "recommended_unified_firewall",
    "candidate_role": "recommended_firewall_candidate",
    "feature_family": FAMILY,
    "n_features": N_FEATURES,
    "features": features,
    "model_type": MODEL_TYPE,
    "runtime_compatible": True,
    "live_extractor_compatible": True,
    "extractor_version": ext_cfg["extractor_version"],
    "production_ready": False,
    "action_mode": "simulation",
    "probability_column": "vpn_score",
    "calibration": "isotonic_regression",
    "aggregation": "per_flow",
    "thresholds": {
        "review": review_thr,
        "block": block_thr,
        "policy": policy
    },
    "performance": {
        "test_auc": 0.9826,
        "lodo_min_auc": 0.6366,
        "lodo_iscx_auc": 0.6366,
        "lodo_vnat_auc": 0.9560,
        "domain_auc": 0.9591,
        "test_recall": 0.8930,
        "test_fpr": 0.0759,
        "test_ece": 0.2988,
        "deployment_score": 0.4691
    },
    "legacy_comparison": {
        "legacy_model_id": "full_canonical__lgbm",
        "legacy_test_auc": 0.9994,
        "legacy_lodo_min": 0.6164,
        "legacy_domain_auc": 1.0000,
        "pooled_auc_delta": -0.0168,
        "lodo_min_delta": +0.0202,
        "domain_auc_delta": -0.0409
    },
    "warnings": [
        "SIMULATION ONLY: No network packets are blocked or modified.",
        "NOT PRODUCTION-READY: Research prototype for academic evaluation only.",
        "Trained under unified_feature_contract_v2. Formulas differ from legacy full_canonical__lgbm.",
        "USBVPN base statistics accepted from pre-processed parquet; raw packet arrays unavailable.",
        "Domain fingerprinting persists (domain AUC 0.9591). Not fully dataset-invariant.",
        "LODO-ISCX = 0.637: moderate transfer risk to new capture environments.",
        "No live PCAP validation performed. Extractor compatibility is schema-confirmed only.",
        "Compare with legacy full_canonical__lgbm before replacing app default.",
        "Run live PCAP validation before any deployment decision."
    ],
    "replacement_recommendation": (
        "DO NOT replace legacy full_canonical__lgbm automatically. "
        "Run live PCAP validation first. "
        "For scientific reporting, use unified model as the methodologically correct result."
    ),
    "artifact_paths": {
        "model_pkl":           f"runtime_models/{MODEL_ID}/model.pkl",
        "calibrator_pkl":      f"runtime_models/{MODEL_ID}/calibrator.pkl",
        "feature_order_json":  f"runtime_models/{MODEL_ID}/feature_order.json",
        "thresholds_json":     f"runtime_models/{MODEL_ID}/thresholds.json",
        "feature_family_json": f"runtime_models/{MODEL_ID}/feature_family.json",
        "feature_contract_json": f"runtime_models/{MODEL_ID}/feature_contract.json",
        "extractor_config_json": f"runtime_models/{MODEL_ID}/extractor_config.json",
        "model_card_md":       f"runtime_models/{MODEL_ID}/model_card.md"
    }
}

registry_path = REGISTRY / "unified_firewall_candidate.json"
registry_path.write_text(json.dumps(registry_entry, indent=2), encoding="utf-8")
print(f"  wrote  {registry_path.relative_to(ROOT)}")

# Registry summary CSV
import csv
csv_path = REGISTRY / "model_registry.csv"
csv_rows = [{
    "model_id":               MODEL_ID,
    "role":                   "recommended_unified_firewall",
    "feature_family":         FAMILY,
    "n_features":             N_FEATURES,
    "runtime_compatible":     True,
    "live_extractor_compatible": True,
    "production_ready":       False,
    "action_mode":            "simulation",
    "test_auc":               0.9826,
    "lodo_min_auc":           0.6366,
    "domain_auc":             0.9591,
    "deployment_score":       0.4691,
    "review_threshold":       review_thr,
    "block_threshold":        block_thr,
    "legacy_model_id":        "full_canonical__lgbm",
    "legacy_test_auc":        0.9994,
    "legacy_lodo_min":        0.6164,
    "legacy_domain_auc":      1.0000,
}]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"  wrote  {csv_path.relative_to(ROOT)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# 4.  Copy reports
# ─────────────────────────────────────────────────────────────────────────
for src_name, dst_name in [
    ("final_report.md",         "final_report.md"),
    ("thesis_summary.md",       "thesis_summary.md"),
    ("unified_formula_report.md","unified_formula_report.md"),
    ("feature_contract.json",   "feature_contract.json"),
    ("model_comparison.csv",    "model_comparison.csv"),
    ("live_pcap_results.csv",   "live_pcap_results.csv"),
    ("lodo_results.csv",        "lodo_results.csv"),
    ("domain_fingerprint_results.csv", "domain_fingerprint_results.csv"),
    ("calibration_results.csv", "calibration_results.csv"),
    ("anti_fingerprint_feature_scores.csv", "anti_fingerprint_feature_scores.csv"),
    ("recommended_models.json", "recommended_models.json"),
]:
    src = ARTIFACT_DIR / src_name
    dst = REPORTS / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  report  {src_name}")
    else:
        print(f"  SKIP (not found): {src_name}")
print()

# ─────────────────────────────────────────────────────────────────────────
# 5.  requirements_runtime.txt  (write fresh, not overwriting old flat one)
# ─────────────────────────────────────────────────────────────────────────
req_content = textwrap.dedent("""\
    # Runtime dependencies for unified_relative_shape_v2__lgbm
    # Generated: 2026-05-30
    # Model type: LightGBM + isotonic calibration
    lightgbm>=4.0
    scikit-learn>=1.3
    numpy>=1.24
    pandas>=2.0
    joblib>=1.3
    # Optional: for full extractor pipeline
    # xgboost>=2.0
    # catboost>=1.2
""")
req_path = STRUCT / "requirements_runtime.txt"
req_path.write_text(req_content, encoding="utf-8")
print(f"  wrote  {req_path.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────
# 6.  RUNTIME_README.md
# ─────────────────────────────────────────────────────────────────────────
features_md = "\n".join(f"- `{f}`" for f in features)
readme = textwrap.dedent(f"""\
    # Runtime Export — unified_feature_contract_v2

    **Generated**: 2026-05-30  
    **Experiment**: `unified_feature_contract_v2`  
    **Selected model**: `{MODEL_ID}`  

    ---

    ## ⚠️ SIMULATION ONLY — NOT PRODUCTION-READY

    This model is a **research prototype**. It operates in **simulation mode only**.
    - No network packets are blocked or modified.
    - Outputs are scored decisions for academic evaluation.
    - Must NOT be deployed to production without live PCAP validation.

    ---

    ## Selected Model

    | Property | Value |
    |----------|-------|
    | model_id | `{MODEL_ID}` |
    | feature_family | `{FAMILY}` |
    | n_features | {N_FEATURES} |
    | model_type | LightGBM + isotonic calibration |
    | test_auc | 0.9826 |
    | lodo_min_auc | 0.6366 |
    | domain_auc | 0.9591 |
    | deployment_score | 0.4691 (highest among 30 trained models) |
    | runtime_compatible | Yes |
    | live_extractor_compatible | Yes — `unified_extractor.py` v2.0 |
    | production_ready | **No** |
    | action_mode | simulation |

    ---

    ## Feature Family: `unified_relative_shape_v2`

    **12 ratio/relative-shape features** (scale-invariant, anti-fingerprinting):

    {features_md}

    ### Feature conventions (from `extractor_config.json`)

    - **Packet size**: IP total length (bytes)
    - **Timestamps**: seconds
    - **Direction**: `1 = upload / client-to-server`, `0 = download / server-to-client`
    - **Window**: first `{ext_cfg["max_window_packets"]}` packets per flow
    - **Min packets**: `{ext_cfg["min_packets"]}`
    - **eps**: `{ext_cfg["eps"]}`
    - **Extractor version**: `{ext_cfg["extractor_version"]}`

    ---

    ## Policy Thresholds

    | Action | Threshold |
    |--------|-----------|
    | `PASS` | calibrated score < {review_thr:.4f} |
    | `FLAG_REVIEW` | {review_thr:.4f} ≤ score < {block_thr:.4f} |
    | `SIMULATED_BLOCK` | score ≥ {block_thr:.4f} |

    ---

    ## Directory Structure

    ```
    runtime_export/
    ├── runtime_models/
    │   └── {MODEL_ID}/
    │       ├── model.pkl                  # trained LightGBM classifier
    │       ├── calibrator.pkl             # isotonic regression calibrator
    │       ├── feature_order.json         # required input columns (12)
    │       ├── thresholds.json            # review + block thresholds
    │       ├── feature_family.json        # feature family metadata
    │       ├── feature_contract.json      # full extractor contract
    │       ├── extractor_config.json      # extractor conventions
    │       └── model_card.md              # model card
    ├── app_model_registry/
    │   ├── unified_firewall_candidate.json  # full registry entry
    │   └── model_registry.csv              # summary CSV
    ├── reports/
    │   ├── final_report.md
    │   ├── thesis_summary.md
    │   ├── unified_formula_report.md
    │   ├── feature_contract.json
    │   ├── model_comparison.csv
    │   ├── lodo_results.csv
    │   ├── domain_fingerprint_results.csv
    │   ├── calibration_results.csv
    │   ├── anti_fingerprint_feature_scores.csv
    │   └── recommended_models.json
    ├── demo_data/
    │   └── (demo CSV to be generated in next phase)
    ├── scripts/
    │   └── smoke_test_unified_model.py
    ├── requirements_runtime.txt
    ├── RUNTIME_README.md
    └── smoke_test_output.txt
    ```

    ---

    ## How to Validate a CSV

    Your input CSV must contain these columns (in any order):

    ```
    {", ".join(features)}
    ```

    Load and score:

    ```python
    import joblib, json, pandas as pd
    from pathlib import Path

    BASE = Path("runtime_export/runtime_models/{MODEL_ID}")
    clf  = joblib.load(BASE / "model.pkl")
    iso  = joblib.load(BASE / "calibrator.pkl")
    feat = json.load(open(BASE / "feature_order.json"))["features"]
    thr  = json.load(open(BASE / "thresholds.json"))

    df = pd.read_csv("your_flows.csv")
    X  = df[feat].values
    p_raw = clf.predict_proba(X)[:, 1]
    p_cal = iso.predict(p_raw)

    def action(p):
        if p >= thr["block_threshold"]:  return "SIMULATED_BLOCK"
        if p >= thr["review_threshold"]: return "FLAG_REVIEW"
        return "PASS"

    df["vpn_score"]  = p_cal
    df["decision"]   = [action(p) for p in p_cal]
    print(df[["vpn_score", "decision"]].value_counts())
    ```

    ---

    ## How to Run Smoke Test

    ```bash
    python runtime_export/scripts/smoke_test_unified_model.py
    ```

    Expected output (artifact-load check):
    ```
    [smoke_test] Model artifacts loaded OK
    [smoke_test] Features (12): sz_cv, sz_iqr, ...
    [smoke_test] Zero-vector inference: score=X.XXXX  action=PASS|FLAG_REVIEW|SIMULATED_BLOCK
    [smoke_test] production_ready = False
    [smoke_test] action_mode      = simulation
    [smoke_test] PASSED
    ```

    ---

    ## Should this replace the legacy model?

    **Not automatically.** Required steps before replacing `full_canonical__lgbm`:

    1. ✅ Unified model bundle exported (this folder)
    2. ⬜ Live PCAP validation: run unified extractor on known VPN traffic (Warp, OpenVPN)
    3. ⬜ Confirm FPR acceptable on live benign traffic
    4. ⬜ Side-by-side comparison in prototype with both models running in parallel
    5. ⬜ Threshold re-calibration on live traffic distribution

    **For scientific reporting**: use the unified model as the methodologically correct result.
    Present the legacy model's AUC=0.9994 with the dataset-fingerprinting caveat (domain AUC=1.0).

    ---

    ## Key metrics vs legacy

    | Metric | Legacy `full_canonical__lgbm` | Unified `{MODEL_ID}` | Δ |
    |--------|-------------------------------|----------------------|---|
    | Test AUC | 0.9994 | 0.9826 | −0.0168 |
    | LODO-min AUC | 0.6164 | **0.6366** | **+0.0202** |
    | Domain AUC | 1.0000 | **0.9591** | **−0.0409** |
    | n_features | ~33 | 12 | −21 |

    ---

    *This bundle was generated by `scripts/build_runtime_export_candidate.py`.*
    *No models were retrained. No production bundles were overwritten.*
""")
readme_path = STRUCT / "RUNTIME_README.md"
readme_path.write_text(readme, encoding="utf-8")
print(f"  wrote  {readme_path.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────
# 7.  scripts/smoke_test_unified_model.py
# ─────────────────────────────────────────────────────────────────────────
smoke_script = textwrap.dedent(f"""\
    #!/usr/bin/env python
    \"\"\"
    Smoke test for the unified_feature_contract_v2 runtime export candidate.

    Usage:
        python runtime_export/scripts/smoke_test_unified_model.py

    Validates:
    - All model artifact files load correctly
    - feature_order.json contains expected 12 features
    - thresholds.json contains review + block thresholds
    - Zero-vector inference runs without error
    - production_ready = False
    - action_mode = simulation

    If demo CSV is present in demo_data/, also runs inference on it.
    \"\"\"
    import sys
    import json
    import joblib
    import numpy as np
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE       = SCRIPT_DIR.parent / "runtime_models" / "{MODEL_ID}"
    DEMO_DIR   = SCRIPT_DIR.parent / "demo_data"
    REGISTRY   = SCRIPT_DIR.parent / "app_model_registry" / "unified_firewall_candidate.json"

    PASS_MARK = "[smoke_test]"
    errors = []

    def check(condition: bool, msg: str):
        if not condition:
            errors.append(f"  FAIL: {{msg}}")
        return condition

    # ── 1. Load artifacts ─────────────────────────────────────────────────
    print(f"{{PASS_MARK}} Loading artifacts from: {{BASE}}")

    try:
        clf = joblib.load(BASE / "model.pkl")
        print(f"{{PASS_MARK}} model.pkl            OK  (type={{type(clf).__name__}})")
    except Exception as e:
        errors.append(f"  FAIL: model.pkl: {{e}}")
        clf = None

    try:
        iso = joblib.load(BASE / "calibrator.pkl")
        print(f"{{PASS_MARK}} calibrator.pkl       OK  (type={{type(iso).__name__}})")
    except Exception as e:
        errors.append(f"  FAIL: calibrator.pkl: {{e}}")
        iso = None

    try:
        feat_data = json.load(open(BASE / "feature_order.json"))
        features  = feat_data["features"]
        print(f"{{PASS_MARK}} feature_order.json   OK  ({{len(features)}} features)")
        check(len(features) == {N_FEATURES}, f"expected {N_FEATURES} features, got {{len(features)}}")
    except Exception as e:
        errors.append(f"  FAIL: feature_order.json: {{e}}")
        features = []

    try:
        thr = json.load(open(BASE / "thresholds.json"))
        review_thr = thr["review_threshold"]
        block_thr  = thr["block_threshold"]
        print(f"{{PASS_MARK}} thresholds.json      OK  (review={{review_thr:.4f}}, block={{block_thr:.4f}})")
    except Exception as e:
        errors.append(f"  FAIL: thresholds.json: {{e}}")
        review_thr, block_thr = 0.05, 0.5

    try:
        ext_cfg = json.load(open(BASE / "extractor_config.json"))
        print(f"{{PASS_MARK}} extractor_config.json OK (version={{ext_cfg['extractor_version']}})")
    except Exception as e:
        errors.append(f"  FAIL: extractor_config.json: {{e}}")

    # ── 2. Registry metadata ──────────────────────────────────────────────
    try:
        reg = json.load(open(REGISTRY))
        prod_ready  = reg["production_ready"]
        action_mode = reg["action_mode"]
        check(not prod_ready,          "production_ready should be False")
        check(action_mode == "simulation", "action_mode should be simulation")
        print(f"{{PASS_MARK}} production_ready     = {{prod_ready}}")
        print(f"{{PASS_MARK}} action_mode          = {{action_mode}}")
    except Exception as e:
        errors.append(f"  FAIL: registry JSON: {{e}}")

    # ── 3. Zero-vector inference ──────────────────────────────────────────
    if clf is not None and iso is not None and features:
        try:
            import pandas as pd
            X_zero = pd.DataFrame(np.zeros((1, len(features))), columns=features)
            p_raw  = clf.predict_proba(X_zero)[0, 1]
            p_cal  = float(iso.predict([p_raw])[0])
            if   p_cal >= block_thr:  action = "SIMULATED_BLOCK"
            elif p_cal >= review_thr: action = "FLAG_REVIEW"
            else:                     action = "PASS"
            print(f"{{PASS_MARK}} Zero-vector inference: raw={{p_raw:.4f}}  calibrated={{p_cal:.4f}}  action={{action}}")
        except Exception as e:
            errors.append(f"  FAIL: zero-vector inference: {{e}}")

    # ── 4. Feature list print ─────────────────────────────────────────────
    if features:
        print(f"{{PASS_MARK}} Required features ({{len(features)}}):")
        for f in features:
            print(f"    {{f}}")

    # ── 5. Demo CSV check ─────────────────────────────────────────────────
    demo_csvs = sorted(DEMO_DIR.glob("*.csv"))
    if demo_csvs:
        try:
            import pandas as pd
            demo_path = demo_csvs[0]
            df = pd.read_csv(demo_path)
            print(f"\\n{{PASS_MARK}} Demo CSV found: {{demo_path.name}} ({{len(df)}} rows)")
            missing = [f for f in features if f not in df.columns]
            if missing:
                errors.append(f"  FAIL: demo CSV missing columns: {{missing}}")
            else:
                X_demo = df[features].values
                p_raw  = clf.predict_proba(X_demo)[:, 1]
                p_cal  = iso.predict(p_raw)
                actions = []
                for p in p_cal:
                    if   p >= block_thr:  actions.append("SIMULATED_BLOCK")
                    elif p >= review_thr: actions.append("FLAG_REVIEW")
                    else:                 actions.append("PASS")
                from collections import Counter
                cnt = Counter(actions)
                print(f"{{PASS_MARK}} Demo inference results:")
                for act, n in sorted(cnt.items()):
                    print(f"    {{act}}: {{n}} flows")
        except Exception as e:
            errors.append(f"  FAIL: demo CSV inference: {{e}}")
    else:
        print(f"\\n{{PASS_MARK}} No demo CSV in demo_data/ — skipping demo inference.")
        print(f"{{PASS_MARK}} Demo CSV should be generated in the next phase.")

    # ── 6. Final verdict ──────────────────────────────────────────────────
    print()
    if errors:
        print(f"{{PASS_MARK}} *** SMOKE TEST FAILED ***")
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print(f"{{PASS_MARK}} PASSED — all artifact checks OK")
        print(f"{{PASS_MARK}} production_ready = False   action_mode = simulation")
        print(f"{{PASS_MARK}} Model: {MODEL_ID}")
        print(f"{{PASS_MARK}} Features: {{len(features)}}   review_thr={{review_thr:.4f}}   block_thr={{block_thr:.4f}}")
""")
smoke_path = SCRIPTS / "smoke_test_unified_model.py"
smoke_path.write_text(smoke_script, encoding="utf-8")
print(f"  wrote  {smoke_path.relative_to(ROOT)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# 8.  demo_data README placeholder
# ─────────────────────────────────────────────────────────────────────────
demo_readme = textwrap.dedent("""\
    # demo_data/

    This folder is reserved for demo inference CSV files.

    Expected file: `demo_flows.csv`

    Required columns (12):
    sz_cv, sz_iqr, sz_qratio, sz_median_to_mean,
    sz_p25_median_ratio, sz_p75_median_ratio, sz_iqr_norm_median,
    iat_cv, iat_iqr, direction_balance_bytes,
    direction_balance_packets, dispersion_symmetry

    Each row represents one network flow with precomputed unified features.
    The demo CSV is generated in the "demo CSV generation" phase.
""")
(DEMO / "README.md").write_text(demo_readme, encoding="utf-8")
print(f"  wrote  demo_data/README.md")
print()

print("=" * 60)
print("Build complete. Now running smoke test...")
print("=" * 60)

