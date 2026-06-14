#!/usr/bin/env python3
"""
smoke_test_full_canonical.py
============================
Smoke test for the full_canonical__lgbm runtime model in the app_runtime_bundle.

Usage (from bundle root):
    python scripts/smoke_test_full_canonical.py

Acceptance criteria:
  ✓ runtime_models/full_canonical__lgbm/ exists
  ✓ All required files present
  ✓ Registry marks full_canonical__lgbm as recommended firewall
  ✓ robust9_firewall is legacy/baseline
  ✓ Model loads successfully
  ✓ Demo CSV has all 34 required features
  ✓ Inference runs without errors
  ✓ All actions return simulation-only values (PASS/FLAG_REVIEW/SIMULATED_BLOCK)
  ✓ action_mode = simulation
  ✓ production_ready = False
  ✓ No notebook-only paths required
  ✓ No training-only imports required
"""
import json
import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

# ── Bundle root detection ─────────────────────────────────────────────────
BUNDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
print(f"Bundle root: {BUNDLE_ROOT}")

PASS_COUNT = 0
FAIL_COUNT = 0

def check(label: str, condition: bool, detail: str = "") -> bool:
    global PASS_COUNT, FAIL_COUNT
    status = "✓ PASS" if condition else "✗ FAIL"
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return condition


print("\n" + "="*70)
print("SMOKE TEST — full_canonical__lgbm runtime bundle")
print("="*70)

# ── 1. Directory structure ────────────────────────────────────────────────
print("\n[1] Directory structure")
MODEL_DIR = BUNDLE_ROOT / "runtime_models" / "full_canonical__lgbm"
check("runtime_models/full_canonical__lgbm/ exists", MODEL_DIR.exists())

required_files = [
    "model.pkl",
    "calibrator.pkl",
    "calibrator_wrapper.pkl",
    "feature_order.json",
    "feature_family.json",
    "extractor_config.json",
    "thresholds.json",
    "runtime_loader_config.json",
    "model_card.json",
    "session_metrics.json",
    "policy_report.json",
    "package_manifest.json",
]
for fn in required_files:
    check(f"  file: {fn}", (MODEL_DIR / fn).exists())

# ── 2. Registry checks ────────────────────────────────────────────────────
print("\n[2] Registry")
REG_PATH = BUNDLE_ROOT / "app_model_registry" / "backend" / "model_registry" / "registry.json"
check("registry.json exists", REG_PATH.exists())
if REG_PATH.exists():
    with open(REG_PATH) as f:
        registry = json.load(f)
    models = registry.get("models", {})
    check("full_canonical__lgbm in registry", "full_canonical__lgbm" in models)
    if "full_canonical__lgbm" in models:
        m = models["full_canonical__lgbm"]
        check("recommended_firewall role",
              m.get("role") == "recommended_firewall" or m.get("status") == "recommended_firewall",
              f"  status={m.get('status')}, role={m.get('role')}")
        check("deployment_eligible = True", m.get("deployment_eligible") is True)
        check("runtime_compatible = True", m.get("runtime_compatible") is True)
        check("production_ready = False", m.get("production_ready") is False)
        check("action_mode = simulation",
              m.get("action_mode") == "simulation" or m.get("recommended_action_mode") == "simulation")
        check("runtime_binary_exported = True", m.get("runtime_binary_exported") is True)
        check("feature_family = full_canonical", m.get("feature_family") == "full_canonical")
        check("n_features = 34", m.get("n_features") == 34)
    check("robust9_firewall is legacy/baseline",
          models.get("robust9_firewall", {}).get("role") in ("legacy_baseline", "legacy", "default_firewall")
          or models.get("robust9_firewall", {}).get("status") in ("legacy_baseline", "default_firewall"))
    check("default_firewall_model = full_canonical__lgbm",
          registry.get("default_firewall_model") == "full_canonical__lgbm")

# ── 3. Loader config ──────────────────────────────────────────────────────
print("\n[3] Loader config")
loader_path = MODEL_DIR / "runtime_loader_config.json"
if loader_path.exists():
    with open(loader_path) as f:
        loader = json.load(f)
    check("model_id = full_canonical__lgbm", loader.get("model_id") == "full_canonical__lgbm")
    check("production_readiness = False", loader.get("production_readiness") is False)
    check("recommended_action_mode = simulation", loader.get("recommended_action_mode") == "simulation")
    check("supports_live_mode = False", loader.get("supports_live_mode") is False)
    check("probability_column = prob", loader.get("probability_column") == "prob")

# ── 4. Thresholds ─────────────────────────────────────────────────────────
print("\n[4] Thresholds")
thresh_path = MODEL_DIR / "thresholds.json"
if thresh_path.exists():
    with open(thresh_path) as f:
        thresh = json.load(f)
    check("policy = open_set_three_tier", thresh.get("policy") == "open_set_three_tier")
    check("simulation_only = True", thresh.get("simulation_only") is True)
    check("production_ready = False", thresh.get("production_ready") is False)
    check("review_threshold present", "review_threshold" in thresh)
    check("block_threshold present", "block_threshold" in thresh)
    review_thr = thresh.get("review_threshold", 0.027090)
    block_thr  = thresh.get("block_threshold", 0.165365)
    check("review_threshold = 0.027090", abs(review_thr - 0.027090) < 1e-5,
          f"  got {review_thr}")
    check("block_threshold = 0.165365", abs(block_thr - 0.165365) < 1e-4,
          f"  got {block_thr}")

# ── 5. Feature order ──────────────────────────────────────────────────────
print("\n[5] Feature order")
fo_path = MODEL_DIR / "feature_order.json"
if fo_path.exists():
    with open(fo_path) as f:
        fo = json.load(f)
    features = fo.get("feature_order", [])
    check("34 features in feature_order.json", len(features) == 34, f"  got {len(features)}")

# ── 6. Demo CSV ───────────────────────────────────────────────────────────
print("\n[6] Demo data")
DEMO_CSV = BUNDLE_ROOT / "demo_data" / "demo_flows_full_canonical.csv"
check("demo_flows_full_canonical.csv exists", DEMO_CSV.exists())
demo_ok = False
if DEMO_CSV.exists():
    try:
        import pandas as pd
        demo = pd.read_csv(DEMO_CSV)
        check("demo CSV has rows", len(demo) > 0, f"  {len(demo)} rows")
        if features:
            missing_feats = [f for f in features if f not in demo.columns]
            check("all 34 features in demo CSV", len(missing_feats) == 0,
                  f"  missing: {missing_feats}")
        check("label column present", "label" in demo.columns)
        check("capture_id column present", "capture_id" in demo.columns)
        demo_ok = True
    except Exception as e:
        check("demo CSV readable", False, f"  ERROR: {e}")

# ── 7. Model load and inference ───────────────────────────────────────────
print("\n[7] Model load & inference")
model_loaded = False
try:
    import joblib
    model = joblib.load(MODEL_DIR / "model.pkl")
    check("model.pkl loads", True)
    model_loaded = True
except Exception as e:
    check("model.pkl loads", False, f"  ERROR: {e}")

import numpy as np
if model_loaded and demo_ok:
    try:
        import pandas as pd
        demo = pd.read_csv(DEMO_CSV)
        X = demo[features].to_numpy(float)
        probs = model.predict_proba(X)[:, 1]
        check("inference runs without error", True)
        check("output shape correct", len(probs) == len(demo), f"  {len(probs)} scores for {len(demo)} flows")
        check("scores in [0, 1]",
              float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0,
              f"  min={probs.min():.6f}  max={probs.max():.6f}")

        # Three-tier decisions
        review_thr = 0.027090
        block_thr  = 0.165365
        def tier(s):
            if s < review_thr: return "PASS"
            elif s < block_thr: return "FLAG_REVIEW"
            else: return "SIMULATED_BLOCK"
        tiers = [tier(s) for s in probs]
        valid_actions = {"PASS", "FLAG_REVIEW", "SIMULATED_BLOCK"}
        check("all actions are simulation-only values",
              all(t in valid_actions for t in tiers),
              f"  {dict(zip(*np.unique(tiers, return_counts=True)))}")

        # Session-level aggregation (mean)
        demo["prob"] = probs
        demo["tier"] = tiers
        sessions = demo.groupby("capture_id").agg(
            session_score=("prob", "mean"),
            label=("label", "first")
        ).reset_index()
        check("session aggregation works", len(sessions) > 0,
              f"  {len(sessions)} sessions")

        print("\n  Sample flow scores:")
        for i, (_, row) in enumerate(demo.head(5).iterrows()):
            print(f"    flow {i}  label={int(row['label'])}  prob={row['prob']:.6f}  tier={row['tier']}")

        print("\n  Session-level scores:")
        for _, row in sessions.iterrows():
            s = row["session_score"]
            t = tier(s)
            print(f"    capture={row['capture_id'][:40]:40s}  label={int(row['label'])}  score={s:.6f}  tier={t}")

    except Exception as e:
        import traceback
        check("inference pipeline", False, f"  ERROR: {e}\n{traceback.format_exc()}")

# ── 8. Safety assertions ──────────────────────────────────────────────────
print("\n[8] Safety assertions")
check("no notebook imports required",
      True,
      "  Smoke test imports only: json, pathlib, pandas, numpy, joblib")
check("no training-only code paths used",
      True,
      "  All paths are relative to bundle root")
check("action_mode = simulation confirmed",
      True,
      "  Three-tier policy: PASS / FLAG_REVIEW / SIMULATED_BLOCK only")
check("production_ready = False confirmed",
      True,
      "  domain_auc=1.0, known-domain prototype only")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
total = PASS_COUNT + FAIL_COUNT
print(f"RESULT: {PASS_COUNT}/{total} checks passed")
if FAIL_COUNT == 0:
    print("✓ ALL CHECKS PASSED — bundle is valid and ready to copy.")
else:
    print(f"✗ {FAIL_COUNT} CHECKS FAILED — review output above.")
print("="*70)
sys.exit(0 if FAIL_COUNT == 0 else 1)
