# AI-VPN-Firewall — Standalone Runtime Bundle

**Updated:** 2026-05-29  
**Default recommended model:** `full_canonical__lgbm`  
**Legacy baseline:** `robust9_firewall`

---

## ⚠️ Safety Notice

**This is a simulation-only research prototype.**

- **No packets are ever blocked.** All SIMULATED_BLOCK decisions are log/audit only.
- `production_ready = false` for every model in this bundle.
- **Dataset fingerprinting is unresolved** (`domain_auc = 1.0` for all models).
  The models can predict dataset origin with perfect accuracy, making them unsuitable
  for deployment to unseen network environments.
- DANN v2 adversarial training did not reduce fingerprinting (`domain_reduction ≈ 0.0003`).
- All actions are labelled: `PASS`, `FLAG_REVIEW`, or `SIMULATED_BLOCK`.

---

## What this is

A self-contained bundle that lets the FastAPI + React demo app run on any machine
without re-training and without the original datasets. It contains:

- `app_model_registry/` — validated metadata registry
- `runtime_models/full_canonical__lgbm/` — **final recommended** model binaries
- `runtime_models/robust9_firewall/` — legacy baseline binaries (still deployable)
- `demo_data/` — sample CSVs for smoke-testing
- `scripts/` — smoke test and utilities
- `reports/` — evaluation reports
- `requirements_runtime.txt` — runtime-only dependencies

---

## Default model: `full_canonical__lgbm`

The default recommended firewall model from the `final_transfer` experiment.

| Property | Value |
|---|---|
| **Algorithm** | LightGBM (single model) |
| **Feature family** | `full_canonical` (34 features) |
| **Pooled AUC** | 0.9994 |
| **LODO-min AUC** | 0.6164 |
| **Domain AUC** | 1.0 (fingerprinting unresolved) |
| **FPR @ 0.5** | 0.0025 |
| **ECE** | 0.0026 |
| **Deployment score** | 0.6836 |
| **Policy** | Open-set three-tier |
| **Probability column** | `prob` (raw LightGBM output) |
| **Session aggregation** | mean per `capture_id` |

### Three-tier policy thresholds

```
Score range                   Action
─────────────────────────────────────────────────────────────
0.000000 – 0.027089   →   PASS              (confident benign)
0.027090 – 0.165364   →   FLAG_REVIEW       (uncertain — route to analyst)
0.165365 – 1.000000   →   SIMULATED_BLOCK   (confident VPN — log only)
```

Thresholds derived from **validation split only** (p95 benign = review threshold,
max benign = block threshold). No test-set contamination.

### Runtime feature contract (34 features)

```
sz_coef_variation, sz_p25_median_ratio, sz_p75_median_ratio, sz_iqr_norm_median,
dispersion_symmetry, direction_balance_bytes, direction_balance_packets,
sz_mean_max, sz_mean_min, sz_std_max, sz_std_min,
iat_all_mean, iat_all_std, iat_all_p25, iat_all_median, iat_all_p75,
iat_mean_max, iat_mean_min, iat_std_max, iat_std_min,
sz_all_mean, sz_all_std, sz_all_median, sz_all_p25, sz_all_p75,
sz_cv, sz_iqr, sz_qratio, sz_median_to_mean,
iat_iqr, iat_cv, iat_median, iat_p25, iat_p75
```

Exact order is in `runtime_models/full_canonical__lgbm/feature_order.json`.

---

## Legacy model: `robust9_firewall`

The previous default firewall. Still loadable. Status: `legacy_baseline`.

- Diverse-bagging ensemble (3 XGB + 3 LGBM + 3 CatBoost bags)
- 9 features (robust9 feature set)
- Uses `prob_iso` (isotonic-calibrated) with p80 session aggregation
- Binary policy (BLOCK/FLAG_REVIEW/PASS)

---

## Inference scope

| Model | Binaries | Role | CSV inference |
|---|---|---|---|
| `full_canonical__lgbm` | ✅ Yes | **recommended_firewall** | ✅ Yes |
| `robust9_firewall` | ✅ Yes | legacy_baseline | ✅ Yes |
| `balanced_bagging_3ds_reference` | ✅ Yes | benchmark/comparison | ✅ Yes |
| `balanced_bagging_xgb_baseline` | ✅ Yes | benchmark/comparison | ✅ Yes |
| `balanced_bagging_baseline` | ✅ Yes | benchmark/comparison | ✅ Yes |
| `robust13_comparison` | ✅ Yes | benchmark/comparison | ✅ Yes |
| `lodo_hold_*` | ❌ No | negative_control | ❌ — not deployable |
| All others | ❌ No | metadata/comparison | ❌ |

---

## How to install this bundle

```bash
# 1) Copy the bundle into the backend folder of the new app
cp -r app_model_registry  ../ai-vpn-firewall-prototype/backend/
cp -r runtime_models      ../ai-vpn-firewall-prototype/backend/
cp -r demo_data           ../ai-vpn-firewall-prototype/backend/
cp requirements_runtime.txt ../ai-vpn-firewall-prototype/backend/

# 2) Install runtime dependencies
cd ../ai-vpn-firewall-prototype/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_runtime.txt

# 3) Point the FastAPI loader at the runtime config
#    backend/runtime_models/full_canonical__lgbm/runtime_loader_config.json

# 4) Run smoke test from bundle root
python scripts/smoke_test_full_canonical.py
```

---

## Smoke test

```bash
# From bundle root:
python scripts/smoke_test_full_canonical.py
```

Expected output: `53/53 checks passed — bundle is valid and ready to copy.`

The smoke test verifies:
- All required files present
- Registry marks `full_canonical__lgbm` as recommended firewall
- `robust9_firewall` is legacy/baseline
- Model loads and runs inference
- Demo CSV contains all 34 features
- All actions are simulation-only (`PASS` / `FLAG_REVIEW` / `SIMULATED_BLOCK`)
- `action_mode = simulation`, `production_ready = False`

---

## Path rewriting

All paths in `runtime_loader_config.json` files are **relative to this bundle root**.
The backend should resolve all model paths from the registry `runtime_loader_config` field.

---

## Demo data

| File | Description |
|---|---|
| `demo_data/demo_flows_full_canonical.csv` | 20 flows (15 benign + 5 VPN), all 34 canonical features |
| `demo_data/demo_flows.csv` | Legacy robust9 demo CSV (9 features) |

---

## Domain fingerprinting — known limitation

All models in this bundle have `domain_auc ≈ 1.0`. A separate classifier can identify
whether a flow came from the ISCX-VPN-2016, USBVPN, or VNAT dataset with perfect
accuracy. This means the models have memorised dataset-specific feature distributions.

**Consequence:** The models perform well on the known training domains but will have
unpredictable performance on unseen network environments.

DANN v2 adversarial training was evaluated and did not reduce fingerprinting
(`domain_reduction ≈ 0.0003`). New capture environments are the proven path forward.

---

## Files not included (by design)

- `notebooks/`, `data/raw/` (PCAPs), training data
- `artifacts/` heavyweight experiment contents
- Model binaries for metadata-only registry entries
- `_logs/`, ablations, window-sensitivity experiments
