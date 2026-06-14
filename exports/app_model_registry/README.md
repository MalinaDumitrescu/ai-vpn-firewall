# AI-VPN-Firewall — App Model Registry (Export Bundle)

Generated: 2026-05-26T09:27:56.071535+00:00

## What this bundle is

This export contains the validated **model registry** that the new FastAPI + React
firewall application consumes. It does **not** contain notebooks, raw datasets, training
data, original model checkpoints, or any unrelated experimental artifacts. Each model
entry is a small, self-describing package of JSON metadata: `model_card.json`,
`loader_config.json`, `feature_order.json` (or `_unavailable.json`), `calibration_info.json`,
`thresholds.json`, `session_metrics.json`, `policy_report.json`, `package_manifest.json`.
Original heavyweight model binaries remain in the source project at the locations recorded
in each package's `source_artifact`.

## Default firewall model

- **`robust9_firewall`** is the only entry with status `default_firewall`.
- Firewall Mode of the application MUST load only this model for any blocking/flagging action.

## Registry entries (by status)

### `alias`
- **`balanced_bagging_3ds_refresh`**: Source artifact is byte-identical to balanced_bagging_firewall_tuned_ensemble (verified by SHA-256 over all files).  — alias of `balanced_bagging_3ds_reference`
- **`historical_3ds_degraded`**: Historical/degraded snapshot of the 3DS reference. Predictions are byte-identical. Not deployable.  — alias of `balanced_bagging_3ds_reference` / warning: historical/degraded artifact; predictions duplicate existing 3DS reference; not deployable.

### `default_firewall`
- **`robust9_firewall`**: All policy files generated; session policy available; no flow_id/capture_id overlap; feature order recovered.

### `negative_control`
- **`lodo_hold_iscx`**: Negative control / LODO stress-test model. Not deployable.  — held-out: iscx / warning: Negative control / LODO stress-test model. Not deployable.
- **`lodo_hold_usbvpn`**: Negative control / LODO stress-test model. Not deployable.  — held-out: usbvpn / warning: Negative control / LODO stress-test model. Not deployable.
- **`lodo_hold_vnat`**: Negative control / LODO stress-test model. Not deployable.  — held-out: vnat / warning: Negative control / LODO stress-test model. Not deployable.

### `policy_computed`
- **`balanced_bagging_3ds_reference`**: Thresholds recomputed from validation; persisted thresholds not adopted.  — warning: comparison-only, not deployment-approved
- **`clean_pooled_benchmark`**: Thresholds recomputed from clean_pipeline val parquet; comparison-only.  — warning: comparison-only, not deployment-approved
- **`single_model_fallback`**: Thresholds recomputed from clean_pipeline val parquet; comparison-only.  — warning: comparison-only, not deployment-approved
- **`balanced_bagging_xgb_baseline`**: Thresholds recomputed from validation; comparison-only.  — warning: comparison-only, not deployment-approved
- **`balanced_bagging_baseline`**: Thresholds recomputed from validation; comparison-only.  — warning: comparison-only, not deployment-approved
- **`balanced_bagging_tuned_baseline`**: Thresholds recomputed from validation; comparison-only.  — warning: comparison-only, not deployment-approved
- **`robust13_comparison`**: Thresholds recomputed from validation; comparison-only.  — warning: comparison-only, not deployment-approved

### `unsupported`
- **`twentyone_feature_experimental`**: No recoverable 21-feature artifact exists in the workspace. Inspected artifacts/clean_pipeline/feature_columns.json (9 features) and artifacts/features/feature_columns.json (7 features). No 21-feature model file, predictions file, or feature manifest was located. Per task constraints, no feature order is invented.  — warning: No recoverable 21-feature artifact exists in the workspace. Inspected artifacts/clean_pipeline/feature_columns.json (9 features) and artifacts/features/feature_columns.json (7 features). No 21-feature model file, predictions file, or feature manifest was located. Per task constraints, no feature order is invented.

## App usage rules

### Firewall Mode
- Firewall Mode MUST use **only** the entry whose `status == "default_firewall"`.
- It must read `thresholds.json` for `strict` and `balanced` and map triggers as:
  `strict_trigger -> BLOCK`, `balanced_trigger only -> FLAG_REVIEW`, otherwise `PASS`.
- Any entry with `strict.action == "DISABLED_DO_NOT_BLOCK"` or `strict.reporting_only == true`
  MUST NOT be exposed in Firewall Mode, even if a user attempts to select it.

### Model Comparison Lab
- Model Comparison Lab MAY display `policy_packaged`, `policy_computed`, `research_only`,
  and `negative_control` entries side-by-side for evaluation.
- The Lab MUST surface the entry's `warnings` (e.g. `"comparison-only, not deployment-approved"`)
  visibly in the UI and disable any control that would trigger live blocking.
- `negative_control` entries (LODO) must show their `held_out_dataset` and a banner indicating
  they are NOT deployable.

### Simulation by default
- Blocking is **simulation-only by default**. Real packet-drop / VM-interception must
  require an explicit operator toggle AND a model entry that satisfies all of:
  - `status == "default_firewall"`
  - `supports_live_mode == true`
  - `recommended_action_mode != "simulation"`
- No entry in this export currently sets `supports_live_mode = true`, so the app stays
  in simulation mode for all entries.

### LODO and robustness
- The Leave-One-Dataset-Out (LODO) entries (`lodo_hold_iscx`, `lodo_hold_usbvpn`,
  `lodo_hold_vnat`) collapse on their held-out dataset (session AUC ≈ 0.18–0.81),
  demonstrating that **no model in this project achieves universal cross-dataset
  production robustness**. LODO entries are negative controls only; they must never
  be exposed as deployable firewall choices.

## Manifest of files in this bundle

- `backend/model_registry/registry.json` — registry index
- `backend/model_registry/<model_id>/` — per-model package
- `scripts/policy_utils.py` — shared policy helpers (wt5 verified from `src/deployment/decision_engine.py::_wt5_agg`)
- `reports/model_registry_validation_report.md` — PASS report
- `reports/model_packaging_summary.md`
- `reports/tables/app_model_registry_summary.csv`
- `reports/tables/app_model_policy_comparison.csv`
- `reports/tables/app_model_capability_matrix.csv`

## Reproducing the bundle

From the project root:

```
python scripts/validate_model_registry.py   # must PASS with 0 errors
python scripts/build_app_export_bundle.py   # rebuilds exports/app_model_registry/
```

_This bundle was generated under read-only constraints: no original artifact was modified,
no notebooks/raw data/training data were copied._
