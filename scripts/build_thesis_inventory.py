"""Build artifact inventory for unified_feature_contract_v2 thesis exports."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
THESIS_DIR   = PROJECT_ROOT / "artifacts" / "unified_feature_contract_v2" / "thesis_exports"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

expected = [
    ('feature_contract',               'artifacts/unified_feature_contract_v2/feature_contract.json', 'json', 'Canonical formulas + extractor version'),
    ('unified_formula_report',         'artifacts/unified_feature_contract_v2/unified_formula_report.md', 'md', 'Per-feature formula reconciliation'),
    ('phase1_data_contract_summary',   'artifacts/unified_feature_contract_v2/PHASE_1_DATA_CONTRACT_SUMMARY.md', 'md', 'Phase 1 outputs summary'),
    ('unified_flows_parquet',          'artifacts/unified_feature_contract_v2/data/unified_flows.parquet', 'parquet', 'Unified dataset (3 datasets, capture-level split)'),
    ('unified_flows_csv',              'artifacts/unified_feature_contract_v2/data/unified_flows.csv', 'csv', 'Unified dataset CSV mirror'),
    ('split_integrity_report',         'artifacts/unified_feature_contract_v2/split_integrity_report.csv', 'csv', 'Capture-level split integrity check'),
    ('model_comparison',               'artifacts/unified_feature_contract_v2/model_comparison.csv', 'csv', 'All 30 unified models compared'),
    ('recommended_models',             'artifacts/unified_feature_contract_v2/recommended_models.json', 'json', 'Selected models per role'),
    ('lodo_results',                   'artifacts/unified_feature_contract_v2/lodo_results.csv', 'csv', 'Leave-one-dataset-out AUCs'),
    ('domain_fingerprint_initial',     'artifacts/unified_feature_contract_v2/domain_fingerprint_initial.csv', 'csv', 'Pre-training domain-classifier diagnostic'),
    ('domain_fingerprint_results',     'artifacts/unified_feature_contract_v2/domain_fingerprint_results.csv', 'csv', 'Per-model domain AUC'),
    ('calibration_results',            'artifacts/unified_feature_contract_v2/calibration_results.csv', 'csv', 'ECE / Brier per model'),
    ('anti_fingerprint_scores',        'artifacts/unified_feature_contract_v2/anti_fingerprint_feature_scores.csv', 'csv', 'Anti-fingerprint feature ranking'),
    ('live_pcap_results',              'artifacts/unified_feature_contract_v2/live_pcap_results.csv', 'csv', 'Live PCAP evaluation (if run)'),
    ('final_report',                   'artifacts/unified_feature_contract_v2/final_report.md', 'md', 'Final experiment report'),
    ('thesis_summary',                 'artifacts/unified_feature_contract_v2/thesis_summary.md', 'md', 'Thesis-ready summary'),
    ('results_html',                   'artifacts/unified_feature_contract_v2/unified_feature_contract_results.html', 'html', 'Phase 2 results HTML'),
    ('thesis_analysis_html',           'artifacts/unified_feature_contract_v2/unified_feature_contract_v2_thesis_analysis.html', 'html', 'This thesis analysis notebook (HTML)'),
    ('figures_dir',                    'artifacts/unified_feature_contract_v2/figures', 'dir', '12 saved thesis figures'),
    ('runtime_model_pkl',              'artifacts/unified_feature_contract_v2/runtime_export/model.pkl', 'pkl', 'Selected LGBM model'),
    ('runtime_calibrator',             'artifacts/unified_feature_contract_v2/runtime_export/calibrator.pkl', 'pkl', 'Isotonic calibrator'),
    ('runtime_feature_order',          'artifacts/unified_feature_contract_v2/runtime_export/feature_order.json', 'json', '12-feature order'),
    ('runtime_thresholds',             'artifacts/unified_feature_contract_v2/runtime_export/thresholds.json', 'json', 'review / block thresholds'),
    ('runtime_extractor_config',       'artifacts/unified_feature_contract_v2/runtime_export/extractor_config.json', 'json', 'Live extractor configuration'),
    ('runtime_model_card',             'artifacts/unified_feature_contract_v2/runtime_export/model_card.md', 'md', 'Runtime model card'),
    ('runtime_readme',                 'artifacts/unified_feature_contract_v2/runtime_export/RUNTIME_README.md', 'md', 'Runtime README + warnings'),
    ('runtime_requirements',           'artifacts/unified_feature_contract_v2/runtime_export/requirements_runtime.txt', 'txt', 'Python deps to run model'),
    ('runtime_smoke_test_output',      'artifacts/unified_feature_contract_v2/runtime_export/smoke_test_output.txt', 'txt', 'Smoke test result log'),
    ('runtime_app_registry_json',      'artifacts/unified_feature_contract_v2/runtime_export/app_model_registry/unified_firewall_candidate.json', 'json', 'App registry candidate entry'),
    ('runtime_app_registry_csv',       'artifacts/unified_feature_contract_v2/runtime_export/app_model_registry/model_registry.csv', 'csv', 'App registry CSV'),
    ('demo_csv',                       'artifacts/unified_feature_contract_v2/runtime_export/demo_data/unified_model_demo_flows.csv', 'csv', 'Demo CSV for app inference'),
    ('demo_manifest',                  'artifacts/unified_feature_contract_v2/runtime_export/demo_data/unified_model_demo_manifest.json', 'json', 'Expected demo results'),
    ('demo_validation',                'artifacts/unified_feature_contract_v2/runtime_export/demo_data/demo_csv_validation.md', 'md', 'Demo CSV validation report'),
    ('demo_benchmark_csv',             'artifacts/unified_feature_contract_v2/runtime_export/demo_data/unified_model_benchmark_flows.csv', 'csv', 'Benchmark CSV (family models scored)'),
    ('final_transfer_model_comparison','artifacts/final_transfer/model_comparison.csv', 'csv', 'Legacy final_transfer comparison (full_canonical etc.)'),
    ('final_transfer_recommended',     'artifacts/final_transfer/recommended_models.json', 'json', 'Legacy recommendations'),
    ('final_transfer_anti_fp',         'artifacts/final_transfer/anti_fingerprint_scores.csv', 'csv', 'Legacy anti-fingerprint scores'),
    ('lood_firewall_summary',          'artifacts/lood_firewall_tuned/lodo_summary.json', 'json', 'Legacy LODO firewall summary'),
    ('ensemble_robust9_metrics',       'artifacts/ensemble/balanced_ensemble_robust9/metrics.json', 'json', 'robust9 ensemble metrics'),
    ('frontend_inventory',             'artifacts/frontend_model_details/model_detail_inventory.csv', 'csv', 'Frontend model inventory'),
    ('frontend_cards',                 'artifacts/frontend_model_details/model_cards_frontend.json', 'json', 'Frontend model cards'),
    ('frontend_metrics',               'artifacts/frontend_model_details/model_metrics_summary.json', 'json', 'Frontend metrics summary'),
]

rows, missing = [], []
for name, rel, kind, purpose in expected:
    p = PROJECT_ROOT / rel
    exists = p.exists()
    if exists and p.is_file():
        size = p.stat().st_size
    elif exists and p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    else:
        size = 0
        missing.append(rel)
    rows.append({
        'artifact_name': name, 'path': rel,
        'exists': 'yes' if exists else 'no',
        'type': kind, 'size_bytes': size,
        'likely_purpose': purpose,
        'notes': '' if exists else 'NOT FOUND on disk',
    })

df = pd.DataFrame(rows)
csv_p = THESIS_DIR / 'artifact_inventory.csv'
md_p  = THESIS_DIR / 'artifact_inventory.md'
df.to_csv(csv_p, index=False)

md_lines = [
    '# Artifact inventory — unified_feature_contract_v2',
    '',
    '_Generated by `notebooks/unified_feature_contract_v2_thesis_analysis.ipynb` (inventory cell)._',
    '',
    '| artifact_name | path | exists | type | size_bytes | likely_purpose |',
    '|---|---|---|---|---:|---|',
]
for r in rows:
    md_lines.append('| {artifact_name} | `{path}` | {exists} | {type} | {size_bytes} | {likely_purpose} |'.format(**r))
md_lines += ['', '## Missing expected artifacts', '']
if missing:
    md_lines += [f'- `{m}`' for m in missing]
else:
    md_lines.append('_None — all expected artifacts found._')
md_p.write_text('\n'.join(md_lines), encoding='utf-8')

print(f'Saved: {csv_p}')
print(f'Saved: {md_p}')
print(f'Total expected : {len(rows)}')
print(f'Found          : {sum(1 for r in rows if r["exists"] == "yes")}')
print(f'Missing        : {len(missing)}')
for m in missing:
    print('  MISSING:', m)
