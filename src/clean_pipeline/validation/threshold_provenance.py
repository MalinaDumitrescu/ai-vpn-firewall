"""
Deployment-threshold provenance export for the clean pipeline.

`artifacts/clean_pipeline/models/evaluation_report.json` already records the
global `policy_fit_split` field. The deployment recommendation in
`eval_v3/clean_deployment_recommendation.json` lists per-policy thresholds
(strict / balanced / flag / monitor / block) but does NOT repeat the fit
split per row. This helper writes a sidecar that records, per policy:

    {
      "policy_name": "strict",
      "threshold":   <float>,
      "fit_split":   "val",
      "fit_metric":  "<carried-over-from-report>"
    }

The data-leakage test suite verifies fit_split == "val" for every policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json

from src.clean_pipeline.validation.leakage_checks import CleanArtifactPaths


# Default policy names we look for in the deployment recommendation.
# `monitor` / `block` are aliases that appear in two-tier configurations.
KNOWN_POLICY_NAMES = (
    "strict", "balanced", "flag", "monitor", "block", "flag_review"
)


def _load_json_or_none(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON at {path}: {e}") from e


def _collect_policies_from_recommendation(rec: dict) -> Dict[str, dict]:
    """Return {policy_name: {threshold, ...}} from the recommendation file."""
    out: Dict[str, dict] = {}
    for key, value in rec.items():
        if not isinstance(value, dict):
            continue
        if "thr" in value or "threshold" in value:
            out[str(key).lower()] = {
                "threshold": float(value.get("thr", value.get("threshold"))),
                "aggregation": value.get("agg"),
                "raw_row": {k: value[k] for k in value if isinstance(value[k], (int, float, str, bool))},
            }
    return out


def ensure_policy_threshold_provenance(
    paths: CleanArtifactPaths,
    overwrite: bool = False,
) -> dict:
    """
    Build (or load) the per-policy threshold-provenance sidecar.

    The function reads `evaluation_report.json` (global `policy_fit_split`,
    `policy_fit_provenance`) and `clean_deployment_recommendation.json`
    (per-policy thresholds) and emits a unified record at
    `artifacts/clean_pipeline/eval_v3/policy_threshold_provenance.json`.

    Returns the loaded provenance dict.
    """
    target = paths.threshold_provenance
    if target.exists() and not overwrite:
        return json.loads(target.read_text(encoding="utf-8"))

    eval_report = _load_json_or_none(paths.evaluation_report)
    if eval_report is None:
        raise FileNotFoundError(
            f"evaluation_report.json not found at {paths.evaluation_report}. "
            "Run model training/evaluation first."
        )

    global_fit_split = eval_report.get("policy_fit_split")
    if global_fit_split is None:
        raise ValueError(
            "evaluation_report.json is missing the `policy_fit_split` field. "
            "Re-emit it from src/eval/metrics.py with the provenance block."
        )

    rec = _load_json_or_none(paths.deployment_recommendation)
    policies: Dict[str, dict] = {}
    if rec is not None:
        policies = _collect_policies_from_recommendation(rec)

    # Stamp every discovered policy with the global fit split.
    for name in policies:
        policies[name]["fit_split"] = global_fit_split

    provenance = {
        "schema_version": 1,
        "global_policy_fit_split": global_fit_split,
        "global_provenance": eval_report.get("policy_fit_provenance", {}),
        "policies": policies,
        "sources": {
            "evaluation_report": str(
                paths.evaluation_report.relative_to(paths.repo_root)
            ).replace("\\", "/"),
            "deployment_recommendation": (
                str(paths.deployment_recommendation.relative_to(paths.repo_root)).replace("\\", "/")
                if paths.deployment_recommendation.exists() else None
            ),
        },
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return provenance
