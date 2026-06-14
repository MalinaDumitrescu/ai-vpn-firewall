#!/usr/bin/env python3
"""Task 6: Validate the model registry against deployment/safety policy rules.

Rules enforced:
  R1  registry.json exists and is valid JSON
  R2  every listed package_dir exists (unless entry is an alias)
  R3  every non-alias model has model_card.json AND loader_config.json
  R4  exactly one model has status default_firewall
  R5  every negative_control model has supports_firewall_actions == false
  R6  every negative_control model has supports_live_mode == false
  R7  every policy_computed model has a warning containing "comparison-only"
  R8  every default_firewall and policy_packaged model has
        thresholds.json, session_metrics.json, policy_report.json
  R9  no model with nonzero flow_id/capture_id overlap is default_firewall or policy_packaged
  R10 no model with strict test FPR > 0 is allowed automatic BLOCK
        (i.e. thresholds.json strict.action == "BLOCK" requires strict_test_fpr == 0)
  R11 if balanced test FPR > 0.01, balanced-only action must be FLAG_REVIEW (not BLOCK)
  R12 duplicate models (byte-identical source artifacts) must be represented as aliases,
       not duplicated full packages
  R13 no LODO model is marked deployable
  R14 no research_only model is marked supports_firewall_actions == true

Outputs:
  reports/model_registry_validation_report.md
  reports/tables/model_registry_validation_errors.csv

Exit code 0 if all rules pass; 1 if any failure (does NOT silently fix anything).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "backend" / "model_registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
REPORT_MD = REPORTS / "model_registry_validation_report.md"
ERROR_CSV = TABLES / "model_registry_validation_errors.csv"

NOW = datetime.now(timezone.utc).isoformat()


# --------------- helpers ---------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir(d: Path) -> dict[str, str]:
    if not d.exists():
        return {}
    return {p.name: _sha256(p) for p in sorted(d.iterdir()) if p.is_file()}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _warnings_contain(payload: Any, needle: str) -> bool:
    needle_l = needle.lower()
    def _walk(x: Any) -> bool:
        if isinstance(x, str):
            return needle_l in x.lower()
        if isinstance(x, dict):
            return any(_walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(_walk(v) for v in x)
        return False
    return _walk(payload)


def _collect_warnings(*sources: Any) -> list[str]:
    out: list[str] = []
    for s in sources:
        if isinstance(s, dict):
            w = s.get("warnings")
            if isinstance(w, list):
                out.extend([str(x) for x in w])
            elif isinstance(w, str):
                out.append(w)
    return out


# --------------- validator ---------------

class Failure:
    __slots__ = ("rule", "model_id", "severity", "message")
    def __init__(self, rule: str, model_id: str, severity: str, message: str):
        self.rule = rule
        self.model_id = model_id
        self.severity = severity
        self.message = message
    def as_row(self) -> dict[str, str]:
        return {"rule": self.rule, "model_id": self.model_id,
                "severity": self.severity, "message": self.message}


def validate() -> tuple[list[Failure], list[Failure], dict[str, Any]]:
    failures: list[Failure] = []
    warnings_log: list[Failure] = []

    # R1: registry exists and is valid JSON
    if not REGISTRY_PATH.exists():
        failures.append(Failure("R1", "<registry>", "error", f"registry.json missing at {REGISTRY_PATH}"))
        return failures, warnings_log, {}
    try:
        reg = _load_json(REGISTRY_PATH)
    except Exception as exc:
        failures.append(Failure("R1", "<registry>", "error", f"registry.json invalid JSON: {exc}"))
        return failures, warnings_log, {}
    if not isinstance(reg, dict) or "models" not in reg or not isinstance(reg["models"], dict):
        failures.append(Failure("R1", "<registry>", "error", "registry.json malformed: missing 'models' dict"))
        return failures, warnings_log, {}

    models: dict[str, dict[str, Any]] = reg["models"]

    # R4: exactly one default_firewall
    defaults = [mid for mid, e in models.items() if e.get("status") == "default_firewall"]
    if len(defaults) != 1:
        failures.append(Failure("R4", "<registry>", "error",
                                f"Expected exactly one default_firewall, found {len(defaults)}: {defaults}"))

    # Per-model checks
    per_model_facts: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, list[str]] = {}  # source_artifact_path -> [model_ids]

    for mid, entry in models.items():
        status = entry.get("status")
        pkg_rel = entry.get("package_dir")
        pkg_dir = (ROOT / pkg_rel) if pkg_rel else None
        src_rel = entry.get("source_artifact")
        src_dir = (ROOT / src_rel) if src_rel else None
        is_alias = (status == "alias")

        # R2: package_dir exists unless alias
        if not is_alias:
            if not pkg_dir or not pkg_dir.exists():
                failures.append(Failure("R2", mid, "error",
                                        f"package_dir missing or not on disk: {pkg_rel}"))

        # R3: non-alias model has model_card.json and loader_config.json
        if not is_alias and pkg_dir and pkg_dir.exists():
            for req in ("model_card.json", "loader_config.json"):
                if not (pkg_dir / req).exists():
                    failures.append(Failure("R3", mid, "error",
                                            f"missing required file: {req}"))

        # Alias rule R12 (positive form): aliases must have alias_of and an alias.json
        if is_alias:
            if not entry.get("alias_of"):
                failures.append(Failure("R12", mid, "error", "alias entry missing 'alias_of'"))
            else:
                tgt = entry["alias_of"]
                if tgt not in models:
                    failures.append(Failure("R12", mid, "error",
                                            f"alias_of points to non-existent model_id: {tgt}"))
            if pkg_dir and pkg_dir.exists() and not (pkg_dir / "alias.json").exists():
                warnings_log.append(Failure("R12", mid, "warning",
                                            "alias package_dir has no alias.json (recommended)"))

        # Load model_card.json + policy_report.json + thresholds.json (if available)
        model_card = None
        thresholds = None
        policy_report = None
        session_metrics = None
        if not is_alias and pkg_dir and pkg_dir.exists():
            if (pkg_dir / "model_card.json").exists():
                try:
                    model_card = _load_json(pkg_dir / "model_card.json")
                except Exception as exc:
                    failures.append(Failure("R3", mid, "error", f"model_card.json invalid JSON: {exc}"))
            if (pkg_dir / "thresholds.json").exists():
                try:
                    thresholds = _load_json(pkg_dir / "thresholds.json")
                except Exception as exc:
                    failures.append(Failure("R8", mid, "error", f"thresholds.json invalid JSON: {exc}"))
            if (pkg_dir / "policy_report.json").exists():
                try:
                    policy_report = _load_json(pkg_dir / "policy_report.json")
                except Exception as exc:
                    failures.append(Failure("R8", mid, "error", f"policy_report.json invalid JSON: {exc}"))
            if (pkg_dir / "session_metrics.json").exists():
                try:
                    session_metrics = _load_json(pkg_dir / "session_metrics.json")
                except Exception as exc:
                    failures.append(Failure("R8", mid, "error", f"session_metrics.json invalid JSON: {exc}"))

        # R8: default_firewall and policy_packaged need thresholds, session_metrics, policy_report
        if status in ("default_firewall", "policy_packaged"):
            for req in ("thresholds.json", "session_metrics.json", "policy_report.json"):
                if not pkg_dir or not (pkg_dir / req).exists():
                    failures.append(Failure("R8", mid, "error", f"{status} model missing {req}"))

        # R5, R6, R13: negative_control / LODO constraints
        if status == "negative_control":
            sfa = entry.get("supports_firewall_actions")
            slm = entry.get("supports_live_mode")
            # Also check model_card.json for the same flags
            mc_sfa = (model_card or {}).get("supports_firewall_actions") if model_card else None
            mc_slm = (model_card or {}).get("supports_live_mode") if model_card else None
            if sfa is not False or (mc_sfa is not None and mc_sfa is not False):
                failures.append(Failure("R5", mid, "error",
                                        f"negative_control must have supports_firewall_actions=false (registry={sfa}, card={mc_sfa})"))
            if slm is not False or (mc_slm is not None and mc_slm is not False):
                failures.append(Failure("R6", mid, "error",
                                        f"negative_control must have supports_live_mode=false (registry={slm}, card={mc_slm})"))
            # R13: no LODO model marked deployable
            mid_lower = mid.lower()
            src_lower = (src_rel or "").lower()
            is_lodo = ("lodo" in mid_lower) or ("lood" in src_lower) or ("lodo" in src_lower)
            if is_lodo:
                deployable = entry.get("deployable")
                mc_deployable = (model_card or {}).get("deployable") if model_card else None
                if deployable is True or mc_deployable is True:
                    failures.append(Failure("R13", mid, "error",
                                            f"LODO model marked deployable=true (registry={deployable}, card={mc_deployable})"))
                if deployable is None and mc_deployable is None:
                    failures.append(Failure("R13", mid, "error",
                                            "LODO model missing explicit deployable=false flag"))

        # R7: policy_computed -> warning contains "comparison-only"
        if status == "policy_computed":
            warns = _collect_warnings(entry, model_card, policy_report)
            has_warning = any("comparison-only" in w.lower() for w in warns)
            if not has_warning:
                failures.append(Failure("R7", mid, "error",
                                        "policy_computed model has no warning containing 'comparison-only' "
                                        "(checked registry entry, model_card.json, policy_report.json)"))

        # R10: strict.action == BLOCK requires strict test FPR == 0
        # R11: balanced.action == BLOCK with balanced test FPR > 0.01 forbidden
        if thresholds:
            strict = thresholds.get("strict") or {}
            balanced = thresholds.get("balanced") or {}
            strict_action = str(strict.get("action") or strict.get("action_if_deployed") or "").upper()
            balanced_action = str(balanced.get("action") or balanced.get("action_if_deployed") or "").upper()
            strict_test_fpr = None
            balanced_test_fpr = None
            if session_metrics:
                strict_test_fpr = (session_metrics.get("strict") or {}).get("fpr")
                balanced_test_fpr = (session_metrics.get("balanced") or {}).get("fpr")
            # Fallback to registry summary fields
            if strict_test_fpr is None:
                strict_test_fpr = entry.get("strict_test_fpr")
            if balanced_test_fpr is None:
                balanced_test_fpr = entry.get("balanced_test_fpr")

            reporting_only = bool(thresholds.get("reporting_only"))

            # R10: only enforce for entries that are actually deployable-style
            if strict_action == "BLOCK" and not reporting_only:
                if strict_test_fpr is None:
                    warnings_log.append(Failure("R10", mid, "warning",
                                                "strict.action=BLOCK but strict_test_fpr is unknown; cannot verify"))
                elif float(strict_test_fpr) > 0.0:
                    failures.append(Failure("R10", mid, "error",
                                            f"strict.action=BLOCK with strict_test_fpr={strict_test_fpr} > 0 "
                                            f"is not allowed (auto-BLOCK requires fpr==0)"))

            # R11: if balanced FPR > 0.01, balanced action must be FLAG_REVIEW (not BLOCK)
            if balanced_test_fpr is not None and float(balanced_test_fpr) > 0.01:
                if balanced_action != "FLAG_REVIEW":
                    failures.append(Failure("R11", mid, "error",
                                            f"balanced_test_fpr={balanced_test_fpr} > 0.01 but "
                                            f"balanced.action='{balanced_action}' (must be FLAG_REVIEW)"))

        # R9: no nonzero overlap on default_firewall or policy_packaged
        any_overlap = entry.get("any_overlap_detected")
        if any_overlap is None and policy_report:
            any_overlap = policy_report.get("any_overlap_detected")
        if status in ("default_firewall", "policy_packaged") and bool(any_overlap):
            failures.append(Failure("R9", mid, "error",
                                    f"{status} model has flow/capture overlap; not allowed"))

        # R14: research_only must not have supports_firewall_actions=true
        if status == "research_only":
            sfa = entry.get("supports_firewall_actions")
            mc_sfa = (model_card or {}).get("supports_firewall_actions") if model_card else None
            if sfa is True or mc_sfa is True:
                failures.append(Failure("R14", mid, "error",
                                        f"research_only must not have supports_firewall_actions=true (registry={sfa}, card={mc_sfa})"))

        # Collect for R12 (duplicate FULL packages from byte-identical source artifacts
        # AND with identical model selection — i.e. truly redundant packages).
        # Two packages from the same source dir that select different probability columns
        # or different model families are NOT duplicates.
        if src_dir and src_dir.exists() and not is_alias and status != "unsupported":
            try:
                h = hashlib.sha256()
                for name, fhash in _hash_dir(src_dir).items():
                    h.update(name.encode("utf-8"))
                    h.update(fhash.encode("utf-8"))
                # Differentiate by the model's selection signature so two distinct
                # packages from the same source dir (e.g. different score columns)
                # are not mis-flagged as duplicates.
                sel_prob = str(entry.get("selected_probability_column") or
                               (model_card or {}).get("selected_probability_column") or "")
                sel_agg = str(entry.get("selected_aggregation") or
                              (model_card or {}).get("selected_aggregation") or "")
                family = ""
                if model_card:
                    family = str((model_card.get("ensemble") or {}).get("family_label") or
                                 (model_card.get("ensemble") or {}).get("families") or "")
                signature = f"{sel_prob}|{sel_agg}|{family}".lower()
                h.update(signature.encode("utf-8"))
                composite = h.hexdigest()
                source_hashes.setdefault(composite, []).append(mid)
            except Exception as exc:
                warnings_log.append(Failure("R12", mid, "warning",
                                            f"could not hash source artifact: {exc}"))

        per_model_facts[mid] = {
            "status": status,
            "package_dir": pkg_rel,
            "source_artifact": src_rel,
            "is_alias": is_alias,
            "warnings": _collect_warnings(entry, model_card, policy_report),
        }

    # R12: if any composite_hash maps to >1 full packages -> failure
    for composite, mids in source_hashes.items():
        if len(mids) > 1:
            failures.append(Failure("R12", ",".join(mids), "error",
                                    f"byte-identical source artifacts present as multiple full packages: {mids} "
                                    f"(should be represented as aliases)"))

    return failures, warnings_log, per_model_facts


def write_outputs(failures: list[Failure], warnings: list[Failure],
                  facts: dict[str, Any]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # CSV
    with ERROR_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "model_id", "severity", "message"])
        w.writeheader()
        for item in failures + warnings:
            w.writerow(item.as_row())

    # MD
    n_err = len(failures)
    n_warn = len(warnings)
    status = "PASS" if n_err == 0 else "FAIL"

    lines = [
        "# Model Registry Validation Report",
        "",
        f"- Generated: {NOW}",
        f"- Registry: `{REGISTRY_PATH.relative_to(ROOT).as_posix()}`",
        f"- Overall status: **{status}**",
        f"- Errors: **{n_err}**",
        f"- Warnings: **{n_warn}**",
        "",
        "## Rules checked",
        "- R1: registry.json exists and is valid JSON",
        "- R2: every listed package_dir exists (unless alias)",
        "- R3: every non-alias model has model_card.json + loader_config.json",
        "- R4: exactly one model has status default_firewall",
        "- R5: negative_control => supports_firewall_actions=false",
        "- R6: negative_control => supports_live_mode=false",
        "- R7: policy_computed => warning contains 'comparison-only'",
        "- R8: default_firewall / policy_packaged => thresholds.json, session_metrics.json, policy_report.json",
        "- R9: nonzero flow/capture overlap => cannot be default_firewall or policy_packaged",
        "- R10: strict.action=BLOCK requires strict test FPR == 0",
        "- R11: balanced test FPR > 0.01 => balanced.action must be FLAG_REVIEW",
        "- R12: byte-identical source artifacts must be aliases, not duplicate packages",
        "- R13: LODO models must not be deployable",
        "- R14: research_only must not have supports_firewall_actions=true",
        "",
        "## Registry contents (per model)",
        "",
        "| model_id | status | package_dir | source_artifact |",
        "|---|---|---|---|",
    ]
    for mid, f in facts.items():
        lines.append(f"| `{mid}` | {f['status']} | `{f['package_dir'] or '-'}` | `{f['source_artifact'] or '-'}` |")

    if failures:
        lines += ["", "## Failures (errors)", "",
                  "| rule | model_id | message |", "|---|---|---|"]
        for x in failures:
            lines.append(f"| {x.rule} | `{x.model_id}` | {x.message} |")
    else:
        lines += ["", "## Failures (errors)", "", "_None._"]

    if warnings:
        lines += ["", "## Warnings (informational)", "",
                  "| rule | model_id | message |", "|---|---|---|"]
        for x in warnings:
            lines.append(f"| {x.rule} | `{x.model_id}` | {x.message} |")

    lines += ["",
              "## Outputs",
              f"- `{ERROR_CSV.relative_to(ROOT).as_posix()}`",
              f"- `{REPORT_MD.relative_to(ROOT).as_posix()}`",
              ""]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    failures, warnings, facts = validate()
    write_outputs(failures, warnings, facts)

    print(f"Validation: {'PASS' if not failures else 'FAIL'}  "
          f"(errors={len(failures)}, warnings={len(warnings)})")
    for x in failures:
        print(f"  [ERROR] {x.rule} {x.model_id}: {x.message}")
    for x in warnings:
        print(f"  [warn ] {x.rule} {x.model_id}: {x.message}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())


