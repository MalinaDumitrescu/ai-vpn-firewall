# demo_firewall/open_set_policy.py
"""
Open-set / uncertainty-aware firewall policy for full_canonical__lgbm.

Three-tier decision system:
  PASS             – score < review_threshold
                     Traffic is below the uncertainty band. Allow.
  FLAG_REVIEW      – review_threshold <= score < block_threshold
                     Score is elevated but not confidently VPN.
                     Route to analyst queue. Do NOT block automatically.
  SIMULATED_BLOCK  – score >= block_threshold
                     High-confidence VPN detection.
                     Simulated block only — no real packet blocking.

Rationale for three tiers under domain shift
--------------------------------------------
Under domain shift (unseen deployment environments), the model produces
elevated but unreliable scores for benign traffic that doesn't match the
training distribution. A binary PASS/BLOCK policy forces high-stakes
decisions in precisely these uncertain regions.

The FLAG_REVIEW band [review_threshold, block_threshold] acts as a
"uncertainty absorber": flows that the model finds suspicious but not
definitively VPN are held for human review instead of being blocked.

Threshold provenance
--------------------
Computed from val_predictions.csv (100 benign sessions, 1 VPN session):
  review_threshold = p95 benign session score = 0.027090
    → ~5% of known-benign sessions enter the review band (val FPR = 0.04)
  block_threshold  = max benign session score  = 0.165365
    → 0% of known-benign sessions are auto-blocked on val (val FPR = 0.01
       due to one borderline benign session above p99)

SIMULATION DISCLAIMER
---------------------
This module does not block real network packets. All SIMULATED_BLOCK
decisions are audit/logging actions only. The system has no integration
with any packet-filtering or firewall infrastructure.
"""
from __future__ import annotations

import datetime
import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("firewall.open_set_policy")


# ─────────────────────────────────────────────────────────────────────────────
# Action enum
# ─────────────────────────────────────────────────────────────────────────────

class FirewallAction(str, Enum):
    """Three-tier firewall action."""
    PASS = "PASS"
    FLAG_REVIEW = "FLAG_REVIEW"
    SIMULATED_BLOCK = "SIMULATED_BLOCK"

    @property
    def severity(self) -> int:
        """Numeric severity (0 = lowest, 2 = highest)."""
        return {"PASS": 0, "FLAG_REVIEW": 1, "SIMULATED_BLOCK": 2}[self.value]

    @property
    def is_blocking(self) -> bool:
        return self == FirewallAction.SIMULATED_BLOCK

    @property
    def requires_review(self) -> bool:
        return self in (FirewallAction.FLAG_REVIEW, FirewallAction.SIMULATED_BLOCK)

    @property
    def display_label(self) -> str:
        labels = {
            "PASS": "✅ PASS",
            "FLAG_REVIEW": " FLAG FOR REVIEW",
            "SIMULATED_BLOCK": " SIMULATED BLOCK",
        }
        return labels[self.value]


# ─────────────────────────────────────────────────────────────────────────────
# Policy thresholds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OpenSetThresholds:
    """Validated threshold pair for the three-tier policy."""
    review_threshold: float   # lower bound of the uncertainty band
    block_threshold: float    # upper bound; scores above this → SIMULATED_BLOCK
    model_id: str = "full_canonical__lgbm"
    source_split: str = "val"
    review_basis: str = "p95 benign session score"
    block_basis: str = "max benign session score"
    simulation_only: bool = True

    def __post_init__(self):
        if not (0.0 <= self.review_threshold < self.block_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: review={self.review_threshold}, "
                f"block={self.block_threshold}. "
                "Must satisfy 0 <= review < block <= 1."
            )

    @classmethod
    def from_json(cls, path: Path) -> "OpenSetThresholds":
        """Load thresholds from the model's thresholds.json file."""
        with open(path) as f:
            d = json.load(f)
        return cls(
            review_threshold=float(d["review_threshold"]),
            block_threshold=float(d["block_threshold"]),
            model_id=d.get("model_id", "full_canonical__lgbm"),
            source_split=d.get("source_split", "val"),
            review_basis=d.get("review_threshold_basis", "p95 benign"),
            block_basis=d.get("block_threshold_basis", "max benign"),
            simulation_only=bool(d.get("simulation_only", True)),
        )

    @classmethod
    def default(cls) -> "OpenSetThresholds":
        """Return the validated default thresholds (no file required)."""
        return cls(
            review_threshold=0.027090,
            block_threshold=0.165365,
        )

    def classify(self, score: float) -> FirewallAction:
        """Classify a single session score into a FirewallAction."""
        if score >= self.block_threshold:
            return FirewallAction.SIMULATED_BLOCK
        if score >= self.review_threshold:
            return FirewallAction.FLAG_REVIEW
        return FirewallAction.PASS

    def confidence_margin(self, score: float) -> float:
        """
        Distance from the nearest tier boundary, normalised to [0, 1].
        High margin = confident decision. Low margin = borderline.
        """
        action = self.classify(score)
        if action == FirewallAction.SIMULATED_BLOCK:
            margin = score - self.block_threshold
        elif action == FirewallAction.FLAG_REVIEW:
            margin = min(
                score - self.review_threshold,
                self.block_threshold - score,
            )
        else:
            margin = self.review_threshold - score
        # Normalise by band width
        band = self.block_threshold - self.review_threshold
        if band > 0:
            return float(np.clip(margin / band, 0.0, 1.0))
        return float(np.clip(margin, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Decision records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """Lightweight per-flow scoring record."""
    flow_id: str
    capture_id: str
    score: float
    action: FirewallAction
    confidence_margin: float
    dataset: Optional[str] = None


@dataclass
class SessionDecisionV2:
    """
    Full three-tier decision for a capture/session.

    Parameters
    ----------
    capture_id : str
    session_score : float
        Aggregated score (mean of flow scores).
    action : FirewallAction
        PASS / FLAG_REVIEW / SIMULATED_BLOCK
    confidence_margin : float
        Distance from nearest tier boundary (normalised).
    n_flows : int
    n_flagged : int
        Flows above review_threshold.
    n_blocked : int
        Flows above block_threshold.
    review_threshold : float
    block_threshold : float
    timestamp : str
    flow_records : list[FlowRecord]
    label : int or None
        Ground-truth label (1=VPN, 0=benign, None=unknown).
    dataset : str or None
        Source dataset name (for diagnostics).
    """
    capture_id: str
    session_score: float
    action: FirewallAction
    confidence_margin: float
    n_flows: int
    n_flagged: int
    n_blocked: int
    review_threshold: float
    block_threshold: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    flow_records: List[FlowRecord] = field(default_factory=list)
    label: Optional[int] = None
    dataset: Optional[str] = None

    # ── explanation templates ──────────────────────────────────────────────
    _EXPLANATIONS: ClassVar[Dict[str, str]] = {
        "PASS": (
            "Score below review threshold. "
            "Traffic pattern consistent with benign baseline."
        ),
        "FLAG_REVIEW": (
            "Score is uncertain; manual review recommended. "
            "Score falls in the uncertainty band between review and block thresholds."
        ),
        "SIMULATED_BLOCK": (
            "Score exceeds block threshold; simulated block recommendation. "
            "[SIMULATION ONLY \u2014 no real packet blocking.]"
        ),
    }

    @property
    def is_correct(self) -> Optional[bool]:
        """True if decision agrees with ground truth label (best effort)."""
        if self.label is None:
            return None
        if self.label == 1:
            return self.action.requires_review
        return not self.action.requires_review

    def _threshold_band(self) -> str:
        """Human-readable label for the tier band the score falls in."""
        if self.action == FirewallAction.SIMULATED_BLOCK:
            return f"score >= {self.block_threshold:.6f}  [SIMULATED_BLOCK]"
        if self.action == FirewallAction.FLAG_REVIEW:
            return (
                f"{self.review_threshold:.6f} <= score < {self.block_threshold:.6f}"
                "  [FLAG_REVIEW]"
            )
        return f"score < {self.review_threshold:.6f}  [PASS]"

    def to_event_dict(self) -> Dict[str, Any]:
        """Compact dict for dashboard event table row.

        Every event includes: score, action, threshold_band, model_id,
        action_mode (always 'simulation'), production_ready, and explanation.
        """
        return {
            "timestamp": self.timestamp,
            "model_id": "full_canonical__lgbm",
            "action_mode": "simulation",
            "simulation_only": True,
            "production_ready": False,
            "capture_id": self.capture_id,
            "dataset": self.dataset or "unknown",
            "score": round(self.session_score, 4),
            "action": self.action.value,
            "action_display": self.action.display_label,
            "threshold_band": self._threshold_band(),
            "confidence_margin": round(self.confidence_margin, 4),
            "n_flows": self.n_flows,
            "n_flagged": self.n_flagged,
            "n_blocked_flows": self.n_blocked,
            "label": self.label,
            "correct": self.is_correct,
            "explanation": self._EXPLANATIONS.get(self.action.value, ""),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialisable dict."""
        d = self.to_event_dict()
        d["review_threshold"] = self.review_threshold
        d["block_threshold"] = self.block_threshold
        d["flow_records"] = [
            {
                "flow_id": fr.flow_id,
                "score": round(fr.score, 6),
                "action": fr.action.value,
                "margin": round(fr.confidence_margin, 4),
            }
            for fr in self.flow_records
        ]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Policy engine
# ─────────────────────────────────────────────────────────────────────────────

class OpenSetFirewallPolicy:
    """
    Open-set uncertainty-aware firewall policy.

    Usage
    -----
    policy = OpenSetFirewallPolicy.from_thresholds_file(path)
    decisions = policy.evaluate_dataframe(val_preds_df)
    report = policy.dashboard_report(decisions)
    print(policy.render_dashboard(report))
    """

    SIMULATION_DISCLAIMER = (
        "[SIMULATION ONLY] No real packet blocking. "
        "SIMULATED_BLOCK is an audit/log action."
    )

    def __init__(
        self,
        thresholds: OpenSetThresholds,
        score_col: str = "prob",
        session_col: str = "capture_id",
        label_col: str = "label",
        dataset_col: str = "dataset",
    ):
        self.thresholds = thresholds
        self.score_col = score_col
        self.session_col = session_col
        self.label_col = label_col
        self.dataset_col = dataset_col
        self._event_log: Deque[SessionDecisionV2] = deque(maxlen=10_000)

    # ── construction helpers ─────────────────────────────────────────────────

    @classmethod
    def from_thresholds_file(
        cls,
        path: Path,
        **kwargs,
    ) -> "OpenSetFirewallPolicy":
        """Load thresholds from JSON and construct policy."""
        thresholds = OpenSetThresholds.from_json(path)
        logger.info(
            f"Loaded thresholds from {path}: "
            f"review={thresholds.review_threshold:.6f}, "
            f"block={thresholds.block_threshold:.6f}"
        )
        return cls(thresholds=thresholds, **kwargs)

    @classmethod
    def default(cls, **kwargs) -> "OpenSetFirewallPolicy":
        """Construct with default validated thresholds (no file needed)."""
        return cls(thresholds=OpenSetThresholds.default(), **kwargs)

    # ── core decision logic ──────────────────────────────────────────────────

    def decide_score(self, score: float) -> Tuple[FirewallAction, float]:
        """
        Classify a single scalar score.

        Returns
        -------
        (action, confidence_margin)
        """
        action = self.thresholds.classify(score)
        margin = self.thresholds.confidence_margin(score)
        return action, margin

    def evaluate_session(
        self,
        flow_df: pd.DataFrame,
    ) -> SessionDecisionV2:
        """
        Evaluate a single session (one capture's flows).

        Parameters
        ----------
        flow_df : pd.DataFrame
            Rows for a single capture. Must contain score_col.

        Returns
        -------
        SessionDecisionV2
        """
        scores = flow_df[self.score_col].values.astype(float)
        session_score = float(np.mean(scores))

        action, margin = self.decide_score(session_score)

        capture_id = str(flow_df[self.session_col].iloc[0])
        label = int(flow_df[self.label_col].max()) if self.label_col in flow_df.columns else None
        dataset = str(flow_df[self.dataset_col].iloc[0]) if self.dataset_col in flow_df.columns else None

        # Per-flow records
        flow_records = []
        for _, row in flow_df.iterrows():
            fs = float(row[self.score_col])
            fa, fm = self.decide_score(fs)
            flow_records.append(FlowRecord(
                flow_id=str(row.get("flow_id", "")),
                capture_id=capture_id,
                score=fs,
                action=fa,
                confidence_margin=fm,
                dataset=dataset,
            ))

        n_flagged = sum(1 for fr in flow_records if fr.action.requires_review)
        n_blocked = sum(1 for fr in flow_records if fr.action.is_blocking)

        decision = SessionDecisionV2(
            capture_id=capture_id,
            session_score=session_score,
            action=action,
            confidence_margin=margin,
            n_flows=len(scores),
            n_flagged=n_flagged,
            n_blocked=n_blocked,
            review_threshold=self.thresholds.review_threshold,
            block_threshold=self.thresholds.block_threshold,
            flow_records=flow_records,
            label=label,
            dataset=dataset,
        )

        self._event_log.append(decision)
        return decision

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> List[SessionDecisionV2]:
        """
        Evaluate all sessions in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Flow-level predictions for multiple captures.

        Returns
        -------
        List[SessionDecisionV2]
        """
        decisions = []
        for cid, group in df.groupby(self.session_col):
            dec = self.evaluate_session(group)
            decisions.append(dec)
        return decisions

    # ── aggregate metrics ────────────────────────────────────────────────────

    def compute_metrics(
        self,
        decisions: List[SessionDecisionV2],
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics across a list of decisions.

        Returns per-action counts, rates, and (if labels present)
        recall/FPR at each tier.
        """
        n = len(decisions)
        if n == 0:
            return {"n_sessions": 0}

        counts = {a: 0 for a in FirewallAction}
        for d in decisions:
            counts[d.action] += 1

        labeled = [d for d in decisions if d.label is not None]
        vpn = [d for d in labeled if d.label == 1]
        benign = [d for d in labeled if d.label == 0]

        metrics: Dict[str, Any] = {
            "n_sessions": n,
            "n_pass": counts[FirewallAction.PASS],
            "n_flag_review": counts[FirewallAction.FLAG_REVIEW],
            "n_simulated_block": counts[FirewallAction.SIMULATED_BLOCK],
            "pass_rate": counts[FirewallAction.PASS] / n,
            "flag_rate": counts[FirewallAction.FLAG_REVIEW] / n,
            "block_rate": counts[FirewallAction.SIMULATED_BLOCK] / n,
            "review_threshold": self.thresholds.review_threshold,
            "block_threshold": self.thresholds.block_threshold,
            "simulation_only": True,
        }

        # Labelled metrics
        if labeled:
            metrics["n_labeled"] = len(labeled)
            metrics["n_vpn"] = len(vpn)
            metrics["n_benign"] = len(benign)

            # VPN recall at each tier
            if vpn:
                vpn_block = sum(1 for d in vpn if d.action == FirewallAction.SIMULATED_BLOCK)
                vpn_flag_or_block = sum(1 for d in vpn if d.action.requires_review)
                metrics["vpn_block_recall"] = vpn_block / len(vpn)
                metrics["vpn_detected_recall"] = vpn_flag_or_block / len(vpn)

            # Benign FPR at each tier
            if benign:
                b_block = sum(1 for d in benign if d.action == FirewallAction.SIMULATED_BLOCK)
                b_flag_or_block = sum(1 for d in benign if d.action.requires_review)
                metrics["benign_block_fpr"] = b_block / len(benign)
                metrics["benign_review_fpr"] = b_flag_or_block / len(benign)

        return metrics

    # ── dashboard ─────────────────────────────────────────────────────────────

    def dashboard_report(
        self,
        decisions: List[SessionDecisionV2],
    ) -> Dict[str, Any]:
        """
        Produce a structured dashboard report dict.

        Contains:
        - status_cards : summary KPI cards
        - recent_events : last N events table
        - metrics : full aggregate metrics
        - policy_info : threshold provenance
        - disclaimer : simulation-only notice
        """
        metrics = self.compute_metrics(decisions)
        n = metrics["n_sessions"]

        # ── Status cards ─────────────────────────────────────────────────────────────
        status_cards = [
            {
                "id": "pass_card",
                "title": "Sessions Passed",
                "value": metrics["n_pass"],
                "rate": f"{metrics['pass_rate']*100:.1f}%",
                "color": "green",
                "icon": "✅",
                "description": f"Score < {self.thresholds.review_threshold:.4f}",
            },
            {
                "id": "review_card",
                "title": "Flagged for Review",
                "value": metrics["n_flag_review"],
                "rate": f"{metrics['flag_rate']*100:.1f}%",
                "color": "orange",
                "icon": "",
                "description": (
                    f"Score in [{self.thresholds.review_threshold:.4f}, "
                    f"{self.thresholds.block_threshold:.4f})"
                ),
            },
            {
                "id": "block_card",
                "title": "Simulated Blocks",
                "value": metrics["n_simulated_block"],
                "rate": f"{metrics['block_rate']*100:.1f}%",
                "color": "red",
                "icon": "",
                "description": (
                    f"Score ≥ {self.thresholds.block_threshold:.4f} "
                    "[SIMULATION ONLY]"
                ),
            },
        ]

        # Optional accuracy card (labelled data only)
        if "vpn_detected_recall" in metrics:
            status_cards.append({
                "id": "recall_card",
                "title": "VPN Detection Rate",
                "value": f"{metrics['vpn_detected_recall']*100:.1f}%",
                "rate": f"{metrics.get('vpn_block_recall', 0)*100:.1f}% auto-blocked",
                "color": "blue",
                "icon": "",
                "description": "FLAG_REVIEW + SIMULATED_BLOCK vs all VPN",
            })

        if "benign_block_fpr" in metrics:
            status_cards.append({
                "id": "fpr_card",
                "title": "Benign Block Rate",
                "value": f"{metrics['benign_block_fpr']*100:.2f}%",
                "rate": f"{metrics.get('benign_review_fpr', 0)*100:.1f}% reviewed",
                "color": "purple",
                "icon": "️",
                "description": "% benign sessions auto-blocked (target: 0%)",
            })

        # ── Recent events table ───────────────────────────────────────────────
        recent = sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:50]
        events_table = [d.to_event_dict() for d in recent]

        # ── Policy info ───────────────────────────────────────────────────────
        policy_info = {
            "model_id": self.thresholds.model_id,
            "policy_type": "open_set_three_tier",
            "review_threshold": self.thresholds.review_threshold,
            "review_threshold_basis": self.thresholds.review_basis,
            "block_threshold": self.thresholds.block_threshold,
            "block_threshold_basis": self.thresholds.block_basis,
            "source_split": self.thresholds.source_split,
            "simulation_only": True,
            "actions": {
                "PASS": f"score < {self.thresholds.review_threshold:.6f}",
                "FLAG_REVIEW": (
                    f"{self.thresholds.review_threshold:.6f} "
                    f"<= score < {self.thresholds.block_threshold:.6f}"
                ),
                "SIMULATED_BLOCK": f"score >= {self.thresholds.block_threshold:.6f}",
            },
        }

        return {
            "generated_at": datetime.datetime.now().isoformat(),
            "status_cards": status_cards,
            "recent_events": events_table,
            "metrics": metrics,
            "policy_info": policy_info,
            "disclaimer": self.SIMULATION_DISCLAIMER,
        }

    def render_dashboard(
        self,
        report: Optional[Dict[str, Any]] = None,
        decisions: Optional[List[SessionDecisionV2]] = None,
    ) -> str:
        """
        Render the dashboard as a human-readable text table.

        Pass either a pre-built `report` dict or raw `decisions`.
        """
        if report is None:
            if decisions is None:
                decisions = list(self._event_log)
            report = self.dashboard_report(decisions)

        lines = [
            "=" * 72,
            "  VPN FIREWALL — OPEN-SET POLICY DASHBOARD",
            f"  {report['disclaimer']}",
            f"  Generated: {report['generated_at']}",
            "=" * 72,
            "",
        ]

        # Status cards
        lines.append("  ── STATUS CARDS ─────────────────────────────────────")
        for card in report["status_cards"]:
            lines.append(
                f"  {card['icon']}  {card['title']:30s}  "
                f"{str(card['value']):>8}  ({card['rate']})  {card['description']}"
            )
        lines.append("")

        # Policy info
        pi = report["policy_info"]
        lines.append("  ── POLICY INFO ──────────────────────────────────────")
        lines.append(f"  Model:              {pi['model_id']}")
        lines.append(f"  Policy type:        {pi['policy_type']}")
        lines.append(f"  review_threshold:   {pi['review_threshold']:.6f}  ({pi['review_threshold_basis']})")
        lines.append(f"  block_threshold:    {pi['block_threshold']:.6f}  ({pi['block_threshold_basis']})")
        lines.append(f"  Source split:       {pi['source_split']}")
        lines.append("")
        lines.append("  Action rules:")
        for action, rule in pi["actions"].items():
            lines.append(f"    {action:20s} → {rule}")
        lines.append("")

        # Aggregate metrics
        m = report["metrics"]
        lines.append("  ── AGGREGATE METRICS ────────────────────────────────")
        lines.append(f"  Total sessions:     {m['n_sessions']}")
        lines.append(f"  ✅  PASS:            {m['n_pass']}  ({m['pass_rate']*100:.1f}%)")
        lines.append(f"    FLAG_REVIEW:     {m['n_flag_review']}  ({m['flag_rate']*100:.1f}%)")
        lines.append(f"    SIMULATED_BLOCK: {m['n_simulated_block']}  ({m['block_rate']*100:.1f}%)")
        if "vpn_detected_recall" in m:
            lines.append("")
            lines.append(f"  VPN recall (flag+block): {m['vpn_detected_recall']*100:.1f}%")
            lines.append(f"  VPN recall (block only): {m.get('vpn_block_recall', 0)*100:.1f}%")
        if "benign_block_fpr" in m:
            lines.append(f"  Benign block FPR:        {m['benign_block_fpr']*100:.2f}%")
            lines.append(f"  Benign review FPR:       {m.get('benign_review_fpr', 0)*100:.1f}%")
        lines.append("")

        # Recent events table
        events = report["recent_events"][:15]
        if events:
            lines.append("  ── RECENT EVENTS (last 15) ──────────────────────────")
            hdr = f"  {'Capture':35s} {'Score':>7} {'Action':20s} {'Margin':>7} {'Flows':>6}"
            lines.append(hdr)
            lines.append("  " + "-" * 80)
            for ev in events:
                correct_marker = ""
                if ev.get("correct") is True:
                    correct_marker = " ✓"
                elif ev.get("correct") is False:
                    correct_marker = " ✗"
                lines.append(
                    f"  {ev['capture_id'][:35]:35s} "
                    f"{ev['score']:7.4f} "
                    f"{ev['action_display']:22s} "
                    f"{ev['confidence_margin']:7.4f} "
                    f"{ev['n_flows']:6d}"
                    f"{correct_marker}"
                )
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ── event log helpers ────────────���───────────────────────────────────────

    def events_as_dataframe(self) -> pd.DataFrame:
        """Return the event log as a tidy DataFrame."""
        if not self._event_log:
            return pd.DataFrame()
        rows = [d.to_event_dict() for d in self._event_log]
        return pd.DataFrame(rows)

    def clear_event_log(self) -> None:
        self._event_log.clear()

    @property
    def n_events(self) -> int:
        return len(self._event_log)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(
    thresholds_json: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> OpenSetFirewallPolicy:
    """
    Load the open-set policy.

    If thresholds_json is None, tries the default location:
        <repo_root>/artifacts/final_transfer/models/full_canonical__lgbm/thresholds.json

    Falls back to built-in defaults if the file is missing.
    """
    if thresholds_json is None and repo_root is not None:
        thresholds_json = (
            repo_root
            / "artifacts"
            / "final_transfer"
            / "models"
            / "full_canonical__lgbm"
            / "thresholds.json"
        )

    if thresholds_json is not None and Path(thresholds_json).exists():
        return OpenSetFirewallPolicy.from_thresholds_file(thresholds_json)

    logger.warning(
        "thresholds.json not found — using built-in default thresholds "
        "(review=0.027090, block=0.165365)."
    )
    return OpenSetFirewallPolicy.default()
