#!/usr/bin/env python3
"""Final promotion audit for top firewall candidates.

Compares:
1) artifacts/ensemble/diverse_bagging_robust9
2) artifacts/balanced_bagging_firewall_tuned_ensemble_3dataset_REFRESH
3) artifacts/balanced_bagging_firewall_tuned_ensemble

Outputs:
- reports/tables/final_promotion_audit.csv
- reports/tables/final_promotion_audit.md
- reports/tables/final_promotion_audit.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "diverse_bagging_robust9": ROOT / "artifacts" / "ensemble" / "diverse_bagging_robust9",
    "3dataset_REFRESH": ROOT / "artifacts" / "balanced_bagging_firewall_tuned_ensemble_3dataset_REFRESH",
    "current_default": ROOT / "artifacts" / "balanced_bagging_firewall_tuned_ensemble",
}

# Ratios from configs/deployment.yaml
STRICT_FLAG_RATIO = 0.95   # 0.8338 / 0.8777
BAL_FLAG_RATIO = 0.50      # 0.0443 / 0.0886


@dataclass
class PolicyEval:
    block_threshold: float
    flag_threshold: float
    val_fpr: float
    test_fpr: float
    test_recall: float
    block_n: int
    flag_n: int
    pass_n: int


def _safe_auc(y: pd.Series, s: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_pr(y: pd.Series, s: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _ece(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p_pred, edges[1:-1], right=True)
    total = len(y_true)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(np.mean(p_pred[m]))
        acc = float(np.mean(y_true[m]))
        ece += (np.sum(m) / total) * abs(acc - conf)
    return float(ece)


def _p90(x: np.ndarray) -> float:
    return float(np.percentile(x, 90)) if len(x) else 0.0


def _wt5(x: np.ndarray) -> float:
    vals = np.sort(np.asarray(x))[::-1][:5]
    if len(vals) == 0:
        return 0.0
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[: len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))


def _session_df(df: pd.DataFrame, score_col: str, agg_fn) -> pd.DataFrame:
    g = df.groupby("capture_id", as_index=False)
    s = g[score_col].apply(lambda x: agg_fn(x.to_numpy(dtype=float))).rename(columns={score_col: "score"})
    y = g["label"].max()
    ds = g["dataset"].first()
    return s.merge(y, on="capture_id").merge(ds, on="capture_id")


def _threshold_at_target_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    b = np.sort(np.asarray(benign_scores, dtype=float))
    if len(b) == 0:
        return 1.0
    if target_fpr <= 0.0:
        return float(np.nextafter(b.max(), np.inf))
    q = 1.0 - target_fpr
    # Conservative threshold: choose "higher" quantile so empirical FPR <= target
    thr = float(np.quantile(b, q, method="higher"))
    return thr


def _class_metrics(scores: np.ndarray, y: np.ndarray, block_thr: float, flag_thr: float) -> tuple[float, float, int, int, int]:
    block = scores >= block_thr
    flag = (scores >= flag_thr) & (~block)
    pas = ~(block | flag)

    y = np.asarray(y)
    vpn = y == 1
    benign = y == 0

    rec = float((block & vpn).sum() / max(vpn.sum(), 1))
    fpr = float((block & benign).sum() / max(benign.sum(), 1))
    return rec, fpr, int(block.sum()), int(flag.sum()), int(pas.sum())


def _eval_policy(val_sess: pd.DataFrame, test_sess: pd.DataFrame, target_fpr: float, flag_ratio: float) -> PolicyEval:
    benign_val = val_sess.loc[val_sess["label"] == 0, "score"].to_numpy()
    block_thr = _threshold_at_target_fpr(benign_val, target_fpr=target_fpr)
    flag_thr = block_thr * flag_ratio

    _, val_fpr, _, _, _ = _class_metrics(
        val_sess["score"].to_numpy(), val_sess["label"].to_numpy(), block_thr, flag_thr
    )
    rec, test_fpr, block_n, flag_n, pass_n = _class_metrics(
        test_sess["score"].to_numpy(), test_sess["label"].to_numpy(), block_thr, flag_thr
    )
    return PolicyEval(
        block_threshold=float(block_thr),
        flag_threshold=float(flag_thr),
        val_fpr=float(val_fpr),
        test_fpr=float(test_fpr),
        test_recall=float(rec),
        block_n=block_n,
        flag_n=flag_n,
        pass_n=pass_n,
    )


def _bootstrap_thr_stability(benign_scores: np.ndarray, target_fpr: float, n_boot: int = 300, seed: int = 42) -> dict:
    r = np.random.default_rng(seed)
    b = np.asarray(benign_scores, dtype=float)
    if len(b) < 10:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan"), "p05": float("nan"), "p95": float("nan")}
    thrs = []
    for _ in range(n_boot):
        samp = r.choice(b, size=len(b), replace=True)
        thrs.append(_threshold_at_target_fpr(samp, target_fpr=target_fpr))
    thrs = np.asarray(thrs)
    m = float(np.mean(thrs))
    sd = float(np.std(thrs))
    cv = float(sd / abs(m)) if m != 0 else float("nan")
    return {
        "mean": m,
        "std": sd,
        "cv": cv,
        "p05": float(np.quantile(thrs, 0.05)),
        "p95": float(np.quantile(thrs, 0.95)),
    }


def _load_lodo_summary() -> dict:
    p = ROOT / "artifacts" / "lood_firewall_tuned" / "lodo_summary.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out = {}
    for e in data:
        hold = e.get("holdout")
        iso = e.get("isotonic", {})
        out[hold] = {
            "auc": iso.get("holdout_auc"),
            "session_auc": iso.get("holdout_session_roc_auc"),
            "block_recall@0": iso.get("holdout_block_recall_at_zero_fp"),
        }
    return out


def main() -> None:
    lodo = _load_lodo_summary()
    rows = []

    for name, d in CANDIDATES.items():
        pred = d / "predictions.csv"
        metrics = d / "metrics.json"
        if not pred.exists() or not metrics.exists():
            continue

        df = pd.read_csv(pred)
        m = json.loads(metrics.read_text())

        # Flow-level headline on isotonic test
        test = df[df["split"] == "test"].copy()
        val = df[df["split"] == "val"].copy()

        flow_auc = _safe_auc(test["label"], test["prob_iso"])
        flow_pr = _safe_pr(test["label"], test["prob_iso"])

        per_ds_auc = {}
        for ds, g in test.groupby("dataset"):
            per_ds_auc[str(ds)] = _safe_auc(g["label"], g["prob_iso"])

        # Session-level pooled AUC with balanced policy score definition (wt5 + prob_iso)
        val_sess_bal = _session_df(val, "prob_iso", _wt5)
        test_sess_bal = _session_df(test, "prob_iso", _wt5)
        sess_auc = _safe_auc(test_sess_bal["label"], test_sess_bal["score"])

        # Block recall at FPR=0 from test-session distribution
        benign_max = test_sess_bal.loc[test_sess_bal["label"] == 0, "score"].max()
        thr0 = float(np.nextafter(benign_max, np.inf)) if pd.notna(benign_max) else 1.0
        block0 = (test_sess_bal["score"] >= thr0).to_numpy()
        y_s = test_sess_bal["label"].to_numpy()
        rec0 = float((block0 & (y_s == 1)).sum() / max((y_s == 1).sum(), 1))

        # Balanced policy (wt5 + prob_iso, target val FPR<=0.01)
        bal = _eval_policy(val_sess_bal, test_sess_bal, target_fpr=0.01, flag_ratio=BAL_FLAG_RATIO)

        # Strict policy (p90 + prob_raw, target val FPR=0)
        val_sess_strict = _session_df(val, "prob_raw", _p90)
        test_sess_strict = _session_df(test, "prob_raw", _p90)
        strict = _eval_policy(val_sess_strict, test_sess_strict, target_fpr=0.0, flag_ratio=STRICT_FLAG_RATIO)

        # Calibration quality on test
        cal_rows = []
        for col, cal in (("prob_raw", "raw"), ("prob_iso", "isotonic"), ("prob_platt", "platt")):
            if col not in test.columns:
                continue
            p = np.clip(test[col].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            y = test["label"].to_numpy(dtype=int)
            cal_rows.append(
                {
                    "cal": cal,
                    "brier": float(brier_score_loss(y, p)),
                    "ece": _ece(y, p, bins=10),
                }
            )
        cal_df = pd.DataFrame(cal_rows).sort_values(["ece", "brier"])
        best_cal = cal_df.iloc[0].to_dict() if len(cal_df) else {"cal": None, "brier": np.nan, "ece": np.nan}

        # Threshold stability (bootstrap on val benign sessions)
        b_bal = val_sess_bal.loc[val_sess_bal["label"] == 0, "score"].to_numpy()
        b_str = val_sess_strict.loc[val_sess_strict["label"] == 0, "score"].to_numpy()
        st_bal = _bootstrap_thr_stability(b_bal, target_fpr=0.01)
        st_str = _bootstrap_thr_stability(b_str, target_fpr=0.0)

        # Presence checks asked by user
        has_calibration = all((d / x).exists() for x in ("isotonic_calibrator.pkl", "platt_calibrator.pkl"))
        has_session_metrics = bool((m.get("isotonic", {}).get("test_overall", {}).get("session_metrics") or {}))
        has_threshold_policy = "fpr_0.0" in (m.get("isotonic", {}).get("test_overall", {}))
        has_block_flag_pass_metrics = has_session_metrics  # stored only when session metrics are present

        lodo_text = "n/a"
        if name in {"3dataset_REFRESH", "current_default"} and lodo:
            vals = [v.get("auc") for v in lodo.values() if v.get("auc") is not None]
            if vals:
                lodo_text = f"isotonic holdout AUC min/mean/max={min(vals):.3f}/{np.mean(vals):.3f}/{max(vals):.3f}"

        rows.append(
            {
                "candidate": name,
                "path": str(d.relative_to(ROOT)).replace("\\", "/"),
                "overall_flow_auc_iso": flow_auc,
                "overall_pr_auc_iso": flow_pr,
                "per_dataset_auc_iso": json.dumps(per_ds_auc, sort_keys=True),
                "session_auc_wt5_iso": sess_auc,
                "block_recall_at_fpr0_wt5_iso": rec0,
                "balanced_recall": bal.test_recall,
                "balanced_test_fpr": bal.test_fpr,
                "balanced_val_fpr": bal.val_fpr,
                "balanced_block_thr": bal.block_threshold,
                "strict_recall": strict.test_recall,
                "strict_test_fpr": strict.test_fpr,
                "strict_val_fpr": strict.val_fpr,
                "strict_block_thr": strict.block_threshold,
                "balanced_BLOCK/FLAG/PASS_test": f"{bal.block_n}/{bal.flag_n}/{bal.pass_n}",
                "strict_BLOCK/FLAG/PASS_test": f"{strict.block_n}/{strict.flag_n}/{strict.pass_n}",
                "calibration_best": best_cal.get("cal"),
                "calibration_best_ece": best_cal.get("ece"),
                "calibration_best_brier": best_cal.get("brier"),
                "thr_stability_bal_cv": st_bal.get("cv"),
                "thr_stability_strict_cv": st_str.get("cv"),
                "thr_stability_bal_p05_p95": f"{st_bal.get('p05'):.4f}..{st_bal.get('p95'):.4f}" if pd.notna(st_bal.get("p05")) else "nan",
                "thr_stability_strict_p05_p95": f"{st_str.get('p05'):.4f}..{st_str.get('p95'):.4f}" if pd.notna(st_str.get("p05")) else "nan",
                "has_calibration_artifacts": has_calibration,
                "has_session_metrics_in_metrics_json": has_session_metrics,
                "has_threshold_policy_metrics_json": has_threshold_policy,
                "has_block_flag_pass_metrics_json": has_block_flag_pass_metrics,
                "lodo_result": lodo_text,
            }
        )

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "final_promotion_audit.csv"
    out_json = OUT_DIR / "final_promotion_audit.json"
    out_md = OUT_DIR / "final_promotion_audit.md"

    out.to_csv(out_csv, index=False)
    out.to_json(out_json, orient="records", indent=2)

    md = [
        "# Final Promotion Audit (Top 3 Candidates)",
        "",
        "| Candidate | Flow AUC | PR-AUC | Session AUC | Block@FPR0 | Balanced R/FPR | Strict R/FPR | Calibration(best) | LODO |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for r in out.to_dict(orient="records"):
        md.append(
            "| `{candidate}` | {overall_flow_auc_iso:.4f} | {overall_pr_auc_iso:.4f} | {session_auc_wt5_iso:.4f} | {block_recall_at_fpr0_wt5_iso:.4f} | "
            "{balanced_recall:.4f}/{balanced_test_fpr:.4f} | {strict_recall:.4f}/{strict_test_fpr:.4f} | {calibration_best} (ECE={calibration_best_ece:.4f}) | {lodo_result} |".format(**r)
        )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv.relative_to(ROOT)}")
    print(f"Wrote: {out_json.relative_to(ROOT)}")
    print(f"Wrote: {out_md.relative_to(ROOT)}")
    print("\nAudit table:")
    with pd.option_context("display.max_colwidth", 120, "display.width", 220):
        print(
            out[
                [
                    "candidate",
                    "overall_flow_auc_iso",
                    "overall_pr_auc_iso",
                    "session_auc_wt5_iso",
                    "block_recall_at_fpr0_wt5_iso",
                    "balanced_recall",
                    "balanced_test_fpr",
                    "strict_recall",
                    "strict_test_fpr",
                    "has_session_metrics_in_metrics_json",
                    "lodo_result",
                ]
            ]
        )


if __name__ == "__main__":
    main()

