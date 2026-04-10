#!/usr/bin/env python
"""
Apply threshold-stability extensions to existing notebooks.

Tasks:
  A — Low-variance threshold selection  (notebook 66)
  B — Conservative policy threshold     (notebook 56)
  C — Threshold stability across seeds  (notebooks 66 + split_stability_audit)

Usage:
  python scripts/apply_threshold_stability_extensions.py
"""

import json
import copy
import uuid
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
NB66 = REPO / "notebooks" / "66_splitter_v2_stability_audit.ipynb"
NB56 = REPO / "notebooks" / "56_firewall_decision_evaluation.ipynb"
NB_SPLIT = REPO / "notebooks" / "split_stability_audit.ipynb"
REPORT_DIR = REPO / "artifacts" / "clean_pipeline"


# ── helpers ──────────────────────────────────────────────────────
def _cell_id():
    return uuid.uuid4().hex[:8]


def _code_cell(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    elif not src:
        src = [""]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def _md_cell(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    elif not src:
        src = [""]
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": src,
    }


def _find_cell_index(nb, anchor_text):
    """Return the index of the first cell whose source contains anchor_text."""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if anchor_text in src:
            return i
    return -1


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 66 — Tasks A + C
# ══════════════════════════════════════════════════════════════════

TASK_C_HEADER = """\
---
## Section 5 — Threshold Stability Analysis (Task C)

**Purpose:** Treat the per-seed best-F1 threshold as a first-class stability
metric—alongside recall, FPR, and AUC—and formally audit whether threshold
selection itself is fragile.

> *This section reuses `met_df` produced in Section 3.*"""

TASK_C_CODE_STATS = '''\
# ── 5a. Per-seed threshold table & statistics (v2 only) ──────────
v2_met = met_df[met_df["version"] == "v2"].copy().reset_index(drop=True)

print("Per-Seed Metrics — V2 (best-F1 threshold rule):\\n")
display(v2_met[["seed", "threshold", "roc_auc", "recall", "precision", "fpr"]])

thr_vals = v2_met["threshold"].values

thr_stats = {
    "mean":   float(np.mean(thr_vals)),
    "std":    float(np.std(thr_vals, ddof=1)) if len(thr_vals) > 1 else 0.0,
    "min":    float(np.min(thr_vals)),
    "max":    float(np.max(thr_vals)),
    "median": float(np.median(thr_vals)),
    "p25":    float(np.percentile(thr_vals, 25)),
    "p75":    float(np.percentile(thr_vals, 75)),
}
thr_stats["iqr"] = thr_stats["p75"] - thr_stats["p25"]
thr_stats["cv"]  = thr_stats["std"] / thr_stats["mean"] if thr_stats["mean"] > 0 else float("nan")

def classify_threshold_stability(cv, std):
    if cv < 0.05 and std < 0.02:
        return "STABLE"
    elif cv < 0.15 and std < 0.05:
        return "MODERATELY STABLE"
    else:
        return "FRAGILE"

thr_verdict = classify_threshold_stability(thr_stats["cv"], thr_stats["std"])

# Build combined stability summary (threshold + metrics)
summary_rows = []
for metric_name in ["threshold", "roc_auc", "recall", "precision", "fpr"]:
    vals = v2_met[metric_name].values
    m, s = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    cv = s / m if m > 0 else float("nan")
    if metric_name == "threshold":
        verd = classify_threshold_stability(cv, s)
    elif metric_name in ("recall", "precision"):
        verd = classify_var(s, (0.01, 0.03))
    elif metric_name == "fpr":
        verd = classify_var(s, (0.005, 0.02))
    else:
        verd = classify_var(s, (0.005, 0.015))
    summary_rows.append({
        "metric": metric_name,
        "mean": round(m, 4), "std": round(s, 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
        "cv": round(cv, 4),
        "verdict": verd,
    })
stability_summary_df = pd.DataFrame(summary_rows)
print("\\nCombined Stability Summary (V2, threshold as first-class metric):")
display(stability_summary_df)

print(f"\\n  Threshold verdict: {thr_verdict}")
print(f"  Mean={thr_stats[\'mean\']:.4f}  Std={thr_stats[\'std\']:.4f}  "
      f"CV={thr_stats[\'cv\']:.4f}  Range=[{thr_stats[\'min\']:.3f}, {thr_stats[\'max\']:.3f}]  "
      f"IQR={thr_stats[\'iqr\']:.4f}")
'''

TASK_C_CODE_PLOTS = '''\
# ── 5b. Threshold stability visualizations ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Threshold by seed
ax = axes[0, 0]
ax.bar(v2_met["seed"].astype(str), v2_met["threshold"], color="#9C27B0", alpha=0.8)
ax.axhline(thr_stats["mean"], ls="--", color="black", alpha=0.7,
           label=f'Mean={thr_stats["mean"]:.3f}')
ax.axhline(thr_stats["median"], ls=":", color="blue", alpha=0.7,
           label=f'Median={thr_stats["median"]:.3f}')
ax.set_title(f'Best-F1 Threshold by Seed  (std={thr_stats["std"]:.4f})')
ax.set_xlabel("Seed"); ax.set_ylabel("Threshold")
ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=45)

# (2) Histogram / KDE of thresholds
ax = axes[0, 1]
ax.hist(thr_vals, bins=min(12, len(thr_vals)), color="#9C27B0",
        alpha=0.6, edgecolor="black", label="count")
ax.axvline(thr_stats["mean"], ls="--", color="red", label=f'Mean={thr_stats["mean"]:.3f}')
ax.axvline(thr_stats["median"], ls=":", color="blue", label=f'Median={thr_stats["median"]:.3f}')
ax.set_title("Distribution of Per-Seed Thresholds")
ax.set_xlabel("Threshold"); ax.set_ylabel("Count"); ax.legend(fontsize=9)

# (3) Threshold vs Recall
ax = axes[1, 0]
ax.scatter(v2_met["threshold"], v2_met["recall"], c="#4CAF50", s=60, edgecolors="black", zorder=3)
z = np.polyfit(v2_met["threshold"], v2_met["recall"], 1)
xs = np.linspace(v2_met["threshold"].min(), v2_met["threshold"].max(), 50)
ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.4)
ax.set_xlabel("Threshold"); ax.set_ylabel("Test Recall")
ax.set_title("Threshold vs Recall (per seed)")

# (4) Threshold vs FPR
ax = axes[1, 1]
ax.scatter(v2_met["threshold"], v2_met["fpr"], c="#EF5350", s=60, edgecolors="black", zorder=3)
z = np.polyfit(v2_met["threshold"], v2_met["fpr"], 1)
ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.4)
ax.set_xlabel("Threshold"); ax.set_ylabel("Test FPR")
ax.set_title("Threshold vs FPR (per seed)")

plt.suptitle("Threshold Stability Analysis — V2 Splitter, 20 Seeds", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
'''

TASK_C_INTERPRETATION = """\
### Interpretation — Threshold Stability

The per-seed best-F1 threshold is treated here as a first-class stability metric.
Key observations:

* **Coefficient of variation (CV)** and **standard deviation** of the threshold
  quantify how much the operating point moves with different random splits.
* If `std(threshold)` is comparable to or larger than `std(recall)`, this confirms
  that threshold selection is a *primary driver* of metric instability—not just
  the model's discriminative ability.
* The scatter plots above show the per-seed relationship between the chosen
  threshold and downstream recall/FPR.  A strong negative correlation between
  threshold and recall is expected (higher threshold → stricter → lower recall).
* **This does NOT imply** that fixing the threshold solves the generalization
  problem.  Representation-level domain shift remains the root cause of
  cross-dataset variance.  Threshold instability is an *amplifier*, not the
  fundamental failure."""

# ── Task A cells ─────────────────────────────────────────────────

TASK_A_HEADER = """\
---
## Section 6 — Low-Variance Threshold Selection (Task A)

**Purpose:** Evaluate alternative threshold rules that trade small recall
losses for substantially lower metric variance across seeds.

**Approach:**
1. Re-run the v2 training loop collecting per-seed test/val probabilities.
2. Sweep a candidate threshold grid, computing mean & std of all metrics.
3. Select and compare: *per-seed best-F1* (baseline), *median threshold*,
   *best mean-F1*, *lowest recall-std*, *balanced stability*."""

TASK_A_CODE_COLLECT = '''\
# ── 6a. Collect per-seed probabilities (v2 only) ────────────────
print("Re-running v2 seed loop with probability collection…")
seed_data = {}
for seed in SEEDS:
    cfg = CleanSplitConfig(seed=seed, splitter_version=2)
    with contextlib.redirect_stdout(io.StringIO()):
        tmp = make_clean_split(df_raw.copy(), cfg)
    tr = tmp[tmp["split"]=="train"]; va = tmp[tmp["split"]=="val"]; te = tmp[tmp["split"]=="test"]
    X_tr = tr[feat_cols].values.astype(np.float32)
    y_tr = tr["label"].values
    X_va = va[feat_cols].values.astype(np.float32)
    y_va = va["label"].values
    X_te = te[feat_cols].values.astype(np.float32)
    y_te = te["label"].values
    for a in [X_tr, X_va, X_te]:
        np.nan_to_num(a, copy=False, nan=0, posinf=0, neginf=0)
    m = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=seed, verbosity=-1, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    seed_data[seed] = {
        "val_prob":  m.predict_proba(X_va)[:, 1],
        "test_prob": m.predict_proba(X_te)[:, 1],
        "y_val":     y_va,
        "y_test":    y_te,
    }
    print(f"  seed={seed} done")
print(f"\\nCollected probabilities for {len(seed_data)} seeds.")
'''

TASK_A_CODE_SWEEP = '''\
# ── 6b. Threshold sweep ─────────────────────────────────────────
from sklearn.metrics import f1_score

# Build candidate grid: quantiles of pooled val scores + regular grid
all_val = np.concatenate([sd["val_prob"] for sd in seed_data.values()])
quantile_grid = np.quantile(all_val, np.linspace(0.05, 0.95, 50))
regular_grid  = np.arange(0.10, 0.95, 0.01)
candidate_thresholds = np.unique(np.round(
    np.concatenate([quantile_grid, regular_grid, v2_met["threshold"].values]),
    decimals=4,
))
candidate_thresholds.sort()

sweep_rows = []
for thr in candidate_thresholds:
    recs, fprs, pres, f1s = [], [], [], []
    for seed in SEEDS:
        sd = seed_data[seed]
        yp = (sd["test_prob"] >= thr).astype(int)
        yt = sd["y_test"]
        rec = recall_score(yt, yp, zero_division=0)
        pre = precision_score(yt, yp, zero_division=0)
        fpr_v = ((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)
        f1v = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
        recs.append(rec); fprs.append(fpr_v); pres.append(pre); f1s.append(f1v)
    sweep_rows.append({
        "threshold":      round(float(thr), 4),
        "mean_recall":    np.mean(recs),  "std_recall":    np.std(recs, ddof=1),
        "mean_fpr":       np.mean(fprs),  "std_fpr":       np.std(fprs, ddof=1),
        "mean_precision": np.mean(pres),  "std_precision": np.std(pres, ddof=1),
        "mean_f1":        np.mean(f1s),   "std_f1":        np.std(f1s, ddof=1),
    })

sweep_df = pd.DataFrame(sweep_rows)

# AUC (threshold-independent, same for all rules)
auc_list = []
for seed in SEEDS:
    sd = seed_data[seed]
    if len(np.unique(sd["y_test"])) > 1:
        auc_list.append(roc_auc_score(sd["y_test"], sd["test_prob"]))
    else:
        auc_list.append(float("nan"))
mean_auc = float(np.nanmean(auc_list))
std_auc  = float(np.nanstd(auc_list, ddof=1))

print(f"Sweep: {len(sweep_df)} candidate thresholds × {len(SEEDS)} seeds")
print(f"AUC (threshold-independent): {mean_auc:.4f} ± {std_auc:.4f}")
'''

TASK_A_CODE_COMPARE = '''\
# ── 6c. Select & compare threshold rules ────────────────────────

def _nearest(df, col, val):
    return df.iloc[(df[col] - val).abs().argsort()[:1]].iloc[0]

best_f1_thresholds = v2_met["threshold"].values

# 1. Baseline: per-seed best-F1
baseline = {
    "threshold_rule": "per_seed_best_F1",
    "threshold_value": "varies",
    "mean_auc": round(mean_auc, 4), "std_auc": round(std_auc, 4),
    "mean_recall":    round(float(v2_met["recall"].mean()), 4),
    "std_recall":     round(float(v2_met["recall"].std(ddof=1)), 4),
    "mean_fpr":       round(float(v2_met["fpr"].mean()), 4),
    "std_fpr":        round(float(v2_met["fpr"].std(ddof=1)), 4),
    "mean_precision": round(float(v2_met["precision"].mean()), 4),
    "std_precision":  round(float(v2_met["precision"].std(ddof=1)), 4),
}

def _rule_row(name, thr_val, r):
    return {
        "threshold_rule": name,
        "threshold_value": round(float(thr_val), 4),
        "mean_auc": round(mean_auc, 4), "std_auc": round(std_auc, 4),
        "mean_recall":    round(float(r["mean_recall"]), 4),
        "std_recall":     round(float(r["std_recall"]), 4),
        "mean_fpr":       round(float(r["mean_fpr"]), 4),
        "std_fpr":        round(float(r["std_fpr"]), 4),
        "mean_precision": round(float(r["mean_precision"]), 4),
        "std_precision":  round(float(r["std_precision"]), 4),
    }

# 2. Median threshold
med_thr = float(np.median(best_f1_thresholds))
r_med   = _nearest(sweep_df, "threshold", med_thr)

# 3. Mean threshold
mn_thr = float(np.mean(best_f1_thresholds))
r_mn   = _nearest(sweep_df, "threshold", mn_thr)

# 4. Best mean-F1 (global fixed)
r_bf1 = sweep_df.loc[sweep_df["mean_f1"].idxmax()]

# 5. Lowest recall-std (with mean_recall > 0.5)
reasonable = sweep_df[sweep_df["mean_recall"] > 0.5].copy()
if len(reasonable) > 0:
    r_lrs = sweep_df.loc[reasonable["std_recall"].idxmin()]
else:
    r_lrs = sweep_df.loc[sweep_df["std_recall"].idxmin()]

# 6. Balanced stability: min( std_recall + std_fpr )  with mean_recall > 0.5
if len(reasonable) > 0:
    reasonable["_stab"] = reasonable["std_recall"] + reasonable["std_fpr"]
    r_bal = sweep_df.loc[reasonable["_stab"].idxmin()]
else:
    sweep_df["_stab"] = sweep_df["std_recall"] + sweep_df["std_fpr"]
    r_bal = sweep_df.loc[sweep_df["_stab"].idxmin()]

comparison_rows = [
    baseline,
    _rule_row("median_threshold",   med_thr,             r_med),
    _rule_row("mean_threshold",     mn_thr,              r_mn),
    _rule_row("best_mean_F1",       r_bf1["threshold"],  r_bf1),
    _rule_row("lowest_recall_std",  r_lrs["threshold"],  r_lrs),
    _rule_row("balanced_stability", r_bal["threshold"],  r_bal),
]
comparison_df = pd.DataFrame(comparison_rows)

print("\\n" + "="*80)
print("  THRESHOLD RULE COMPARISON  (V2 splitter, 20 seeds)")
print("="*80 + "\\n")
display(comparison_df)

# Save artifact
out_path = REPORT_DIR / "threshold_rule_comparison.csv"
comparison_df.to_csv(out_path, index=False)
print(f"\\nSaved → {out_path}")
'''

TASK_A_CODE_PLOTS = '''\
# ── 6d. Threshold sweep visualization ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) threshold vs std(recall)
ax = axes[0, 0]
ax.plot(sweep_df["threshold"], sweep_df["std_recall"], color="#4CAF50", lw=1.5)
for _, row in comparison_df.iterrows():
    if row["threshold_value"] != "varies":
        tv = float(row["threshold_value"])
        sr = float(row["std_recall"])
        ax.scatter(tv, sr, s=80, zorder=5, edgecolors="black",
                   label=row["threshold_rule"])
ax.set_xlabel("Threshold"); ax.set_ylabel("std(Recall) across seeds")
ax.set_title("Threshold vs Recall Instability")
ax.legend(fontsize=7, loc="best")

# (2) threshold vs std(FPR)
ax = axes[0, 1]
ax.plot(sweep_df["threshold"], sweep_df["std_fpr"], color="#EF5350", lw=1.5)
for _, row in comparison_df.iterrows():
    if row["threshold_value"] != "varies":
        tv = float(row["threshold_value"])
        sf = float(row["std_fpr"])
        ax.scatter(tv, sf, s=80, zorder=5, edgecolors="black",
                   label=row["threshold_rule"])
ax.set_xlabel("Threshold"); ax.set_ylabel("std(FPR) across seeds")
ax.set_title("Threshold vs FPR Instability")
ax.legend(fontsize=7, loc="best")

# (3) threshold vs mean(recall)  — cost curve
ax = axes[1, 0]
ax.plot(sweep_df["threshold"], sweep_df["mean_recall"], color="#2196F3", lw=1.5)
for _, row in comparison_df.iterrows():
    if row["threshold_value"] != "varies":
        tv = float(row["threshold_value"])
        mr = float(row["mean_recall"])
        ax.scatter(tv, mr, s=80, zorder=5, edgecolors="black",
                   label=row["threshold_rule"])
ax.set_xlabel("Threshold"); ax.set_ylabel("mean(Recall)")
ax.set_title("Threshold vs Mean Recall (trade-off)")
ax.legend(fontsize=7, loc="best")

# (4) Combined stability objective  J(t) = std_recall + std_fpr
ax = axes[1, 1]
combined = sweep_df["std_recall"] + sweep_df["std_fpr"]
ax.plot(sweep_df["threshold"], combined, color="#FF9800", lw=1.5)
for _, row in comparison_df.iterrows():
    if row["threshold_value"] != "varies":
        tv = float(row["threshold_value"])
        cs = float(row["std_recall"]) + float(row["std_fpr"])
        ax.scatter(tv, cs, s=80, zorder=5, edgecolors="black",
                   label=row["threshold_rule"])
ax.set_xlabel("Threshold"); ax.set_ylabel("std(Recall) + std(FPR)")
ax.set_title("Combined Stability Objective")
ax.legend(fontsize=7, loc="best")

plt.suptitle("Threshold Sweep — Stability vs Performance Trade-off", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
'''

TASK_A_INTERPRETATION = """\
### Interpretation — Low-Variance Threshold Selection

The comparison table and sweep plots above quantify the trade-off between
threshold stability and detection performance.

**Key questions answered:**

1. **Does replacing per-seed best-F1 with a fixed threshold reduce instability?**
   Compare `std_recall` and `std_fpr` across rules.  A meaningful reduction
   (e.g., >30% drop in std) supports using a fixed threshold for deployment.

2. **How much recall is sacrificed?**
   The `mean_recall` column shows the cost.  If the balanced-stability rule
   loses <2pp recall while halving std(recall), it is a strong candidate.

3. **Is median threshold a sensible default?**
   Median is robust to outlier seeds and requires no extra computation.
   If its stability metrics are close to the sweep-optimized rules, prefer it
   for simplicity.

**Important caveat:** Threshold stability improvements do NOT solve the
fundamental domain-shift problem.  They only reduce *amplification* of
variance at the operating-point selection stage.  Cross-dataset generalization
failures originate in representation space, not in threshold arithmetic."""

TASK_A_CODE_SAVE = '''\
# ── 6e. Save threshold stability artifacts ───────────────────────

# Per-seed table
perseed_path = REPORT_DIR / "threshold_stability_per_seed.csv"
v2_met[["seed","threshold","roc_auc","recall","precision","fpr"]].to_csv(perseed_path, index=False)
print(f"Saved → {perseed_path}")

# Stability summary
summary_path = REPORT_DIR / "threshold_stability_summary.csv"
stability_summary_df.to_csv(summary_path, index=False)
print(f"Saved → {summary_path}")

# Comparison already saved above
'''

TASK_A_CODE_REPORT = '''\
# ── 6f. Generate threshold stability report ──────────────────────
from IPython.display import Markdown, display as ipy_display

report_lines = []
report_lines.append("# Threshold Stability Report")
report_lines.append("")
report_lines.append(f"**Generated:** {pd.Timestamp.now().isoformat()}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# Section: Threshold stability
report_lines.append("## 1. Threshold Stability Across Seeds")
report_lines.append("")
report_lines.append(f"- Seeds evaluated: **{len(SEEDS)}** ({min(SEEDS)}–{max(SEEDS)})")
report_lines.append(f"- Mean threshold: **{thr_stats['mean']:.4f}**")
report_lines.append(f"- Std threshold:  **{thr_stats['std']:.4f}**")
report_lines.append(f"- CV: **{thr_stats['cv']:.4f}**")
report_lines.append(f"- Range: [{thr_stats['min']:.4f}, {thr_stats['max']:.4f}]")
report_lines.append(f"- IQR: **{thr_stats['iqr']:.4f}**")
report_lines.append(f"- **Verdict: {thr_verdict}**")
report_lines.append("")

# Section: Threshold rule comparison
report_lines.append("## 2. Threshold Rule Comparison")
report_lines.append("")
report_lines.append(comparison_df.to_markdown(index=False))
report_lines.append("")

# Section: Interpretation
report_lines.append("## 3. Honest Assessment")
report_lines.append("")
report_lines.append("### Is threshold instability a real problem?")
report_lines.append("")
if thr_verdict == "FRAGILE":
    report_lines.append(
        "Yes. The best-F1 threshold varies substantially across seeds "
        f"(CV={thr_stats['cv']:.2f}), confirming that threshold selection "
        "amplifies the inherent split-composition variance.")
elif thr_verdict == "MODERATELY STABLE":
    report_lines.append(
        "Partially. Threshold variation is moderate "
        f"(CV={thr_stats['cv']:.2f}). It contributes to metric instability "
        "but is not the sole driver—data-level variance is the root cause.")
else:
    report_lines.append(
        "The threshold is relatively stable across seeds "
        f"(CV={thr_stats['cv']:.2f}). Metric instability is driven primarily "
        "by split-composition variance, not threshold selection.")
report_lines.append("")

report_lines.append("### Does a fixed threshold materially help?")
report_lines.append("")
report_lines.append(
    "Compare `std_recall` between `per_seed_best_F1` and the best alternative "
    "in the table above.  A >30% reduction in std with <2pp recall loss "
    "would justify switching.  If the improvement is marginal (<15%), "
    "the benefit is primarily interpretive (deployment simplicity) rather "
    "than statistical.")
report_lines.append("")

report_lines.append("### Recommendation for thesis")
report_lines.append("")
report_lines.append(
    "1. **Primary strategy:** Report per-seed best-F1 results (current practice) "
    "with explicit uncertainty intervals.\\n"
    "2. **Supplementary evidence:** Include the median-threshold and "
    "balanced-stability results as robustness evidence.\\n"
    "3. **Deployment recommendation:** Use a fixed threshold (median or "
    "balanced-stability) for deployment, clearly noting the recall trade-off.\\n"
    "4. **Report threshold uncertainty explicitly** — include CV, IQR, and "
    "the stability verdict in the thesis methodology section.")
report_lines.append("")
report_lines.append("---")
report_lines.append("*Report generated automatically by threshold stability extension.*")

threshold_report_text = "\\n".join(report_lines)

report_path = REPORT_DIR / "threshold_stability_report.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(threshold_report_text, encoding="utf-8")
print(f"\\nReport saved → {report_path}")

ipy_display(Markdown(threshold_report_text))
'''


def _nb66_new_cells():
    """Return the list of new cells for notebook 66."""
    return [
        _md_cell(TASK_C_HEADER),
        _code_cell(TASK_C_CODE_STATS),
        _code_cell(TASK_C_CODE_PLOTS),
        _md_cell(TASK_C_INTERPRETATION),
        _md_cell(TASK_A_HEADER),
        _code_cell(TASK_A_CODE_COLLECT),
        _code_cell(TASK_A_CODE_SWEEP),
        _code_cell(TASK_A_CODE_COMPARE),
        _code_cell(TASK_A_CODE_PLOTS),
        _md_cell(TASK_A_INTERPRETATION),
        _code_cell(TASK_A_CODE_SAVE),
        _code_cell(TASK_A_CODE_REPORT),
    ]


def apply_nb66():
    """Inject Task A + C cells into notebook 66."""
    nb = json.loads(NB66.read_text(encoding="utf-8"))

    # Check idempotency
    if _find_cell_index(nb, "Threshold Stability Analysis (Task C)") >= 0:
        print(f"  ⏭  NB66 already extended — skipping.")
        return

    # Insert before "Section 4 — Verdict"
    anchor = _find_cell_index(nb, "Section 4 — Verdict")
    if anchor < 0:
        # Fallback: insert before last two cells
        anchor = max(0, len(nb["cells"]) - 2)

    new_cells = _nb66_new_cells()
    for i, cell in enumerate(new_cells):
        nb["cells"].insert(anchor + i, cell)

    # Renumber existing sections after insertion
    # (Section 4 → Section 8, Section 5 → Section 9)
    for cell in nb["cells"][anchor + len(new_cells):]:
        src = "".join(cell.get("source", []))
        if "## Section 4" in src:
            cell["source"] = [s.replace("Section 4", "Section 8") for s in cell["source"]]
        elif "## Section 5" in src:
            cell["source"] = [s.replace("Section 5", "Section 9") for s in cell["source"]]

    NB66.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ NB66 extended — {len(new_cells)} cells added.")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 56 — Task B
# ══════════════════════════════════════════════════════════════════

TASK_B_HEADER = """\
---
## Conservative Policy Threshold Analysis (Task B)

**Purpose:** Compare the current operating thresholds (`T_block`, `T_monitor`)
with more conservative alternatives derived from the benign score distribution.

A conservative threshold sacrifices some VPN recall in exchange for lower
false-block risk, producing a more defensible deployment operating point.

**Methodology:**
- `current` = existing `T_block` / `T_monitor` from validation calibration
- `p95_benign` = 95th-percentile of benign test scores + small margin
- `p99_benign` = 99th-percentile of benign test scores
- `strict` = max(benign test score) + margin  (≈ zero false-block target)
- `conservative` = midpoint between current `T_block` and strict threshold"""

TASK_B_CODE_ANALYSIS = '''\
# ── B1. Compute conservative threshold candidates ────────────────
benign_scores = p_test[y_test == 0]
vpn_scores    = p_test[y_test == 1]

# Also load validation scores if available
y_val = val_df["label"].values
p_val = val_df["ensemble_score"].values
benign_val_scores = p_val[y_val == 0]

# Threshold candidates
margin = 0.005  # small safety margin

thresholds = {
    "current_T_block":  T_block,
    "current_T_monitor": T_monitor,
    "p95_benign":       float(np.percentile(benign_scores, 95)) + margin,
    "p99_benign":       float(np.percentile(benign_scores, 99)),
    "strict_max_benign": float(np.max(benign_scores)) + margin,
    "conservative":     (T_block + float(np.max(benign_scores)) + margin) / 2,
}

print("Threshold Candidates:")
for name, val in thresholds.items():
    print(f"  {name:25s} = {val:.6f}")
print()

# Compute metrics for each threshold
def eval_threshold(thr, y_true, y_prob):
    yp = (y_prob >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    tn = int(((yp == 0) & (y_true == 0)).sum())
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    fpr_val = fp / max(fp + tn, 1)
    return {
        "recall": rec, "precision": pre, "fpr": fpr_val,
        "false_block_rate": fpr_val,
        "missed_vpn_rate": 1.0 - rec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

# Build comparison table
policy_rows = []
for name in ["current_T_block", "p95_benign", "p99_benign", "conservative", "strict_max_benign"]:
    thr = thresholds[name]
    ev = eval_threshold(thr, y_test, p_test)
    policy_rows.append({
        "policy_threshold_rule": name,
        "threshold": round(thr, 6),
        "block_rate_benign": round(ev["false_block_rate"], 6),
        "miss_rate_vpn": round(ev["missed_vpn_rate"], 6),
        "recall": round(ev["recall"], 4),
        "fpr": round(ev["fpr"], 6),
        "precision": round(ev["precision"], 4),
        "fp": ev["fp"],
        "fn": ev["fn"],
        "notes": "",
    })

policy_df = pd.DataFrame(policy_rows)
print("Conservative Threshold Comparison:")
display(policy_df)

# Save artifact
out_path = Path(str(EVAL)) / "conservative_threshold_analysis.csv"
policy_df.to_csv(out_path, index=False)
print(f"\\nSaved → {out_path}")
'''

TASK_B_CODE_SENSITIVITY = '''\
# ── B2. Threshold sensitivity plot ───────────────────────────────
thr_range = np.linspace(0.01, 0.99, 300)
sens_recs, sens_fprs = [], []
for t in thr_range:
    ev = eval_threshold(t, y_test, p_test)
    sens_recs.append(ev["missed_vpn_rate"])
    sens_fprs.append(ev["false_block_rate"])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(thr_range, sens_fprs, color="#EF5350", lw=2, label="False block rate (benign)")
ax.plot(thr_range, sens_recs, color="#2196F3", lw=2, label="Missed VPN rate")

# Mark thresholds
colors = {"current_T_block": "#D32F2F", "p99_benign": "#FF9800",
          "conservative": "#4CAF50", "strict_max_benign": "#9C27B0"}
for name, color in colors.items():
    thr = thresholds[name]
    ax.axvline(thr, ls="--", color=color, alpha=0.8, label=f"{name}={thr:.3f}")

ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Rate", fontsize=12)
ax.set_title("Threshold Sensitivity: False Block Rate vs Missed VPN Rate", fontsize=14)
ax.legend(fontsize=9, loc="center left")
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-0.01, 0.5)
plt.tight_layout()
plt.show()
'''

TASK_B_CODE_ZONES = '''\
# ── B3. Zone composition under different policy thresholds ───────
zone_rows = []
for rule_name, block_thr_name, monitor_thr_name in [
    ("Current policy", "current_T_block", "current_T_monitor"),
    ("Conservative",   "conservative",    "current_T_monitor"),
    ("Strict",         "strict_max_benign", "current_T_block"),
]:
    t_blk = thresholds[block_thr_name]
    t_mon = thresholds.get(monitor_thr_name, thresholds["current_T_monitor"])
    if t_mon > t_blk:
        t_mon = t_blk * 0.8  # Ensure monitor < block

    blocked  = p_test >= t_blk
    monitored = (p_test >= t_mon) & (p_test < t_blk)
    passed   = p_test < t_mon

    for zone, mask, zname in [(blocked, blocked, "BLOCK"),
                               (monitored, monitored, "MONITOR"),
                               (passed, passed, "PASS")]:
        n_vpn = int((mask & (y_test == 1)).sum())
        n_ben = int((mask & (y_test == 0)).sum())
        zone_rows.append({
            "policy": rule_name,
            "zone": zname,
            "T_block": round(t_blk, 4),
            "T_monitor": round(t_mon, 4),
            "total": int(mask.sum()),
            "vpn": n_vpn, "benign": n_ben,
        })

zone_df = pd.DataFrame(zone_rows)
print("Zone Composition Under Different Policies:")
display(zone_df)
'''

TASK_B_INTERPRETATION = """\
### Interpretation — Conservative Policy Threshold

**Does a conservative threshold improve operational safety?**

A conservative threshold (e.g., `p99_benign` or `midpoint`) reduces the
false-block rate—meaning fewer benign sessions are wrongly disrupted. This
directly improves operational safety and user trust.

**What recall is sacrificed?**

Compare `miss_rate_vpn` between the current threshold and the conservative
alternative. A modest increase (e.g., 5–10pp) may be acceptable if the
false-block reduction is substantial.

**Deployment recommendation:**

- If the current `T_block` already achieves near-zero false-block rate,
  a more conservative threshold provides only marginal safety improvement
  at the cost of detection capability.
- If `false_block_rate > 1%`, a conservative threshold is strongly recommended.
- For thesis deployment recommendations, present both the optimized and
  conservative thresholds with explicit trade-off analysis.

**Note:** Conservative thresholding addresses *operating-point safety*, not
the underlying domain-shift problem. It makes deployment interpretation
more honest but does not improve cross-dataset generalization."""


def _nb56_new_cells():
    return [
        _md_cell(TASK_B_HEADER),
        _code_cell(TASK_B_CODE_ANALYSIS),
        _code_cell(TASK_B_CODE_SENSITIVITY),
        _code_cell(TASK_B_CODE_ZONES),
        _md_cell(TASK_B_INTERPRETATION),
    ]


def apply_nb56():
    """Inject Task B cells into notebook 56."""
    nb = json.loads(NB56.read_text(encoding="utf-8"))

    # Check idempotency
    if _find_cell_index(nb, "Conservative Policy Threshold Analysis (Task B)") >= 0:
        print(f"  ⏭  NB56 already extended — skipping.")
        return

    # Insert before "Summary & Verdict" or before gallery section
    anchor = _find_cell_index(nb, "Summary & Verdict")
    if anchor < 0:
        anchor = _find_cell_index(nb, "Results Gallery")
    if anchor < 0:
        anchor = _find_cell_index(nb, "End of Notebook 56")
    if anchor < 0:
        anchor = len(nb["cells"]) - 2  # before last 2 cells

    new_cells = _nb56_new_cells()
    for i, cell in enumerate(new_cells):
        nb["cells"].insert(anchor + i, cell)

    NB56.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ NB56 extended — {len(new_cells)} cells added.")


# ══════════════════════════════════════════════════════════════════
# SPLIT STABILITY AUDIT — Task C extension
# ══════════════════════════════════════════════════════════════════

SPLIT_AUDIT_EXTENSION = '''
#%% md
---
## Section 11 — Threshold Stability Across Seeds

**Purpose:** Formally audit threshold selection stability. The per-seed
best-F1 threshold is treated as a first-class stability metric, comparable
to recall, FPR, and AUC.

This extends the metric sensitivity analysis from Section 8 by adding
threshold-specific statistics, visualizations, and a stability verdict.
#%%
# ── 11a. Threshold statistics ────────────────────────────────────
if RUN_METRIC_EXPERIMENT:
    thr_vals = metrics_df["threshold"].values
    thr_mean   = float(np.mean(thr_vals))
    thr_std    = float(np.std(thr_vals, ddof=1)) if len(thr_vals) > 1 else 0.0
    thr_min    = float(np.min(thr_vals))
    thr_max    = float(np.max(thr_vals))
    thr_median = float(np.median(thr_vals))
    thr_p25    = float(np.percentile(thr_vals, 25))
    thr_p75    = float(np.percentile(thr_vals, 75))
    thr_iqr    = thr_p75 - thr_p25
    thr_cv     = thr_std / thr_mean if thr_mean > 0 else float("nan")

    # Classify threshold stability
    if thr_cv < 0.05 and thr_std < 0.02:
        thr_stability_verdict = "STABLE"
    elif thr_cv < 0.15 and thr_std < 0.05:
        thr_stability_verdict = "MODERATELY STABLE"
    else:
        thr_stability_verdict = "FRAGILE"

    print("Threshold Stability Across Seeds:")
    print(f"  Mean:   {thr_mean:.4f}")
    print(f"  Std:    {thr_std:.4f}")
    print(f"  CV:     {thr_cv:.4f}")
    print(f"  Min:    {thr_min:.4f}")
    print(f"  Max:    {thr_max:.4f}")
    print(f"  Median: {thr_median:.4f}")
    print(f"  IQR:    {thr_iqr:.4f}")
    print(f"  Verdict: {thr_stability_verdict}")

    # Extended summary table including threshold
    ext_summary_rows = []
    for met_name in ["threshold", "roc_auc", "recall", "precision", "fpr"]:
        vals = metrics_df[met_name].values
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = s / m if m > 0 else float("nan")
        if met_name == "threshold":
            verd = thr_stability_verdict
        else:
            verd = classify_variation(s, (0.01, 0.03) if met_name in ("recall","precision")
                                       else (0.005, 0.02) if met_name == "fpr"
                                       else (0.005, 0.015))
        ext_summary_rows.append({
            "metric": met_name,
            "mean": round(m, 4), "std": round(s, 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "cv": round(cv, 4),
            "verdict": verd,
        })
    ext_summary_df = pd.DataFrame(ext_summary_rows)
    print("\\nExtended Stability Summary (threshold as first-class metric):")
    display(ext_summary_df)

    # Save artifact
    ext_summary_path = REPORT_DIR / "threshold_stability_summary.csv"
    ext_summary_df.to_csv(ext_summary_path, index=False)
    print(f"\\nSaved → {ext_summary_path}")
else:
    print("Metric sensitivity experiment was not executed. Skipping threshold stability.")
    thr_stability_verdict = "not_evaluated"
#%%
# ── 11b. Threshold stability visualizations ──────────────────────
if RUN_METRIC_EXPERIMENT:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Threshold by seed
    ax = axes[0, 0]
    ax.bar(metrics_df["seed"].astype(str), metrics_df["threshold"],
           color="#9C27B0", alpha=0.8)
    ax.axhline(thr_mean, ls="--", color="black", alpha=0.7,
               label=f"Mean={thr_mean:.3f}")
    ax.axhline(thr_median, ls=":", color="blue", alpha=0.7,
               label=f"Median={thr_median:.3f}")
    ax.set_title(f"Best-F1 Threshold by Seed (std={thr_std:.4f})")
    ax.set_xlabel("Seed"); ax.set_ylabel("Threshold")
    ax.legend(); ax.tick_params(axis="x", rotation=45)

    # (2) Boxplot
    ax = axes[0, 1]
    bp = ax.boxplot(thr_vals, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#CE93D8")
    ax.set_title("Threshold Distribution (Boxplot)")
    ax.set_ylabel("Threshold")
    ax.set_xticklabels(["Best-F1 threshold"])

    # (3) Threshold vs Recall
    ax = axes[1, 0]
    ax.scatter(metrics_df["threshold"], metrics_df["recall"],
               c="#4CAF50", s=60, edgecolors="black")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Test Recall")
    ax.set_title("Threshold vs Recall")

    # (4) Threshold vs FPR
    ax = axes[1, 1]
    ax.scatter(metrics_df["threshold"], metrics_df["fpr"],
               c="#EF5350", s=60, edgecolors="black")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Test FPR")
    ax.set_title("Threshold vs FPR")

    plt.suptitle("Threshold Stability — Split Stability Audit",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("Skipped (no metric experiment).")
#%% md
### Interpretation — Threshold Stability

The per-seed best-F1 threshold is reported here as a **first-class stability
metric**.  If the threshold coefficient of variation (CV) exceeds ~0.10, or
its standard deviation exceeds ~0.03, the operating point is meaningfully
unstable across random split seeds.

This instability *amplifies* the underlying split-composition variance:
different seeds yield different validation sets → different optimal thresholds
→ different recall / FPR on test.

**Recommendations:**
- Report threshold uncertainty explicitly (mean ± std, CV, IQR).
- Consider using a **median or fixed threshold** across seeds for deployment.
- Do **not** frame threshold stabilization as solving domain shift.

See also: `notebooks/66_splitter_v2_stability_audit.ipynb` Section 6 for a
detailed threshold rule comparison and low-variance threshold selection.
#%%
# ── 11c. Update report text with threshold stability section ─────
if RUN_METRIC_EXPERIMENT:
    thr_section = []
    thr_section.append("")
    thr_section.append("## 8. Threshold Stability Across Seeds")
    thr_section.append("")
    thr_section.append(f"- Mean threshold: **{thr_mean:.4f}**")
    thr_section.append(f"- Std threshold:  **{thr_std:.4f}**")
    thr_section.append(f"- CV: **{thr_cv:.4f}**")
    thr_section.append(f"- Range: [{thr_min:.4f}, {thr_max:.4f}]")
    thr_section.append(f"- IQR: **{thr_iqr:.4f}**")
    thr_section.append(f"- **Stability verdict: {thr_stability_verdict}**")
    thr_section.append("")
    thr_section.append("The per-seed best-F1 threshold is included as a first-class ")
    thr_section.append("stability metric. If the threshold is FRAGILE or MODERATELY STABLE, ")
    thr_section.append("downstream metric variance is partly driven by unstable threshold ")
    thr_section.append("selection rather than purely by model discriminative ability.")
    thr_section.append("")
    thr_section.append("**Recommendation:** Use a fixed threshold (median across seeds) ")
    thr_section.append("for deployment, and report threshold uncertainty explicitly.")
    thr_section.append("")

    # Append to existing report text
    report_text_extended = report_text.replace(
        "---\\n\\n*Report generated automatically.",
        "\\n".join(thr_section) + "\\n---\\n\\n*Report generated automatically."
    )

    # Save updated report
    report_path = REPORT_DIR / "split_stability_report.md"
    report_path.write_text(report_text_extended, encoding="utf-8")
    print(f"Updated report saved → {report_path}")
else:
    print("Skipped (no metric experiment).")
'''


def apply_split_audit():
    """Append Task C sections to split_stability_audit.ipynb."""
    content = NB_SPLIT.read_text(encoding="utf-8")

    # Check idempotency
    if "Threshold Stability Across Seeds" in content:
        print(f"  ⏭  split_stability_audit already extended — skipping.")
        return

    # Determine format
    is_percent = content.lstrip().startswith("#%%") or content.lstrip().startswith("# %%")
    is_json = content.lstrip().startswith("{")

    if is_percent:
        # Append percent-format cells
        content += "\n" + SPLIT_AUDIT_EXTENSION
        NB_SPLIT.write_text(content, encoding="utf-8")
        print(f"  ✅ split_stability_audit extended (percent format) — 4 sections added.")

    elif is_json:
        # Parse as ipynb JSON
        nb = json.loads(content)
        if _find_cell_index(nb, "Threshold Stability Across Seeds") >= 0:
            print(f"  ⏭  split_stability_audit already extended — skipping.")
            return

        # Parse percent-format cells into ipynb cells
        new_cells = _parse_percent_to_cells(SPLIT_AUDIT_EXTENSION)
        # Insert before the last cell (display report)
        anchor = _find_cell_index(nb, "Display the generated report")
        if anchor < 0:
            anchor = _find_cell_index(nb, "ipy_display(Markdown(report_text))")
        if anchor < 0:
            anchor = len(nb["cells"])

        for i, cell in enumerate(new_cells):
            nb["cells"].insert(anchor + i, cell)

        NB_SPLIT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  ✅ split_stability_audit extended (ipynb JSON) — {len(new_cells)} cells added.")
    else:
        print(f"  ⚠️  Unrecognized format for split_stability_audit — skipping.")


def _parse_percent_to_cells(text):
    """Parse percent-format text into a list of ipynb cell dicts."""
    cells = []
    current_type = None
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "#%% md" or stripped == "# %% md":
            if current_type is not None:
                cells.append(_flush_cell(current_type, current_lines))
            current_type = "markdown"
            current_lines = []
        elif stripped == "#%%" or stripped == "# %%":
            if current_type is not None:
                cells.append(_flush_cell(current_type, current_lines))
            current_type = "code"
            current_lines = []
        else:
            if current_type is not None:
                current_lines.append(line)

    if current_type is not None and current_lines:
        cells.append(_flush_cell(current_type, current_lines))

    return cells


def _flush_cell(cell_type, lines):
    """Create a cell dict from accumulated lines."""
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    source = "\n".join(lines)
    if cell_type == "markdown":
        return _md_cell(source)
    else:
        return _code_cell(source)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Applying threshold stability extensions")
    print("=" * 60)
    print()

    print("Task A + C → NB66:")
    apply_nb66()
    print()

    print("Task B → NB56:")
    apply_nb56()
    print()

    print("Task C → split_stability_audit:")
    apply_split_audit()
    print()

    # Create report directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Create placeholder for threshold_stability_report.md
    # (actual content generated when notebook is run)
    placeholder = REPORT_DIR / "threshold_stability_report.md"
    if not placeholder.exists():
        placeholder.write_text(
            "# Threshold Stability Report\n\n"
            "**Status:** Pending — run `notebooks/66_splitter_v2_stability_audit.ipynb` "
            "to generate this report.\n",
            encoding="utf-8",
        )
        print(f"  Created placeholder → {placeholder}")

    print()
    print("=" * 60)
    print("  Done. Run the following notebooks to generate results:")
    print("    1. notebooks/66_splitter_v2_stability_audit.ipynb")
    print("    2. notebooks/56_firewall_decision_evaluation.ipynb")
    print("    3. notebooks/split_stability_audit.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()

