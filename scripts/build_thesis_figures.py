"""Build thesis figures for unified_feature_contract_v2.

Loads the comparison CSV + (for figs 5–6) runs inference-only with the saved
model.pkl / calibrator.pkl on the test split of unified_flows.parquet.
No training, no threshold modification, no runtime-bundle modification.
"""
from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
EXP_DIR      = PROJECT_ROOT / "artifacts" / "unified_feature_contract_v2"
THESIS_DIR   = EXP_DIR / "thesis_exports"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

CMP_CSV = THESIS_DIR / "final_vs_legacy_model_comparison.csv"
cmp = pd.read_csv(CMP_CSV)
cmp = cmp.set_index("model_id", drop=False)

SELECTED   = "unified_relative_shape_v2__lgbm"
LEGACY     = "full_canonical__lgbm"

def short(mid: str) -> str:
    """Compact label for plots."""
    return (mid.replace("unified_", "u_")
               .replace("__lgbm", "")
               .replace("__", "."))

# ====================================================================
# Fig 1 — AUC comparison (test / lodo_min / domain)
# ====================================================================
sub = cmp.dropna(subset=["test_auc"]).copy()
labels = [short(m) for m in sub["model_id"]]
x = np.arange(len(sub))
w = 0.27

fig, ax = plt.subplots(figsize=(11, 5.2))
b1 = ax.bar(x - w, sub["test_auc"],     w, label="test AUC",       color="#4C78A8", edgecolor="black", linewidth=0.4)
b2 = ax.bar(x,     sub["lodo_min_auc"], w, label="LODO-min AUC",   color="#F58518", edgecolor="black", linewidth=0.4)
b3 = ax.bar(x + w, sub["domain_auc"],   w, label="domain AUC",     color="#E45756", edgecolor="black", linewidth=0.4)
for bars in (b1, b2, b3):
    for b in bars:
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=0)
ax.set_ylim(0, 1.13)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5)
ax.set_ylabel("AUC")
ax.set_title("Model comparison — test, LODO-min, and domain AUC\n(domain AUC lower is better)")
ax.legend(loc="upper right", ncol=3, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
# Highlight the selected model
sel_idx = list(sub["model_id"]).index(SELECTED) if SELECTED in sub["model_id"].values else None
if sel_idx is not None:
    ax.axvspan(sel_idx - 0.5, sel_idx + 0.5, color="green", alpha=0.07)
fig.tight_layout()
fig.savefig(THESIS_DIR / "fig_model_comparison_auc.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig 1: fig_model_comparison_auc.png")

# ====================================================================
# Fig 2 — LODO-min vs domain AUC scatter
# ====================================================================
sub2 = cmp.dropna(subset=["domain_auc", "lodo_min_auc"]).copy()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#54A24B" if m == SELECTED else ("#E45756" if m == LEGACY else "#4C78A8")
          for m in sub2["model_id"]]
sizes  = [180 if m == SELECTED else (140 if m == LEGACY else 80) for m in sub2["model_id"]]
ax.scatter(sub2["domain_auc"], sub2["lodo_min_auc"], c=colors, s=sizes,
           edgecolor="black", linewidth=0.7, zorder=3)
for _, r in sub2.iterrows():
    label = short(r["model_id"])
    if r["model_id"] in (SELECTED, LEGACY):
        ax.annotate(label, (r["domain_auc"], r["lodo_min_auc"]),
                    xytext=(-14, 12), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    else:
        ax.annotate(label, (r["domain_auc"], r["lodo_min_auc"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, alpha=0.75)
ax.set_xlabel("domain AUC  (lower = less dataset fingerprinting)")
ax.set_ylabel("LODO-min AUC  (higher = better cross-dataset transfer)")
ax.set_title("Transfer vs. fingerprinting\nGood models lie in the upper-left quadrant")
ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5)
ax.axvline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xlim(0.4, 1.05); ax.set_ylim(0.25, 0.75)
# Quadrant arrow
ax.annotate("better →", xy=(0.45, 0.72), fontsize=9, color="grey", style="italic")
fig.tight_layout()
fig.savefig(THESIS_DIR / "fig_lodo_vs_domain_auc.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig 2: fig_lodo_vs_domain_auc.png")

# ====================================================================
# Fig 3 — REMOVED.
# The legacy-vs-unified comparison figure (`fig_legacy_vs_unified_metrics.png`)
# has been retired from the thesis. The unified-only runtime-candidate
# comparison is produced by `scripts/plot_unified_runtime_candidate_metrics.py`
# and saved to `figures/unified_runtime_candidate_metrics.{png,pdf}`.
# ====================================================================
_legacy_fig = THESIS_DIR / "fig_legacy_vs_unified_metrics.png"
if _legacy_fig.exists():
    _legacy_fig.unlink()
    print("Removed retired fig: fig_legacy_vs_unified_metrics.png")

# ====================================================================
# Fig 4 — Domain fingerprinting reduction
# ====================================================================
fp = cmp.dropna(subset=["domain_auc"]).copy()
fp = fp.sort_values("domain_auc", ascending=False)
labels = [short(m) for m in fp["model_id"]]
colors = ["#E45756" if m == LEGACY else ("#54A24B" if m == SELECTED else "#4C78A8")
          for m in fp["model_id"]]
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(labels, fp["domain_auc"], color=colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, fp["domain_auc"]):
    ax.text(v + 0.003, b.get_y() + b.get_height()/2, f"{v:.4f}",
            va="center", fontsize=9)
ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6, label="chance (no fingerprint)")
ax.axvline(1.0, color="red", lw=0.8, ls=":", alpha=0.6, label="perfect fingerprint")
ax.set_xlim(0, 1.08)
ax.set_xlabel("domain classifier AUC  (lower is better)")
ax.set_title("Domain fingerprinting per model\n"
             f"Legacy `{LEGACY}` = 1.0  →  Unified `{SELECTED}` ≈ {float(cmp.loc[SELECTED,'domain_auc']):.3f}")
ax.legend(loc="lower right", fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(THESIS_DIR / "fig_domain_fingerprinting_reduction.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig 4: fig_domain_fingerprinting_reduction.png")

# ====================================================================
# Inference-only: load model + calibrator + run on test split
# ====================================================================
MODEL_DIR = EXP_DIR / "models" / SELECTED
DATA_PARQ = EXP_DIR / "data" / "unified_flows.parquet"
THR_PATH  = MODEL_DIR / "thresholds.json"

predictions_available = False
y_true = y_prob = None
review_thr = block_thr = None

if MODEL_DIR.exists() and DATA_PARQ.exists() and THR_PATH.exists():
    try:
        model      = joblib.load(MODEL_DIR / "model.pkl")
        calibrator = joblib.load(MODEL_DIR / "calibrator.pkl")
        feat_order = json.loads((MODEL_DIR / "feature_order.json").read_text(encoding="utf-8"))
        feats      = feat_order["features"] if isinstance(feat_order, dict) else feat_order
        thr        = json.loads(THR_PATH.read_text(encoding="utf-8"))
        review_thr = float(thr["review_threshold"])
        block_thr  = float(thr["block_threshold"])

        df = pd.read_parquet(DATA_PARQ)
        if "split" not in df.columns:
            raise RuntimeError("`split` column missing from unified_flows.parquet")
        test = df[df["split"] == "test"].copy()
        if test.empty:
            raise RuntimeError("test split is empty")
        missing_cols = [c for c in feats if c not in test.columns]
        if missing_cols:
            raise RuntimeError(f"missing feature columns in parquet: {missing_cols}")

        X = test[feats].astype("float32").to_numpy()
        y_true = test["label"].astype(int).to_numpy()

        # Raw model probability — this is what the training script used to compute the
        # reported test_recall / test_fpr / test_ece values in model_comparison.csv
        # (block_threshold=0.425 was tuned on raw probabilities). The calibrator.pkl is
        # saved separately for the runtime export and is NOT applied here, so that the
        # thesis figures reproduce the artifact numbers exactly.
        raw = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        y_prob = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        # Calibrated probability (kept for reference, used only in the reliability sub-panel)
        try:
            y_prob_calibrated = np.clip(np.asarray(calibrator.predict(raw), dtype=float), 0.0, 1.0)
        except Exception:
            y_prob_calibrated = None

        predictions_available = True
        print(f"Inference OK: n_test={len(y_true)}, pos_rate={y_true.mean():.4f}")
    except Exception as e:
        print("Inference failed:", e)
        predictions_available = False
else:
    print("Inference inputs missing:",
          "model_dir=" + str(MODEL_DIR.exists()),
          "parquet="   + str(DATA_PARQ.exists()),
          "thr="       + str(THR_PATH.exists()))

# ====================================================================
# Fig 5 — Calibration / reliability curve
# ====================================================================
calib_fig = THESIS_DIR / "fig_unified_calibration_curve.png"
calib_note = THESIS_DIR / "missing_calibration_plot_note.md"

if predictions_available:
    n_bins = 12
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if m.sum() > 0:
            mean_pred.append(y_prob[m].mean())
            frac_pos.append(y_true[m].mean())
            counts.append(int(m.sum()))
        else:
            mean_pred.append(np.nan); frac_pos.append(np.nan); counts.append(0)
    counts = np.asarray(counts, dtype=float)
    # ECE
    valid = counts > 0
    ece = np.sum(counts[valid] * np.abs(np.asarray(mean_pred)[valid] - np.asarray(frac_pos)[valid])) / counts.sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
    ax1.plot(mean_pred, frac_pos, marker="o", color="#54A24B", lw=2, label=SELECTED)
    ax1.scatter(mean_pred, frac_pos, color="#54A24B", s=40, zorder=4)
    ax1.set_ylabel("observed fraction of VPN")
    ax1.set_ylim(-0.02, 1.05); ax1.set_xlim(-0.02, 1.02)
    ax1.set_title(f"Reliability curve — {SELECTED} (test split)\nECE = {ece:.4f}  ·  n = {len(y_true)}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    # Annotate ECE
    ax1.text(0.98, 0.06, f"ECE = {ece:.4f}", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="black", lw=0.7))
    # Bottom: bin histogram
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax2.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.9, color="#4C78A8", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel("predicted probability (calibrated)")
    ax2.set_ylabel("count (log)")
    ax2.grid(axis="y", alpha=0.3); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(calib_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig 5: {calib_fig.name} (ECE={ece:.4f})")
    if calib_note.exists():
        calib_note.unlink()
else:
    calib_note.write_text(
        "# Missing calibration plot\n\n"
        "No pre-saved prediction CSV was available, and inference also failed.\n\n"
        f"Paths checked:\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/model.pkl`\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/calibrator.pkl`\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/feature_order.json`\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/thresholds.json`\n"
        f"- `{DATA_PARQ.relative_to(PROJECT_ROOT)}` (test split)\n", encoding="utf-8")
    print("Saved:", calib_note.name)

# ====================================================================
# Fig 6 — Confusion matrix at block threshold (with FLAG_REVIEW band)
# ====================================================================
cm_fig  = THESIS_DIR / "fig_unified_confusion_matrix.png"
cm_note = THESIS_DIR / "missing_confusion_matrix_note.md"

if predictions_available and review_thr is not None and block_thr is not None:
    # 3-way action labels: PASS / FLAG_REVIEW / SIMULATED_BLOCK
    action = np.where(y_prob >= block_thr, "SIMULATED_BLOCK",
              np.where(y_prob >= review_thr, "FLAG_REVIEW", "PASS"))
    label_names = ["non-VPN (0)", "VPN (1)"]
    action_names = ["PASS", "FLAG_REVIEW", "SIMULATED_BLOCK"]

    # Build a 2x3 table: rows=true label, cols=action
    table = np.zeros((2, 3), dtype=int)
    for li, lab in enumerate([0, 1]):
        for ai, act in enumerate(action_names):
            table[li, ai] = int(((y_true == lab) & (action == act)).sum())

    # Binary confusion matrix at block threshold (predicted positive = SIMULATED_BLOCK)
    pred_bin = (y_prob >= block_thr).astype(int)
    tp = int(((y_true == 1) & (pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (pred_bin == 0)).sum())
    tn = int(((y_true == 0) & (pred_bin == 0)).sum())
    fp = int(((y_true == 0) & (pred_bin == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr    = fp / (fp + tn) if (fp + tn) else float("nan")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.1, 1.4]})
    # Left: binary 2x2 at block threshold
    cm2 = np.array([[tn, fp], [fn, tp]])
    im = axL.imshow(cm2, cmap="Blues")
    axL.set_xticks([0, 1]); axL.set_xticklabels(["pred non-VPN", "pred VPN (BLOCK)"])
    axL.set_yticks([0, 1]); axL.set_yticklabels(["true non-VPN", "true VPN"])
    axL.set_title(f"Binary confusion @ block threshold = {block_thr:.3f}\n"
                  f"recall = {recall:.4f}   ·   FPR = {fpr:.4f}")
    for i in range(2):
        for j in range(2):
            v = cm2[i, j]
            axL.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm2.max()/2 else "black", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axL, fraction=0.046, pad=0.04)

    # Right: 3-way action-by-true-label heatmap
    im2 = axR.imshow(table, cmap="Oranges", aspect="auto")
    axR.set_xticks(range(3)); axR.set_xticklabels(action_names)
    axR.set_yticks(range(2)); axR.set_yticklabels(label_names)
    axR.set_title(f"3-way action counts (review={review_thr:.3f}, block={block_thr:.3f})\n"
                  f"n_test = {len(y_true):,}")
    for i in range(2):
        for j in range(3):
            v = table[i, j]
            axR.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > table.max()/2 else "black", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axR, fraction=0.046, pad=0.04)

    fig.suptitle(f"Confusion matrices — {SELECTED}  (test split, n={len(y_true):,})", fontsize=12)
    fig.tight_layout()
    fig.savefig(cm_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig 6: {cm_fig.name}  (TP={tp}, FP={fp}, TN={tn}, FN={fn})")
    if cm_note.exists():
        cm_note.unlink()
else:
    cm_note.write_text(
        "# Missing confusion matrix\n\n"
        "Could not produce the confusion matrix because predictions or thresholds were unavailable.\n\n"
        f"Paths checked:\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/model.pkl`\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/calibrator.pkl`\n"
        f"- `{MODEL_DIR.relative_to(PROJECT_ROOT)}/thresholds.json`\n"
        f"- `{DATA_PARQ.relative_to(PROJECT_ROOT)}` (test split)\n", encoding="utf-8")
    print("Saved:", cm_note.name)

print("\nDone.")

