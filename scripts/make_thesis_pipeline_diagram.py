"""
Generate Figure 3.1 — Experimental pipeline diagram for the thesis.

Workflow diagram only. No result plots, no heatmaps, no numeric content.
Exports both PNG (300 dpi) and PDF (vector) to:
  - figures/fig_3_1_experimental_pipeline.{png,pdf}
  - artifacts/unified_feature_contract_v2/thesis_exports/fig_3_1_experimental_pipeline.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Academic neutral palette
C_INPUT = "#D9E6F2"   # light blue   - data sources
C_PROC = "#ECECEC"    # light gray   - processing stages
C_PART = "#E6E0D4"    # warm gray    - splitting
C_TRAIN = "#DDE7D5"   # pale green   - modelling
C_OUT = "#F4DFC8"     # pale orange  - outputs/branches
EDGE = "#444444"
TXT = "#111111"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "figures",
    ROOT / "artifacts" / "unified_feature_contract_v2" / "thesis_exports",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Figure size tuned for one thesis page (portrait) ~ 7.0 x 9.5 inches
FIG_W, FIG_H = 8.5, 11.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 140)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, title, lines, color, fontsize=9, title_size=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.1, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    # Title
    ax.text(cx, y + h - 2.2, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=TXT)
    # Bullet lines
    if lines:
        text = "\n".join(f"• {ln}" for ln in lines)
        ax.text(cx, y + h - 5.0, text, ha="center", va="top",
                fontsize=fontsize, color=TXT, linespacing=1.25)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=EDGE,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Layout (top-down).  y increases upward.
# ---------------------------------------------------------------------------
# 1) Three input datasets (top row)
ds_w, ds_h = 24, 7
ds_y = 130
ax.text(50, 138.5, "Experimental pipeline for cross-dataset VPN traffic analysis",
        ha="center", va="center", fontsize=11, fontweight="bold", color=TXT)

box(6,  ds_y, ds_w, ds_h, "ISCXVPN2016", ["raw .pcap captures"], C_INPUT, fontsize=8.5)
box(38, ds_y, ds_w, ds_h, "USBVPN",      ["raw .pcap captures"], C_INPUT, fontsize=8.5)
box(70, ds_y, ds_w, ds_h, "VNAT",        ["raw .pcap captures"], C_INPUT, fontsize=8.5)

# 2) Dataset-specific loaders
ld_x, ld_y, ld_w, ld_h = 14, 113, 72, 9
box(ld_x, ld_y, ld_w, ld_h,
    "Dataset-specific loaders",
    ["parse native capture formats",
     "extract timestamps, packet sizes, directions, capture IDs, labels"],
    C_PROC, fontsize=8.5)

# arrows from datasets to loader
for cx in (18, 50, 82):
    arrow(cx, ds_y, cx, ld_y + ld_h)

# 3) Unified flow schema
us_x, us_y, us_w, us_h = 14, 99, 72, 10
box(us_x, us_y, us_w, us_h,
    "Unified flow schema",
    ["timestamp sequence  •  packet-size sequence  •  direction sequence",
     "capture_id  •  VPN / non-VPN label"],
    C_PROC, fontsize=8.5)
arrow(50, ld_y, 50, us_y + us_h)

# 4) Streaming feature extraction
fe_x, fe_y, fe_w, fe_h = 14, 84, 72, 11
box(fe_x, fe_y, fe_w, fe_h,
    "Streaming feature extraction",
    ["packet-length statistics",
     "inter-arrival-time statistics",
     "flow duration  •  packet rate / byte rate  •  variability descriptors"],
    C_PROC, fontsize=8.5)
arrow(50, us_y, 50, fe_y + fe_h)

# 5) Dataset harmonization
dh_x, dh_y, dh_w, dh_h = 14, 69, 72, 11
box(dh_x, dh_y, dh_w, dh_h,
    "Dataset harmonization",
    ["absolute packet sizes  •  consistent statistical formulas",
     "direction-invariant descriptors",
     "no target-domain normalization in strict transfer evaluation"],
    C_PROC, fontsize=8.5)
arrow(50, fe_y, 50, dh_y + dh_h)

# 6) Capture-level partitioning
cp_x, cp_y, cp_w, cp_h = 14, 54, 72, 11
box(cp_x, cp_y, cp_w, cp_h,
    "Capture-level partitioning",
    ["train  /  validation  /  test",
     "all flows from the same capture stay in the same split",
     "prevents flow-level leakage"],
    C_PART, fontsize=8.5)
arrow(50, dh_y, 50, cp_y + cp_h)

# 7) Model training and threshold selection
mt_x, mt_y, mt_w, mt_h = 14, 39, 72, 11
box(mt_x, mt_y, mt_w, mt_h,
    "Model training and threshold selection",
    ["train models on the training split",
     "select thresholds only on the validation split",
     "apply unchanged to the test split"],
    C_TRAIN, fontsize=8.5)
arrow(50, cp_y, 50, mt_y + mt_h)

# 8) Evaluation protocols
ev_x, ev_y, ev_w, ev_h = 14, 24, 72, 11
box(ev_x, ev_y, ev_w, ev_h,
    "Evaluation protocols",
    ["pooled evaluation",
     "leave-one-dataset-out transfer",
     "temporal / capture-level diagnostics"],
    C_TRAIN, fontsize=8.5)
arrow(50, mt_y, 50, ev_y + ev_h)

# 9) Three output branches
br_w, br_h = 28, 14
br_y = 4
# A — Cross-dataset evaluation (left)
box(2, br_y, br_w, br_h,
    "A. Cross-dataset evaluation",
    ["pooled vs. LODO transfer",
     "feature-subset screening",
     "normalization / alignment"],
    C_OUT, fontsize=8.0)
# B — Structural-shift analysis (center)
box(36, br_y, br_w, br_h,
    "B. Structural-shift analysis",
    ["dataset fingerprinting",
     "distribution shift  •  correlation instability",
     "feature-effect sign reversal",
     "feature-importance instability"],
    C_OUT, fontsize=7.8)
# C — Firewall-oriented simulation (right)
box(70, br_y, br_w, br_h,
    "C. Firewall-oriented simulation",
    ["packet-sequence models",
     "decision policy:",
     "PASS  /  FLAG_REVIEW  /  SIMULATED_BLOCK"],
    C_OUT, fontsize=8.0)

# Branch arrows from evaluation protocols
arrow(30, ev_y, 16, br_y + br_h)   # to A
arrow(50, ev_y, 50, br_y + br_h)   # to B
arrow(70, ev_y, 84, br_y + br_h)   # to C

# Side legend / note
ax.text(99.5, 1.0,
        "Workflow diagram — Chapter 3",
        ha="right", va="bottom",
        fontsize=7.5, color="#666666", style="italic")

plt.tight_layout()

png_paths = []
pdf_paths = []
for d in OUT_DIRS:
    p_png = d / "fig_3_1_experimental_pipeline.png"
    p_pdf = d / "fig_3_1_experimental_pipeline.pdf"
    fig.savefig(p_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(p_pdf,             bbox_inches="tight", facecolor="white")
    png_paths.append(p_png)
    pdf_paths.append(p_pdf)

plt.close(fig)

print("Saved:")
for p in png_paths + pdf_paths:
    print(" -", p)



