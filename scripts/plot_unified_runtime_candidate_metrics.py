"""Generate unified runtime-candidate comparison bar chart for thesis.

Replaces the deprecated `fig_legacy_vs_unified_metrics.png` figure. Contains
ONLY unified feature-contract candidates with confirmed metrics. No legacy
or full-canonical model is referenced.

Bars: Test AUC, LODO-min AUC, Domain AUC, Deployment score.
ECE and FPR are intentionally not plotted (lower-is-better metrics reported
in the comparison table instead).

Outputs:
    figures/unified_runtime_candidate_metrics.png
    figures/unified_runtime_candidate_metrics.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"key": "size_shape",        "n": 16, "test": 0.9926, "lodo": 0.5623, "domain": 0.9479, "dep": 0.4004},
    {"key": "directionless",     "n": 30, "test": 0.9955, "lodo": 0.3888, "domain": 0.9802, "dep": 0.1451},
    {"key": "relative_shape_v2", "n": 12, "test": 0.9826, "lodo": 0.6366, "domain": 0.9591, "dep": 0.4691},
]
SELECTED = "relative_shape_v2"

METRICS = [
    ("test",   "Test AUC",        "#4C78A8"),
    ("lodo",   "LODO-min AUC",    "#F58518"),
    ("domain", "Domain AUC",      "#72B7B2"),
    ("dep",    "Deployment score","#9D7BB0"),
]


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    x = np.arange(len(MODELS))
    n_metrics = len(METRICS)
    width = 0.78 / n_metrics

    for i, (key, label, color) in enumerate(METRICS):
        vals = [m[key] for m in MODELS]
        offset = (i - (n_metrics - 1) / 2.0) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.012,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8.5,
            )

    # X-axis labels with feature count and selection marker.
    tick_labels = []
    for m in MODELS:
        marker = "  ★" if m["key"] == SELECTED else ""
        tick_labels.append(f"{m['key']}{marker}\n(n={m['n']})")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10.5)

    # Bold + green tick label for the selected candidate (no background band).
    sel_idx = next(i for i, m in enumerate(MODELS) if m["key"] == SELECTED)
    for i, tick in enumerate(ax.get_xticklabels()):
        if i == sel_idx:
            tick.set_fontweight("bold")
            tick.set_color("#2a6f3b")

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("metric value")
    ax.set_title("Unified runtime-candidate comparison", fontsize=13, pad=36)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=n_metrics,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    png_path = FIG_DIR / "unified_runtime_candidate_metrics.png"
    pdf_path = FIG_DIR / "unified_runtime_candidate_metrics.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote: {png_path}")
    print(f"wrote: {pdf_path}")


if __name__ == "__main__":
    main()


