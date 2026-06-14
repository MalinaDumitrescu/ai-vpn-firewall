"""Generate unified runtime-candidate trade-off scatter plot for thesis.

X: LODO-min AUC
Y: Domain AUC
Size: number of features
Highlight: relative_shape_v2 (selected runtime candidate).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "key": "size_shape",
        "n_features": 16,
        "lodo_min": 0.5623,
        "domain_auc": 0.9479,
        "selected": False,
    },
    {
        "key": "directionless",
        "n_features": 30,
        "lodo_min": 0.3888,
        "domain_auc": 0.9802,
        "selected": False,
    },
    {
        "key": "relative_shape_v2",
        "n_features": 12,
        "lodo_min": 0.6366,
        "domain_auc": 0.9591,
        "selected": True,
    },
]


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # Size scaling: marker area proportional to n_features.
    size_scale = 36.0  # area per feature

    for m in MODELS:
        area = m["n_features"] * size_scale
        if m["selected"]:
            face = "#1f77b4"
            edge = "#0b3d66"
            lw = 2.0
            alpha = 0.95
        else:
            face = "#bfbfbf"
            edge = "#4d4d4d"
            lw = 1.2
            alpha = 0.85
        ax.scatter(
            m["lodo_min"],
            m["domain_auc"],
            s=area,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
        )

    # Text labels with slight offsets to avoid overlap with marker.
    label_offsets = {
        "size_shape": (0.012, 0.004),
        "directionless": (0.012, 0.004),
        "relative_shape_v2": (-0.012, -0.012),
    }
    h_align = {
        "size_shape": "left",
        "directionless": "left",
        "relative_shape_v2": "right",
    }
    for m in MODELS:
        dx, dy = label_offsets[m["key"]]
        weight = "bold" if m["selected"] else "normal"
        color = "#0b3d66" if m["selected"] else "#1a1a1a"
        suffix = "  (selected)" if m["selected"] else ""
        ax.annotate(
            f"{m['key']}{suffix}\nn={m['n_features']}",
            xy=(m["lodo_min"], m["domain_auc"]),
            xytext=(m["lodo_min"] + dx, m["domain_auc"] + dy),
            ha=h_align[m["key"]],
            va="center",
            fontsize=11,
            fontweight=weight,
            color=color,
            zorder=4,
        )

    ax.set_xlabel("LODO-min ROC-AUC  (higher = better worst-domain transfer)")
    ax.set_ylabel("Domain ROC-AUC  (lower = weaker dataset fingerprinting)")
    ax.set_title("Runtime candidate trade-off under unified feature contract",
                 fontsize=13, pad=12)

    # Sensible axis bounds with padding.
    xs = [m["lodo_min"] for m in MODELS]
    ys = [m["domain_auc"] for m in MODELS]
    ax.set_xlim(min(xs) - 0.08, max(xs) + 0.08)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.02)

    ax.grid(True, which="major", linestyle="--", alpha=0.35, zorder=0)
    ax.tick_params(direction="out", length=4)

    fig.tight_layout()

    png_path = FIG_DIR / "unified_runtime_tradeoff_scatter.png"
    pdf_path = FIG_DIR / "unified_runtime_tradeoff_scatter.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote: {png_path}")
    print(f"wrote: {pdf_path}")


if __name__ == "__main__":
    main()
