"""Execute the model stability audit cells to produce figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "validation"
FIG = REPO / "figures" / "validation"
FIG.mkdir(parents=True, exist_ok=True)

seed_metrics = pd.read_csv(ART / "model_seed_stability.csv")
summary = pd.read_csv(ART / "model_stability_summary.csv")

# 1. Metric variability across seeds
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
seed_metrics.boxplot(column=["test_auc", "test_recall", "test_fpr"], ax=axes[0])
axes[0].set_title("Test Metrics Variability")
axes[1].plot(seed_metrics["seed"], seed_metrics["threshold"], marker="o")
axes[1].set_title("Thresholds Across Seeds")
axes[1].set_xlabel("Seed")
axes[1].set_ylabel("Threshold")
axes[2].boxplot(
    [seed_metrics["test_auc"], seed_metrics["test_recall"], seed_metrics["test_fpr"]],
    labels=["AUC", "Recall", "FPR"],
)
axes[2].set_title("Metric Distribution")
plt.tight_layout()
plt.savefig(FIG / "metric_variability_across_seeds.png")
plt.close()

# 2. Threshold distribution
plt.figure(figsize=(6, 4))
seed_metrics["threshold"].plot(kind="hist", bins=10, alpha=0.7)
plt.title("Threshold Distribution Across Seeds")
plt.xlabel("Threshold")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG / "threshold_distribution_across_seeds.png")
plt.close()

# 3. Bootstrap CIs
metrics = ["auc", "recall", "fpr"]
means = [summary[m][0] for m in metrics]
lowers = [summary[f"{m}_ci_lower"][0] for m in metrics]
uppers = [summary[f"{m}_ci_upper"][0] for m in metrics]
plt.figure(figsize=(7, 4))
plt.errorbar(
    metrics,
    means,
    yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
    fmt="o",
    capsize=5,
)
plt.title("Bootstrap Confidence Intervals")
plt.ylabel("Metric value")
plt.tight_layout()
plt.savefig(FIG / "bootstrap_confidence_intervals.png")
plt.close()

# Save JSON summary
summary.to_json(ART / "model_stability_summary.json", orient="records", indent=2)

print("Saved model stability figures to", FIG)
print(summary)
