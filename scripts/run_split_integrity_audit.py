"""Execute the split integrity audit cells to produce figures."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
ART_CP = REPO / "artifacts" / "clean_pipeline"
ART_V = REPO / "artifacts" / "validation"
FIG = REPO / "figures" / "validation"
FIG.mkdir(parents=True, exist_ok=True)
ART_V.mkdir(parents=True, exist_ok=True)

features = pd.read_parquet(ART_CP / "features.parquet")
manifest = json.load(open(ART_CP / "clean_split_manifest.json"))

# 1. Split composition per dataset
fig, ax = plt.subplots(figsize=(8, 5))
split_counts = features.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0)
split_counts = split_counts.rename(columns={0: "nonVPN", 1: "VPN"})
split_counts[["nonVPN", "VPN"]].plot(kind="bar", stacked=True, ax=ax)
plt.title("Split Composition per Dataset")
plt.ylabel("Flow count")
plt.tight_layout()
plt.savefig(FIG / "split_composition_per_dataset.png")
plt.close()

# 2. Capture size distribution
fig, ax = plt.subplots(figsize=(8, 5))
cap_sizes = features.groupby(["dataset", "split", "capture_id"]).size().reset_index(name="n_flows")
sns.boxplot(data=cap_sizes, x="split", y="n_flows", hue="dataset", ax=ax)
ax.set_yscale("log")
plt.title("Capture Size Distribution by Split and Dataset")
plt.ylabel("Flows per capture (log scale)")
plt.tight_layout()
plt.savefig(FIG / "capture_size_distribution.png")
plt.close()

# 4. Save summary
summary = []
for ds in manifest["per_dataset_split"]:
    for split, stats in manifest["per_dataset_split"][ds].items():
        summary.append({"dataset": ds, "split": split, **stats})
summary_df = pd.DataFrame(summary)
summary_df.to_csv(ART_V / "split_integrity_summary.csv", index=False)
summary_df.to_json(ART_V / "split_integrity_summary.json", orient="records", indent=2)
print(summary_df)
print("Saved split integrity figures to", FIG)
