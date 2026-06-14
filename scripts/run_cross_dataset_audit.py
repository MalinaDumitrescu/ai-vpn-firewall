"""Execute the cross-dataset domain shift audit cells to produce figures."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "validation"
FIG = REPO / "figures" / "validation"
FIG.mkdir(parents=True, exist_ok=True)

# 1. LODO protocol integrity
lodo = pd.read_csv(ART / "lodo_protocol_integrity.csv")
print("LODO protocol integrity:\n", lodo, "\n")

# 2. Domain fingerprinting confusion matrix
cm = pd.read_csv(ART / "domain_fingerprinting_confusion_matrix.csv", index_col=0)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Domain Fingerprinting Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG / "domain_fingerprinting_confusion_matrix.png")
plt.close()

# 3. Top shifted features by JSD
shift = pd.read_csv(ART / "feature_distribution_shift.csv")
top = shift.groupby("feature")["jsd"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 4))
top.plot(kind="barh")
plt.title("Top 10 Features by Mean JSD")
plt.xlabel("Mean JSD")
plt.tight_layout()
plt.savefig(FIG / "top_shifted_features_jsd.png")
plt.close()

# 4. Sign reversal heatmap
signs = pd.read_csv(ART / "sign_reversal_audit.csv")
pivot = signs.pivot(index="feature", columns="dataset", values="smd")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, center=0, cmap="coolwarm", annot=False)
plt.title("Feature Sign Reversal Heatmap (SMD VPN - nonVPN)")
plt.tight_layout()
plt.savefig(FIG / "sign_reversal_heatmap.png")
plt.close()

# 5. LODO transfer performance
perf = pd.read_csv(ART / "lodo_transfer_performance.csv")
plt.figure(figsize=(6, 4))
perf.set_index("target")["auc"].plot(kind="bar")
plt.title("LODO Transfer AUC by Target")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig(FIG / "lodo_transfer_auc_by_target.png")
plt.close()

print("Saved cross-dataset figures to", FIG)
