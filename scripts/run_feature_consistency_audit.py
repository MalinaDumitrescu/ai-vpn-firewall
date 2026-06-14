"""Execute the feature consistency audit cells to produce figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ART_CP = REPO / "artifacts" / "clean_pipeline"
ART_V = REPO / "artifacts" / "validation"
FIG = REPO / "figures" / "validation"
FIG.mkdir(parents=True, exist_ok=True)
ART_V.mkdir(parents=True, exist_ok=True)

features = pd.read_parquet(ART_CP / "features.parquet")
with open(ART_CP / "feature_columns.json") as f:
    feature_columns = json.load(f)

# 1. Required schema
required = ["flow_id", "capture_id", "dataset", "label", "timestamps", "sizes", "directions"]
schema_present = all(col in features.columns for col in required)
print("Schema present:", schema_present)

# 3. Risky features
risky_patterns = ["fwd", "bwd", "forward", "backward", "src", "dst", "client", "server", "directional_ratio"]
risky = [f for f in feature_columns if any(p in f.lower() for p in risky_patterns)]
print("Risky features:", risky)

# 4. Rate feature formula consistency
df = features[features["flow_duration"] > 0].copy()
recomputed_packet_rate = df["total_packets"] / df["flow_duration"]
recomputed_byte_rate = df["total_bytes"] / df["flow_duration"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(df["packet_rate"], recomputed_packet_rate, alpha=0.3, s=5)
mn, mx = df["packet_rate"].min(), df["packet_rate"].max()
axes[0].plot([mn, mx], [mn, mx], "r--")
axes[0].set_xlabel("Stored packet_rate")
axes[0].set_ylabel("Recomputed packet_rate")
axes[0].set_title("Packet Rate Consistency")
axes[0].set_xscale("log"); axes[0].set_yscale("log")

axes[1].scatter(df["byte_rate"], recomputed_byte_rate, alpha=0.3, s=5)
mn, mx = df["byte_rate"].min(), df["byte_rate"].max()
axes[1].plot([mn, mx], [mn, mx], "r--")
axes[1].set_xlabel("Stored byte_rate")
axes[1].set_ylabel("Recomputed byte_rate")
axes[1].set_title("Byte Rate Consistency")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig(FIG / "rate_feature_formula_consistency.png")
plt.close()

max_packet_rate_error = float(np.max(np.abs(df["packet_rate"] - recomputed_packet_rate)))
max_byte_rate_error = float(np.max(np.abs(df["byte_rate"] - recomputed_byte_rate)))
print("Max abs packet_rate error:", max_packet_rate_error)
print("Max abs byte_rate error:", max_byte_rate_error)

# 5. Summary
summary = [{
    "check_name": "rate_formula_consistency",
    "status": "PASS" if max_packet_rate_error < 1e-6 and max_byte_rate_error < 1e-6 else "FAIL",
    "feature_count": len(feature_columns),
    "missing_features": [f for f in feature_columns if f not in features.columns],
    "extra_features": [f for f in features.columns if f not in feature_columns],
    "max_absolute_formula_error": max(max_packet_rate_error, max_byte_rate_error),
    "notes": "Synthetic data",
}]
pd.DataFrame(summary).to_csv(ART_V / "feature_consistency_summary.csv", index=False)
(ART_V / "feature_consistency_summary.json").write_text(json.dumps(summary, indent=2))
print("Saved feature consistency figure to", FIG)
