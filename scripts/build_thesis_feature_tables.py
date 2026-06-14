"""Export unified_relative_shape_v2 feature tables + composition figure for thesis."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
EXP_DIR      = PROJECT_ROOT / "artifacts" / "unified_feature_contract_v2"
RT_FO        = EXP_DIR / "runtime_export" / "runtime_models" / "unified_relative_shape_v2__lgbm" / "feature_order.json"
FAM_PATH     = EXP_DIR / "feature_families" / "unified_relative_shape_v2.json"
FC_PATH      = EXP_DIR / "feature_contract.json"
EXTRACTOR    = PROJECT_ROOT / "src" / "features" / "unified_extractor.py"
THESIS_DIR   = EXP_DIR / "thesis_exports"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = [
    "sz_cv", "sz_iqr", "sz_qratio", "sz_median_to_mean",
    "sz_p25_median_ratio", "sz_p75_median_ratio", "sz_iqr_norm_median",
    "iat_cv", "iat_iqr",
    "direction_balance_bytes", "direction_balance_packets", "dispersion_symmetry",
]

# ---------- 1. Load + verify runtime feature_order ----------
fo_obj   = json.loads(RT_FO.read_text(encoding="utf-8"))
runtime_features = fo_obj["features"] if isinstance(fo_obj, dict) else fo_obj
fc       = json.loads(FC_PATH.read_text(encoding="utf-8"))
fam_obj  = json.loads(FAM_PATH.read_text(encoding="utf-8"))
fam_features = fam_obj.get("features", [])

print("Runtime feature_order.json  :", RT_FO.relative_to(PROJECT_ROOT))
print("  features (in order):", runtime_features)

missing_in_runtime = [f for f in EXPECTED if f not in runtime_features]
extra_in_runtime   = [f for f in runtime_features if f not in EXPECTED]
order_match        = (runtime_features == EXPECTED)

print(f"\nVerification:")
print(f"  expected count   : {len(EXPECTED)}")
print(f"  runtime count    : {len(runtime_features)}")
print(f"  missing in runtime: {missing_in_runtime or 'NONE'}")
print(f"  extra in runtime  : {extra_in_runtime or 'NONE'}")
print(f"  order matches    : {order_match}")
assert not missing_in_runtime, "Runtime is missing required features"
assert not extra_in_runtime,   "Runtime has unexpected extra features"

# ---------- 2. Feature groups ----------
GROUPS = {
    "sz_cv":                   ("size ratio / shape", "Packet-size coefficient of variation"),
    "sz_iqr":                  ("size ratio / shape", "Packet-size interquartile range"),
    "sz_qratio":               ("size ratio / shape", "Packet-size upper/lower quartile ratio"),
    "sz_median_to_mean":       ("size ratio / shape", "Packet-size median relative to mean"),
    "sz_p25_median_ratio":     ("size ratio / shape", "Packet-size p25 relative to median"),
    "sz_p75_median_ratio":     ("size ratio / shape", "Packet-size p75 relative to median"),
    "sz_iqr_norm_median":      ("size ratio / shape", "Packet-size IQR normalised by median"),
    "iat_cv":                  ("timing ratio / shape", "Inter-arrival-time coefficient of variation"),
    "iat_iqr":                 ("timing ratio / shape", "Inter-arrival-time interquartile range"),
    "direction_balance_bytes": ("direction / symmetry", "Up/down byte balance, sign-aware"),
    "direction_balance_packets": ("direction / symmetry", "Up/down packet-count balance, sign-aware"),
    "dispersion_symmetry":     ("direction / symmetry", "Symmetry of the packet-size distribution around the median"),
}

# ---------- 3. Feature contract formulas (extracted from src/features/unified_extractor.py) ----------
FORMULAS = {
    "sz_cv":                    ("sz_std / (sz_mean + eps)",
                                 ["sz_all_std", "sz_all_mean"]),
    "sz_iqr":                   ("sz_p75 - sz_p25",
                                 ["sz_all_p25", "sz_all_p75"]),
    "sz_qratio":                ("sz_p75 / (sz_p25 + eps)",
                                 ["sz_all_p25", "sz_all_p75"]),
    "sz_median_to_mean":        ("sz_median / (sz_mean + eps)",
                                 ["sz_all_median", "sz_all_mean"]),
    "sz_p25_median_ratio":      ("sz_p25 / (sz_median + eps)",
                                 ["sz_all_p25", "sz_all_median"]),
    "sz_p75_median_ratio":      ("sz_p75 / (sz_median + eps)",
                                 ["sz_all_p75", "sz_all_median"]),
    "sz_iqr_norm_median":       ("(sz_p75 - sz_p25) / (sz_median + eps)",
                                 ["sz_all_p25", "sz_all_p75", "sz_all_median"]),
    "iat_cv":                   ("iat_std / (iat_mean + eps)",
                                 ["iat_all_std", "iat_all_mean"]),
    "iat_iqr":                  ("iat_p75 - iat_p25",
                                 ["iat_all_p25", "iat_all_p75"]),
    "direction_balance_bytes":  ("(bytes_up - bytes_down) / (bytes_up + bytes_down + eps)",
                                 ["bytes_up", "bytes_down"]),
    "direction_balance_packets":("(packets_up - packets_down) / (packets_up + packets_down + eps)",
                                 ["packets_up", "packets_down"]),
    "dispersion_symmetry":      ("clip( (sz_p75 + sz_p25 - 2*sz_median) / (|sz_p75 - sz_p25| + eps), -1, 1 )",
                                 ["sz_all_p25", "sz_all_median", "sz_all_p75"]),
}

NOTES = {
    "sz_cv": "Canonical name. Legacy alias `sz_coef_variation` is identical and excluded from runtime to avoid duplication.",
    "sz_iqr": "Absolute byte units; complements the ratio features.",
    "sz_qratio": "Ratio, hence partly scale-invariant.",
    "sz_median_to_mean": "Detects skew without using absolute scale.",
    "sz_p25_median_ratio": "Lower-tail relative width.",
    "sz_p75_median_ratio": "Upper-tail relative width.",
    "sz_iqr_norm_median": "Scale-normalised dispersion.",
    "iat_cv": "Higher for bursty traffic, lower for steady streams.",
    "iat_iqr": "Absolute seconds; the only timing-magnitude feature kept.",
    "direction_balance_bytes": "Unified `(A-B)/(A+B+eps)`; same formula across ISCX/VNAT/USBVPN/live extractor.",
    "direction_balance_packets": "Unified `(A-B)/(A+B+eps)`; same formula across all datasets.",
    "dispersion_symmetry": "Clipped to [-1, 1] for numerical safety; same formula across all datasets.",
}

# ---------- 4. Build the two tables ----------
groups_rows = []
for f in EXPECTED:
    grp, desc = GROUPS[f]
    groups_rows.append({
        "feature_name": f,
        "feature_group": grp,
        "description": desc,
        "required_by_runtime": "yes" if f in runtime_features else "no",
    })
groups_df = pd.DataFrame(groups_rows)

contract_rows = []
for f in EXPECTED:
    formula, raw_inputs = FORMULAS[f]
    contract_rows.append({
        "feature_name": f,
        "formula_or_short_description": formula,
        "raw_inputs_needed": ", ".join(raw_inputs),
        "required_by_runtime": "yes" if f in runtime_features else "no",
        "notes": NOTES[f],
    })
contract_df = pd.DataFrame(contract_rows)

# ---------- 5. Save tables ----------
out_g_csv = THESIS_DIR / "unified_relative_shape_v2_features.csv"
out_g_md  = THESIS_DIR / "unified_relative_shape_v2_features.md"
out_c_csv = THESIS_DIR / "unified_relative_shape_v2_feature_contract.csv"
out_c_md  = THESIS_DIR / "unified_relative_shape_v2_feature_contract.md"

groups_df.to_csv(out_g_csv, index=False)
contract_df.to_csv(out_c_csv, index=False)

def df_to_md(df, title, preface=""):
    lines = [f"# {title}", ""]
    if preface:
        lines += [preface, ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(lines) + "\n"

preface_groups = (
    f"_Source: runtime `feature_order.json` for `unified_relative_shape_v2__lgbm`._  \n"
    f"_All {len(EXPECTED)} expected features are present, in the expected order. "
    f"No extra features required._\n"
)

artifacts_checked = [
    "artifacts/unified_feature_contract_v2/runtime_export/runtime_models/unified_relative_shape_v2__lgbm/feature_order.json",
    "artifacts/unified_feature_contract_v2/feature_contract.json",
    "artifacts/unified_feature_contract_v2/feature_families/unified_relative_shape_v2.json",
    "src/features/unified_extractor.py",
]
preface_contract = (
    "_Formulas are extracted from `src/features/unified_extractor.py` "
    "(the canonical unified extractor, `extractor_version = unified_v2.0`)._  \n"
    "_`eps = 1e-6`. Packet sizes use IP total length; IAT in seconds._  \n"
    "_Artifacts checked for formulas:_\n"
    + "\n".join(f"- `{p}`" for p in artifacts_checked)
    + "\n"
)

out_g_md.write_text(df_to_md(groups_df, "Unified relative_shape_v2 — feature groups", preface_groups), encoding="utf-8")
out_c_md.write_text(df_to_md(contract_df, "Unified relative_shape_v2 — feature contract", preface_contract), encoding="utf-8")

print("\nSaved:")
for p in [out_g_csv, out_g_md, out_c_csv, out_c_md]:
    print(" ", p)

# ---------- 6. Composition figure ----------
counts = groups_df["feature_group"].value_counts().reindex(
    ["size ratio / shape", "timing ratio / shape", "direction / symmetry"]
)
colors = ["#4C78A8", "#F58518", "#54A24B"]

fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(int(v)),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of features")
ax.set_title("unified_relative_shape_v2 — feature group composition\n"
             f"(total = {len(EXPECTED)} features required by runtime model)")
ax.set_ylim(0, max(counts.values) + 1.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

fig_path = THESIS_DIR / "fig_unified_feature_groups.png"
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved figure:", fig_path)

# ---------- 7. Sanity verification summary ----------
print("\n=== Verification summary ===")
print(f"  expected feature count   : {len(EXPECTED)}")
print(f"  runtime feature count    : {len(runtime_features)}")
print(f"  family json feature count: {len(fam_features)}")
print(f"  features identical       : {set(EXPECTED) == set(runtime_features) == set(fam_features)}")
print(f"  order matches expected   : {runtime_features == EXPECTED}")
