# src/clean_pipeline/feature_families.py
"""
Feature family definitions for the CLEAN pipeline.

Each family is a frozen set of feature names that can be computed identically
across all three datasets (VNAT, ISCX, USBVPN) from raw packet arrays.

Feature safety classifications:
  SAFE              — same formula, same semantic meaning, verified across all 3 datasets
  SEMANTICALLY_RISKY — formula matches but semantic meaning may differ (e.g. direction)
  BLOCKED           — known corruption or stub in at least one dataset
  REJECTED          — proven mismatched across datasets

PERMANENTLY EXCLUDED from CLEAN experiments:
  - stored dispersion_symmetry (ISCX corrupted)
  - stored direction_balance_bytes (ISCX corrupted)
  - stored direction_balance_packets (ISCX corrupted)
  - USBVPN iat_mean_max / iat_mean_min (== iat_all_mean, stub)
  - USBVPN iat_std_max / iat_std_min (stub)
  - any VNAT features.parquet pre-transformed columns
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


class FeatureSafety(str, Enum):
    SAFE = "SAFE"
    SEMANTICALLY_RISKY = "SEMANTICALLY_RISKY"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    formula: str
    source_fields: Tuple[str, ...]
    safety: FeatureSafety
    notes: str = ""


# ──────────────────────────────────────────────────────
# Feature registry — every clean feature must be here
# ──────────────────────────────────────────────────────

FEATURE_REGISTRY: Dict[str, FeatureSpec] = {}


def _register(spec: FeatureSpec) -> None:
    FEATURE_REGISTRY[spec.name] = spec


# === SAFE CORE 10 features (direction-INDEPENDENT) ===

_register(FeatureSpec(
    name="total_packets",
    formula="len(sizes)",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Count of packets in the flow window. Identical across all datasets.",
))

_register(FeatureSpec(
    name="total_bytes",
    formula="sum(abs(sizes))",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Sum of absolute packet sizes. Uses abs() for USBVPN signed bytes.",
))

_register(FeatureSpec(
    name="mean_pkt_len",
    formula="mean(abs(sizes))",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Mean absolute packet length.",
))

_register(FeatureSpec(
    name="std_pkt_len",
    formula="std(abs(sizes), ddof=0)",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Standard deviation of absolute packet lengths (population).",
))

_register(FeatureSpec(
    name="median_pkt_len",
    formula="median(abs(sizes))",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Median absolute packet length.",
))

_register(FeatureSpec(
    name="p25_pkt_len",
    formula="percentile(abs(sizes), 25)",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="25th percentile of absolute packet lengths.",
))

_register(FeatureSpec(
    name="p75_pkt_len",
    formula="percentile(abs(sizes), 75)",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="75th percentile of absolute packet lengths.",
))

_register(FeatureSpec(
    name="iat_mean",
    formula="mean(diff(timestamps))",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Mean inter-arrival time. Computed from sorted timestamps.",
))

_register(FeatureSpec(
    name="iat_std",
    formula="std(diff(timestamps), ddof=0)",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Std of inter-arrival times (population).",
))

_register(FeatureSpec(
    name="iat_median",
    formula="median(diff(timestamps))",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Median inter-arrival time.",
))

# === SAFE CORE PLUS DURATION features ===

_register(FeatureSpec(
    name="flow_duration",
    formula="timestamps[-1] - timestamps[0]",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Flow duration in seconds. Requires >= 2 packets.",
))

_register(FeatureSpec(
    name="packet_rate",
    formula="total_packets / max(flow_duration, eps)",
    source_fields=("timestamps", "sizes"),
    safety=FeatureSafety.SAFE,
    notes="Packets per second.",
))

_register(FeatureSpec(
    name="byte_rate",
    formula="total_bytes / max(flow_duration, eps)",
    source_fields=("timestamps", "sizes"),
    safety=FeatureSafety.SAFE,
    notes="Bytes per second.",
))

_register(FeatureSpec(
    name="max_pkt_len",
    formula="max(abs(sizes))",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Maximum absolute packet length.",
))

_register(FeatureSpec(
    name="min_pkt_len",
    formula="min(abs(sizes))",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Minimum absolute packet length.",
))

# === TEMPORAL FEATURES ===

_register(FeatureSpec(
    name="iat_cv",
    formula="iat_std / max(iat_mean, eps)",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Coefficient of variation of IAT. Burstiness proxy.",
))

_register(FeatureSpec(
    name="iat_p25",
    formula="percentile(diff(timestamps), 25)",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="25th percentile of inter-arrival times.",
))

_register(FeatureSpec(
    name="iat_p75",
    formula="percentile(diff(timestamps), 75)",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="75th percentile of inter-arrival times.",
))

_register(FeatureSpec(
    name="iat_iqr",
    formula="iat_p75 - iat_p25",
    source_fields=("timestamps",),
    safety=FeatureSafety.SAFE,
    notes="Interquartile range of IAT. Timing variability.",
))

_register(FeatureSpec(
    name="pkt_len_cv",
    formula="std_pkt_len / max(mean_pkt_len, eps)",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Coefficient of variation of packet lengths.",
))

_register(FeatureSpec(
    name="pkt_len_iqr",
    formula="p75_pkt_len - p25_pkt_len",
    source_fields=("sizes",),
    safety=FeatureSafety.SAFE,
    notes="Interquartile range of absolute packet lengths.",
))

# === DIRECTION-AUGMENTED features (SEMANTICALLY_RISKY) ===
# Only usable if direction semantics are harmonized across datasets.
# USBVPN uses sign convention (positive=src→dst, negative=dst→src).
# VNAT/ISCX use canonical IP sorting (1=A→B, 0=B→A).
# Direction semantics ARE different → these are RISKY.

_register(FeatureSpec(
    name="fwd_packets",
    formula="count(dir == 1)",
    source_fields=("directions",),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Forward packets. Direction semantics differ: VNAT/ISCX=canonical sort, USBVPN=sign.",
))

_register(FeatureSpec(
    name="bwd_packets",
    formula="count(dir == 0)",
    source_fields=("directions",),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Backward packets. Same direction-mismatch issue.",
))

_register(FeatureSpec(
    name="fwd_bytes",
    formula="sum(abs(sizes[dir == 1]))",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Forward bytes.",
))

_register(FeatureSpec(
    name="bwd_bytes",
    formula="sum(abs(sizes[dir == 0]))",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Backward bytes.",
))

_register(FeatureSpec(
    name="fwd_mean_pkt_len",
    formula="mean(abs(sizes[dir == 1]))",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Mean forward packet length.",
))

_register(FeatureSpec(
    name="bwd_mean_pkt_len",
    formula="mean(abs(sizes[dir == 0]))",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Mean backward packet length.",
))

_register(FeatureSpec(
    name="packet_ratio",
    formula="fwd_packets / max(bwd_packets, 1)",
    source_fields=("directions",),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Ratio of forward to backward packets.",
))

_register(FeatureSpec(
    name="byte_ratio",
    formula="fwd_bytes / max(bwd_bytes, eps)",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SEMANTICALLY_RISKY,
    notes="Ratio of forward to backward bytes.",
))

# === DIRECTION-INVARIANT alternatives to direction features ===
# These use min/max of per-direction stats, so direction label doesn't matter.

_register(FeatureSpec(
    name="dir_pkt_ratio_minmax",
    formula="min(fwd_pkts, bwd_pkts) / max(fwd_pkts, bwd_pkts, 1)",
    source_fields=("directions",),
    safety=FeatureSafety.SAFE,
    notes="Min/max packet ratio — direction-invariant symmetry measure.",
))

_register(FeatureSpec(
    name="dir_bytes_ratio_minmax",
    formula="min(fwd_bytes, bwd_bytes) / max(fwd_bytes, bwd_bytes, eps)",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SAFE,
    notes="Min/max byte ratio — direction-invariant.",
))

_register(FeatureSpec(
    name="dir_mean_pkt_max",
    formula="max(fwd_mean_pkt_len, bwd_mean_pkt_len)",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SAFE,
    notes="Larger mean packet size of the two directions. Direction-invariant.",
))

_register(FeatureSpec(
    name="dir_mean_pkt_min",
    formula="min(fwd_mean_pkt_len, bwd_mean_pkt_len)",
    source_fields=("sizes", "directions"),
    safety=FeatureSafety.SAFE,
    notes="Smaller mean packet size of the two directions. Direction-invariant.",
))


# ──────────────────────────────────────────────────────
# Feature families
# ──────────────────────────────────────────────────────

SAFE_CORE_10: Tuple[str, ...] = (
    "total_packets",
    "total_bytes",
    "mean_pkt_len",
    "std_pkt_len",
    "median_pkt_len",
    "p25_pkt_len",
    "p75_pkt_len",
    "iat_mean",
    "iat_std",
    "iat_median",
)

SAFE_CORE_PLUS_DURATION: Tuple[str, ...] = SAFE_CORE_10 + (
    "flow_duration",
    "packet_rate",
    "byte_rate",
    "max_pkt_len",
    "min_pkt_len",
)

SAFE_CORE_PLUS_TEMPORAL: Tuple[str, ...] = SAFE_CORE_PLUS_DURATION + (
    "iat_cv",
    "iat_p25",
    "iat_p75",
    "iat_iqr",
    "pkt_len_cv",
    "pkt_len_iqr",
)

DIRECTION_INVARIANT_AUGMENTED: Tuple[str, ...] = SAFE_CORE_PLUS_TEMPORAL + (
    "dir_pkt_ratio_minmax",
    "dir_bytes_ratio_minmax",
    "dir_mean_pkt_max",
    "dir_mean_pkt_min",
)

DIRECTION_AUGMENTED: Tuple[str, ...] = DIRECTION_INVARIANT_AUGMENTED + (
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_mean_pkt_len",
    "bwd_mean_pkt_len",
    "packet_ratio",
    "byte_ratio",
)

FAMILY_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "safe_core_10": SAFE_CORE_10,
    "safe_core_plus_duration": SAFE_CORE_PLUS_DURATION,
    "safe_core_plus_temporal": SAFE_CORE_PLUS_TEMPORAL,
    "direction_invariant_augmented": DIRECTION_INVARIANT_AUGMENTED,
    "direction_augmented": DIRECTION_AUGMENTED,
}

# Permanently excluded legacy columns
PERMANENTLY_EXCLUDED: FrozenSet[str] = frozenset({
    "dispersion_symmetry",
    "direction_balance_bytes",
    "direction_balance_packets",
    "iat_mean_max",
    "iat_mean_min",
    "iat_std_max",
    "iat_std_min",
    "sz_coef_variation",
    "sz_p25_median_ratio",
    "sz_p75_median_ratio",
    "sz_iqr_norm_median",
})


def get_family(name: str) -> Tuple[str, ...]:
    """Get feature names for a family."""
    if name not in FAMILY_REGISTRY:
        raise ValueError(
            f"Unknown feature family '{name}'. "
            f"Available: {sorted(FAMILY_REGISTRY.keys())}"
        )
    return FAMILY_REGISTRY[name]


def get_family_safety(name: str) -> Dict[str, FeatureSafety]:
    """Get safety classification for each feature in a family."""
    features = get_family(name)
    result = {}
    for f in features:
        if f in FEATURE_REGISTRY:
            result[f] = FEATURE_REGISTRY[f].safety
        else:
            result[f] = FeatureSafety.REJECTED
    return result


def family_has_risky_features(name: str) -> bool:
    """Check if a family contains SEMANTICALLY_RISKY features."""
    safety = get_family_safety(name)
    return any(s == FeatureSafety.SEMANTICALLY_RISKY for s in safety.values())


def validate_family_in_dataframe(
    df,
    family_name: str,
) -> List[str]:
    """
    Check that all features of a family exist in a DataFrame.
    Returns list of missing feature names (empty if all present).
    """
    features = get_family(family_name)
    return [f for f in features if f not in df.columns]

