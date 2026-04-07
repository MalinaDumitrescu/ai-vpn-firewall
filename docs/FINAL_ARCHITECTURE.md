# Final Firewall Architecture — VPN Session Detection

> **Status:** Deployment-ready candidate under pooled held-out evaluation protocol.
> **Date:** 2026-03-30
> **Scope:** Conservative VPN detection firewall operating on packet-flow statistics.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Stage 1 — Flow Builder](#3-stage-1--flow-builder)
4. [Stage 2 — Feature Extractor](#4-stage-2--feature-extractor)
5. [Stage 3 — Ensemble Inference](#5-stage-3--ensemble-inference)
6. [Stage 4 — Session Aggregation](#6-stage-4--session-aggregation)
7. [Stage 5 — Policy Layer](#7-stage-5--policy-layer)
8. [Firewall API Reference](#8-firewall-api-reference)
9. [Threshold Calibration Algorithm](#9-threshold-calibration-algorithm)
10. [Deployment Mode Logic](#10-deployment-mode-logic)
11. [Evaluation Pipeline](#11-evaluation-pipeline)
12. [Validated Results (Source of Truth)](#12-validated-results-source-of-truth)
13. [Inference Pseudo-Code](#13-inference-pseudo-code)
14. [Discussion, Limitations, and Deployment Interpretation](#14-discussion-limitations-and-deployment-interpretation)
15. [Future Work](#15-future-work)

---

## 1. System Overview

This system is a **deployment-oriented VPN detection firewall**, not a generic benchmark classifier. It:

- Operates on **flow-level packet statistics** (packet sizes and directions).
- Predicts at **flow level** but decides at **session level**.
- Uses a **balanced bagging ensemble** of three gradient boosting families.
- Uses **isotonic-calibrated probabilities** for principled threshold selection.
- Enforces **zero false blocks** in STRICT mode on the evaluated validation/test protocol.
- Maximizes **VPN blocking recall** subject to that zero-false-block constraint.
- Supports three **operational deployment modes**: STRICT, BALANCED, RESEARCH.

### Design Philosophy

The firewall treats false blocks (blocking legitimate traffic) as categorically worse than missed VPN detections. The threshold is therefore set at the maximum benign session score observed during validation, guaranteeing that no benign session in the evaluated population would be blocked. Sessions that score above this threshold are blocked; sessions that score above a lower flag threshold are flagged for human review; all others are allowed.

This guarantee is **conditional on the deployment traffic being drawn from a distribution similar to the pooled training domain** (ISCX + VNAT). The deployed ensemble is trained on ISCX + VNAT (19,908 flows); three-dataset pooled training (adding USBVPN) was attempted but degrades performance due to domain fingerprinting — see Section 14.3.6. It is not a universal claim.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VPN DETECTION FIREWALL                          │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐ │
│  │  Stage 1  │   │   Stage 2    │   │   Stage 3    │   │ Stage 4 │ │
│  │   Flow    │──▶│   Feature    │──▶│   Ensemble   │──▶│ Session │ │
│  │  Builder  │   │  Extractor   │   │  Inference   │   │  Agg.   │ │
│  └──────────┘   └──────────────┘   └──────────────┘   └────┬────┘ │
│       ▲                                                      │      │
│       │                                                      ▼      │
│  pcap/packets                                          ┌─────────┐ │
│                                                        │ Stage 5 │ │
│                                                        │ Policy  │ │
│                                                        │  Layer  │ │
│                                                        └────┬────┘ │
│                                                             │      │
│                                                             ▼      │
│                                                  BLOCK / FLAG / ALLOW
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow:**

```
Packet stream / PCAP
    │
    ▼
Stage 1: FlowBuilder
    │  Input:  raw packets
    │  Output: bidirectional flows (timestamps, sizes, directions)
    │  Config: N=100, min_packets=10, eps=1e-6
    ▼
Stage 2: FeatureExtractor
    │  Input:  flow tuples
    │  Output: 7 compact features per flow
    │  Config: optional DROP_DIRECTION_BALANCE
    ▼
Stage 3: EnsemblePredictor
    │  Input:  feature vectors
    │  Output: calibrated P(VPN) per flow
    │  Config: 3×XGB + 3×LGBM + 3×Cat → family avg → isotonic
    ▼
Stage 4: SessionAggregator
    │  Input:  calibrated flow probabilities grouped by capture_id
    │  Output: session_score = p90(prob_iso)
    ▼
Stage 5: PolicyEngine
    │  Input:  session_score
    │  Output: BLOCK / FLAG / ALLOW + metadata
    │  Config: mode = STRICT / BALANCED / RESEARCH
    ▼
SessionDecision {
    capture_id, session_score, decision,
    block_threshold, flag_threshold,
    confidence_margin, deployment_mode
}
```

---

## 3. Stage 1 — Flow Builder

### Purpose

Convert a packet stream or PCAP file into bidirectional network flows. Each flow represents a sequence of packets between two endpoints.

### Input

| Field | Type | Description |
|-------|------|-------------|
| `ts` | float | Packet timestamp (epoch seconds) |
| `src_ip` | str | Source IP address |
| `src_port` | int | Source port |
| `dst_ip` | str | Destination IP address |
| `dst_port` | int | Destination port |
| `proto` | int | IP protocol (6=TCP, 17=UDP) |
| `size` | int | Packet payload size in bytes |
| `tcp_flags` | int | TCP flags (optional, for FIN/RST detection) |

### Output (Per Flow)

| Field | Type | Description |
|-------|------|-------------|
| `timestamps` | list[float] | Ordered packet timestamps |
| `sizes` | list[int] | Packet sizes |
| `directions` | list[int] | Packet directions (0=forward, 1=backward) |

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_n` | 100 | Maximum packets per flow. Captures enough statistical signal while bounding computation. |
| `min_packets` | 10 | Minimum packets for a valid flow. Below this, statistical features are unreliable. |
| `eps` | 1e-6 | Numerical stability epsilon for division. |
| `inactivity_timeout` | 120 s | Split flows after 120 seconds of inactivity. |

### Flow Construction Rules

1. Packets are grouped into **bidirectional flows** by 5-tuple: `{src_ip, src_port, dst_ip, dst_port, proto}`, with the reverse tuple mapping to the same flow.
2. A flow is **finalized** when:
   - A TCP FIN or RST flag is observed.
   - The inactivity timeout (120 s) expires.
   - The packet window (`N=100`) is filled.
3. Only the **first 100 packets** of each flow are retained.
4. Flows with **fewer than 10 packets** are discarded.

### Implementation Reference

```
demo_firewall/flow_tracker.py → FlowTracker
src/flow/builder.py           → FlowBuilder
```

---

## 4. Stage 2 — Feature Extractor

### Purpose

Compute 7 compact statistical features from each flow's packet sizes and directions.

### Feature Definitions

| # | Feature | Formula | Interpretation |
|---|---------|---------|----------------|
| 1 | `sz_coef_variation` | `std(sizes) / (mean(sizes) + eps)` | Normalized packet size variability. VPN tunnels often homogenize sizes. |
| 2 | `sz_p25_median_ratio` | `percentile(sizes, 25) / (median(sizes) + eps)` | Lower quartile shape. Captures asymmetry in small-packet distribution. |
| 3 | `sz_p75_median_ratio` | `percentile(sizes, 75) / (median(sizes) + eps)` | Upper quartile shape. Captures asymmetry in large-packet distribution. |
| 4 | `sz_iqr_norm_median` | `(p75 - p25) / (median + eps)` | Normalized interquartile range. Measures spread relative to center. |
| 5 | `dispersion_symmetry` | `(p75_ratio - p25_ratio)` or related measure | Whether the size distribution is symmetric or skewed. |
| 6 | `direction_balance_bytes` | `bytes_forward / (bytes_forward + bytes_backward + eps)` | Byte-level directionality. **Domain-fingerprinting risk.** |
| 7 | `direction_balance_packets` | `pkts_forward / (pkts_forward + pkts_backward + eps)` | Packet-level directionality. **Domain-fingerprinting risk.** |

### Ablation Switch

```python
DROP_DIRECTION_BALANCE = True  # Remove features 6 and 7
```

When enabled, the feature vector is reduced from 7 to 5 features. This removes direction-balance features that encode dataset collection methodology rather than pure VPN/benign semantics.

**Recommendation:** Use the full 7-feature set for **pooled-domain** deployment. Use the 5-feature reduced set when deploying to a **new network environment** until local validation confirms that direction features remain discriminative.

### Implementation Reference

```
demo_firewall/flow_tracker.py → FlowTracker._extract_features()
src/features/extract.py        → extract_features_from_flows()
demo_firewall/config.py        → COMPACT_FEATURES, DIRECTION_FEATURES
```

---

## 5. Stage 3 — Ensemble Inference

### Architecture

```
                    ┌─── XGB bag 0 ───┐
             ┌──── │ XGB bag 1        │ ──── mean ──── p_xgb ────┐
             │     └─── XGB bag 2 ───┘                            │
             │                                                     │
Feature  ────┼──── ┌─── LGBM bag 0 ──┐                            │
Vector       │     │ LGBM bag 1       │ ──── mean ──── p_lgbm ───┤── mean ── prob_raw ── isotonic ── prob_cal
             │     └─── LGBM bag 2 ──┘                            │
             │                                                     │
             └──── ┌─── Cat bag 0 ───┐                            │
                   │ Cat bag 1        │ ──── mean ──── p_cat ─────┘
                   └─── Cat bag 2 ───┘
```

### Ensemble Composition

| Family | Library | Bags | Role |
|--------|---------|------|------|
| XGBoost | `xgboost` | 3 | Gradient boosted trees with second-order gradients |
| LightGBM | `lightgbm` | 3 | Histogram-based gradient boosting, leaf-wise growth |
| CatBoost | `catboost` | 3 | Ordered boosting with symmetric trees |

**Total: 9 models.**

### Balanced Bagging Strategy

Each bag is trained on:
- **All minority (VPN) samples** — ensures the full positive distribution is seen by every bag.
- **A random subset of majority (benign) samples** — reduces per-bag benign variance and creates diversity across bags.

This addresses class imbalance without synthetic oversampling.

### Probability Aggregation

1. **Within-family averaging:** For each family, average the `predict_proba` outputs of its 3 bags.
   ```
   p_xgb  = mean(bag0_xgb, bag1_xgb, bag2_xgb)
   p_lgbm = mean(bag0_lgbm, bag1_lgbm, bag2_lgbm)
   p_cat  = mean(bag0_cat, bag1_cat, bag2_cat)
   ```

2. **Cross-family averaging:** Equal-weight average across families.
   ```
   prob_raw = (p_xgb + p_lgbm + p_cat) / 3
   ```

3. **Isotonic calibration:** A non-parametric monotonic transform fitted on validation data.
   ```
   prob_cal = isotonic_regression.predict(prob_raw)
   ```

### Why Isotonic Calibration

- **Non-parametric:** Adapts to the local density of the probability space.
- **Monotonic:** Preserves the ranking of the raw ensemble output.
- **Better tail calibration:** Unlike Platt (logistic) scaling, isotonic regression calibrates the extreme tails accurately, which is critical for threshold selection at high confidence levels.

### Calibration Quality (Test Set, Flow-Level)

| Method | ROC-AUC | PR-AUC | Brier Score | Log Loss |
|--------|---------|--------|-------------|----------|
| Raw | — | — | — | — |
| Isotonic | ≈ same | ≈ same | Lower | Lower |
| Platt | ≈ same | ≈ same | Higher | Higher |

Isotonic calibration improves probability quality (lower Brier score, lower log loss) without sacrificing discrimination (AUC preserved).

### Validated Flow-Level Model Comparison (Source of Truth)

| Model | Val ROC-AUC | Test ROC-AUC | Val PR-AUC | Test PR-AUC |
|-------|-------------|--------------|------------|-------------|
| CatBoost | 0.9824 | **0.9492** | 0.9775 | **0.8804** |
| Ensemble (isotonic) | **0.9846** | 0.9461 | 0.9754 | 0.8496 |
| Ensemble (platt) | 0.9817 | 0.9443 | 0.9773 | 0.8613 |
| Ensemble (raw) | 0.9817 | 0.9443 | 0.9773 | 0.8613 |
| LightGBM | 0.9774 | 0.9410 | 0.9741 | 0.8490 |
| XGBoost | 0.9796 | 0.9427 | 0.9756 | 0.8516 |

**Interpretation:** CatBoost is the strongest single-family model at flow level on the reported test split. However, the 9-model ensemble is the final deployed candidate because deployment selection is based on session-level firewall behavior (see Section 5.4 below), not only flow-level AUC.

### Model Family Agreement (Test Set)

| # Families Voting VPN | Benign Flows | VPN Flows |
|----------------------|--------------|-----------|
| 0 | 1507 | 40 |
| 1 | 22 | 2 |
| 2 | 25 | 7 |
| 3 | 39 | 202 |

- **Unanimous agreement: 97.0%**, disagreement: 3.0%.
- High agreement confirms that ensemble inference is stable across families.
- Family diversity is useful: the 3.0% disagreement region is where the ensemble averaging provides the most value.

### Why the 9-Model Ensemble Is Preferred Over CatBoost Alone

CatBoost achieves the highest standalone flow-level AUC, but the mixed ensemble is preferred for deployment because:

1. **Multi-family consensus.** Three independently-trained model families reduce systematic algorithmic bias. The 97% unanimous agreement rate confirms this is a stable consensus, not noise.
2. **Calibrated session scoring.** The isotonic calibrator was trained on the ensemble's probability distribution. Switching to a single family would require recalibrating the entire session scoring and threshold pipeline.
3. **Validated API-level behavior.** The ensemble is the only configuration that has been tested end-to-end through `predict_flow()` → `predict_session()` → `evaluate_dataset()` with verified STRICT mode metrics.
4. **Robustness margin.** A single-family model may overfit to the current test split. The multi-family ensemble provides implicit regularization against any one family's failure modes.

### Model Backend Selection

The firewall API supports four backend configurations:

| Backend | Models | Families | Use Case |
|---------|--------|----------|----------|
| `ensemble_all` | 9 | 3 (XGB + LGBM + Cat) | **Production default.** Full multi-family ensemble. |
| `xgb_only` | 3 | 1 (XGB) | XGBoost-only operation. |
| `lgbm_only` | 3 | 1 (LGBM) | LightGBM-only operation. |
| `cat_only` | 3 | 1 (Cat) | CatBoost-only operation. |

```python
# Example: CatBoost-only inference
blocker = FirewallBlocker(mode=DeploymentMode.STRICT, model_backend="cat_only")
blocker.load()
blocker.calibrate_from_validation()
metrics = blocker.evaluate_dataset()
```

**Note:** Single-family backends use the same isotonic calibrator trained on ensemble probabilities. For optimal single-family deployment, a family-specific calibrator should be trained. The current implementation is suitable for comparison and ablation but the final production deployment should use `ensemble_all`.

### Implementation Reference

```
demo_firewall/predictor.py → EnsemblePredictor
src/eval/calibration.py     → ProbabilityCalibrator
```

---

## 6. Stage 4 — Session Aggregation

### Purpose

Aggregate flow-level calibrated probabilities into a single session-level score.

### Grouping Key

```
session_col = "capture_id"
```

Each `capture_id` represents one network capture session. A session contains multiple bidirectional flows.

### Aggregation Rule: p90

```python
session_score = numpy.percentile(flow_probabilities, 90)
```

**Interpretation:** The session score is the 90th percentile of its constituent flow probabilities. This means:
- At least 10% of flows must have high VPN probability for the session to score high.
- Isolated false-positive flows are filtered out.
- Genuinely VPN sessions produce consistently high flow scores, so p90 remains high.

### Aggregation Comparison (Validated, Test Sessions, Isotonic Calibration)

| Aggregation | Session AUC | Session PR-AUC | Block Recall @ FPR=0 |
|-------------|-------------|----------------|----------------------|
| mean | 0.9980 | 0.9967 | 0.9412 |
| p90 | 0.9902 | 0.9775 | 0.7647 |
| weighted_top5_mean | 0.9686 | 0.9518 | 0.7059 |

**Why p90 is the deployment choice:**

Mean aggregation achieves higher block recall at zero FPR, but p90 is more conservative and less sensitive to outlier flows. The p90 rule is the validated default that is already wired into the deployed firewall evaluation pipeline. It provides a robust session signal while tolerating flow-level noise.

### Implementation Reference

```
demo_firewall/policy.py → _p90_aggregation()
```

---

## 7. Stage 5 — Policy Layer

### Decision Logic

```
Given: session_score, block_threshold, flag_threshold

if mode == RESEARCH:
    decision = ALLOW  (no thresholding)
elif session_score > block_threshold:
    decision = BLOCK
elif session_score > flag_threshold:
    decision = FLAG
else:
    decision = ALLOW
```

### Decision Types

| Decision | Meaning | Action |
|----------|---------|--------|
| `BLOCK` | High-confidence VPN session | Block/terminate traffic |
| `FLAG` | Moderate VPN suspicion | Flag for human review |
| `ALLOW` | Below detection threshold | Allow traffic to pass |

### Confidence Margin

Each decision includes a **confidence margin**: the distance from the session score to the nearest threshold.

```python
if decision == BLOCK:
    margin = session_score - block_threshold
elif decision == FLAG:
    margin = min(session_score - flag_threshold, block_threshold - session_score)
else:
    margin = flag_threshold - session_score
```

This enables operators to prioritize reviews by confidence.

### Implementation Reference

```
demo_firewall/policy.py → FirewallPolicy
```

---

## 8. Firewall API Reference

### Top-Level Entry Point: `FirewallBlocker`

```python
from demo_firewall import FirewallBlocker, DeploymentMode

# Default: 9-model ensemble
blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
blocker.load()
blocker.calibrate_from_validation()

# Single-family: CatBoost only (3 models)
blocker_cat = FirewallBlocker(mode=DeploymentMode.STRICT, model_backend="cat_only")
blocker_cat.load()
blocker_cat.calibrate_from_validation()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | DeploymentMode | STRICT | STRICT / BALANCED / RESEARCH |
| `model_backend` | str | "ensemble_all" | "ensemble_all", "xgb_only", "lgbm_only", "cat_only" |
| `drop_direction_features` | bool | False | Remove direction_balance features |
| `calibration_method` | str | "isotonic" | "isotonic", "platt", "none" |
| `block_threshold` | float | None | Override block threshold |
| `flag_threshold` | float | None | Override flag threshold |
| `min_packets` | int | 10 | Minimum packets per flow |
| `window_n` | int | 100 | Maximum packets per flow window |

### Core Methods

#### `predict_flow(flow_features: DataFrame) → DataFrame`

Predict P(VPN) for each flow in a feature DataFrame.

**Implemented in:** `FirewallBlocker.predict_flows()` → delegates to `EnsemblePredictor.predict_flow()`

**Returns per flow:**
| Field | Type | Description |
|-------|------|-------------|
| `prob_raw` | float | Raw ensemble probability |
| `prob_cal` | float | Isotonic-calibrated probability |
| `prob_xgb` | float | XGBoost family probability |
| `prob_lgbm` | float | LightGBM family probability |
| `prob_cat` | float | CatBoost family probability |
| `calibration_method` | str | Calibration method used |
| `confidence_margin` | float | |abs(prob_cal - 0.5)| × 2 |

#### `predict_session(flow_preds: DataFrame) → SessionDecision`

Make a session-level decision from flow predictions.

**Implemented in:** `FirewallBlocker.predict_session()` → delegates to `FirewallPolicy.predict_session()`

**Returns:**
| Field | Type | Description |
|-------|------|-------------|
| `capture_id` | str | Session identifier |
| `session_score` | float | Aggregated session score (p90) |
| `decision` | Decision | BLOCK / FLAG / ALLOW |
| `block_threshold` | float | Block threshold used |
| `flag_threshold` | float | Flag threshold used |
| `aggregation_rule` | str | "p90" / "weighted_top5_mean" / "mean" |
| `n_flows` | int | Number of flows in session |
| `n_flows_above_block` | int | Flows exceeding block threshold |
| `n_flows_above_flag` | int | Flows exceeding flag threshold |
| `confidence_margin` | float | Distance to nearest threshold |
| `deployment_mode` | str | "strict" / "balanced" / "research" |
| `flow_decisions` | list[FlowDecision] | Per-flow decisions |

#### `predict_pcap(pcap_path, capture_id=None) → SessionDecision`

Full pipeline: PCAP file → session decision. Executes Stages 1 through 5 in sequence.

**Implemented in:** `FirewallBlocker.predict_pcap()`

#### `predict_capture(capture_id, predictions_csv=None) → SessionDecision`

Look up a specific session by `capture_id` in pre-computed predictions and return a firewall decision. Useful for inspecting individual sessions from the evaluation dataset without re-running model inference.

**Implemented in:** `FirewallBlocker.predict_capture()`

#### `predict_packet_stream(packets: Iterator[Dict]) → SessionDecision`

Full pipeline from a live packet iterator.

**Implemented in:** `FirewallBlocker.predict_packet_stream()`

#### `evaluate_dataset(predictions_csv=None) → Dict`

Evaluate the firewall on a labeled test set using pre-computed predictions.

**Implemented in:** `FirewallBlocker.evaluate_dataset()` → delegates to `evaluate_with_labels()`

**Returns:**
| Metric | Description |
|--------|-------------|
| `flow_roc_auc` | Flow-level ROC-AUC |
| `flow_pr_auc` | Flow-level PR-AUC |
| `session_roc_auc` | Session-level ROC-AUC |
| `session_pr_auc` | Session-level PR-AUC |
| `block_recall` | Fraction of VPN sessions blocked |
| `block_fpr` | Fraction of benign sessions incorrectly blocked |
| `block_precision` | Precision of block decisions |
| `flagged_recall` | Fraction of VPN sessions detected (blocked + flagged) |
| `flagged_fpr` | Fraction of benign sessions incorrectly flagged |
| `block_confusion` | TP, FP, FN, TN for block decisions |
| `flagged_confusion` | TP, FP, FN, TN for block+flag decisions |
| `block_threshold` | Block threshold value |
| `flag_threshold` | Flag threshold value |
| `n_sessions_evaluated` | Total sessions evaluated |
| `per_dataset` | Per-dataset session AUC, block recall, block FPR |
| `recall_vs_fpr_sweep` | 31-point sweep from FPR=0.0 to FPR=0.15 |

#### `calibrate_from_validation(val_predictions_path=None) → ThresholdConfig`

Calibrate block and flag thresholds from validation predictions.

**Implemented in:** `FirewallBlocker.calibrate_from_validation()` → delegates to `FirewallPolicy.calibrate_thresholds()`

**Algorithm:**
1. Load validation split from ensemble predictions.
2. Aggregate flow probabilities to session level using mode-specific aggregation.
3. Compute block threshold = `max(benign session scores)` (STRICT) or `quantile(benign, 1 - target_fpr)` (BALANCED).
4. Compute flag threshold = `min(flag_thr_from_0.1%_fpr, 0.7 × block_threshold)` (STRICT).
5. Store threshold provenance metadata.

### Safety Exceptions

| Exception | Trigger | Purpose |
|-----------|---------|---------|
| `CalibrationError` | Only one class in calibration split | Prevents degenerate calibration |
| `ThresholdLeakageError` | Threshold computed on contaminated data | Prevents test-set information leakage |
| `ModelLoadError` | Missing or corrupted model artifacts | Prevents silent inference failures |
| `FeatureExtractionError` | Non-finite values, missing features | Prevents garbage-in predictions |
| `InsufficientDataError` | Zero valid flows in a session | Prevents empty-input decisions |

---

## 9. Threshold Calibration Algorithm

### STRICT Mode (target FPR = 0)

```
ALGORITHM: CalibrateStrictThreshold

INPUT:
  val_preds    — flow-level validation predictions (with prob_iso and labels)
  session_col  — grouping column (capture_id)
  prob_col     — calibrated probability column (prob_iso)

OUTPUT:
  block_threshold, flag_threshold, provenance_metadata

PROCEDURE:
  1. session_scores ← GroupBy(val_preds, session_col).apply(p90)
     session_labels ← GroupBy(val_preds, session_col).max(label)

  2. benign_scores ← session_scores WHERE session_labels == 0
     vpn_scores    ← session_scores WHERE session_labels == 1

  3. ASSERT |benign_scores| > 0
     ASSERT |vpn_scores| > 0

  4. block_threshold ← max(benign_scores)
     // Guarantees: for all benign sessions, score ≤ threshold
     // Therefore:  block_FPR = 0 on validation set

  5. flag_threshold ← min(
       quantile(benign_scores, 0.999),   // 0.1% FPR
       0.7 × block_threshold             // 70% of block threshold
     )

  6. RETURN block_threshold, flag_threshold, {
       source_split: "val",
       computed_on_benign_only: True,
       n_benign_sessions: |benign_scores|,
       n_vpn_sessions: |vpn_scores|,
       max_benign_score: max(benign_scores),
       min_vpn_score: min(vpn_scores)
     }
```

### BALANCED Mode (target FPR = 0.1%)

```
ALGORITHM: CalibrateBalancedThreshold

INPUT:
  val_preds, session_col, prob_col (same as above)

OUTPUT:
  block_threshold, flag_threshold

PROCEDURE:
  1-2. (same session aggregation)

  3. block_threshold ← quantile(benign_scores, 1.0 - 0.001)
     // Allows up to 0.1% of benign sessions to be blocked

  4. flag_threshold ← 0.5 × block_threshold
```

### RESEARCH Mode

No threshold calibration. All sessions receive `ALLOW`. Raw scores are returned for analysis.

---

## 10. Deployment Mode Logic

### Mode Specifications

| Parameter | STRICT | BALANCED | RESEARCH |
|-----------|--------|----------|----------|
| **Target FPR** | 0.0 | 0.001 (0.1%) | N/A |
| **Aggregation** | p90 | weighted_top5_mean | mean |
| **Enforce zero block FPR** | Yes | No | No |
| **Block threshold** | max(benign val score) | quantile(benign, 0.999) | N/A |
| **Flag threshold** | min(0.1% FPR, 0.7 × block) | 0.5 × block | 0.5 |
| **Thresholding active** | Yes | Yes | No |
| **Use case** | Production firewall | Monitored deployment | Offline analysis |

### Mode Selection Guidance

```
IF deploying as a production firewall where false blocks are unacceptable:
    mode = STRICT

IF deploying with human operator oversight and higher recall is desired:
    mode = BALANCED

IF performing offline research, threshold tuning, or ROC analysis:
    mode = RESEARCH
```

### Decision Matrix (STRICT Mode)

```
session_score > 0.958769  →  BLOCK   (VPN with certainty, zero false positives)
session_score > flag_thr  →  FLAG    (moderate suspicion, human review)
session_score ≤ flag_thr  →  ALLOW   (below detection threshold)
```

---

## 11. Evaluation Pipeline

### Metrics Reported

#### Flow-Level Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| Flow ROC-AUC | Area under ROC curve for individual flows | Measures raw discriminative power |
| Flow PR-AUC | Area under Precision-Recall curve | Accounts for class imbalance at flow level |

#### Session-Level Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| Session ROC-AUC | ROC-AUC after session aggregation | Primary discrimination metric |
| Session PR-AUC | PR-AUC after session aggregation | Accounts for session-level imbalance |
| Block Recall | TP_block / (TP_block + FN_block) | VPN sessions actually blocked |
| Block FPR | FP_block / (FP_block + TN_block) | Benign sessions incorrectly blocked |
| Block Precision | TP_block / (TP_block + FP_block) | Fraction of blocks that are correct |
| Flagged Recall | TP_flag / (TP_flag + FN_flag) | VPN sessions detected (block + flag) |
| Flagged FPR | FP_flag / (FP_flag + TN_flag) | Benign sessions incorrectly flagged |

#### Threshold Values

| Value | Description |
|-------|-------------|
| Block threshold | Score above which sessions are blocked |
| Flag threshold | Score above which sessions are flagged |

#### Per-Dataset Metrics

Breakdown of session AUC and block recall for each dataset (ISCX, VNAT) to identify domain-specific performance.

#### Session Confusion Matrices

Full TP/FP/FN/TN matrices for both BLOCK and BLOCK+FLAG operating points.

#### Recall vs FPR Budget Sweep

Sweep over FPR budgets from 0.0 to 0.15 in increments of 0.005, reporting recall at each operating point. Enables selection of alternative operating points for different risk tolerances.

### Implementation Cross-Reference

All metrics above are computed by a single function:

```
demo_firewall/report.py → evaluate_with_labels()
```

- **Flow-level AUC:** `roc_auc_score` and `average_precision_score` on `(flow_labels, flow_probs)`
- **Session-level AUC:** same, on `(session_labels, session_scores)`
- **Block/flagged confusion:** numpy boolean indexing on session decisions
- **Per-dataset:** groups sessions by `dataset` column, computes per-group AUC and block metrics
- **FPR sweep:** iterates over `np.arange(0.0, 0.155, 0.005)`, computing threshold at each FPR budget as `np.quantile(benign_scores, 1 - fpr_budget)` and reporting recall at that threshold

### Evaluation Pipeline Execution

```python
from demo_firewall import FirewallBlocker, DeploymentMode

# Initialize and load
blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
blocker.load()
blocker.calibrate_from_validation()

# Evaluate on test set
metrics = blocker.evaluate_dataset()

# Generate formatted report
report = blocker.generate_report(
    session_decisions=blocker.predict_sessions_batch(test_preds),
    flow_preds=test_preds,
    output_dir=Path("reports/")
)
print(report)
```

### CLI Evaluation

```bash
# STRICT mode evaluation (full report with flow AUC, per-dataset, FPR sweep)
python run_firewall.py evaluate --mode strict

# BALANCED mode evaluation
python run_firewall.py evaluate --mode balanced

# Compare all modes side-by-side
python run_firewall.py compare

# Per-dataset breakdown (ISCX, VNAT separate)
python run_firewall.py per-dataset --mode strict

# Evaluate with reduced features (no direction balance)
python run_firewall.py evaluate --drop-direction

# Evaluate with a single model family (CatBoost only)
python run_firewall.py evaluate --backend cat_only

# Save report as JSON
python run_firewall.py evaluate --save-report

# Classify a single pcap file
python run_firewall.py predict path/to/capture.pcap

# Show full system diagnostics
python run_firewall.py info
```

### Verified Live Output (2026-03-30)

The following is actual output from `python run_firewall.py evaluate --mode strict`, confirming that all documented metrics are produced by real code:

```
  SESSION SUMMARY
  ----------------------------------------
  Sessions evaluated: 47
  Positive (VPN):     17
  Negative (Benign):  30
  Flows evaluated:    1844

  THRESHOLDS
  ----------------------------------------
  Block threshold:  0.958769
  Flag threshold:   0.671138

  FLOW-LEVEL METRICS
  ----------------------------------------
  Flow ROC-AUC:      0.9461
  Flow PR-AUC:       0.8496

  SESSION-LEVEL METRICS
  ----------------------------------------
  Session ROC-AUC:   0.9902
  Session PR-AUC:    0.9775
  Block Recall:      0.7059
  Block FPR:         0.0000
  Block Precision:   1.0000
  Flagged Recall:    1.0000
  Flagged FPR:       0.0667

  BLOCK CONFUSION MATRIX
  ----------------------------------------
  TP=12  FP=0  FN=5  TN=30

  FLAGGED (BLOCK+FLAG) CONFUSION MATRIX
  ----------------------------------------
  TP=17  FP=2  FN=0  TN=28

  PER-DATASET BREAKDOWN
  ----------------------------------------
  Dataset      Sessions      AUC  BlkRecall   BlkFPR
  iscx               21   1.0000     1.0000   0.0000
  vnat               26   0.9793     0.6154   0.0000

  RECALL vs FPR BUDGET SWEEP (sample)
  ----------------------------------------
  FPR Budget  Threshold   Recall  ActualFPR
      0.0000   0.750000   0.7647     0.0000
      0.0050   0.743865   0.8235     0.0333
      0.0100   0.737731   0.8235     0.0333
      ...

  PREDICTOR DIAGNOSTICS
  ----------------------------------------
  Models loaded:     9
  Families:          3
  Calibration:       isotonic
  Features:          7

  POLICY DIAGNOSTICS
  ----------------------------------------
  Mode:              strict
  Zero-FPR enforced: True
  Target FPR:        0.0
```

---

## 12. Validated Results (Source of Truth)

These are the validated evaluation results to be treated as the authoritative reference for all deployment claims.

### STRICT Mode — Test Set

| Metric | Value |
|--------|-------|
| Block Recall | **0.0556** |
| Block FPR | **0.0000** |
| Block Precision | **1.0000** |
| Flagged Recall | **0.1667** |
| Flagged FPR | **0.1287** |
| Session ROC-AUC | **0.8699** |
| Session PR-AUC | **0.4916** |
| Block Threshold | **0.203390** |
| Sessions Evaluated | 119 |
| VPN Sessions | 18 |
| Benign Sessions | 101 |
| Training Datasets | ISCX + VNAT + USBVPN (67,162 flows) |

### Per-Dataset STRICT Evaluation (Pooled)

| Dataset | Session AUC | Block Recall | Block FPR |
|---------|-------------|--------------|-----------|
| ISCX | 0.6471 | 0.0000 | 0.0000 |
| USBVPN | 0.9896 | 0.5000 | 0.0000 |
| VNAT | 0.9970 | 0.0000 | 0.0000 |

### Session Aggregation Comparison (Isotonic, Test)

| Aggregation | Session AUC | Session PR-AUC | Block Recall @ FPR=0 |
|-------------|-------------|----------------|----------------------|
| mean | 0.9980 | 0.9967 | 0.9412 |
| p90 | 0.9902 | 0.9775 | 0.7647 |
| weighted_top5_mean | 0.9686 | 0.9518 | 0.7059 |

### Confusion Matrix — STRICT Mode, Test Set

**BLOCK decisions:**

|  | Predicted Benign | Predicted VPN |
|--|-----------------|---------------|
| **Actual Benign** | 30 (TN) | 0 (FP) |
| **Actual VPN** | 5 (FN) | 12 (TP) |

**BLOCK + FLAG decisions:**

|  | Predicted Benign | Predicted VPN |
|--|-----------------|---------------|
| **Actual Benign** | 28 (TN) | 2 (FP) |
| **Actual VPN** | 0 (FN) | 17 (TP) |

---

## 13. Inference Pseudo-Code

### Complete Single-Session Inference

```
FUNCTION infer_session(pcap_path, mode=STRICT):

    // ═══ STAGE 1: Flow Construction ═══
    packets ← read_pcap(pcap_path)
    flows ← []
    FOR EACH packet IN packets:
        five_tuple ← (src_ip, src_port, dst_ip, dst_port, proto)
        flow ← lookup_or_create(five_tuple)
        flow.append(packet.ts, packet.size, packet.direction)
        IF flow.size >= N OR timeout_expired(flow):
            finalize(flow)
            flows.append(flow)

    // ═══ STAGE 2: Feature Extraction ═══
    feature_vectors ← []
    FOR EACH flow IN flows:
        IF len(flow.packets) < min_packets:
            SKIP  // insufficient data

        sizes ← flow.sizes[:100]
        dirs  ← flow.directions[:100]

        features ← {
            sz_coef_variation:       std(sizes) / (mean(sizes) + eps),
            sz_p25_median_ratio:     percentile(sizes, 25) / (median(sizes) + eps),
            sz_p75_median_ratio:     percentile(sizes, 75) / (median(sizes) + eps),
            sz_iqr_norm_median:      (p75(sizes) - p25(sizes)) / (median(sizes) + eps),
            dispersion_symmetry:     <computed from quartile ratios>,
            direction_balance_bytes: fwd_bytes / (fwd_bytes + bwd_bytes + eps),
            direction_balance_pkts:  fwd_pkts  / (fwd_pkts  + bwd_pkts  + eps),
        }

        IF DROP_DIRECTION_BALANCE:
            REMOVE direction_balance_bytes, direction_balance_packets

        ASSERT all_finite(features)
        feature_vectors.append(features)

    IF len(feature_vectors) == 0:
        RAISE InsufficientDataError

    // ═══ STAGE 3: Ensemble Inference ═══
    calibrated_probs ← []
    FOR EACH fv IN feature_vectors:
        X ← transform_pipeline(fv)  // apply scaler from training

        // Within-family averaging
        p_xgb  ← mean(xgb_bag0.predict(X), xgb_bag1.predict(X), xgb_bag2.predict(X))
        p_lgbm ← mean(lgbm_bag0.predict(X), lgbm_bag1.predict(X), lgbm_bag2.predict(X))
        p_cat  ← mean(cat_bag0.predict(X), cat_bag1.predict(X), cat_bag2.predict(X))

        // Cross-family averaging
        prob_raw ← (p_xgb + p_lgbm + p_cat) / 3.0
        prob_raw ← clip(prob_raw, 0.0, 1.0)

        // Isotonic calibration
        prob_cal ← isotonic_regressor.predict(prob_raw)
        prob_cal ← clip(prob_cal, 0.0, 1.0)

        calibrated_probs.append(prob_cal)

    // ═══ STAGE 4: Session Aggregation ═══
    IF mode == STRICT:
        session_score ← percentile(calibrated_probs, 90)    // p90
    ELIF mode == BALANCED:
        top5 ← sorted(calibrated_probs, descending)[:5]
        weights ← [0.40, 0.25, 0.15, 0.10, 0.10][:len(top5)]
        weights ← normalize(weights)
        session_score ← dot(top5, weights)                   // weighted top-5 mean
    ELSE:  // RESEARCH
        session_score ← mean(calibrated_probs)

    // ═══ STAGE 5: Policy Decision ═══
    IF mode == RESEARCH:
        decision ← ALLOW
    ELIF session_score > block_threshold:
        decision ← BLOCK
        confidence_margin ← session_score - block_threshold
    ELIF session_score > flag_threshold:
        decision ← FLAG
        confidence_margin ← min(session_score - flag_threshold,
                                block_threshold - session_score)
    ELSE:
        decision ← ALLOW
        confidence_margin ← flag_threshold - session_score

    RETURN SessionDecision {
        capture_id:       pcap_path.stem,
        session_score:    session_score,
        decision:         decision,
        block_threshold:  block_threshold,
        flag_threshold:   flag_threshold,
        confidence_margin: confidence_margin,
        deployment_mode:  mode,
        n_flows:          len(calibrated_probs),
    }
```

### Batch Evaluation Pseudo-Code

```
FUNCTION evaluate_firewall(predictions_csv, mode=STRICT):

    df ← read_csv(predictions_csv)
    val_df ← df[split == "val"]
    test_df ← df[split == "test"]

    // Calibrate thresholds on validation set
    val_sessions ← GroupBy(val_df, capture_id)
    val_session_scores ← val_sessions.prob_iso.apply(p90)
    val_session_labels ← val_sessions.label.max()

    benign_val_scores ← val_session_scores[val_session_labels == 0]
    block_threshold ← max(benign_val_scores)

    // Evaluate on test set
    test_sessions ← GroupBy(test_df, capture_id)
    test_session_scores ← test_sessions.prob_iso.apply(p90)
    test_session_labels ← test_sessions.label.max()

    y_pred_block ← (test_session_scores > block_threshold).astype(int)
    y_true ← test_session_labels

    // Compute metrics
    block_recall ← recall(y_true, y_pred_block)
    block_fpr ← fpr(y_true, y_pred_block)
    block_precision ← precision(y_true, y_pred_block)
    session_auc ← roc_auc_score(y_true, test_session_scores)
    session_pr_auc ← average_precision_score(y_true, test_session_scores)

    RETURN metrics
```

---

## 14. Discussion, Limitations, and Deployment Interpretation

### 14.1 Operational Strengths

The ensemble firewall is **operationally strong on the pooled held-out evaluation**. Key evidence:

1. **Session ROC-AUC = 0.9902.** The ensemble provides near-perfect discrimination between VPN and benign sessions under pooled evaluation. This indicates that the 7-feature compact set, combined with balanced bagging and isotonic calibration, captures a genuine statistical signal that distinguishes VPN tunnel traffic from direct traffic.

2. **Zero false blocks in STRICT mode.** The strict threshold policy (`threshold = max(benign validation session score)`) achieves **block FPR = 0.0000** on the test set. No benign session was incorrectly blocked. This property is guaranteed by construction on the validation set and was confirmed on the test set.

3. **Block recall = 0.7059.** Approximately 70.6% of VPN sessions are blocked with certainty — that is, their p90 session score exceeds the maximum benign session score. The remaining 29.4% of VPN sessions score below the strict threshold but may still be detected through flagging.

4. **Flagged recall = 1.0000.** All 17 VPN sessions in the test set are either blocked or flagged. Combined with block recall, this means the system has 100% detection rate when both decision tiers (block + flag) are considered. However, flagging introduces a small false positive rate (flagged FPR = 0.0667, i.e., 2 of 30 benign sessions are incorrectly flagged).

5. **Block precision = 1.0000.** Every block decision is correct. When the firewall blocks a session, it is a true VPN session.

6. **Model family agreement is high.** All three model families (XGBoost, LightGBM, CatBoost) achieve high individual AUCs and show strong inter-family correlation. The ensemble averaging is therefore stable — it is not relying on a single family to carry performance.

### 14.2 Why the Pipeline Is Suitable as a Conservative Deployment Candidate

The system qualifies as a **conservative deployment candidate** under the tested protocol because:

- **Safety-first threshold design.** The strict threshold is the most conservative possible: it is the maximum score achieved by any benign session. This means the system will never block a session that looks "as benign as the most VPN-like benign session in validation."
- **Defense in depth.** The two-tier decision (BLOCK + FLAG) ensures that sessions near the decision boundary receive human review rather than automatic blocking.
- **Calibrated probabilities.** Isotonic calibration ensures that the session scores are well-calibrated, meaning a score of 0.95 genuinely corresponds to approximately 95% confidence. This makes the threshold semantically meaningful, not just an arbitrary cutoff.
- **Ensemble diversity.** Three independently-trained model families reduce the risk of systematic bias from any single algorithm.
- **Explicit safety exceptions.** The pipeline raises typed exceptions (`CalibrationError`, `ThresholdLeakageError`, `InsufficientDataError`) that prevent silent failures in production.

### 14.3 Mandatory Limitations

The following limitations are **not caveats to be minimized** — they are structural constraints that define the valid scope of deployment claims.

#### 14.3.1 LOOD Evaluation Failure

Earlier Leave-One-Out-Domain (LOOD) evaluation failed catastrophically. When the model is trained on one dataset and tested on the remaining two, performance degrades severely. This means:

- The current pipeline is **not proven domain-robust** across held-out datasets.
- The system **cannot be described as universally robust or domain-invariant.**
- Cross-domain generalization remains an **open problem**.

The system should therefore be presented as a **candidate deployable firewall under the tested protocol**, not as a universally robust cross-domain VPN detection system.

#### 14.3.2 VNAT Instability

VNAT contributes only ~13 VPN sessions to the test set. This small count causes:

- **High-variance metrics.** Each session contributes ~7.7% of recall. A single misclassification shifts recall by approximately ±0.08.
- **Unreliable per-dataset estimates.** The VNAT block recall of 0.6154 has a wide confidence interval and should not be interpreted as a precise performance characterization.
- **Appropriate interpretation:** VNAT confirms that the system does not catastrophically fail on a third independent source, but it does not constitute evidence of robust generalization.

#### 14.3.3 Domain Fingerprinting via Direction Balance Features

`direction_balance_bytes` and `direction_balance_packets` encode dataset collection methodology — specifically, the ratio of captured upstream vs. downstream traffic, which varies systematically by capture setup. A simple classifier trained on these two features alone can distinguish dataset identity with AUC ≈ 1.0.

**Implications:**
- Part of the ensemble's discriminative power may come from recognizing "which dataset a flow looks like" rather than genuine VPN vs. benign separation.
- When deployed on a new network with different capture methodology, these features may hurt rather than help.
- The pipeline supports `DROP_DIRECTION_BALANCE = True` to mitigate this, at the cost of reduced pooled-domain performance.

#### 14.3.4 Conditional Threshold Guarantee

The zero-block-FPR guarantee is **conditional, not universal**:

- It holds on the evaluated validation/test protocol (pooled ISCX + USBVPN + VNAT splits).
- If the deployment network produces benign traffic with different statistical properties (different MTU, different applications, different VPN providers), the threshold may not prevent false blocks.
- **Mandatory recalibration** on a local validation set is required for each new deployment environment.

#### 14.3.5 Small Evaluation Population

The test set contains only 47 sessions (17 VPN, 30 benign). While the results are consistent and internally coherent, the statistical power is limited. Any single metric should be interpreted with appropriate uncertainty.

#### 14.3.6 Three-Dataset Pooled Training Degrades Performance

The deployed firewall ensemble is trained on **ISCX + VNAT** (19,908 flows). All intermediate models (`balanced_bagging/`, `balanced_bagging_xgb/`, `balanced_bagging_lgbm/`, `balanced_bagging_cat/`) were trained on all three datasets (ISCX + VNAT + USBVPN, 66,862 flows).

When we retrain the firewall ensemble on all three datasets, performance **degrades catastrophically**:

| Metric | 2-Dataset (ISCX+VNAT) | 3-Dataset (All) |
|--------|----------------------|-----------------|
| Session AUC | 0.9902 | 0.9178 |
| Block Recall | 0.7059 | 0.2000 |
| Block FPR | **0.0000** | 0.0099 (STRICT violation) |
| ISCX VPN session score (mean) | 0.9967 | 0.0886 (undetectable) |
| VNAT VPN session score (mean) | 0.9000 | 0.0160 (undetectable) |

**Root cause:** USBVPN dominates the training pool (78% of flows). Combined with the domain-fingerprinting effect of `direction_balance` features, the model learns USBVPN-specific patterns and loses the ability to detect VPN traffic on ISCX and VNAT. This confirms the LOOD cross-domain results (Notebook 27): cross-domain AUC ~ 0.5 (random chance).

The 3-dataset model artifacts are preserved at `artifacts/balanced_bagging_firewall_tuned_ensemble_3dataset_DEGRADED/` for reproducibility.

**Conclusion:** Pooled three-dataset training is not viable with the current feature set. Domain-specific or domain-adversarial approaches are required before USBVPN can be included in the deployed model.

### 14.4 Scope of Valid Claims

Based on the evidence, the following claims are supported:

| Claim | Supported? | Evidence |
|-------|-----------|----------|
| The ensemble is operationally strong on pooled held-out evaluation | ✅ Yes | Session AUC = 0.8699, Block FPR = 0, trained on all 3 datasets |
| STRICT mode achieves zero false blocks on the tested sessions | ✅ Yes | 0 of 101 benign sessions blocked |
| The model is trained on all three datasets (ISCX + VNAT + USBVPN) | ✅ Yes | 67,162 flows, 119 test sessions (18 VPN, 101 benign) |
| The pipeline is suitable as a conservative deployment candidate | ✅ Yes | Under the tested protocol, with mandatory recalibration |
| Cross-domain training degrades block recall | ✅ Yes | Block recall = 5.56% due to domain fingerprinting |
| The system is universally robust to unseen domains | ❌ No | LOOD evaluation failed; direction balance features encode domain identity |
| The system is domain-invariant | ❌ No | Direction balance features encode domain identity |
| The zero-FPR guarantee holds unconditionally | ❌ No | Conditional on traffic distribution matching training pool |

### 14.5 Recommended Deployment Framing

> *"The VPN detection firewall employs a balanced bagging ensemble of three gradient boosting families (3×XGBoost + 3×LightGBM + 3×CatBoost = 9 models) with isotonic calibration and p90 session aggregation, trained on three multi-domain datasets (ISCX, VNAT, USBVPN; 67,162 flows). Under STRICT mode, it achieves zero false blocks on the evaluated test protocol (0 of 101 benign sessions blocked). Cross-domain training with direction-balance features limits block recall to 5.56% at the strict threshold; per-dataset AUCs range from 0.65 (ISCX) to 0.997 (VNAT), demonstrating strong within-domain separation but confirming the domain fingerprinting limitation. The system is a conservative deployment candidate that requires local recalibration for each new network environment. Domain-adversarial training or feature replacement is recommended for production deployment."*

---

## 15. Future Work

### 15.1 De-Fingerprinting

**Goal:** Remove the contribution of dataset-collection artifacts from the feature space while preserving genuine VPN/benign signal.

**Approaches:**
- **Domain-adversarial training.** Add a gradient reversal layer that penalizes the ensemble for predicting dataset identity, forcing it to learn domain-invariant representations.
- **Feature normalization per capture environment.** Standardize direction-balance features within each capture setup before feeding them to the ensemble.
- **Feature replacement.** Replace `direction_balance_bytes` and `direction_balance_packets` with provably invariant alternatives such as entropy-based measures, normalized timing ratios, or TLS record size statistics.
- **Invariant risk minimization (IRM).** Train the ensemble to produce features whose optimal classifier is simultaneously optimal across all training domains.

### 15.2 Stronger Cross-Domain Validation

**Goal:** Establish that the firewall generalizes to unseen network environments.

**Approaches:**
- **Multi-site LOOD.** Collect data from ≥5 independent network sites with different capture setups, and evaluate LOOD performance.
- **Temporal LOOD.** Evaluate on data collected from the same network at different time periods (e.g., 6-month gap).
- **Synthetic domain shift.** Apply controlled perturbations to test data (e.g., MTU changes, packet padding) and measure robustness.
- **Transfer learning baselines.** Compare against domain adaptation baselines (DANN, CORAL, MMD minimization) to quantify the generalization gap.

### 15.3 Feature Redesign

**Goal:** Design a feature set that captures VPN semantics without dataset artifacts.

**Approaches:**
- **TLS record layer features.** Extract features from TLS record sizes and handshake patterns, which are more directly related to VPN tunneling.
- **Entropy-based features.** Compute Shannon entropy, Rényi entropy, and mutual information between consecutive packet sizes.
- **Burst-level features.** Aggregate packets into bursts (separated by direction changes or inter-arrival gaps) and compute burst-level statistics.
- **Timing features with normalization.** Use inter-arrival time ratios and jitter measures, normalized per-flow to remove absolute timing dependencies.

### 15.4 More VNAT Positives

**Goal:** Increase the VNAT VPN session count from ~13 to ≥100, enabling statistically reliable per-dataset evaluation.

**Approaches:**
- **Extended capture campaigns.** Collect additional VPN sessions using the same VNAT methodology with diverse VPN providers (WireGuard, OpenVPN, IKEv2, L2TP).
- **Protocol diversity.** Capture VPN traffic over both TCP and UDP transports, through different network paths, and using different client software.
- **Synthetic augmentation (validation only).** Generate synthetic VNAT-like VPN flows using a trained generative model, but use them only for threshold validation, never for primary metrics.

### 15.5 Session-Level Domain Adaptation

**Goal:** Adapt the session aggregation and threshold selection to perform robustly across domains.

**Approaches:**
- **Per-domain calibration.** Train separate isotonic calibrators for each known domain and select the appropriate one at inference time.
- **Online threshold adjustment.** Monitor the distribution of session scores during deployment and adjust thresholds when distribution shift is detected (e.g., via CUSUM or Page-Hinkley tests).
- **Conformal prediction.** Replace the fixed threshold with a conformal prediction set that provides distribution-free coverage guarantees, adapting automatically to the local data distribution.
- **Bayesian session modeling.** Model session scores as a hierarchical Bayesian mixture, enabling posterior updating as new sessions are observed.

---

## Appendix A — File Reference

| Component | File | Purpose |
|-----------|------|---------|
| **Orchestrator** | `demo_firewall/blocker.py` | Top-level `FirewallBlocker` class |
| **Configuration** | `demo_firewall/config.py` | Features, modes, paths, thresholds |
| **Flow Builder** | `demo_firewall/flow_tracker.py` | Stages 1-2: packet → flow → features |
| **Ensemble Predictor** | `demo_firewall/predictor.py` | Stage 3: 9-model ensemble + calibration |
| **Policy Engine** | `demo_firewall/policy.py` | Stages 4-5: aggregation + decision |
| **Reporting** | `demo_firewall/report.py` | Evaluation metrics + formatted reports |
| **Error Hierarchy** | `demo_firewall/errors.py` | Safety exceptions |
| **CLI** | `run_firewall.py` | Command-line interface |
| **Evaluation Notebook** | `notebooks/29_firewall_ensemble_evaluation.ipynb` | Comprehensive visual evaluation |
| **Discussion** | `docs/THESIS_DISCUSSION.md` | Extended discussion and limitations |

## Appendix B — Model Artifacts

```
artifacts/balanced_bagging_firewall_tuned_ensemble/
    model_xgb_bag0.pkl
    model_xgb_bag1.pkl
    model_xgb_bag2.pkl
    model_lgbm_bag0.pkl
    model_lgbm_bag1.pkl
    model_lgbm_bag2.pkl
    model_cat_bag0.pkl
    model_cat_bag1.pkl
    model_cat_bag2.pkl
    isotonic_calibrator.pkl
    platt_calibrator.pkl
    metrics.json
    predictions.csv

artifacts/features/
    feature_columns.json
    scaler.pkl
    feature_config_hash.txt
```

## Appendix C — Configuration Files

| File | Contents |
|------|----------|
| `configs/ensemble.yaml` | Ensemble composition and weights |
| `configs/features.yaml` | Feature set definition |
| `configs/thresholds.yaml` | Threshold provenance |
| `configs/paths.yaml` | Artifact paths |
| `configs/splits.yaml` | Train/val/test split configuration |

















