# Discussion and Limitations

## System Summary

This work presents a zero-block-FPR VPN detection firewall operating on packet-flow
statistical features. The system employs a balanced bagging ensemble of three gradient
boosting families (XGBoost, LightGBM, CatBoost), isotonic probability calibration,
and session-level aggregation to classify network traffic as VPN or benign.
The deployment operating point achieves **block recall ≈ 0.7059** with
**block FPR = 0** and **flagged FPR = 0** under strict-mode evaluation.

---

## Why Pooled Evaluation Succeeds

The pooled-domain evaluation, where training and test data are drawn from the same
combined pool of ISCX, USBVPN, and VNAT datasets, succeeds because:

1. **Statistical consistency within the pooled domain.** When all three datasets
   are mixed and split capture-aware, the feature distributions in training and test
   overlap sufficiently for the ensemble to generalize within this pooled domain.

2. **Ensemble diversity absorbs noise.** The balanced bagging strategy, which creates
   three bags per family with varying majority-to-minority ratios, produces classifiers
   that vote independently. Their averaged probability smooths per-bag variance and
   produces a stable session-level score.

3. **Session aggregation amplifies signal.** By aggregating flow-level predictions
   to session level using p90 (or weighted top-5 mean), the system leverages the
   fact that VPN sessions consistently produce multiple high-probability flows,
   while benign sessions rarely produce even one. This separation enables a clear
   threshold with zero false positives on benign sessions.

4. **Compact feature set reduces overfitting.** The seven-feature compact set
   (`sz_coef_variation`, `sz_p25_median_ratio`, `sz_p75_median_ratio`,
   `sz_iqr_norm_median`, `dispersion_symmetry`, `direction_balance_bytes`,
   `direction_balance_packets`) was selected for domain-invariant statistical
   properties, primarily packet-size distribution shape and directional balance.
   These features capture fundamental differences between encrypted tunnel traffic
   and direct traffic regardless of application-layer content.

---

## Why LOOD Fails

Leave-One-Out-Domain (LOOD) evaluation, where the model trains on one dataset and
tests on the remaining two, fails because:

1. **Domain shift.** Each dataset (ISCX, USBVPN, VNAT) was collected under
   different network conditions, VPN providers, protocols, and time periods. The
   statistical profile of packet sizes and inter-arrival times shifts between
   collection environments. What the model learns as "VPN-like" from ISCX may not
   transfer to the USBVPN or VNAT distributions.

2. **Dataset identity is separable.** A simple classifier trained on direction
   balance features alone achieves AUC ≈ 1.0 at distinguishing which dataset a
   flow comes from. This means the feature space encodes dataset provenance, not
   just VPN/benign semantics. When tested on an unseen domain, the model
   encounters a distribution it has never seen, causing catastrophic performance
   degradation.

3. **Class imbalance interacts with domain shift.** VNAT contributes only 12
   positive sessions (VPN). When VNAT is the sole training domain, the model lacks
   sufficient positive diversity. When VNAT is in the test set, its small positive
   count makes metrics unstable — a single misclassified session shifts recall
   by ~8%.

4. **This is a fundamental limitation, not a fixable bug.** LOOD failure reflects
   the reality that the deployed VPN signatures are dataset-specific to some degree.
   The system is designed for deployment within a pooled domain, not for
   zero-shot generalization to unseen network environments.

---

## Why VNAT is Unstable

VNAT contains only 12 VPN sessions in the test split. This small count causes:

1. **High-variance metrics.** Each session contributes ~8.3% of recall. Bootstrap
   analysis shows recall standard deviation of ±0.15 when resampling VNAT alone.

2. **Non-representative positive distribution.** 12 sessions cannot cover the
   diversity of VPN traffic patterns, making VNAT unsuitable as a primary robustness
   benchmark.

3. **Appropriate role: auxiliary validation.** VNAT serves as a supplementary
   check that the system does not catastrophically fail on a third independent
   source, but it cannot serve as evidence of generalization.

---

## Why Domain Fingerprinting Persists

The `direction_balance_bytes` feature enables dataset-identity classification with
near-perfect AUC. This occurs because:

1. **Collection methodology differences.** Different packet capture tools and network
   taps systematically alter the ratio of captured upstream vs. downstream bytes.
   Some captures are taken at the client, others at a gateway, producing consistent
   directional biases per dataset.

2. **Not a VPN signal per se.** The directional balance differences are artifacts of
   collection setup, not fundamental VPN vs. benign differences. However, within the
   pooled domain where this artifact is consistent, the feature still correlates with
   VPN status and improves discrimination.

3. **Mitigation supported.** The pipeline supports a `drop_direction_features=True`
   flag that removes both `direction_balance_bytes` and `direction_balance_packets`.
   The reduced-feature model has lower pooled AUC but is more domain-robust. For
   deployment in a new network environment, the reduced-feature mode should be
   preferred until a local calibration set validates the full-feature model.

---

## Why Ensemble Aggregation Improves Stability

The three-stage aggregation (within-bag → within-family → cross-family → session)
improves stability through:

1. **Balanced bagging reduces minority-class variance.** Each bag contains all
   minority (VPN) samples but a different random subset of majority (benign) samples.
   This ensures every bag sees the full positive distribution while reducing the
   noise from the benign class.

2. **Cross-family diversity.** XGBoost, LightGBM, and CatBoost have different
   algorithmic biases (split strategies, regularization, gradient handling). Their
   disagreement regions are uncorrelated, so averaging cancels individual errors.

3. **Session-level aggregation is robust to flow-level noise.** A single flow may be
   misclassified with high confidence, but it is unlikely that multiple flows in a
   benign session are simultaneously misclassified. The p90 aggregation rule requires
   that at least 10% of flows in a session are high-confidence VPN before triggering
   a block, making the system robust to isolated false positives.

4. **Isotonic calibration preserves ranking while fixing probability scale.**
   Isotonic regression is a non-parametric monotonic transform that maps raw
   ensemble averages to calibrated probabilities. Unlike Platt scaling (logistic),
   isotonic calibration adapts to the local density of the probability space,
   producing better-calibrated outputs especially in the tail regions critical for
   threshold selection.

---

## Why Strict-Mode Deployment Remains Valid Despite LOOD Limits

The strict-mode deployment guarantee (block FPR = 0) is valid because:

1. **The threshold is computed on the validation split of the same pooled domain.**
   The deployment assumption is that incoming traffic is drawn from a distribution
   similar to the training pool. Within this assumption, the max-benign-score
   threshold guarantees zero false positives by construction.

2. **The guarantee is conditional, not universal.** If the deployment network has
   fundamentally different traffic patterns (new applications, different MTU settings,
   different VPN providers), the threshold may not hold. This is explicitly
   acknowledged as a deployment constraint: the system must be recalibrated on a
   local validation set for each new environment.

3. **Conservative design prioritizes safety.** By choosing the maximum benign session
   score as the block threshold, the system accepts reduced recall (≈70% vs. ≈83%
   at default threshold) in exchange for zero false blocking. This trade-off is
   appropriate for a firewall where false blocks disrupt legitimate traffic.

4. **The flagging layer adds defense in depth.** Sessions above the flag threshold
   but below the block threshold are marked for human review, capturing additional
   VPN sessions without risking false blocks.

---

## Limitations Summary

| Limitation | Impact | Mitigation |
|---|---|---|
| LOOD failure | Cannot generalize to unseen network domains | Pooled-domain deployment only; recalibrate for new environments |
| VNAT instability | Metrics unreliable on 12-session subset | Use as auxiliary check only; do not report as primary evidence |
| Domain fingerprinting | `direction_balance_bytes` encodes collection setup | Optional reduced-feature mode; flag in deployment report |
| Calibration sensitivity | Isotonic calibration requires both classes | `CalibrationError` safety check; falls back gracefully |
| Threshold conditionality | Zero-FPR guarantee is pooled-domain-specific | Mandatory recalibration on local validation for new deployments |
| Single-model tuning invalidity | Label collapse in XGBoost calibration split | All deployment uses ensemble only; single-model results excluded |

---

## Future Work

1. **Federated calibration.** Deploy local calibration models at each network site,
   retaining a shared ensemble backbone but site-specific thresholds.

2. **Temporal robustness.** Evaluate drift over time within a single network to
   determine recalibration frequency.

3. **Expanded dataset collection.** Add ≥5 additional VPN providers and ≥3 additional
   capture methodologies to the training pool to reduce domain separability.

4. **Online learning.** Implement incremental calibration updates from confirmed
   blocks/allows to maintain threshold validity as traffic patterns evolve.

5. **Adversarial robustness.** Test against traffic shaping and padding attacks that
   may defeat packet-size-based features.

6. **Feature engineering for domain invariance.** Investigate features that are
   provably invariant to capture methodology, such as normalized timing ratios or
   entropy-based measures.

7. **Multi-threshold deployment.** Implement confidence-binned actions (e.g., hard
   block above 0.95, soft block above threshold, flag below) for more granular
   operational control.

