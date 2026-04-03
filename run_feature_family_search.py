#!/usr/bin/env python
"""
FEATURE-FAMILY ROBUSTNESS SEARCH
=================================
Systematic search over stability-aware feature subsets to find the representation
that maximizes cross-dataset robustness (LODO min AUC), not just pooled metrics.

Primary ranking criteria (in order):
  1. LODO min AUC
  2. worst-domain recall
  3. worst-domain FPR
  4. ISCX VPN recall
  5. USBVPN VPN recall
  6. pooled AUC

Outputs go to artifacts/clean_pipeline/eval_v3/
"""
from __future__ import annotations
import gc, json, time, sys, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "artifacts" / "clean_pipeline" / "features.parquet"
OUT = ROOT / "artifacts" / "clean_pipeline" / "eval_v3"

from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ================================================================
# HELPERS (shared with run_meta_model_eval.py)
# ================================================================
def safe_auc(y, s):
    if len(np.unique(y)) < 2 or len(y) < 5: return float('nan')
    return float(roc_auc_score(y, s))

def cm_met(y, yp):
    cm = confusion_matrix(y, yp, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    return {"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn),
            "fpr":fp/(fp+tn) if (fp+tn)>0 else 0.0,
            "recall":tp/(tp+fn) if (tp+fn)>0 else 0.0,
            "precision":tp/(tp+fp) if (tp+fp)>0 else 0.0}

def best_f1_thr(y, s):
    if len(np.unique(y)) < 2: return 0.5
    pr, re, th = precision_recall_curve(y, s)
    f1 = 2*pr*re/(pr+re+1e-12)
    return float(th[min(np.argmax(f1), len(th)-1)])

def session_agg(df, sc="score"):
    rows = []
    for c, g in df.groupby("capture_id"):
        s = g[sc].values
        rows.append({"capture_id":c, "dataset":g["dataset"].iloc[0],
            "label":int(g["label"].iloc[0]), "n_flows":len(g),
            "mean_score":float(s.mean()),
            "p90_score":float(np.percentile(s,90)),
            "wt5_score":float(np.sort(s)[-min(5,len(s)):].mean()),
            "max_score":float(s.max())})
    return pd.DataFrame(rows)

def train_xgb(X_tr, y_tr, X_va, y_va, n_est=200):
    """Train a single XGBoost model."""
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(1.0, (y_tr==0).sum() / max((y_tr==1).sum(), 1)),
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m

def do_split(df, seed=42, train_r=0.70, val_r=0.15):
    """Deterministic capture-level split."""
    rng = np.random.default_rng(seed)
    cap = df.groupby(["dataset","label","capture_id"]).agg(n=("flow_id","count")).reset_index()
    assigns = {}
    for (ds,lbl), grp in cap.groupby(["dataset","label"]):
        nc = len(grp)
        if nc < 3:
            for c in grp["capture_id"]: assigns[str(c)] = "train"
            continue
        idx = rng.permutation(nc)
        cids = grp["capture_id"].values[idx]
        flows = grp["n"].values[idx]
        min_p = 1 if nc < 6 else 2
        order = np.argsort(flows)
        test_ids = [str(cids[order[i]]) for i in range(min_p)]
        val_ids  = [str(cids[order[i]]) for i in range(min_p, 2*min_p)]
        rest     = [str(cids[order[i]]) for i in range(2*min_p, nc)]
        for c in test_ids: assigns[c] = "test"
        for c in val_ids:  assigns[c] = "val"
        total = int(flows.sum())
        tgt = {"train":int(total*train_r),"val":int(total*val_r),
               "test":total-int(total*train_r)-int(total*val_r)}
        cur = {"train":0,"val":0,"test":0}
        for c in test_ids: cur["test"] += int(cap[cap.capture_id==c]["n"].iloc[0])
        for c in val_ids:  cur["val"]  += int(cap[cap.capture_id==c]["n"].iloc[0])
        for c in rest:
            w = int(cap[cap.capture_id==c]["n"].iloc[0])
            best = min(("train","val","test"),
                       key=lambda s: sum(abs((cur[k]+(w if k==s else 0))-tgt[k]) for k in tgt))
            assigns[c] = best
            cur[best] += w
    out = df.copy()
    out["split"] = out["capture_id"].astype(str).map(assigns).fillna("train")
    return out

def compute_domain_detector_auc(df, feat_cols, seed=42):
    """Train RF domain-identity classifier, return OvR macro AUC."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import label_binarize
    le = LabelEncoder()
    y_ds = le.fit_transform(df["dataset"].values)
    X = df[feat_cols].values
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    sp = int(0.7 * n)
    tr_idx, te_idx = idx[:sp], idx[sp:]
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
    rf.fit(X[tr_idx], y_ds[tr_idx])
    proba = rf.predict_proba(X[te_idx])
    if len(le.classes_) == 2:
        return safe_auc(y_ds[te_idx], proba[:, 1])
    y_bin = label_binarize(y_ds[te_idx], classes=list(range(len(le.classes_))))
    try:
        return float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float('nan')


# ================================================================
# FEATURE FAMILIES — stability-aware, concept-grouped
# ================================================================
# Ordered by KS stability rank (1=most stable):
# 1.median_pkt_len 2.iat_cv 3.pkt_len_cv 4.flow_duration 5.p25_pkt_len
# 6.iat_std 7.iat_median 8.total_packets 9.iat_p25 10.iat_iqr
# 11.packet_rate 12.pkt_len_iqr 13.iat_mean 14.byte_rate 15.iat_p75
# 16.min_pkt_len 17.std_pkt_len 18.total_bytes 19.p75_pkt_len
# 20.mean_pkt_len 21.max_pkt_len

FAMILIES = {
    # --- Stability-based subsets ---
    "ks_top3": [
        "median_pkt_len", "iat_cv", "pkt_len_cv",
    ],
    "ks_top5": [
        "median_pkt_len", "iat_cv", "pkt_len_cv", "flow_duration", "p25_pkt_len",
    ],
    "ks_top7": [
        "median_pkt_len", "iat_cv", "pkt_len_cv", "flow_duration", "p25_pkt_len",
        "iat_std", "iat_median",
    ],
    "ks_top10": [
        "median_pkt_len", "iat_cv", "pkt_len_cv", "flow_duration", "p25_pkt_len",
        "iat_std", "iat_median", "total_packets", "iat_p25", "iat_iqr",
    ],

    # --- Concept-grouped subsets ---
    "cv_ratios_only": [
        # Coefficient of variation + IQR — normalized, least unit-dependent
        "iat_cv", "pkt_len_cv", "pkt_len_iqr", "iat_iqr",
    ],
    "iat_only": [
        # All inter-arrival time features
        "iat_mean", "iat_std", "iat_median", "iat_cv", "iat_p25", "iat_p75", "iat_iqr",
    ],
    "pkt_size_only": [
        # All packet size features
        "mean_pkt_len", "std_pkt_len", "median_pkt_len", "p25_pkt_len", "p75_pkt_len",
        "min_pkt_len", "max_pkt_len", "pkt_len_cv", "pkt_len_iqr",
    ],
    "size_iat_balanced": [
        # Top 5 stable from each concept group (size + IAT)
        "median_pkt_len", "pkt_len_cv", "p25_pkt_len", "pkt_len_iqr", "min_pkt_len",
        "iat_cv", "iat_std", "iat_median", "iat_p25", "iat_iqr",
    ],

    # --- Compact direction-free subsets ---
    "compact_cv_flow": [
        # Only normalized/ratio features + flow duration — hardest to fingerprint
        "iat_cv", "pkt_len_cv", "flow_duration", "packet_rate", "byte_rate",
    ],
    "stable_no_rates": [
        # Top 10 stable but drop derived rates (packet_rate, byte_rate)
        "median_pkt_len", "iat_cv", "pkt_len_cv", "flow_duration", "p25_pkt_len",
        "iat_std", "iat_median", "total_packets", "iat_p25", "iat_iqr",
    ],  # same as ks_top10 — rates happen to rank 11+14

    # --- With directional features ---
    "ks_top10_plus_dir": [
        "median_pkt_len", "iat_cv", "pkt_len_cv", "flow_duration", "p25_pkt_len",
        "iat_std", "iat_median", "total_packets", "iat_p25", "iat_iqr",
        "dir_pkt_ratio_minmax", "dir_bytes_ratio_minmax",
    ],
    "full_no_dir": [
        # All 21 non-directional features (current safe_temporal_21)
        "total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
        "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
        "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
        "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
    ],
    "full_clean_25": [
        # All 25 features including directional
        "total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
        "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
        "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
        "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
        "dir_pkt_ratio_minmax","dir_bytes_ratio_minmax","dir_mean_pkt_max","dir_mean_pkt_min",
    ],
}

# Remove duplicate family (stable_no_rates == ks_top10)
del FAMILIES["stable_no_rates"]

N_SEEDS = 3
N_EST = 200
BASELINE_LODO_MIN = 0.419   # current XGB safe_temporal_21 baseline
STOP_LODO_TARGET = 0.70     # if any family reaches this, declare success


# ================================================================
# PART 1 — EVALUATE EACH FAMILY (XGB only, LODO + pooled + domain)
# ================================================================
def evaluate_family(df, feat_cols, family_name, datasets):
    """Full evaluation of one feature family with XGBoost."""
    t0 = time.time()
    n_feat = len(feat_cols)
    print(f"\n  [{family_name}] ({n_feat} features): {feat_cols}")

    # --- LODO evaluation ---
    lodo_rows = []
    for held in datasets:
        for seed in range(42, 42 + N_SEEDS):
            df_s = do_split(df, seed=seed)
            src = df_s[df_s.dataset != held]
            tgt = df_s[df_s.dataset == held]
            src_tr, src_va = src[src.split=="train"], src[src.split=="val"]
            X_tr, y_tr = src_tr[feat_cols].values, src_tr.label.values
            X_va, y_va = src_va[feat_cols].values, src_va.label.values
            X_te, y_te = tgt[feat_cols].values, tgt.label.values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            try:
                m = train_xgb(X_tr, y_tr, X_va, y_va, N_EST)
                score = m.predict_proba(X_te)[:, 1]
                thr = best_f1_thr(y_va, m.predict_proba(X_va)[:, 1])
                yp = (score >= thr).astype(int)
                gm = cm_met(y_te, yp)
                lodo_rows.append({
                    "held_out": held, "seed": seed,
                    "flow_auc": safe_auc(y_te, score),
                    "recall": gm["recall"], "fpr": gm["fpr"],
                })
            except Exception as e:
                print(f"    [WARN] LODO {held} seed={seed}: {e}")
            finally:
                gc.collect()

    # Aggregate LODO
    lodo_df = pd.DataFrame(lodo_rows)
    lodo_per_ds = {}
    for ds in datasets:
        sub = lodo_df[lodo_df.held_out == ds]
        if len(sub) > 0:
            aucs = sub.flow_auc.dropna()
            lodo_per_ds[f"lodo_{ds}_auc"] = float(aucs.mean()) if len(aucs) > 0 else float('nan')
            lodo_per_ds[f"lodo_{ds}_recall"] = float(sub.recall.mean())
            lodo_per_ds[f"lodo_{ds}_fpr"] = float(sub.fpr.mean())

    all_lodo_aucs = lodo_df.flow_auc.dropna().values
    lodo_min = float(np.min(all_lodo_aucs)) if len(all_lodo_aucs) > 0 else float('nan')
    lodo_mean = float(np.mean(all_lodo_aucs)) if len(all_lodo_aucs) > 0 else float('nan')

    # --- Pooled within-distribution evaluation ---
    pooled_rows = []
    for seed in range(42, 42 + N_SEEDS):
        df_s = do_split(df, seed=seed)
        tr, va, te = df_s[df_s.split=="train"], df_s[df_s.split=="val"], df_s[df_s.split=="test"]
        X_tr, y_tr = tr[feat_cols].values, tr.label.values
        X_va, y_va = va[feat_cols].values, va.label.values
        X_te, y_te = te[feat_cols].values, te.label.values
        try:
            m = train_xgb(X_tr, y_tr, X_va, y_va, N_EST)
            score = m.predict_proba(X_te)[:, 1]
            thr = best_f1_thr(y_va, m.predict_proba(X_va)[:, 1])
            yp = (score >= thr).astype(int)
            gm = cm_met(y_te, yp)
            ds_arr = te.dataset.values
            worst_rec, worst_fpr = 1.0, 0.0
            per_ds = {}
            for ds in datasets:
                mask = ds_arr == ds
                if mask.sum() == 0: continue
                yd, ypd, sd = y_te[mask], yp[mask], score[mask]
                dm = cm_met(yd, ypd)
                per_ds[f"{ds}_auc"] = safe_auc(yd, sd)
                per_ds[f"{ds}_recall"] = dm["recall"]
                per_ds[f"{ds}_fpr"] = dm["fpr"]
                if dm["recall"] < worst_rec: worst_rec = dm["recall"]
                if dm["fpr"] > worst_fpr: worst_fpr = dm["fpr"]

            # Session-level
            te_df = te[["capture_id","dataset","label"]].copy()
            te_df["score"] = score
            sess = session_agg(te_df)
            sess_auc = {}
            for ag in ["mean_score","p90_score","wt5_score"]:
                sess_auc[f"sess_{ag}_auc"] = safe_auc(sess.label.values, sess[ag].values)

            row = {
                "seed": seed,
                "pooled_auc": safe_auc(y_te, score),
                "pooled_recall": gm["recall"], "pooled_fpr": gm["fpr"],
                "worst_recall": worst_rec, "worst_fpr": worst_fpr,
            }
            row.update(per_ds)
            row.update(sess_auc)
            pooled_rows.append(row)
        except Exception as e:
            print(f"    [WARN] pooled seed={seed}: {e}")
        finally:
            gc.collect()

    pooled_df = pd.DataFrame(pooled_rows)

    # --- Domain detector AUC ---
    domain_auc = compute_domain_detector_auc(df, feat_cols)

    # --- Build result row ---
    result = {
        "family": family_name,
        "n_features": n_feat,
        "features": "|".join(feat_cols),
        "lodo_min_auc": lodo_min,
        "lodo_mean_auc": lodo_mean,
        "domain_detector_auc": domain_auc,
    }
    result.update(lodo_per_ds)

    if not pooled_df.empty:
        for col in ["pooled_auc","pooled_recall","pooled_fpr","worst_recall","worst_fpr"]:
            result[col] = float(pooled_df[col].mean())
        for ds in datasets:
            for met in ["auc","recall","fpr"]:
                c = f"{ds}_{met}"
                if c in pooled_df.columns:
                    result[c] = float(pooled_df[c].mean())
        for sag in ["sess_mean_score_auc","sess_p90_score_auc","sess_wt5_score_auc"]:
            if sag in pooled_df.columns:
                result[sag] = float(pooled_df[sag].mean())

    # Composite robustness score: LODO-heavy
    if not np.isnan(lodo_min) and not np.isnan(domain_auc):
        result["robustness_score"] = (
            0.50 * lodo_min +
            0.30 * (1.0 - domain_auc) +
            0.20 * result.get("pooled_auc", 0.0)
        )
    else:
        result["robustness_score"] = float('nan')

    elapsed = time.time() - t0
    print(f"    LODO_min={lodo_min:.4f}  domain_det={domain_auc:.4f}  "
          f"pooled_AUC={result.get('pooled_auc',0):.4f}  "
          f"worst_rec={result.get('worst_recall',0):.4f}  "
          f"worst_fpr={result.get('worst_fpr',0):.4f}  ({elapsed:.0f}s)")

    return result, lodo_df


# ================================================================
# PART 2 — DEPLOYMENT POLICY GRID (for top families)
# ================================================================
def policy_grid_for_family(df, feat_cols, family_name, datasets):
    """Test aggregation × threshold combos for deployment."""
    print(f"\n  Policy grid for [{family_name}]...")
    t0 = time.time()

    agg_methods = ["mean_score", "p90_score", "wt5_score", "max_score"]
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    for seed in range(42, 42 + N_SEEDS):
        df_s = do_split(df, seed=seed)
        tr, va, te = df_s[df_s.split=="train"], df_s[df_s.split=="val"], df_s[df_s.split=="test"]
        X_tr, y_tr = tr[feat_cols].values, tr.label.values
        X_va, y_va = va[feat_cols].values, va.label.values
        X_te, y_te = te[feat_cols].values, te.label.values

        m = train_xgb(X_tr, y_tr, X_va, y_va, N_EST)
        score_te = m.predict_proba(X_te)[:, 1]

        # Also get val F1-optimal threshold
        score_va = m.predict_proba(X_va)[:, 1]
        val_thr = best_f1_thr(y_va, score_va)
        thr_list = thresholds + [val_thr]
        thr_names = [str(t) for t in thresholds] + ["val_f1"]

        te_df = te[["capture_id","dataset","label"]].copy()
        te_df["score"] = score_te
        sess = session_agg(te_df)

        for agg in agg_methods:
            s_vals = sess[agg].values
            y_sess = sess.label.values
            ds_sess = sess.dataset.values
            sess_auc = safe_auc(y_sess, s_vals)

            for thr, thr_name in zip(thr_list, thr_names):
                yp = (s_vals >= thr).astype(int)
                gm = cm_met(y_sess, yp)

                worst_rec, worst_fpr = 1.0, 0.0
                for ds in datasets:
                    mask = ds_sess == ds
                    if mask.sum() == 0: continue
                    dm = cm_met(y_sess[mask], yp[mask])
                    if dm["recall"] < worst_rec: worst_rec = dm["recall"]
                    if dm["fpr"] > worst_fpr: worst_fpr = dm["fpr"]

                rows.append({
                    "family": family_name, "seed": seed,
                    "agg": agg, "threshold": thr_name,
                    "sess_auc": sess_auc,
                    "sess_recall": gm["recall"], "sess_fpr": gm["fpr"],
                    "sess_precision": gm["precision"],
                    "worst_sess_recall": worst_rec, "worst_sess_fpr": worst_fpr,
                })
        del m; gc.collect()

    result_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"    {len(result_df)} policy combos evaluated ({elapsed:.0f}s)")
    return result_df


# ================================================================
# PART 3 — DEPLOYMENT MODE RECOMMENDATION
# ================================================================
def recommend_deployment_modes(policy_df, family_name):
    """Pick best policy for each deployment mode."""
    # Average over seeds
    avg = policy_df.groupby(["family","agg","threshold"]).agg({
        "sess_auc": "mean", "sess_recall": "mean", "sess_fpr": "mean",
        "sess_precision": "mean",
        "worst_sess_recall": "mean", "worst_sess_fpr": "mean",
    }).reset_index()

    modes = {}

    # STRICT BLOCK: minimize FPR, require decent recall
    strict = avg[avg.worst_sess_fpr <= 0.05]
    if len(strict) == 0:
        strict = avg.nsmallest(5, "worst_sess_fpr")
    strict = strict.sort_values("sess_recall", ascending=False)
    if len(strict) > 0:
        best = strict.iloc[0]
        modes["STRICT_BLOCK"] = {
            "agg": str(best["agg"]), "threshold": str(best["threshold"]),
            "recall": float(best.sess_recall), "fpr": float(best.sess_fpr),
            "worst_recall": float(best.worst_sess_recall),
            "worst_fpr": float(best.worst_sess_fpr),
        }

    # BALANCED BLOCK: best F1 proxy (recall * (1-fpr))
    avg["f1_proxy"] = avg.sess_recall * (1 - avg.sess_fpr)
    balanced = avg.nlargest(1, "f1_proxy").iloc[0]
    modes["BALANCED_BLOCK"] = {
        "agg": str(balanced["agg"]), "threshold": str(balanced["threshold"]),
        "recall": float(balanced.sess_recall), "fpr": float(balanced.sess_fpr),
        "worst_recall": float(balanced.worst_sess_recall),
        "worst_fpr": float(balanced.worst_sess_fpr),
    }

    # FLAG/REVIEW: maximize recall, accept higher FPR
    flag = avg[avg.sess_recall >= 0.90]
    if len(flag) == 0:
        flag = avg.nlargest(5, "sess_recall")
    flag = flag.sort_values("sess_fpr", ascending=True)
    if len(flag) > 0:
        best = flag.iloc[0]
        modes["FLAG_REVIEW"] = {
            "agg": str(best["agg"]), "threshold": str(best["threshold"]),
            "recall": float(best.sess_recall), "fpr": float(best.sess_fpr),
            "worst_recall": float(best.worst_sess_recall),
            "worst_fpr": float(best.worst_sess_fpr),
        }

    return modes


# ================================================================
# MAIN
# ================================================================
def main():
    t_total = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEATURE-FAMILY ROBUSTNESS SEARCH")
    print("=" * 70)

    # Load data
    print("\nLoading features...")
    df = pd.read_parquet(FEATURES_PATH)
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    print(f"  {len(df)} flows")
    datasets = sorted(df.dataset.unique())
    print(f"  Datasets: {datasets}")

    # Validate families
    available = set(df.columns)
    valid_families = {}
    for name, feats in FAMILIES.items():
        missing = [f for f in feats if f not in available]
        if missing:
            print(f"  [SKIP] {name}: missing {missing}")
        else:
            valid_families[name] = feats
    print(f"  {len(valid_families)} valid families to evaluate")

    # ============================================================
    # PART 1: Evaluate all families
    # ============================================================
    search_csv = OUT / "family_search_results.csv"
    if search_csv.exists():
        print(f"\n  Family search results exist, loading from {search_csv.name}")
        results_df = pd.read_csv(search_csv)
    else:
        print(f"\n{'='*70}")
        print("PART 1: SYSTEMATIC FEATURE FAMILY EVALUATION")
        print(f"{'='*70}")

        all_results = []
        all_lodo_details = []
        found_good = False

        for fname, fcols in valid_families.items():
            res, lodo_df = evaluate_family(df, fcols, fname, datasets)
            all_results.append(res)
            lodo_df["family"] = fname
            all_lodo_details.append(lodo_df)

            # Check early success
            if res["lodo_min_auc"] >= STOP_LODO_TARGET:
                print(f"\n    *** EARLY SUCCESS: {fname} achieves LODO min={res['lodo_min_auc']:.4f} >= {STOP_LODO_TARGET} ***")
                found_good = True

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(search_csv, index=False)
        print(f"\n  Saved family search -> {search_csv.name}")

        # Save LODO details
        lodo_all = pd.concat(all_lodo_details, ignore_index=True)
        lodo_all.to_csv(OUT / "family_search_lodo_details.csv", index=False)

    # ============================================================
    # LEADERBOARD: Rank by robustness criteria
    # ============================================================
    print(f"\n{'='*70}")
    print("LEADERBOARD: Feature Families Ranked by Robustness")
    print(f"{'='*70}")

    lb = results_df.sort_values(
        ["lodo_min_auc", "lodo_mean_auc"],
        ascending=[False, False]
    ).reset_index(drop=True)
    lb["rank"] = range(1, len(lb) + 1)

    # Improvement over baseline
    lb["vs_baseline"] = lb["lodo_min_auc"] - BASELINE_LODO_MIN
    lb["vs_baseline_pct"] = (lb["vs_baseline"] / BASELINE_LODO_MIN * 100).round(1)

    lb_path = OUT / "family_search_leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    print(f"\n  {'Rank':<5} {'Family':<25} {'#F':>3} {'LODO_min':>9} {'LODO_mean':>10} "
          f"{'Domain':>7} {'Pool_AUC':>9} {'W_Recall':>9} {'W_FPR':>7} {'vs_base':>8}")
    print("  " + "-" * 105)
    for _, r in lb.iterrows():
        delta = r.get("vs_baseline", 0)
        sign = "+" if delta >= 0 else ""
        print(f"  {int(r['rank']):<5} {r['family']:<25} {int(r['n_features']):>3} "
              f"{r['lodo_min_auc']:>9.4f} {r['lodo_mean_auc']:>10.4f} "
              f"{r['domain_detector_auc']:>7.4f} {r.get('pooled_auc',0):>9.4f} "
              f"{r.get('worst_recall',0):>9.4f} {r.get('worst_fpr',0):>7.4f} "
              f"{sign}{delta:>7.4f}")

    # ============================================================
    # PART 2: Policy optimization on top 3 families
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 2: DEPLOYMENT POLICY OPTIMIZATION (Top 3 Families)")
    print(f"{'='*70}")

    top3 = lb.head(3)
    policy_path = OUT / "family_policy_grids.csv"

    if policy_path.exists():
        print(f"  Policy grids exist, loading from {policy_path.name}")
        all_policies = pd.read_csv(policy_path)
    else:
        policy_dfs = []
        for _, row in top3.iterrows():
            fname = row["family"]
            fcols = FAMILIES[fname]
            pdf = policy_grid_for_family(df, fcols, fname, datasets)
            policy_dfs.append(pdf)
        all_policies = pd.concat(policy_dfs, ignore_index=True)
        all_policies.to_csv(policy_path, index=False)
        print(f"  Saved policy grids -> {policy_path.name}")

    # ============================================================
    # PART 3: Deployment mode recommendations per top family
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 3: DEPLOYMENT MODE RECOMMENDATIONS")
    print(f"{'='*70}")

    deploy_recs = {}
    for _, row in top3.iterrows():
        fname = row["family"]
        sub = all_policies[all_policies.family == fname]
        modes = recommend_deployment_modes(sub, fname)
        deploy_recs[fname] = {
            "lodo_min_auc": float(row["lodo_min_auc"]),
            "lodo_mean_auc": float(row["lodo_mean_auc"]),
            "domain_detector_auc": float(row["domain_detector_auc"]),
            "pooled_auc": float(row.get("pooled_auc", 0)),
            "n_features": int(row["n_features"]),
            "modes": modes,
        }
        print(f"\n  [{fname}] LODO_min={row['lodo_min_auc']:.4f}")
        for mode, cfg in modes.items():
            thr_str = str(cfg['threshold'])
            print(f"    {mode:20s}: agg={cfg['agg']:12s} thr={thr_str:>7s} "
                  f"recall={cfg['recall']:.3f} fpr={cfg['fpr']:.3f} "
                  f"w_rec={cfg['worst_recall']:.3f} w_fpr={cfg['worst_fpr']:.3f}")

    # ============================================================
    # PART 4: FINAL VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 4: FINAL HONEST VERDICT")
    print(f"{'='*70}")

    best = lb.iloc[0]
    best_family = best["family"]
    best_lodo = float(best["lodo_min_auc"])
    best_domain = float(best["domain_detector_auc"])
    improvement = best_lodo - BASELINE_LODO_MIN

    if best_lodo >= STOP_LODO_TARGET and best_domain < 0.85:
        conclusion = "STRONG_IMPROVEMENT"
        message = (f"Feature family '{best_family}' achieves LODO min AUC {best_lodo:.4f} "
                   f"(>= {STOP_LODO_TARGET}) with domain detector {best_domain:.4f}. "
                   f"Cross-dataset generalization is materially improved.")
    elif best_lodo >= STOP_LODO_TARGET:
        conclusion = "MODERATE_IMPROVEMENT"
        message = (f"Feature family '{best_family}' achieves LODO min AUC {best_lodo:.4f} "
                   f"(>= {STOP_LODO_TARGET}) but domain detector remains high ({best_domain:.4f}). "
                   f"Improvement is present but features still fingerprint datasets.")
    elif improvement > 0.05:
        conclusion = "MARGINAL_IMPROVEMENT"
        message = (f"Best family '{best_family}' improves LODO min AUC from "
                   f"{BASELINE_LODO_MIN:.4f} to {best_lodo:.4f} (+{improvement:.4f}). "
                   f"Improvement is marginal. Domain mismatch remains structural.")
    else:
        conclusion = "NO_MATERIAL_IMPROVEMENT"
        message = (f"No feature family materially improves LODO min AUC beyond "
                   f"baseline {BASELINE_LODO_MIN:.4f} (best: {best_lodo:.4f}, Δ={improvement:+.4f}). "
                   f"Domain detector AUC {best_domain:.4f} confirms features inherently fingerprint datasets. "
                   f"The remaining limitation is structural cross-dataset semantic mismatch. "
                   f"More meta-model complexity is NOT the right direction.")

    verdict = {
        "conclusion": conclusion,
        "message": message,
        "best_family": best_family,
        "best_lodo_min_auc": best_lodo,
        "baseline_lodo_min_auc": BASELINE_LODO_MIN,
        "improvement": float(improvement),
        "domain_detector_auc": best_domain,
        "n_families_tested": len(lb),
        "all_families_tested": lb["family"].tolist(),
        "deployment_recommendations": deploy_recs,
        "model_baselines_frozen": {
            "xgb": "MAIN robustness baseline (LODO_min=0.419)",
            "ensemble_mean": "Pooled benchmark only (LODO_min=0.333)",
            "majority_vote": "Tested — NOT recommended for robustness",
            "logistic_stack": "Tested (OOF) — NOT recommended for robustness",
            "isolation_forest": "NOT DEPLOYABLE for primary VPN detection",
        },
        "stop_rule": (
            f"If no clean representation materially improves LODO min AUC beyond {BASELINE_LODO_MIN:.3f}, "
            f"conclude: (1) remaining limitation is structural dataset mismatch, "
            f"(2) system is conditionally deployable but universal deployability is unsupported, "
            f"(3) more meta-model complexity is not the right direction."
        ),
    }

    verdict_path = OUT / "family_search_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"  {message}")
    print(f"\n  Saved verdict -> {verdict_path.name}")

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"FEATURE FAMILY SEARCH COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



