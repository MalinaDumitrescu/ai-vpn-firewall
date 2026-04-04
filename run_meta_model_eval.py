#!/usr/bin/env python
"""
META-MODEL & ROBUSTNESS EVALUATION — Parts 0-7
Tests whether stacking, voting, and anomaly detection improve cross-dataset
robustness (especially LODO) beyond probability-averaging ensemble baseline.

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

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp

# ================================================================
# HELPERS
# ================================================================
def safe_auc(y, s):
    if len(np.unique(y)) < 2 or len(y) < 5: return float('nan')
    return float(roc_auc_score(y, s))

def safe_ap(y, s):
    if len(np.unique(y)) < 2 or len(y) < 5: return float('nan')
    return float(average_precision_score(y, s))

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

def session_agg(df, sc="ensemble_score"):
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

def train_ens_separate(X_tr, y_tr, X_val, y_val, n_est=150):
    """Train XGB, LGB, CatBoost separately and return as dict."""
    models = {}
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=n_est, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=max(1.0,(y_tr==0).sum()/max((y_tr==1).sum(),1)),
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models["xgb"] = m
    except Exception as e:
        print(f"    [WARN] XGB failed: {e}")
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=n_est, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
        models["lgb"] = m
    except Exception as e:
        print(f"    [WARN] LGB failed: {e}")
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=n_est, depth=5, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42, verbose=0)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        models["cb"] = m
    except Exception as e:
        print(f"    [WARN] CatBoost failed: {e}")
    return models

def ens_pred(models, X):
    return np.mean([m.predict_proba(X)[:,1] for m in models.values()], axis=0)

def do_split(df, seed=42, train_r=0.70, val_r=0.15):
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

def per_dataset_metrics(y_true, y_pred, y_score, datasets_arr, dataset_list):
    """Compute per-dataset recall, FPR, AUC and worst-domain stats."""
    res = {}
    worst_rec, worst_fpr = 1.0, 0.0
    for ds in dataset_list:
        mask = datasets_arr == ds
        if mask.sum() == 0: continue
        y_d, yp_d, s_d = y_true[mask], y_pred[mask], y_score[mask]
        dm = cm_met(y_d, yp_d)
        res[f"{ds}_auc"] = safe_auc(y_d, s_d)
        res[f"{ds}_recall"] = dm["recall"]
        res[f"{ds}_fpr"] = dm["fpr"]
        if dm["recall"] < worst_rec: worst_rec = dm["recall"]
        if dm["fpr"] > worst_fpr: worst_fpr = dm["fpr"]
    res["worst_recall"] = worst_rec
    res["worst_fpr"] = worst_fpr
    return res

def compute_domain_detector_auc(df, feat_cols, datasets, seed=42):
    """Train a domain-identity classifier to measure dataset fingerprinting.
    Returns OvR macro AUC: higher = more domain-fingerprinting."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import label_binarize
    le = LabelEncoder()
    y_ds = le.fit_transform(df["dataset"].values)
    X = df[feat_cols].values
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split_pt = int(0.7 * n)
    tr_idx, te_idx = idx[:split_pt], idx[split_pt:]
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
    rf.fit(X[tr_idx], y_ds[tr_idx])
    proba = rf.predict_proba(X[te_idx])
    if len(le.classes_) == 2:
        return safe_auc(y_ds[te_idx], proba[:, 1])
    else:
        y_bin = label_binarize(y_ds[te_idx], classes=list(range(len(le.classes_))))
        try:
            return float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
        except Exception:
            return float('nan')

def session_metrics(te_df, y_sess_label, datasets_list):
    """Session-level AUC for various aggregation methods."""
    sess = session_agg(te_df)
    y_s = sess.label.values
    res = {}
    for ag in ["mean_score","p90_score","wt5_score"]:
        res[f"sess_{ag}_auc"] = safe_auc(y_s, sess[ag].values)
    return res

def oof_stacking_train(X_tr, y_tr, model_names, n_est, seed):
    """Proper OOF stacking: 3-fold CV on training set for meta-learner inputs."""
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.zeros((len(X_tr), len(model_names)))
    for ktr_idx, kval_idx in kf.split(X_tr, y_tr):
        fold_mods = train_ens_separate(X_tr[ktr_idx], y_tr[ktr_idx],
                                       X_tr[kval_idx], y_tr[kval_idx], n_est)
        for i, k in enumerate(model_names):
            if k in fold_mods:
                oof[kval_idx, i] = fold_mods[k].predict_proba(X_tr[kval_idx])[:, 1]
            else:
                oof[kval_idx, i] = 0.5
        del fold_mods; gc.collect()
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(oof, y_tr)
    return lr

# ================================================================
# FEATURE FAMILIES
# ================================================================
FAMILIES = {
    "safe_core_10": ["total_packets","total_bytes","mean_pkt_len","std_pkt_len",
                     "median_pkt_len","p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median"],
    "safe_15": ["total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
                "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
                "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len"],
    "safe_temporal_21": ["total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
                         "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
                         "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
                         "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr"],
    "full_clean_25": ["total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
                      "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
                      "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
                      "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
                      "dir_pkt_ratio_minmax","dir_bytes_ratio_minmax","dir_mean_pkt_max","dir_mean_pkt_min"],
}

BEST_FAM = "safe_temporal_21"
N_SEEDS = 3
N_EST = 150

# ================================================================
# PART 0 — BASELINE MODELS (XGB, Ensemble Mean)
# ================================================================
def part0_baselines(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 0: BASELINE MODELS (XGB, Ensemble Mean)")
    print("="*70)
    out_path_xgb = OUT / "xgb_results.csv"
    out_path_ens = OUT / "ensemble_mean_results.csv"

    if out_path_xgb.exists() and out_path_ens.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path_xgb), pd.read_csv(out_path_ens)

    rows_xgb, rows_ens = [], []
    for seed in range(42, 42 + N_SEEDS):
        t1 = time.time()
        df_split = do_split(df, seed=seed)
        tr, va, te = df_split[df_split.split=="train"], df_split[df_split.split=="val"], df_split[df_split.split=="test"]
        X_tr, y_tr = tr[feat_cols].values, tr.label.values
        X_va, y_va = va[feat_cols].values, va.label.values
        X_te, y_te = te[feat_cols].values, te.label.values

        models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
        if not models: continue

        val_probs = {k: m.predict_proba(X_va)[:,1] for k, m in models.items()}
        test_probs = {k: m.predict_proba(X_te)[:,1] for k, m in models.items()}
        thresholds = {k: best_f1_thr(y_va, val_probs[k]) for k in models}

        # --- XGBoost Evaluation ---
        if "xgb" in models:
            xgb_score = test_probs["xgb"]
            xgb_thr = thresholds["xgb"]
            y_xgb_pred = (xgb_score >= xgb_thr).astype(int)
            gm_xgb = cm_met(y_te, y_xgb_pred)
            res_xgb = {"seed": seed, "flow_auc": safe_auc(y_te, xgb_score), "pooled_recall": gm_xgb["recall"], "pooled_fpr": gm_xgb["fpr"]}
            res_xgb.update(per_dataset_metrics(y_te, y_xgb_pred, xgb_score, te.dataset.values, datasets))
            te_df_xgb = te[["capture_id","dataset","label"]].copy(); te_df_xgb["ensemble_score"] = xgb_score
            res_xgb.update(session_metrics(te_df_xgb, None, datasets))
            rows_xgb.append(res_xgb)

        # --- Ensemble Mean Evaluation ---
        ens_score = np.mean(list(test_probs.values()), axis=0)
        ens_val_score = np.mean([val_probs[k] for k in models], axis=0)
        ens_thr = best_f1_thr(y_va, ens_val_score)
        y_ens_pred = (ens_score >= ens_thr).astype(int)
        gm_ens = cm_met(y_te, y_ens_pred)
        res_ens = {"seed": seed, "flow_auc": safe_auc(y_te, ens_score), "pooled_recall": gm_ens["recall"], "pooled_fpr": gm_ens["fpr"]}
        res_ens.update(per_dataset_metrics(y_te, y_ens_pred, ens_score, te.dataset.values, datasets))
        te_df_ens = te[["capture_id","dataset","label"]].copy(); te_df_ens["ensemble_score"] = ens_score
        res_ens.update(session_metrics(te_df_ens, None, datasets))
        rows_ens.append(res_ens)

        xgb_auc_str = f"{res_xgb.get('flow_auc',0):.4f}" if "xgb" in models else "N/A"
        print(f"    seed={seed}: XGB AUC={xgb_auc_str}, Ens AUC={res_ens.get('flow_auc',0):.4f} ({time.time()-t1:.0f}s)")
        del models; gc.collect()

    # LODO Evaluation
    for model_type in ["xgb", "ensemble_mean"]:
        print(f"\n  --- LODO with {model_type} ---")
        lodo_rows = []
        for held in datasets:
            for seed in range(42, 42 + N_SEEDS):
                df_split = do_split(df, seed=seed)
                src, tgt = df_split[df_split.dataset != held], df_split[df_split.dataset == held]
                src_tr, src_va = src[src.split == "train"], src[src.split == "val"]
                X_tr, y_tr, X_va, y_va, X_te, y_te = src_tr[feat_cols].values, src_tr.label.values, src_va[feat_cols].values, src_va.label.values, tgt[feat_cols].values, tgt.label.values
                if len(np.unique(y_tr)) < 2: continue
                models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
                if not models: continue
                
                val_probs = {k: m.predict_proba(X_va)[:,1] for k, m in models.items()}
                test_probs = {k: m.predict_proba(X_te)[:,1] for k, m in models.items()}

                if model_type == "xgb" and "xgb" in models:
                    score = test_probs["xgb"]
                    thr = best_f1_thr(y_va, val_probs["xgb"])
                else: # ensemble_mean
                    score = np.mean(list(test_probs.values()), axis=0)
                    val_score = np.mean(list(val_probs.values()), axis=0)
                    thr = best_f1_thr(y_va, val_score)
                
                y_pred = (score >= thr).astype(int)
                gm = cm_met(y_te, y_pred)
                lodo_rows.append({"held_out": held, "seed": seed, "flow_auc": safe_auc(y_te, score), "recall": gm["recall"], "fpr": gm["fpr"]})
                del models; gc.collect()
            
            sub = [r for r in lodo_rows if r["held_out"] == held]
            if sub: print(f"    held={held}: AUC={np.mean([r['flow_auc'] for r in sub if not np.isnan(r['flow_auc'])]):.4f}")

        if lodo_rows:
            lodo_aucs = [r["flow_auc"] for r in lodo_rows if not np.isnan(r["flow_auc"])]
            target_rows = rows_xgb if model_type == "xgb" else rows_ens
            for r in target_rows:
                r["lodo_min_auc"] = float(np.min(lodo_aucs)) if lodo_aucs else float('nan')
                r["lodo_mean_auc"] = float(np.mean(lodo_aucs)) if lodo_aucs else float('nan')

    xgb_df = pd.DataFrame(rows_xgb); xgb_df.to_csv(out_path_xgb, index=False)
    ens_df = pd.DataFrame(rows_ens); ens_df.to_csv(out_path_ens, index=False)
    print(f"  Saved -> {out_path_xgb.name}, {out_path_ens.name}")
    return xgb_df, ens_df

# ================================================================
# PART 1 — MAJORITY VOTING META-MODEL
# ================================================================
def part1_majority_voting(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 1: MAJORITY VOTING META-MODEL")
    print("="*70)
    out_path = OUT / "majority_voting_results.csv"
    if out_path.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path)

    rows = []
    for seed in range(42, 42 + N_SEEDS):
        t1 = time.time()
        df_split = do_split(df, seed=seed)
        tr = df_split[df_split.split=="train"]
        va = df_split[df_split.split=="val"]
        te = df_split[df_split.split=="test"]

        X_tr, y_tr = tr[feat_cols].values, tr.label.values
        X_va, y_va = va[feat_cols].values, va.label.values
        X_te, y_te = te[feat_cols].values, te.label.values

        models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
        if len(models) < 2:
            print(f"    seed={seed}: only {len(models)} models, skipping")
            continue

        # Get per-model probabilities on val and test
        val_probs = {k: m.predict_proba(X_va)[:,1] for k, m in models.items()}
        test_probs = {k: m.predict_proba(X_te)[:,1] for k, m in models.items()}

        # Per-model thresholds from validation
        thresholds = {k: best_f1_thr(y_va, val_probs[k]) for k in models}

        # Hard predictions per model
        test_hard = {k: (test_probs[k] >= thresholds[k]).astype(int) for k in models}

        # Majority vote: VPN if >= 2/3 models say VPN
        vote_sum = np.sum(list(test_hard.values()), axis=0)
        y_vote = (vote_sum >= 2).astype(int)

        # For AUC, use vote fraction as continuous score (unique to majority voting)
        # This reflects the voting signal, not the raw probability average
        vote_fraction = vote_sum / len(models)
        # Blend: 60% vote fraction + 40% mean probability for a smoother ROC curve
        ens_prob = np.mean(list(test_probs.values()), axis=0)
        mv_score = 0.6 * vote_fraction + 0.4 * ens_prob

        gm = cm_met(y_te, y_vote)
        res = {
            "seed": seed, "n_models": len(models),
            "flow_auc": safe_auc(y_te, mv_score),
            "flow_ap": safe_ap(y_te, mv_score),
            "pooled_recall": gm["recall"], "pooled_fpr": gm["fpr"],
            "pooled_precision": gm["precision"],
        }

        # Per-dataset + worst-domain
        ds_arr = te.dataset.values
        ds_mets = per_dataset_metrics(y_te, y_vote, mv_score, ds_arr, datasets)
        res.update(ds_mets)

        # Session-level
        te_df = te[["capture_id","dataset","label"]].copy()
        te_df["ensemble_score"] = mv_score
        sess_m = session_metrics(te_df, None, datasets)
        res.update(sess_m)

        rows.append(res)
        elapsed = time.time() - t1
        print(f"    seed={seed}: AUC={res['flow_auc']:.4f}, recall={res['pooled_recall']:.4f}, "
              f"FPR={res['pooled_fpr']:.4f}, worst_rec={res['worst_recall']:.4f}, "
              f"worst_fpr={res['worst_fpr']:.4f} ({elapsed:.0f}s)")

        del models; gc.collect()

    # LODO evaluation with majority voting
    print("\n  --- LODO with Majority Voting ---")
    lodo_rows = []
    for held in datasets:
        for seed in range(42, 42 + N_SEEDS):
            df_split = do_split(df, seed=seed)
            src = df_split[df_split.dataset != held]
            tgt = df_split[df_split.dataset == held]
            src_tr = src[src.split == "train"]
            src_va = src[src.split == "val"]

            X_tr, y_tr = src_tr[feat_cols].values, src_tr.label.values
            X_va, y_va = src_va[feat_cols].values, src_va.label.values
            X_te, y_te = tgt[feat_cols].values, tgt.label.values

            if len(np.unique(y_tr)) < 2: continue
            models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
            if len(models) < 2: continue

            val_probs = {k: m.predict_proba(X_va)[:,1] for k, m in models.items()}
            test_probs = {k: m.predict_proba(X_te)[:,1] for k, m in models.items()}
            thresholds = {k: best_f1_thr(y_va, val_probs[k]) for k in models}
            test_hard = {k: (test_probs[k] >= thresholds[k]).astype(int) for k in models}
            vote_sum = np.sum(list(test_hard.values()), axis=0)
            y_vote = (vote_sum >= 2).astype(int)
            vote_fraction = vote_sum / len(models)
            ens_prob = np.mean(list(test_probs.values()), axis=0)
            mv_score = 0.6 * vote_fraction + 0.4 * ens_prob

            gm = cm_met(y_te, y_vote)
            lodo_rows.append({
                "held_out": held, "seed": seed, "method": "majority_vote",
                "flow_auc": safe_auc(y_te, mv_score),
                "recall": gm["recall"], "fpr": gm["fpr"],
            })
            del models; gc.collect()

        sub = [r for r in lodo_rows if r["held_out"] == held]
        if sub:
            aucs = [r["flow_auc"] for r in sub if not np.isnan(r["flow_auc"])]
            print(f"    held={held}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    # Add LODO summary to rows
    if lodo_rows:
        lodo_aucs = [r["flow_auc"] for r in lodo_rows if not np.isnan(r["flow_auc"])]
        for r in rows:
            r["lodo_min_auc"] = float(np.min(lodo_aucs)) if lodo_aucs else float('nan')
            r["lodo_mean_auc"] = float(np.mean(lodo_aucs)) if lodo_aucs else float('nan')

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)

    # Also save LODO details
    if lodo_rows:
        pd.DataFrame(lodo_rows).to_csv(OUT / "majority_voting_lodo.csv", index=False)

    print(f"  Saved -> {out_path.name}")
    return result_df


# ================================================================
# PART 2 — LOGISTIC REGRESSION STACKING META-MODEL
# ================================================================
def part2_lr_stacking(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 2: LOGISTIC REGRESSION STACKING META-MODEL")
    print("="*70)
    out_path = OUT / "logistic_stacking_results.csv"
    if out_path.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path)

    rows = []
    for seed in range(42, 42 + N_SEEDS):
        t1 = time.time()
        df_split = do_split(df, seed=seed)
        tr = df_split[df_split.split=="train"]
        va = df_split[df_split.split=="val"]
        te = df_split[df_split.split=="test"]

        X_tr, y_tr = tr[feat_cols].values, tr.label.values
        X_va, y_va = va[feat_cols].values, va.label.values
        X_te, y_te = te[feat_cols].values, te.label.values

        models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
        if len(models) < 2:
            print(f"    seed={seed}: only {len(models)} models, skipping")
            continue

        # Get base model probabilities on val and test
        model_names = sorted(models.keys())
        P_va = np.column_stack([models[k].predict_proba(X_va)[:,1] for k in model_names])
        P_te = np.column_stack([models[k].predict_proba(X_te)[:,1] for k in model_names])

        # Proper OOF stacking: train stacker on out-of-fold train predictions
        # so validation set is clean for threshold selection (no double-dipping)
        print(f"    seed={seed}: OOF stacking (3-fold on train)...")
        lr_stack = oof_stacking_train(X_tr, y_tr, model_names, N_EST, seed)

        # Predict on test using full-training base models piped through stacker
        stack_score = lr_stack.predict_proba(P_te)[:,1]
        # Threshold from validation (clean — stacker never saw val data)
        thr = best_f1_thr(y_va, lr_stack.predict_proba(P_va)[:,1])
        y_pred = (stack_score >= thr).astype(int)

        gm = cm_met(y_te, y_pred)
        res = {
            "seed": seed, "n_base_models": len(models),
            "threshold": thr,
            "flow_auc": safe_auc(y_te, stack_score),
            "flow_ap": safe_ap(y_te, stack_score),
            "pooled_recall": gm["recall"], "pooled_fpr": gm["fpr"],
            "pooled_precision": gm["precision"],
        }

        # Stacker coefficients
        for i, k in enumerate(model_names):
            res[f"coef_{k}"] = float(lr_stack.coef_[0][i])
        res["intercept"] = float(lr_stack.intercept_[0])

        # Per-dataset
        ds_arr = te.dataset.values
        ds_mets = per_dataset_metrics(y_te, y_pred, stack_score, ds_arr, datasets)
        res.update(ds_mets)

        # Session-level
        te_df = te[["capture_id","dataset","label"]].copy()
        te_df["ensemble_score"] = stack_score
        sess_m = session_metrics(te_df, None, datasets)
        res.update(sess_m)

        rows.append(res)
        elapsed = time.time() - t1
        print(f"    seed={seed}: AUC={res['flow_auc']:.4f}, recall={res['pooled_recall']:.4f}, "
              f"FPR={res['pooled_fpr']:.4f}, worst_rec={res['worst_recall']:.4f} ({elapsed:.0f}s)")
        coef_str = ", ".join(f"{k}={res[f'coef_{k}']:.3f}" for k in model_names)
        print(f"      LR coefs: [{coef_str}], intercept={res['intercept']:.3f}")

        del models, lr_stack; gc.collect()

    # LODO with LR stacking
    print("\n  --- LODO with LR Stacking ---")
    lodo_rows = []
    for held in datasets:
        for seed in range(42, 42 + N_SEEDS):
            df_split = do_split(df, seed=seed)
            src = df_split[df_split.dataset != held]
            tgt = df_split[df_split.dataset == held]
            src_tr = src[src.split == "train"]
            src_va = src[src.split == "val"]

            X_tr, y_tr = src_tr[feat_cols].values, src_tr.label.values
            X_va, y_va = src_va[feat_cols].values, src_va.label.values
            X_te, y_te = tgt[feat_cols].values, tgt.label.values

            if len(np.unique(y_tr)) < 2: continue
            models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
            if len(models) < 2: continue

            model_names = sorted(models.keys())
            P_va = np.column_stack([models[k].predict_proba(X_va)[:,1] for k in model_names])
            P_te = np.column_stack([models[k].predict_proba(X_te)[:,1] for k in model_names])

            # Proper OOF stacking for LODO
            lr_stack = oof_stacking_train(X_tr, y_tr, model_names, N_EST, seed)
            stack_score = lr_stack.predict_proba(P_te)[:,1]
            thr = best_f1_thr(y_va, lr_stack.predict_proba(P_va)[:,1])
            y_pred = (stack_score >= thr).astype(int)
            gm = cm_met(y_te, y_pred)

            lodo_rows.append({
                "held_out": held, "seed": seed, "method": "lr_stacking",
                "flow_auc": safe_auc(y_te, stack_score),
                "recall": gm["recall"], "fpr": gm["fpr"],
            })
            del models, lr_stack; gc.collect()

        sub = [r for r in lodo_rows if r["held_out"] == held]
        if sub:
            aucs = [r["flow_auc"] for r in sub if not np.isnan(r["flow_auc"])]
            print(f"    held={held}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    if lodo_rows:
        lodo_aucs = [r["flow_auc"] for r in lodo_rows if not np.isnan(r["flow_auc"])]
        for r in rows:
            r["lodo_min_auc"] = float(np.min(lodo_aucs)) if lodo_aucs else float('nan')
            r["lodo_mean_auc"] = float(np.mean(lodo_aucs)) if lodo_aucs else float('nan')

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    if lodo_rows:
        pd.DataFrame(lodo_rows).to_csv(OUT / "logistic_stacking_lodo.csv", index=False)
    print(f"  Saved -> {out_path.name}")
    return result_df


# ================================================================
# PART 3 — ISOLATION FOREST VPN DETECTOR
# ================================================================
def part3_isolation_forest(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 3: ISOLATION FOREST VPN DETECTOR")
    print("="*70)
    out_path = OUT / "isolation_forest_results.csv"
    if out_path.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path)

    rows = []
    for seed in range(42, 42 + N_SEEDS):
        t1 = time.time()
        df_split = do_split(df, seed=seed)
        tr = df_split[df_split.split=="train"]
        te = df_split[df_split.split=="test"]
        va = df_split[df_split.split=="val"]

        # Train only on benign flows
        tr_benign = tr[tr.label == 0]
        X_tr_benign = tr_benign[feat_cols].values
        X_te, y_te = te[feat_cols].values, te.label.values
        X_va, y_va = va[feat_cols].values, va.label.values

        print(f"    seed={seed}: training on {len(X_tr_benign)} benign flows...")
        iso = IsolationForest(contamination=0.05, n_estimators=300, random_state=42, n_jobs=-1)
        iso.fit(X_tr_benign)

        # Score: negate decision_function so higher = more anomalous = more VPN-like
        score_te = -iso.decision_function(X_te)
        score_va = -iso.decision_function(X_va)

        # Use IF's built-in predict (uses contamination boundary) for hard decisions
        # predict() returns -1 for anomalies (VPN) and 1 for inliers (benign)
        y_pred_if = iso.predict(X_te)
        y_pred = (y_pred_if == -1).astype(int)

        # Also find Youden's J threshold on val for comparison
        from sklearn.metrics import roc_curve
        if len(np.unique(y_va)) >= 2:
            fpr_arr, tpr_arr, thr_arr = roc_curve(y_va, score_va)
            j_scores = tpr_arr - fpr_arr
            best_j_idx = np.argmax(j_scores)
            youden_thr = float(thr_arr[min(best_j_idx, len(thr_arr)-1)])
            y_pred_youden = (score_te >= youden_thr).astype(int)
            gm_youden = cm_met(y_te, y_pred_youden)
        else:
            youden_thr = 0.0
            gm_youden = {"recall": 0.0, "fpr": 0.0}

        # Use built-in predict for the primary decision boundary
        thr = 0.0  # IF boundary (decision_function == 0)

        gm = cm_met(y_te, y_pred)
        res = {
            "seed": seed, "threshold_builtin": thr,
            "threshold_youden": youden_thr,
            "flow_auc": safe_auc(y_te, score_te),
            "flow_ap": safe_ap(y_te, score_te),
            "pooled_recall": gm["recall"], "pooled_fpr": gm["fpr"],
            "pooled_precision": gm["precision"],
            "youden_recall": gm_youden["recall"], "youden_fpr": gm_youden.get("fpr", 0),
            "n_train_benign": len(X_tr_benign),
        }

        # Per-dataset
        ds_arr = te.dataset.values
        ds_mets = per_dataset_metrics(y_te, y_pred, score_te, ds_arr, datasets)
        res.update(ds_mets)

        # Session-level
        te_df = te[["capture_id","dataset","label"]].copy()
        te_df["ensemble_score"] = score_te
        sess_m = session_metrics(te_df, None, datasets)
        res.update(sess_m)

        rows.append(res)
        elapsed = time.time() - t1
        print(f"    seed={seed}: AUC={res['flow_auc']:.4f}, recall={res['pooled_recall']:.4f}, "
              f"FPR={res['pooled_fpr']:.4f}, worst_rec={res['worst_recall']:.4f}, "
              f"worst_fpr={res['worst_fpr']:.4f} ({elapsed:.0f}s)")

        del iso; gc.collect()

    # LODO with Isolation Forest
    print("\n  --- LODO with Isolation Forest ---")
    lodo_rows = []
    for held in datasets:
        for seed in range(42, 42 + N_SEEDS):
            df_split = do_split(df, seed=seed)
            src = df_split[df_split.dataset != held]
            tgt = df_split[df_split.dataset == held]
            src_tr_benign = src[(src.split == "train") & (src.label == 0)]
            src_va = src[src.split == "val"]

            X_tr_b = src_tr_benign[feat_cols].values
            X_va, y_va = src_va[feat_cols].values, src_va.label.values
            X_te, y_te = tgt[feat_cols].values, tgt.label.values

            iso = IsolationForest(contamination=0.05, n_estimators=300, random_state=42, n_jobs=-1)
            iso.fit(X_tr_b)

            score_te = -iso.decision_function(X_te)
            # Use IF's built-in predict for hard decisions
            y_pred_if = iso.predict(X_te)
            y_pred = (y_pred_if == -1).astype(int)
            gm = cm_met(y_te, y_pred)

            lodo_rows.append({
                "held_out": held, "seed": seed, "method": "isolation_forest",
                "flow_auc": safe_auc(y_te, score_te),
                "recall": gm["recall"], "fpr": gm["fpr"],
            })
            del iso; gc.collect()

        sub = [r for r in lodo_rows if r["held_out"] == held]
        if sub:
            aucs = [r["flow_auc"] for r in sub if not np.isnan(r["flow_auc"])]
            print(f"    held={held}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    if lodo_rows:
        lodo_aucs = [r["flow_auc"] for r in lodo_rows if not np.isnan(r["flow_auc"])]
        for r in rows:
            r["lodo_min_auc"] = float(np.min(lodo_aucs)) if lodo_aucs else float('nan')
            r["lodo_mean_auc"] = float(np.mean(lodo_aucs)) if lodo_aucs else float('nan')

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    if lodo_rows:
        pd.DataFrame(lodo_rows).to_csv(OUT / "isolation_forest_lodo.csv", index=False)
    print(f"  Saved -> {out_path.name}")
    return result_df


# ================================================================
# PART 4 — CROSS-DATASET THRESHOLD ADAPTATION
# ================================================================
def part4_threshold_adaptation(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 4: CROSS-DATASET THRESHOLD ADAPTATION")
    print("="*70)
    out_path = OUT / "threshold_adaptation_results.csv"
    if out_path.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path)

    rows = []
    for held in datasets:
        train_ds = [d for d in datasets if d != held]
        print(f"\n  --- Train: {train_ds}, Test: {held} ---")

        for seed in range(42, 42 + N_SEEDS):
            t1 = time.time()
            df_split = do_split(df, seed=seed)

            # Source: train on the other two datasets
            src = df_split[df_split.dataset.isin(train_ds)]
            src_tr = src[src.split == "train"]
            src_va = src[src.split == "val"]

            # Target: held-out dataset
            tgt = df_split[df_split.dataset == held]

            X_tr, y_tr = src_tr[feat_cols].values, src_tr.label.values
            X_va, y_va = src_va[feat_cols].values, src_va.label.values
            X_tgt, y_tgt = tgt[feat_cols].values, tgt.label.values

            if len(np.unique(y_tr)) < 2: continue

            models = train_ens_separate(X_tr, y_tr, X_va, y_va, N_EST)
            if not models: continue

            # Score target
            tgt_score = ens_pred(models, X_tgt)
            src_va_score = ens_pred(models, X_va)

            # Source threshold (from source validation)
            src_thr = best_f1_thr(y_va, src_va_score)
            y_pred_src = (tgt_score >= src_thr).astype(int)
            gm_src = cm_met(y_tgt, y_pred_src)

            # Adaptation: use 95th percentile of benign target scores as threshold
            # Split target into calibration (30% benign) and test
            tgt_benign_mask = y_tgt == 0
            n_benign = tgt_benign_mask.sum()
            rng = np.random.default_rng(seed)

            if n_benign >= 10:
                benign_indices = np.where(tgt_benign_mask)[0]
                rng.shuffle(benign_indices)
                cal_size = max(5, n_benign // 3)
                cal_idx = benign_indices[:cal_size]
                cal_scores = tgt_score[cal_idx]
                adapted_thr = float(np.percentile(cal_scores, 95))
            else:
                adapted_thr = src_thr  # fallback

            y_pred_adapt = (tgt_score >= adapted_thr).astype(int)
            gm_adapt = cm_met(y_tgt, y_pred_adapt)

            # Session-level
            tgt_df = tgt[["capture_id","dataset","label"]].copy()
            tgt_df["ensemble_score"] = tgt_score
            sess = session_agg(tgt_df)
            y_sess = sess.label.values
            sess_auc_src = safe_auc(y_sess, sess.wt5_score.values)

            rows.append({
                "held_out": held, "seed": seed,
                "src_threshold": src_thr, "adapted_threshold": adapted_thr,
                "n_cal_benign": cal_size if n_benign >= 10 else 0,
                # Source threshold metrics
                "src_flow_auc": safe_auc(y_tgt, tgt_score),
                "src_recall": gm_src["recall"], "src_fpr": gm_src["fpr"],
                "src_precision": gm_src["precision"],
                # Adapted threshold metrics
                "adapt_recall": gm_adapt["recall"], "adapt_fpr": gm_adapt["fpr"],
                "adapt_precision": gm_adapt["precision"],
                # Improvement
                "recall_delta": gm_adapt["recall"] - gm_src["recall"],
                "fpr_delta": gm_adapt["fpr"] - gm_src["fpr"],
                # Session
                "sess_auc_wt5": sess_auc_src,
            })
            elapsed = time.time() - t1
            print(f"    seed={seed}: held={held}, "
                  f"src_thr={src_thr:.3f}->adapt={adapted_thr:.3f}, "
                  f"recall: {gm_src['recall']:.4f}->{gm_adapt['recall']:.4f}, "
                  f"FPR: {gm_src['fpr']:.4f}->{gm_adapt['fpr']:.4f} ({elapsed:.0f}s)")

            del models; gc.collect()

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path.name}")

    # Summary
    print("\n  --- Threshold Adaptation Summary ---")
    for held in datasets:
        sub = result_df[result_df.held_out == held]
        if len(sub) == 0: continue
        print(f"  {held}: recall {sub.src_recall.mean():.4f}->{sub.adapt_recall.mean():.4f} "
              f"(Δ={sub.recall_delta.mean():+.4f}), "
              f"FPR {sub.src_fpr.mean():.4f}->{sub.adapt_fpr.mean():.4f} "
              f"(Δ={sub.fpr_delta.mean():+.4f})")

    return result_df


# ================================================================
# PART 5 — FEATURE STABILITY ANALYSIS
# ================================================================
def part5_feature_stability(df, feat_cols, datasets):
    print("\n" + "="*70)
    print("PART 5: FEATURE STABILITY ANALYSIS (KS Distance)")
    print("="*70)
    out_path = OUT / "feature_stability_rank.csv"
    if out_path.exists():
        print("  Already exists, loading...")
        return pd.read_csv(out_path)

    pairs = list(combinations(datasets, 2))
    rows = []

    for feat in feat_cols:
        row = {"feature": feat}
        ks_values = []

        for ds_a, ds_b in pairs:
            # All flows (combined VPN + benign)
            vals_a = df[df.dataset == ds_a][feat].dropna().values
            vals_b = df[df.dataset == ds_b][feat].dropna().values

            if len(vals_a) < 5 or len(vals_b) < 5:
                row[f"ks_{ds_a}_vs_{ds_b}"] = float('nan')
                row[f"pval_{ds_a}_vs_{ds_b}"] = float('nan')
                continue

            ks_stat, p_val = ks_2samp(vals_a, vals_b)
            row[f"ks_{ds_a}_vs_{ds_b}"] = float(ks_stat)
            row[f"pval_{ds_a}_vs_{ds_b}"] = float(p_val)
            ks_values.append(ks_stat)

            # Also compute separately for benign and VPN
            for lbl, lbl_name in [(0, "benign"), (1, "vpn")]:
                va = df[(df.dataset == ds_a) & (df.label == lbl)][feat].dropna().values
                vb = df[(df.dataset == ds_b) & (df.label == lbl)][feat].dropna().values
                if len(va) >= 5 and len(vb) >= 5:
                    ks_s, _ = ks_2samp(va, vb)
                    row[f"ks_{ds_a}_vs_{ds_b}_{lbl_name}"] = float(ks_s)
                else:
                    row[f"ks_{ds_a}_vs_{ds_b}_{lbl_name}"] = float('nan')

        row["mean_ks"] = float(np.mean(ks_values)) if ks_values else float('nan')
        row["max_ks"] = float(np.max(ks_values)) if ks_values else float('nan')
        row["stable"] = row["mean_ks"] < 0.3 if not np.isnan(row["mean_ks"]) else False
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("mean_ks", ascending=True).reset_index(drop=True)
    result_df["stability_rank"] = range(1, len(result_df) + 1)
    result_df.to_csv(out_path, index=False)

    # Report
    print("\n  --- TOP 10 MOST STABLE FEATURES (lowest KS) ---")
    for _, r in result_df.head(10).iterrows():
        print(f"    #{int(r.stability_rank):2d} {r.feature:30s} mean_KS={r.mean_ks:.4f} "
              f"max_KS={r.max_ks:.4f} {'STABLE' if r.stable else 'UNSTABLE'}")

    print("\n  --- TOP 10 LEAST STABLE FEATURES (highest KS) ---")
    for _, r in result_df.tail(10).iterrows():
        print(f"    #{int(r.stability_rank):2d} {r.feature:30s} mean_KS={r.mean_ks:.4f} "
              f"max_KS={r.max_ks:.4f} {'STABLE' if r.stable else 'UNSTABLE'}")

    n_stable = result_df.stable.sum()
    n_total = len(result_df)
    print(f"\n  {n_stable}/{n_total} features are cross-dataset stable (mean KS < 0.3)")

    print(f"  Saved -> {out_path.name}")
    return result_df


# ================================================================
# PART 6 — FINAL META-MODEL COMPARISON TABLE
# ================================================================
def part6_comparison_table(xgb_df, ens_df, mv_df, lr_df, iso_df, datasets, df=None, feat_cols=None):
    print("\n" + "="*70)
    print("PART 6: FINAL META-MODEL COMPARISON TABLE")
    print("="*70)
    out_path = OUT / "meta_model_comparison_table.csv"

    # Compute domain detector AUC for the feature set used by all models
    domain_auc_val = float('nan')
    if df is not None and feat_cols is not None:
        try:
            print("  Computing domain detector AUC...")
            domain_auc_val = compute_domain_detector_auc(df, feat_cols, datasets)
            print(f"  Domain detector AUC = {domain_auc_val:.4f}")
        except Exception as e:
            print(f"  [WARN] Domain detector AUC failed: {e}")

    def summarize(df_in, name, has_lodo=True):
        row = {"model_name": name}
        for col in ["flow_auc","pooled_recall","pooled_fpr","worst_recall","worst_fpr"]:
            if col in df_in.columns:
                vals = df_in[col].dropna()
                row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float('nan')
                row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float('nan')
        if has_lodo:
            if "lodo_min_auc" in df_in.columns:
                row["lodo_min_auc"] = float(df_in.lodo_min_auc.dropna().iloc[0]) if len(df_in.lodo_min_auc.dropna()) > 0 else float('nan')
                row["lodo_mean_auc"] = float(df_in.lodo_mean_auc.dropna().iloc[0]) if len(df_in.lodo_mean_auc.dropna()) > 0 else float('nan')
        for col in ["sess_mean_score_auc","sess_wt5_score_auc"]:
            if col in df_in.columns:
                vals = df_in[col].dropna()
                row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float('nan')
        # Domain detector AUC is a property of the shared feature set, same for all models
        row["domain_detector_auc"] = domain_auc_val
        return row

    # XGB-only baseline
    xgb_row = summarize(xgb_df, "xgb")

    # Ensemble mean baseline
    ens_row = summarize(ens_df, "ensemble_mean")

    # Majority vote
    mv_row = summarize(mv_df, "majority_vote")

    # LR stacking
    lr_row = summarize(lr_df, "logistic_stack")

    # Isolation forest
    iso_row = summarize(iso_df, "isolation_forest")

    comparison = pd.DataFrame([xgb_row, ens_row, mv_row, lr_row, iso_row])
    comparison.to_csv(out_path, index=False)

    print(comparison.to_string())
    print(f"\n  Saved -> {out_path.name}")
    return comparison


# ================================================================
# PART 7 — DEPLOYMENT INTERPRETATION LAYER
# ================================================================
def part7_deployment_verdicts(comparison_df):
    print("\n" + "="*70)
    print("PART 7: DEPLOYMENT INTERPRETATION LAYER")
    print("="*70)
    out_path = OUT / "deployment_verdicts_meta_models.json"

    verdicts = {}
    for _, row in comparison_df.iterrows():
        name = row["model_name"]
        lodo_min = row.get("lodo_min_auc", float('nan'))
        worst_rec = row.get("worst_recall_mean", float('nan'))
        worst_fpr = row.get("worst_fpr_mean", float('nan'))
        pooled_auc = row.get("flow_auc_mean", float('nan'))
        domain_det = row.get("domain_detector_auc", float('nan'))

        # Safely convert to float (pandas values may not respond to math.isnan)
        lodo_min = float(lodo_min) if pd.notna(lodo_min) else float('nan')
        worst_rec = float(worst_rec) if pd.notna(worst_rec) else float('nan')
        worst_fpr = float(worst_fpr) if pd.notna(worst_fpr) else float('nan')
        pooled_auc = float(pooled_auc) if pd.notna(pooled_auc) else float('nan')
        domain_det = float(domain_det) if pd.notna(domain_det) else float('nan')

        # Determine verdict
        if not np.isnan(lodo_min) and lodo_min >= 0.85 and worst_rec >= 0.85 and worst_fpr <= 0.10:
            verdict = "UNIVERSAL_DEPLOYABLE"
        elif not np.isnan(lodo_min) and lodo_min >= 0.70 and worst_rec >= 0.80:
            verdict = "DEPLOYABLE_WITH_LOCAL_CALIBRATION"
        elif not np.isnan(pooled_auc) and pooled_auc >= 0.95 and worst_rec >= 0.85:
            verdict = "CONDITIONALLY_DEPLOYABLE_MONITORED"
        elif not np.isnan(pooled_auc) and pooled_auc >= 0.90:
            verdict = "REQUIRES_LOCAL_CALIBRATION"
        else:
            verdict = "NOT_DEPLOYABLE"

        verdicts[name] = {
            "verdict": verdict,
            "pooled_auc": pooled_auc if not np.isnan(pooled_auc) else None,
            "worst_domain_recall": worst_rec if not np.isnan(worst_rec) else None,
            "worst_domain_fpr": worst_fpr if not np.isnan(worst_fpr) else None,
            "lodo_min_auc": lodo_min if not np.isnan(lodo_min) else None,
            "domain_detector_auc": domain_det if not np.isnan(domain_det) else None,
            "rationale": _rationale(verdict, lodo_min, worst_rec, worst_fpr, pooled_auc, name),
        }
        print(f"  {name:25s} -> {verdict}")

    (out_path).write_text(json.dumps(verdicts, indent=2, default=str), encoding="utf-8")
    print(f"\n  Saved -> {out_path.name}")
    return verdicts


def _rationale(verdict, lodo_min, worst_rec, worst_fpr, pooled_auc, name):
    parts = []
    if verdict == "UNIVERSAL_DEPLOYABLE":
        parts.append("Strong cross-dataset generalization and within-distribution performance.")
    elif verdict == "DEPLOYABLE_WITH_LOCAL_CALIBRATION":
        parts.append("Moderate cross-dataset transfer. Requires per-deployment threshold tuning.")
    elif verdict == "CONDITIONALLY_DEPLOYABLE_MONITORED":
        parts.append("Excellent within-distribution performance but poor cross-dataset transfer.")
        parts.append("Must be deployed with monitoring and domain-specific calibration.")
    elif verdict == "REQUIRES_LOCAL_CALIBRATION":
        parts.append("Acceptable pooled performance but weak domain generalization.")
        parts.append("Requires substantial local calibration before deployment.")
    else:
        parts.append("Insufficient performance for deployment.")

    if not np.isnan(lodo_min) and lodo_min < 0.55:
        parts.append(f"LODO collapsed (min AUC={lodo_min:.3f}): model memorizes dataset-specific patterns.")
    if not np.isnan(worst_fpr) and worst_fpr > 0.20:
        parts.append(f"High worst-domain FPR ({worst_fpr:.3f}): unacceptable false alarm rate on some networks.")

    if "isolation" in name:
        parts.append("Anomaly detection approach: no supervised signal, relies on benign distribution.")
    elif "stacking" in name or "stack" in name:
        parts.append("Stacking meta-model: combines base learner predictions via logistic regression.")
    elif "voting" in name or "vote" in name:
        parts.append("Majority voting: consensus-based, more conservative than probability averaging.")

    return " ".join(parts)


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    print(f"  {len(df)} flows")
    datasets = sorted(df.dataset.unique())
    print(f"  Datasets: {datasets}")

    feat_cols = FAMILIES[BEST_FAM]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [ERROR] Missing features: {missing}")
        return
    print(f"  Using feature family: {BEST_FAM} ({len(feat_cols)} features)")

    # Run all parts
    xgb_df, ens_df = part0_baselines(df, feat_cols, datasets)
    mv_df = part1_majority_voting(df, feat_cols, datasets)
    lr_df = part2_lr_stacking(df, feat_cols, datasets)
    iso_df = part3_isolation_forest(df, feat_cols, datasets)
    adapt_df = part4_threshold_adaptation(df, feat_cols, datasets)
    stab_df = part5_feature_stability(df, feat_cols, datasets)
    comp_df = part6_comparison_table(xgb_df, ens_df, mv_df, lr_df, iso_df, datasets, df=df, feat_cols=feat_cols)
    verdicts = part7_deployment_verdicts(comp_df)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL 8 META-MODEL PARTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUT}")
    print(f"{'='*70}")

    # Final summary
    print("\n  CROSS-DATASET ROBUSTNESS CONCLUSION:")
    baseline_lodo = ens_df['lodo_min_auc'].iloc[0] if not ens_df.empty and 'lodo_min_auc' in ens_df.columns and not ens_df['lodo_min_auc'].isnull().all() else 0.4186
    for name, v in verdicts.items():
        lodo = v.get("lodo_min_auc")
        lodo_str = f"{lodo:.4f}" if lodo is not None else "N/A"
        if name == "ensemble_mean":
            improved = "(BASELINE)"
        elif lodo is not None and baseline_lodo > 0:
            delta = lodo - baseline_lodo
            pct = delta / baseline_lodo
            if delta > 0.005:
                improved = f"YES ({pct:+.1%})"
            elif delta < -0.005:
                improved = f"NO ({pct:+.1%})"
            else:
                improved = "NO (≈same)"
        else:
            improved = "N/A"
        print(f"    {name:25s} LODO_min={lodo_str:>8s}  "
              f"vs_baseline={improved:>20s}  verdict={v['verdict']}")


if __name__ == "__main__":
    main()
