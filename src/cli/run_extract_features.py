# src/cli/run_extract_features.py

from __future__ import annotations

import pandas as pd

from src.utils.paths import load_paths
from src.utils.logging import setup_logger

from src.features.extract import extract_features_from_flows, load_feature_config, feature_config_hash_text
from src.pipeline.artifacts import default_feature_artifacts
from src.pipeline.feature_pipeline import FeaturePipeline
from src.splits.io import load_splits


def main() -> None:
    paths = load_paths()
    paths.ensure_dirs()
    logger = setup_logger(level="INFO")

    flows_parquet = paths.data_processed / "vnat" / "flows.parquet"
    features_yaml = paths.configs_dir / "features.yaml"

    # Split lists are expected in data/splits (same as your manifest paths)
    train_list = paths.data_splits / "vnat_train_captures.txt"
    val_list = paths.data_splits / "vnat_val_captures.txt"
    test_list = paths.data_splits / "vnat_test_captures.txt"

    if not flows_parquet.exists():
        raise FileNotFoundError(f"Missing: {flows_parquet}")
    if not features_yaml.exists():
        raise FileNotFoundError(f"Missing: {features_yaml}")
    for p in [train_list, val_list, test_list]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing split list: {p}\n"
                f"Fix: rerun your split notebook or run src/cli/run_split.py to regenerate lists."
            )

    cfg = load_feature_config(features_yaml)
    cfg_hash = feature_config_hash_text(features_yaml)

    logger.info(f"flows.parquet: {flows_parquet}")
    logger.info(f"features.yaml: {features_yaml} (hash={cfg_hash[:12]}...)")

    splits = load_splits(train_list, val_list, test_list)
    s_train, s_val, s_test = set(splits["train"]), set(splits["val"]), set(splits["test"])

    cols_needed = [
        "flow_id",
        "capture_id",
        "label",
        "timestamps",
        "sizes",
        "directions",
        "packet_count",
        "window_complete",
        "min_packets_ok",
    ]
    flows = pd.read_parquet(flows_parquet, columns=cols_needed)
    flows["capture_id"] = flows["capture_id"].astype(str)
    flows["flow_id"] = flows["flow_id"].astype(str)

    def split_of(cid: str) -> str:
        if cid in s_train:
            return "train"
        if cid in s_val:
            return "val"
        if cid in s_test:
            return "test"
        return "unknown"

    flows["split"] = flows["capture_id"].map(split_of)

    n_unknown = int((flows["split"] == "unknown").sum())
    if n_unknown:
        raise ValueError(
            f"{n_unknown} flows belong to capture_ids not present in train/val/test lists. "
            f"Split lists and flows.parquet are out of sync."
        )

    logger.info("Extracting features from flows...")
    feats = extract_features_from_flows(flows, cfg)

    # Fit pipeline ONLY on VNAT TRAIN flows where min_packets_ok == True
    train_mask = (flows["split"] == "train") & (flows["min_packets_ok"].astype(bool))
    train_flow_ids = set(flows.loc[train_mask, "flow_id"].tolist())

    feats_train = feats[feats["flow_id"].astype(str).isin(train_flow_ids)].copy()
    if len(feats_train) == 0:
        raise ValueError(
            "No training flows with min_packets_ok==True. "
            "Check configs/features.yaml window.min_packets and VNAT preprocessing."
        )

    def pct_ok(split_name: str) -> float:
        sub = flows[flows["split"] == split_name]
        if len(sub) == 0:
            return 0.0
        return float(sub["min_packets_ok"].astype(bool).mean() * 100.0)

    logger.info(
        "min_packets_ok coverage: "
        f"train={pct_ok('train'):.2f}%, val={pct_ok('val'):.2f}%, test={pct_ok('test'):.2f}%"
    )
    logger.info(f"Fitting feature pipeline on TRAIN(min_packets_ok=True): {len(feats_train)} flows")

    pipe = FeaturePipeline().fit(feats_train)

    feats_scaled = pipe.transform(feats)

    out_dir = paths.data_processed / "vnat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    feats_scaled.to_parquet(out_path, index=False)

    logger.info(f"Saved features parquet: {out_path} (rows={len(feats_scaled)}, cols={len(feats_scaled.columns)})")

    art = default_feature_artifacts(paths.artifacts_features)
    pipe.save(art, feature_config_hash=cfg_hash)
    logger.info(f"Saved feature artifacts under: {paths.artifacts_features}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
