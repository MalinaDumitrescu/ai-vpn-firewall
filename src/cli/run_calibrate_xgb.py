from pathlib import Path

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.models.xgb_calibrate import calibrate_xgb_predictions


def main():
    paths = load_paths()
    paths.ensure_dirs()
    logger = setup_logger(level="INFO")

    calib_yaml = paths.configs_dir / "xgb_calibrate.yaml"
    if not calib_yaml.exists():
        raise FileNotFoundError(f"Missing config: {calib_yaml}")

    logger.info(f"Repo root: {paths.repo_root}")
    logger.info(f"Calibration config: {calib_yaml}")

    res = calibrate_xgb_predictions(paths=paths, calib_yaml=calib_yaml)

    logger.info(f"Saved calibrator: {res.calibrator_path}")
    logger.info(f"Saved metrics: {res.metrics_path}")
    logger.info(f"Saved calibrated preds: {res.preds_calibrated_path}")


if __name__ == "__main__":
    main()
