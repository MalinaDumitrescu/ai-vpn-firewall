"""
Leave-One-Out-Dataset (LOOD) evaluation framework.

Instead of fixed train/val/test splits per dataset, we rotate which dataset
is held out for testing while combining all other datasets for training.

This increases training signal significantly:
- VNAT train increases from 374 VPN samples → ~450 VPN samples (ISCX + USBVPN)
- ISCX train increases from 2029 VPN samples → ~830 VPN samples (VNAT + USBVPN)
- USBVPN train increases from 352 VPN samples → ~723 VPN samples (ISCX + VNAT)

LOOD Splits:
1. train_iscx_usbvpn, test_vnat
2. train_vnat_usbvpn, test_iscx
3. train_vnat_iscx, test_usbvpn
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict

from src.utils.paths import load_paths
from src.utils.logging import setup_logger

logger = setup_logger()


@dataclass
class LOODFold:
    """
    Represents one LOOD fold:
    - train_datasets: List of dataset names to use for training
    - test_dataset: Dataset name held out for testing
    - fold_name: Human-readable fold identifier
    """
    train_datasets: List[str]
    test_dataset: str
    fold_name: str

    @property
    def fold_id(self) -> str:
        """Unique identifier for this fold."""
        train_str = "_".join(sorted(self.train_datasets))
        return f"fold_{train_str}_vs_{self.test_dataset}"


class LOODEvaluator:
    """
    Leave-One-Out-Dataset evaluator for unified model training with rotating test sets.
    """

    def __init__(self):
        self.folds: List[LOODFold] = []
        self.results: Dict[str, Dict] = {}

    def create_folds(self, datasets: List[str]) -> List[LOODFold]:
        """
        Create all LOOD folds from a list of datasets.

        Args:
            datasets: List of dataset names (e.g., ["vnat", "iscx", "usbvpn"])

        Returns:
            List of LOODFold objects
        """
        self.folds = []
        for test_ds in datasets:
            train_ds = [ds for ds in datasets if ds != test_ds]
            fold_name = f"Train on {', '.join(train_ds)} | Test on {test_ds}"
            fold = LOODFold(
                train_datasets=train_ds,
                test_dataset=test_ds,
                fold_name=fold_name
            )
            self.folds.append(fold)
            logger.info(f"Created fold: {fold_name}")

        return self.folds

    def prepare_lood_data(
        self,
        df_all: pd.DataFrame,
        fold: LOODFold,
        split_col: str = "split",
        dataset_col: str = "dataset"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and test data for a single LOOD fold.

        Args:
            df_all: Combined dataframe with all datasets
            fold: LOODFold specification
            split_col: Column name for split (train/val/test)
            dataset_col: Column name for dataset

        Returns:
            (df_train, df_test): Training and test dataframes
        """
        # Training: combine train splits from specified datasets
        train_masks = [
            (df_all[dataset_col] == ds) & (df_all[split_col] == "train")
            for ds in fold.train_datasets
        ]
        df_train = df_all[pd.concat(train_masks, axis=1).any(axis=1)].copy()

        # Validation: combine val splits from specified datasets
        val_masks = [
            (df_all[dataset_col] == ds) & (df_all[split_col] == "val")
            for ds in fold.train_datasets
        ]
        df_val = df_all[pd.concat(val_masks, axis=1).any(axis=1)].copy()

        # Test: use test split from held-out dataset
        df_test = df_all[
            (df_all[dataset_col] == fold.test_dataset) &
            (df_all[split_col] == "test")
        ].copy()

        logger.info(f"LOOD Fold: {fold.fold_name}")
        logger.info(f"  Train: {len(df_train)} flows")
        logger.info(f"  Val:   {len(df_val)} flows")
        logger.info(f"  Test:  {len(df_test)} flows")
        
        # Show VPN counts per dataset in train
        logger.info(f"  Train VPN by dataset:")
        for ds in fold.train_datasets:
            vpn_count = len(df_train[(df_train[dataset_col] == ds) & (df_train["label"] == 1)])
            logger.info(f"    {ds}: {vpn_count}")

        # Combine train and val for fitting pipeline
        df_train_val = pd.concat([df_train, df_val], ignore_index=True)

        return df_train_val, df_test

    def prepare_all_lood_data(
        self,
        df_all: pd.DataFrame,
        split_col: str = "split",
        dataset_col: str = "dataset"
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Prepare data for all LOOD folds.

        Returns:
            Dict mapping fold_id -> (df_train, df_test)
        """
        if not self.folds:
            raise ValueError("No LOOD folds created. Call create_folds() first.")

        data_dict = {}
        for fold in self.folds:
            df_train, df_test = self.prepare_lood_data(
                df_all, fold, split_col, dataset_col
            )
            data_dict[fold.fold_id] = (df_train, df_test)

        return data_dict

    def save_fold_results(
        self,
        fold: LOODFold,
        metrics: Dict,
        preds: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """
        Save results for a single LOOD fold.

        Args:
            fold: LOODFold specification
            metrics: Evaluation metrics dict
            preds: Predictions dataframe
            output_dir: Output directory for results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        fold_dir = output_dir / fold.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = fold_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
        logger.info(f"Saved metrics: {metrics_path}")

        # Save predictions
        preds_path = fold_dir / "predictions.parquet"
        preds.to_parquet(preds_path, index=False)
        logger.info(f"Saved predictions: {preds_path}")

        # Save fold metadata
        metadata = {
            "fold_id": fold.fold_id,
            "fold_name": fold.fold_name,
            "train_datasets": fold.train_datasets,
            "test_dataset": fold.test_dataset,
        }
        metadata_path = fold_dir / "fold_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

    def summarize_lood_results(
        self,
        results_dir: Path,
        metric_name: str = "auc"
    ) -> Dict[str, float]:
        """
        Summarize results across all LOOD folds.

        Args:
            results_dir: Directory containing fold results
            metric_name: Metric to summarize (e.g., "auc", "ap")

        Returns:
            Dict with summary statistics
        """
        fold_metrics = {}

        for fold in self.folds:
            fold_dir = results_dir / fold.fold_id
            metrics_path = fold_dir / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fold_metrics[fold.fold_id] = metrics.get(metric_name, np.nan)
                logger.info(f"{fold.fold_id}: {metric_name}={fold_metrics[fold.fold_id]:.4f}")

        summary = {
            "fold_metrics": fold_metrics,
            "mean": np.mean(list(fold_metrics.values())),
            "std": np.std(list(fold_metrics.values())),
            "min": np.min(list(fold_metrics.values())),
            "max": np.max(list(fold_metrics.values())),
        }

        return summary

    def print_lood_summary(self) -> None:
        """Print summary of all LOOD folds."""
        logger.info("\n" + "="*70)
        logger.info("LOOD (Leave-One-Out-Dataset) Evaluation Plan")
        logger.info("="*70)

        for i, fold in enumerate(self.folds, 1):
            logger.info(f"\nFold {i}: {fold.fold_name}")
            logger.info(f"  Fold ID: {fold.fold_id}")
            logger.info(f"  Training on: {', '.join(fold.train_datasets)}")
            logger.info(f"  Testing on: {fold.test_dataset}")

        logger.info(f"\nTotal folds: {len(self.folds)}")
        logger.info("="*70 + "\n")


def get_lood_folds(datasets: Optional[List[str]] = None) -> List[LOODFold]:
    """
    Convenience function to get all LOOD folds for standard datasets.

    Args:
        datasets: List of dataset names. If None, uses ["vnat", "iscx", "usbvpn"]

    Returns:
        List of LOODFold objects
    """
    if datasets is None:
        datasets = ["vnat", "iscx", "usbvpn"]

    evaluator = LOODEvaluator()
    return evaluator.create_folds(datasets)


if __name__ == "__main__":
    # Example usage
    evaluator = LOODEvaluator()
    folds = evaluator.create_folds(["vnat", "iscx", "usbvpn"])
    evaluator.print_lood_summary()

