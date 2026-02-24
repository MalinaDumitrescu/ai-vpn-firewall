from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

def plot_roc_curves(
    df_preds: pd.DataFrame,
    label_col: str = "label",
    prob_col: str = "p_raw",
    split_col: str = "split",
    title: str = "ROC Curves",
    save_path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(8, 6))
    
    splits = sorted(df_preds[split_col].unique())
    for split in splits:
        subset = df_preds[df_preds[split_col] == split]
        if len(subset) == 0 or subset[label_col].nunique() < 2:
            continue
            
        y_true = subset[label_col]
        y_score = subset[prob_col]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{split} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_curves(
    df_preds: pd.DataFrame,
    label_col: str = "label",
    prob_col: str = "p_raw",
    split_col: str = "split",
    title: str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(8, 6))
    
    splits = sorted(df_preds[split_col].unique())
    for split in splits:
        subset = df_preds[df_preds[split_col] == split]
        if len(subset) == 0 or subset[label_col].nunique() < 2:
            continue
            
        y_true = subset[label_col]
        y_score = subset[prob_col]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{split} (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrices(
    df_preds: pd.DataFrame,
    label_col: str = "label",
    prob_col: str = "p_raw",
    split_col: str = "split",
    threshold: float = 0.5,
    save_dir: Optional[Path] = None,
) -> None:
    splits = sorted(df_preds[split_col].unique())
    
    for split in splits:
        subset = df_preds[df_preds[split_col] == split]
        if len(subset) == 0:
            continue
            
        y_true = subset[label_col]
        y_pred = (subset[prob_col] >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {split} (Thr={threshold})')
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"confusion_matrix_{split}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_probability_distributions(
    df_preds: pd.DataFrame,
    label_col: str = "label",
    prob_col: str = "p_raw",
    split_col: str = "split",
    save_dir: Optional[Path] = None,
) -> None:
    splits = sorted(df_preds[split_col].unique())
    
    for split in splits:
        subset = df_preds[df_preds[split_col] == split]
        if len(subset) == 0:
            continue
            
        plt.figure(figsize=(8, 6))
        sns.histplot(data=subset, x=prob_col, hue=label_col, bins=50, kde=True, element="step", common_norm=False)
        plt.title(f'Probability Distribution - {split}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count (Density)')
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"prob_dist_{split}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
