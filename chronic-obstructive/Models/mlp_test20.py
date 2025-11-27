"""
Run the MLP neural network against the 20% test splits (test20.csv)
for both the Chronic Obstructive dataset and the Single Cluster dataset.

This script reuses the existing MLP ensemble defined in mlp_neural_network.py.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from mlp_neural_network import train_mlp


def load_split(dataset: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[LabelEncoder]]:
    """
    Load processed data and return train vs test20 split for the requested dataset.

    Parameters
    ----------
    dataset : str
        Either "chronic" (chronic obstructive) or "single" (single cluster).

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test20 : pd.DataFrame
    y_test20 : pd.Series
    label_encoder : Optional[LabelEncoder]
        Label encoder used for the single cluster dataset (None for chronic).
    """
    dataset = dataset.lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if dataset == "chronic":
        base_dir = os.path.join(project_root, "chronic-obstructive")
        id_col = "patient_id"
        target_col = "has_copd_risk"
        label_encoder = None
    elif dataset == "single":
        base_dir = os.path.join(project_root, "Single_cluster_classification")
        id_col = "sample_id"
        target_col = "category"
        label_encoder = LabelEncoder()
    else:
        raise ValueError("dataset must be either 'chronic' or 'single'")

    # Load processed features and targets
    X_full = pd.read_csv(os.path.join(base_dir, "processed_train_features.csv"))
    y_full = pd.read_csv(os.path.join(base_dir, "train_target.csv"))[target_col]

    # Encode labels if needed (single cluster dataset)
    if label_encoder is not None:
        y_full = pd.Series(label_encoder.fit_transform(y_full), name=target_col)
    else:
        y_full = y_full.astype(int)

    # Load original IDs to align rows
    train_ids = pd.read_csv(os.path.join(base_dir, "train.csv"))[id_col]
    if len(train_ids) != len(X_full):
        raise ValueError("Mismatch between train.csv and processed_train_features.csv lengths")

    # Load the 20% split file
    test20_path = os.path.join(base_dir, "test20.csv")
    if not os.path.exists(test20_path):
        raise FileNotFoundError(f"{test20_path} not found. Please generate test20.csv first.")
    test20_ids = pd.read_csv(test20_path)[id_col]

    mask = train_ids.isin(test20_ids)
    if mask.sum() == 0:
        raise ValueError("test20.csv has no overlapping IDs with train.csv")

    X_test20 = X_full[mask.values].reset_index(drop=True)
    y_test20 = y_full[mask.values].reset_index(drop=True)
    X_train = X_full[~mask.values].reset_index(drop=True)
    y_train = y_full[~mask.values].reset_index(drop=True)

    print(f"\nDataset: {dataset.capitalize()}")
    print(f"Train samples: {X_train.shape[0]}, Test20 samples: {X_test20.shape[0]}")
    return X_train, y_train, X_test20, y_test20, label_encoder


def evaluate_on_test20(dataset: str):
    """
    Train the MLP ensemble on the 80% training split and evaluate on the 20% split.
    """
    X_train, y_train, X_test20, y_test20, label_encoder = load_split(dataset)

    print("\n" + "=" * 80)
    print(f"TRAINING MLP ENSEMBLE FOR {dataset.upper()} DATASET")
    print("=" * 80)

    model = train_mlp(X_train, y_train)

    print("\nEvaluating on test20 split...")
    preds = model.predict(X_test20.to_numpy())

    if label_encoder is not None:
        # Convert predictions back to original labels for readability
        preds_labels = label_encoder.inverse_transform(preds)
        true_labels = label_encoder.inverse_transform(y_test20.to_numpy())
    else:
        preds_labels = preds
        true_labels = y_test20.to_numpy()

    avg = "macro" if dataset == "single" else "binary"
    f1 = f1_score(true_labels, preds_labels, average=avg)
    print(f"\nTest20 F1-score ({avg}): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds_labels))


def main():
    for dataset in ["chronic", "single"]:
        evaluate_on_test20(dataset)


if __name__ == "__main__":
    main()

