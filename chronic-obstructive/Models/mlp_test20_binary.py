"""
Evaluate the MLP ensemble (from mlp_neural_network.py) on the 20% holdout
split for either the chronic obstructive dataset (binary) or the
Single Cluster dataset (multiclass, using its data_preprocessing.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from mlp_neural_network import train_mlp


def ensure_processed_features(base_dir: str, dataset: str):
    """
    Ensure processed feature files exist. For the Single Cluster dataset,
    call its data_preprocessing.py to generate them if missing.
    """
    processed_path = os.path.join(base_dir, "processed_train_features.csv")
    if os.path.exists(processed_path):
        return

    if dataset == "single":
        sys.path.append(base_dir)
        from data_preprocessing import preprocess_data  # type: ignore

        preprocess_data(
            train_path=os.path.join(base_dir, "train.csv"),
            test_path=os.path.join(base_dir, "test.csv"),
            feature_types=["linear", "quadratic", "logarithmic", "other", "statistical"],
            use_scaling=True,
            scaler_type="standard",
            save_processed=True,
        )
        print("Generated processed features for Single Cluster dataset.")
    else:
        raise FileNotFoundError(
            f"{processed_path} missing. Please run chronic data preprocessing first."
        )


def load_split(dataset: str):
    """
    Load 80/20 split for the requested dataset.

    Parameters
    ----------
    dataset : str
        'chronic' or 'single'
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
        raise ValueError("dataset must be 'chronic' or 'single'")

    ensure_processed_features(base_dir, dataset)

    X_full = pd.read_csv(os.path.join(base_dir, "processed_train_features.csv"))
    y_full = pd.read_csv(os.path.join(base_dir, "train_target.csv"))[target_col]

    if label_encoder is not None:
        y_full = pd.Series(label_encoder.fit_transform(y_full), name=target_col)
    else:
        y_full = y_full.astype(int)

    ids = pd.read_csv(os.path.join(base_dir, "train.csv"))[id_col]

    test20_path = os.path.join(base_dir, "test20.csv")
    if not os.path.exists(test20_path):
        raise FileNotFoundError(f"{test20_path} not found. Run the splitter first.")
    test20_ids = pd.read_csv(test20_path)[id_col]

    mask = ids.isin(test20_ids)
    if mask.sum() == 0:
        raise ValueError("test20 split did not match any ID entries.")

    X_test20 = X_full[mask.values].reset_index(drop=True)
    y_test20 = y_full[mask.values].reset_index(drop=True)
    X_train = X_full[~mask.values].reset_index(drop=True)
    y_train = y_full[~mask.values].reset_index(drop=True)

    print(f"{dataset.capitalize()} data -> train: {X_train.shape[0]}, test20: {X_test20.shape[0]}")
    return X_train, y_train, X_test20, y_test20, label_encoder


def evaluate(dataset: str):
    """Train on the 80% split and evaluate on the dataset-specific test20 split."""
    X_train, y_train, X_test20, y_test20, label_encoder = load_split(dataset)

    print("\n" + "=" * 80)
    print(f"TRAINING MLP ENSEMBLE ({dataset.upper()})")
    print("=" * 80)
    model = train_mlp(X_train, y_train)

    print(f"\nEvaluating on {dataset} test20 split...")
    preds = model.predict(X_test20.to_numpy())

    if label_encoder is not None:
        preds_labels = label_encoder.inverse_transform(preds)
        true_labels = label_encoder.inverse_transform(y_test20.to_numpy())
        average = "macro"
    else:
        preds_labels = preds
        true_labels = y_test20.to_numpy()
        average = "binary"

    f1 = f1_score(true_labels, preds_labels, average=average)
    print(f"\nF1-score ({average}): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds_labels))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP ensemble on test20 split.")
    parser.add_argument(
        "--dataset",
        choices=["chronic", "single"],
        default="chronic",
        help="Dataset to evaluate: 'chronic' (binary) or 'single' (multiclass).",
    )
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()

