"""
Evaluate the MLP ensemble (from mlp_neural_network.py) on the 20% holdout
split for the Single Cluster dataset (multiclass classification).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from mlp_neural_network import train_mlp


def _patch_f1_score_for_multiclass():
    """
    Monkeypatch sklearn.metrics.f1_score so that any call without an 'average'
    argument defaults to 'macro'. This prevents train_mlp (which assumes binary)
    from crashing when we feed multiclass targets.
    """
    import sklearn.metrics as sk_metrics

    original_f1 = sk_metrics.f1_score

    def patched_f1(y_true, y_pred, *args, **kwargs):
        kwargs.setdefault("average", "macro")
        return original_f1(y_true, y_pred, *args, **kwargs)

    sk_metrics.f1_score = patched_f1


def load_single_split():
    """Load 80/20 split for the Single Cluster dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "Single_cluster_classification")

    X_full = pd.read_csv(os.path.join(base_dir, "processed_train_features.csv"))
    y_full = pd.read_csv(os.path.join(base_dir, "train_target.csv"))["category"]
    ids = pd.read_csv(os.path.join(base_dir, "train.csv"))["sample_id"]

    label_encoder = LabelEncoder()
    y_full_encoded = pd.Series(label_encoder.fit_transform(y_full), name="category_encoded")

    test20_path = os.path.join(base_dir, "test20.csv")
    if not os.path.exists(test20_path):
        raise FileNotFoundError(f"{test20_path} not found. Run the splitter first.")
    test20_ids = pd.read_csv(test20_path)["sample_id"]

    mask = ids.isin(test20_ids)
    if mask.sum() == 0:
        raise ValueError("test20 split did not match any sample_id entries.")

    X_test20 = X_full[mask.values].reset_index(drop=True)
    y_test20 = y_full_encoded[mask.values].reset_index(drop=True)
    X_train = X_full[~mask.values].reset_index(drop=True)
    y_train = y_full_encoded[~mask.values].reset_index(drop=True)

    print(f"Single-cluster data -> train: {X_train.shape[0]}, test20: {X_test20.shape[0]}")
    return X_train, y_train, X_test20, y_test20, label_encoder


def evaluate_multiclass():
    """Train on the 80% split and evaluate on the single-cluster test20 split."""
    _patch_f1_score_for_multiclass()
    X_train, y_train, X_test20, y_test20, label_encoder = load_single_split()

    print("\n" + "=" * 80)
    print("TRAINING MLP ENSEMBLE (SINGLE CLUSTER)")
    print("=" * 80)
    model = train_mlp(X_train, y_train)

    print("\nEvaluating on single-cluster test20 split...")
    preds = model.predict(X_test20.to_numpy())

    # Convert back to original labels for readability
    true_labels = label_encoder.inverse_transform(y_test20.to_numpy())
    pred_labels = label_encoder.inverse_transform(preds)

    f1 = f1_score(true_labels, pred_labels, average="macro")
    print(f"\nMacro F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))


def main():
    evaluate_multiclass()


if __name__ == "__main__":
    main()

