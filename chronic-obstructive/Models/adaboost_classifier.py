"""
AdaBoost Classifier training pipeline for COPD risk prediction.
Loads the preprocessed feature matrices to stay consistent with other models.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(data_dir: str = None):
    """Load preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR ADABOOST")
    print("=" * 80)

    # Get the script's directory and resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_root, "chronic-obstructive")
    
    X_train = pd.read_csv(os.path.join(data_dir, "processed_train_features.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["has_copd_risk"]
    X_test = pd.read_csv(os.path.join(data_dir, "processed_test_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_patient_ids = test_df["patient_id"]

    # Basic cleanup so the model does not crash.
    if X_train.isnull().values.any():
        print("Detected NaNs in training features -> filling with 0.")
        X_train = X_train.fillna(0)
    if X_test.isnull().values.any():
        print("Detected NaNs in test features -> filling with 0.")
        X_test = X_test.fillna(0)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    print(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, test_patient_ids


def build_adaboost() -> AdaBoostClassifier:
    """
    AdaBoost Classifier with Decision Tree as base estimator.
    """
    base_estimator = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    
    return AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )


def train_adaboost(X_train: pd.DataFrame, y_train: pd.Series) -> AdaBoostClassifier:
    """Train an AdaBoost classifier."""
    print("\n" + "=" * 80)
    print("TRAINING ADABOOST")
    print("=" * 80)

    X_np = X_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        stratify=y_np,
        random_state=42,
    )

    model = build_adaboost()
    print("Fitting AdaBoost Classifier...")
    model.fit(X_tr, y_tr)

    print("\n" + "-" * 80)
    print("VALIDATION METRICS")
    print("-" * 80)
    val_pred = model.predict(X_val)

    print(f"Accuracy: {(val_pred == y_val).mean():.4f}")
    print(f"F1-score: {f1_score(y_val, val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))

    print("\nRetraining on the full dataset...")
    model.fit(X_np, y_np)
    return model


def make_predictions(model: AdaBoostClassifier, X_test: pd.DataFrame):
    """Return class predictions and probabilities for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = model.predict(X_test_np)
    return preds, None


def create_submission(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = None,
):
    """Create submission matching sample_submission format."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "chronic-obstructive", "submission_adaboost.csv")
    
    submission = pd.DataFrame(
        {
            "patient_id": patient_ids.values,
            "has_copd_risk": predictions.astype(int),
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print(submission.head())
    return submission


def main():
    X_train, y_train, X_test, patient_ids = load_data()
    model = train_adaboost(X_train, y_train)
    predictions, _ = make_predictions(model, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    main()

