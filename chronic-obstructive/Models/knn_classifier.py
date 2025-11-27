"""
K-Nearest Neighbors Classifier pipeline for COPD risk prediction.
Loads the preprocessed chronic-obstructive feature matrices so the workflow stays
consistent with the other second-half models.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def load_data(data_dir: str | None = None):
    """Load *simple* preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR KNN CLASSIFIER")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if data_dir is None:
        data_dir = os.path.join(project_root, "chronic-obstructive")

    train_path = os.path.join(data_dir, "train_processed_simple.csv")
    test_path = os.path.join(data_dir, "test_processed_simple.csv")
    ids_path = os.path.join(data_dir, "ids_simple.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Simple preprocessing files not found. Run data_preprocessing_simple.py first."
        )

    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    if "has_copd_risk" in X_train.columns:
        y_train = X_train.pop("has_copd_risk")
    else:
        y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["has_copd_risk"]

    if os.path.exists(ids_path):
        test_patient_ids = pd.read_csv(ids_path)["patient_id"]
    else:
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        test_patient_ids = test_df["patient_id"]

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


def build_knn_pipeline(n_neighbors=15, weights="distance", metric="minkowski", p=2):
    """
    Build a StandardScaler + KNN pipeline.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to consult.
    weights : str
        'uniform' or 'distance'. Distance weighting helps with class imbalance.
    metric : str
        Distance metric for neighbors (e.g., 'minkowski', 'manhattan').
    p : int
        Power parameter for Minkowski metric (2 = Euclidean, 1 = Manhattan).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_knn_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    """Grid search for KNN hyperparameters using stratified 5-fold CV."""
    print("\n" + "=" * 80)
    print("GRID SEARCH FOR KNN")
    print("=" * 80)

    pipeline = build_knn_pipeline()

    param_grid = {
        "knn__n_neighbors": [7, 11, 15, 21, 31],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski", "manhattan"],
        "knn__p": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV F1-score: {grid.best_score_:.4f}")
    return grid.best_params_


def train_knn_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train the final KNN model after hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("TRAINING KNN CLASSIFIER")
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

    best_params = tune_knn_hyperparameters(X_tr, y_tr)

    model = build_knn_pipeline(
        n_neighbors=best_params["knn__n_neighbors"],
        weights=best_params["knn__weights"],
        metric=best_params["knn__metric"],
        p=best_params["knn__p"],
    )

    print("\nFitting model on training split...")
    model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    print("\n" + "-" * 80)
    print("VALIDATION METRICS")
    print("-" * 80)
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"F1-score: {f1_score(y_val, val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))

    print("\nRetraining best model on full dataset...")
    model.fit(X_np, y_np)
    return model


def make_predictions(model: Pipeline, X_test: pd.DataFrame):
    """Return class predictions for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = model.predict(X_test_np)
    return preds, None


def create_submission(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str | None = None,
):
    """Create submission matching sample_submission format."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "chronic-obstructive", "submission_knn.csv")

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
    model = train_knn_classifier(X_train, y_train)
    predictions, _ = make_predictions(model, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    main()


