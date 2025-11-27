"""
Logistic Regression Classifier with K-Fold Cross-Validation for COPD risk prediction.
Uses 5-fold CV to select best hyperparameters and validation split.
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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: str = None):
    """Load preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR LOGISTIC REGRESSION")
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


def build_logistic_regression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000):
    """
    Build Logistic Regression pipeline with scaling.
    
    Parameters:
    -----------
    C : float
        Inverse of regularization strength (smaller = stronger regularization)
    penalty : str
        Regularization type: 'l1' or 'l2'
    solver : str
        Solver algorithm: 'lbfgs', 'liblinear', 'saga'
    max_iter : int
        Maximum iterations
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ))
    ])


def k_fold_cross_validation(X_train, y_train, k=5):
    """
    Perform K-fold cross-validation and return best performing fold.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training labels
    k : int
        Number of folds
    
    Returns:
    --------
    dict: Best fold information and all fold results
    """
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION (K={k})")
    print("=" * 80)
    
    X_np = X_train.to_numpy(dtype=np.float32) if isinstance(X_train, pd.DataFrame) else X_train
    y_np = y_train.to_numpy(dtype=np.int64) if isinstance(y_train, pd.Series) else y_train
    
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Hyperparameters to try
    hyperparameters = [
        {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
        {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'},
        {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'},
        {'C': 1.0, 'penalty': 'l1', 'solver': 'saga'},
        {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'},
    ]
    
    best_score = -1
    best_params = None
    best_fold_idx = -1
    all_results = []
    
    print(f"\nTesting {len(hyperparameters)} hyperparameter combinations...")
    
    for param_idx, params in enumerate(hyperparameters):
        print(f"\n--- Testing: C={params['C']}, penalty={params['penalty']}, solver={params['solver']} ---")
        
        fold_scores = []
        fold_details = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            X_tr_fold = X_np[train_idx]
            X_val_fold = X_np[val_idx]
            y_tr_fold = y_np[train_idx]
            y_val_fold = y_np[val_idx]
            
            # Build and train model
            model = build_logistic_regression(
                C=params['C'],
                penalty=params['penalty'],
                solver=params['solver']
            )
            model.fit(X_tr_fold, y_tr_fold)
            
            # Evaluate on validation fold
            val_pred = model.predict(X_val_fold)
            val_f1 = f1_score(y_val_fold, val_pred)
            val_acc = accuracy_score(y_val_fold, val_pred)
            
            fold_scores.append(val_f1)
            fold_details.append({
                'fold': fold_idx + 1,
                'f1_score': val_f1,
                'accuracy': val_acc,
                'params': params
            })
            
            print(f"  Fold {fold_idx + 1}: F1={val_f1:.4f}, Acc={val_acc:.4f}")
        
        # Average score across folds for this hyperparameter set
        avg_score = np.mean(fold_scores)
        print(f"  Average F1-score: {avg_score:.4f}")
        
        all_results.append({
            'params': params,
            'avg_f1': avg_score,
            'fold_scores': fold_scores,
            'fold_details': fold_details
        })
        
        # Track best hyperparameters
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_fold_idx = param_idx
    
    # Find best individual fold across all hyperparameters
    best_individual_fold = None
    best_individual_score = -1
    
    for result in all_results:
        for fold_detail in result['fold_details']:
            if fold_detail['f1_score'] > best_individual_score:
                best_individual_score = fold_detail['f1_score']
                best_individual_fold = fold_detail
                best_individual_fold['params'] = result['params']
    
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest hyperparameters (average across folds):")
    print(f"  C={best_params['C']}, penalty={best_params['penalty']}, solver={best_params['solver']}")
    print(f"  Average F1-score: {best_score:.4f}")
    
    print(f"\nBest individual fold:")
    print(f"  Fold {best_individual_fold['fold']}")
    print(f"  F1-score: {best_individual_fold['f1_score']:.4f}")
    print(f"  Accuracy: {best_individual_fold['accuracy']:.4f}")
    print(f"  Parameters: C={best_individual_fold['params']['C']}, penalty={best_individual_fold['params']['penalty']}")
    
    return {
        'best_params': best_params,
        'best_avg_score': best_score,
        'best_individual_fold': best_individual_fold,
        'all_results': all_results
    }


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train Logistic Regression with K-fold CV to select best configuration."""
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 80)

    X_np = X_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)

    print(f"Training samples: {X_np.shape[0]}")
    print(f"Features: {X_np.shape[1]}")
    print(f"Class distribution - Class 0: {np.sum(y_np == 0)}, Class 1: {np.sum(y_np == 1)}")

    # Perform 5-fold cross-validation
    cv_results = k_fold_cross_validation(X_train, y_train, k=5)
    
    # Use best hyperparameters from CV
    best_params = cv_results['best_params']
    
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    print(f"Using best hyperparameters: C={best_params['C']}, penalty={best_params['penalty']}, solver={best_params['solver']}")
    
    # Train final model on full dataset
    final_model = build_logistic_regression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver']
    )
    
    print("Training on full dataset...")
    final_model.fit(X_np, y_np)
    
    # Evaluate on training set for reference
    train_pred = final_model.predict(X_np)
    train_f1 = f1_score(y_np, train_pred)
    train_acc = accuracy_score(y_np, train_pred)
    
    print(f"\nTraining set metrics:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  F1-score: {train_f1:.4f}")
    
    return final_model


def make_predictions(model: Pipeline, X_test: pd.DataFrame):
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
        output_path = os.path.join(project_root, "chronic-obstructive", "submission_lr.csv")
    
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
    model = train_logistic_regression(X_train, y_train)
    predictions, _ = make_predictions(model, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    main()

