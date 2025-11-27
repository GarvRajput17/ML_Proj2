"""
Advanced Ensemble Classifier combining Random Forest and XGBoost
for Signal Cluster Classification. Tests multiple ensemble methods:
- Weighted voting ensemble
- Stacking ensemble with meta-learner
- Optimized probability blending
Selects the best performing method automatically.
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
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    raise ImportError("XGBoost is required for this ensemble. Please install it using: pip install xgboost")


def load_data(data_dir: str = None):
    """Load preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR ENSEMBLE CLASSIFIER")
    print("=" * 80)

    # Get the script's directory and resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_root, "Signal_cluster_classification")
    
    X_train = pd.read_csv(os.path.join(data_dir, "processed_data", "processed_train_features.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["category"]
    X_test = pd.read_csv(os.path.join(data_dir, "processed_data", "processed_test_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_sample_ids = test_df["sample_id"]

    # Basic cleanup so the models do not crash.
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
    return X_train, y_train, X_test, test_sample_ids


def build_base_models():
    """Build base models for the ensemble - tuned for diversity (RF + XGBoost)."""
    models = []
    
    # Random Forest - aggressively tuned for maximum performance
    rf = RandomForestClassifier(
        n_estimators=500,  # More trees for better performance
        max_depth=None,  # No depth limit (use min_samples_split to control)
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features='sqrt',  # Feature subsampling for diversity
        bootstrap=True,
        oob_score=True,  # Out-of-bag scoring
    )
    models.append(("rf", rf))
    
    # XGBoost - aggressively tuned for maximum performance
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,  # More trees
            max_depth=10,  # Deeper
            learning_rate=0.01,  # Lower learning rate with more trees
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # Will be adjusted
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=2.0,  # L2 regularization
            gamma=0.1,  # Minimum loss reduction
            min_child_weight=1,
            max_delta_step=0,
        )
        models.append(("xgb", xgb_model))
    
    return models


def build_ensemble(models, voting="soft", weights=None):
    """
    Build voting ensemble from base models with optional weights.
    
    Parameters:
    -----------
    models : list of (name, model) tuples
    voting : str, 'hard' or 'soft'
        Hard voting uses class predictions, soft voting uses probabilities
    weights : list or None
        Weights for each model (if None, equal weights)
    """
    return VotingClassifier(
        estimators=models,
        voting=voting,
        weights=weights,  # Add weights for weighted voting
        n_jobs=-1,
    )


def build_stacking_ensemble(models):
    """
    Build stacking ensemble with meta-learner.
    
    Parameters:
    -----------
    models : list of (name, model) tuples
    
    Returns:
    --------
    StackingClassifier
    """
    # Meta-learner: Logistic Regression
    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    
    return StackingClassifier(
        estimators=models,
        final_estimator=meta_learner,
        cv=5,  # 5-fold CV for meta-features
        stack_method='predict_proba',  # Use probabilities
        n_jobs=-1,
    )


def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series, voting="soft"):
    """Train the ensemble classifier."""
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIER")
    print("=" * 80)

    X_np = X_train.to_numpy(dtype=np.float32)
    # For categorical target, keep as string/object type
    y_np = y_train.to_numpy()

    # Calculate scale_pos_weight for XGBoost if available
    if XGBOOST_AVAILABLE:
        class_counts = np.bincount(y_np)
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        print(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        stratify=y_np,
        random_state=42,
    )

    # Build base models
    print("\nBuilding base models...")
    base_models = build_base_models()
    
    # Update XGBoost scale_pos_weight if available
    if XGBOOST_AVAILABLE:
        for name, model in base_models:
            if name == "xgb":
                model.set_params(scale_pos_weight=scale_pos_weight)
                break
    
    print(f"Base models: {[name for name, _ in base_models]}")
    
    # Build ensemble
    print(f"\nCreating {voting} voting ensemble...")
    ensemble = build_ensemble(base_models, voting=voting)
    
    print("Training ensemble...")
    ensemble.fit(X_tr, y_tr)

    print("\n" + "-" * 80)
    print("VALIDATION METRICS")
    print("-" * 80)
    val_pred = ensemble.predict(X_val)

    print(f"Accuracy: {(val_pred == y_val).mean():.4f}")
    print(f"F1-score: {f1_score(y_val, val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))

    # Individual model performance and calculate weights
    print("\n" + "-" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE (on validation set)")
    print("-" * 80)
    model_scores = []
    for name, model in base_models:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        acc = (pred == y_val).mean()
        f1 = f1_score(y_val, pred)
        model_scores.append(f1)
        print(f"{name:15s} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    
    # Calculate weights based on F1 scores (normalized)
    # Higher F1 = higher weight
    if len(model_scores) > 0:
        # Normalize scores to sum to 1, but give more weight to better models
        total_score = sum(model_scores)
        if total_score > 0:
            weights = [score / total_score for score in model_scores]
            # Boost better models: square the normalized weights and renormalize
            weights = [w**2 for w in weights]
            weights = [w / sum(weights) for w in weights]
        else:
            weights = [1.0 / len(model_scores)] * len(model_scores)
        
        print(f"\nModel weights (based on F1-scores): {[f'{w:.3f}' for w in weights]}")
        
        # Test different ensemble methods
        print("\n" + "=" * 80)
        print("TESTING DIFFERENT ENSEMBLE METHODS")
        print("=" * 80)
        
        # 1. Equal weights voting
        print("\n1. Testing equal weights voting...")
        equal_ensemble = build_ensemble(base_models, voting=voting, weights=None)
        equal_ensemble.fit(X_tr, y_tr)
        equal_pred = equal_ensemble.predict(X_val)
        equal_f1 = f1_score(y_val, equal_pred)
        print(f"   Equal weights F1: {equal_f1:.4f}")
        
        # 2. Weighted voting
        print("\n2. Testing weighted voting...")
        weighted_ensemble = build_ensemble(base_models, voting=voting, weights=weights)
        weighted_ensemble.fit(X_tr, y_tr)
        weighted_pred = weighted_ensemble.predict(X_val)
        weighted_f1 = f1_score(y_val, weighted_pred)
        print(f"   Weighted voting F1: {weighted_f1:.4f}")
        
        # 3. Stacking ensemble
        print("\n3. Testing stacking ensemble (with meta-learner)...")
        stacking_ensemble = build_stacking_ensemble(base_models)
        stacking_ensemble.fit(X_tr, y_tr)
        stacking_pred = stacking_ensemble.predict(X_val)
        stacking_f1 = f1_score(y_val, stacking_pred)
        print(f"   Stacking F1: {stacking_f1:.4f}")
        
        # 4. Optimized probability blending with threshold tuning
        print("\n4. Testing optimized probability blending with threshold tuning...")
        # Get probabilities from all models
        rf_proba = None
        xgb_proba = None
        for name, model in base_models:
            model.fit(X_tr, y_tr)
            if name == "rf":
                rf_proba = model.predict_proba(X_val)[:, 1]
            elif name == "xgb":
                xgb_proba = model.predict_proba(X_val)[:, 1]
        
        # Try different blend weights AND thresholds
        best_blend_f1 = 0
        best_blend_weights = None
        best_blend_threshold = 0.5
        if rf_proba is not None and xgb_proba is not None:
            # Grid search over weights and thresholds
            for w1 in np.arange(0.2, 0.9, 0.05):
                w2 = 1.0 - w1
                blended_proba = w1 * rf_proba + w2 * xgb_proba
                # Try different thresholds
                for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
                    blended_pred = (blended_proba > threshold).astype(int)
                    blend_f1 = f1_score(y_val, blended_pred)
                    if blend_f1 > best_blend_f1:
                        best_blend_f1 = blend_f1
                        best_blend_weights = (w1, w2)
                        best_blend_threshold = threshold
            print(f"   Best blend F1: {best_blend_f1:.4f}")
            print(f"   Best weights: RF={best_blend_weights[0]:.2f}, XGB={best_blend_weights[1]:.2f}")
            print(f"   Best threshold: {best_blend_threshold:.2f}")
        
        # Select best ensemble method
        ensemble_results = [
            ("Equal weights voting", equal_ensemble, equal_f1, None, 0.5),
            ("Weighted voting", weighted_ensemble, weighted_f1, None, 0.5),
            ("Stacking", stacking_ensemble, stacking_f1, None, 0.5),
        ]
        
        if rf_proba is not None and xgb_proba is not None:
            ensemble_results.append(("Optimized blending", None, best_blend_f1, best_blend_weights, best_blend_threshold))
        
        best_name, best_ensemble, best_f1, best_blend_weights_final, best_threshold_final = max(ensemble_results, key=lambda x: x[2])
        
        print("\n" + "=" * 80)
        print(f"BEST ENSEMBLE METHOD: {best_name} (F1: {best_f1:.4f})")
        print("=" * 80)
        
        # If blending is best, create a custom ensemble class
        if best_name == "Optimized blending":
            class BlendedEnsemble:
                def __init__(self, models, weights, threshold=0.5):
                    self.models = models
                    self.weights = weights
                    self.threshold = threshold
                
                def fit(self, X, y):
                    for _, model in self.models:
                        model.fit(X, y)
                    return self
                
                def predict(self, X):
                    rf_proba = None
                    xgb_proba = None
                    for name, model in self.models:
                        if name == "rf":
                            rf_proba = model.predict_proba(X)[:, 1]
                        elif name == "xgb":
                            xgb_proba = model.predict_proba(X)[:, 1]
                    blended = self.weights[0] * rf_proba + self.weights[1] * xgb_proba
                    return (blended > self.threshold).astype(int)
            
            final_ensemble = BlendedEnsemble(base_models, best_blend_weights_final, best_threshold_final)
        else:
            final_ensemble = best_ensemble
    else:
        print("\nRetraining ensemble on the full dataset...")
        final_ensemble = ensemble
    
    print("\nRetraining best ensemble on full dataset...")
    final_ensemble.fit(X_np, y_np)
    return final_ensemble


def make_predictions(ensemble, X_test: pd.DataFrame):
    """Return class predictions and probabilities for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = ensemble.predict(X_test_np)
    return preds, None


def create_submission(
    sample_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = None,
):
    """Create submission matching sample_submission format."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "Signal_cluster_classification", "submissions", "submission_ensemble.csv")
    
    submission = pd.DataFrame(
        {
            "sample_id": sample_ids.values,
            "category": predictions,
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print(submission.head())
    return submission


def main():
    X_train, y_train, X_test, sample_ids = load_data()
    
    # Train with soft voting (uses probabilities when available)
    # Change to "hard" for majority voting
    ensemble = train_ensemble(X_train, y_train, voting="soft")
    
    predictions, _ = make_predictions(ensemble, X_test)
    create_submission(sample_ids, predictions)


if __name__ == "__main__":
    main()

