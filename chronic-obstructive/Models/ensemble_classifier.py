"""
Advanced Ensemble Classifier combining Random Forest and XGBoost
for COPD risk prediction. Tests multiple ensemble methods:
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
)
from sklearn.linear_model import LogisticRegression

# Import necessary libraries for data processing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --- XGBOOST CHECK ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost is NOT installed. The ensemble will run with only Random Forest.")
    
    

def load_data():
    """Load preprocessed feature matrices and targets from current directory/data/."""
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA FOR ENSEMBLE")
    print("=" * 80)
    
    # Load files assuming standard names from previous steps
    try:
        X_train = pd.read_csv('data/train_processed_simple.csv')
        y_train = pd.read_csv('data/train_target.csv')["has_copd_risk"]
        X_test = pd.read_csv('data/test_processed_simple.csv')
        test_patient_ids = pd.read_csv('data/test.csv')["patient_id"]
        
        # Drop the target from the X_train features loaded above (if still present)
        X_train = X_train.drop(["has_copd_risk"], axis=1, errors='ignore')
        
    except FileNotFoundError:
        print("FATAL ERROR: Required processed files not found. Check FILE_PATH_PREFIX.")
        raise
        
    # --- Final Data Cleaning (Robustness Check) ---
    for df in [X_train, X_test]:
        if df.isnull().values.any():
            df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
    print(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, test_patient_ids


def build_base_models(scale_pos_weight: float):
    """Build base models for the ensemble (RF + XGBoost)."""
    models = []
    
    # Random Forest - tuned for diversity and performance
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced", # Handles imbalance
        random_state=42,
        n_jobs=-1,
        max_features='sqrt',
        bootstrap=True,
    )
    models.append(("rf", rf))
    
    # XGBoost - tuned for maximum performance with class imbalance
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5, # Reduced depth for stability
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight, # Critical for F1 optimization
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            reg_alpha=0.1, 
            reg_lambda=2.0,
        )
        models.append(("xgb", xgb_model))
    
    return models


def build_ensemble(models: list, voting: str = "soft", weights: list = None):
    """Build voting ensemble from base models."""
    return VotingClassifier(
        estimators=models,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )


def build_stacking_ensemble(models: list):
    """Build stacking ensemble with Logistic Regression meta-learner."""
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
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
    )


def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series, voting: str = "soft"):
    """Train the ensemble classifier and test all combination methods."""
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIER AND TESTING METHODS")
    print("=" * 80)

    # 1. Calculate class weight for XGBoost
    X_np = X_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)
    class_counts = np.bincount(y_np)
    scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0

    # 2. Split for internal validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        stratify=y_np,
        random_state=42,
    )

    # 3. Build and Configure Base Models
    base_models = build_base_models(scale_pos_weight)
    
    # 4. Individual Model Performance and Calculate Weights (for weighted voting)
    model_scores = []
    print("\n--- Individual Model Validation Scores ---")
    for name, model in base_models:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred)
        model_scores.append(f1)
        print(f"{name:15s} - F1-score: {f1:.4f}")
    
    # Normalize scores to create weights
    if sum(model_scores) > 0:
        weights = [score / sum(model_scores) for score in model_scores]
        # Squared normalization to boost influence of better models
        weights = [w**2 for w in weights]
        weights = [w / sum(weights) for w in weights]
    else:
        weights = [1.0 / len(base_models)] * len(base_models)
    print(f"\nModel weights (for soft voting): {[f'{w:.3f}' for w in weights]}")
    
    
    # 5. Testing Different Ensemble Methods
    ensemble_results = []
    
    # a. Weighted Voting
    weighted_ensemble = build_ensemble(base_models, voting=voting, weights=weights)
    weighted_ensemble.fit(X_tr, y_tr)
    weighted_f1 = f1_score(y_val, weighted_ensemble.predict(X_val))
    ensemble_results.append(("Weighted Voting", weighted_ensemble, weighted_f1))
    
    # b. Stacking
    stacking_ensemble = build_stacking_ensemble(base_models)
    stacking_ensemble.fit(X_tr, y_tr)
    stacking_f1 = f1_score(y_val, stacking_ensemble.predict(X_val))
    ensemble_results.append(("Stacking Ensemble", stacking_ensemble, stacking_f1))
    
    # c. Optimized Blending (Uses RF and XGBoost probabilities)
    if XGBOOST_AVAILABLE:
        best_blend_f1, best_blend_weights, best_blend_threshold = _optimize_blending(base_models, X_tr, y_tr, X_val, y_val)
        ensemble_results.append(("Optimized Blending", None, best_blend_f1, best_blend_weights, best_blend_threshold))
    
    
    # 6. Select Best Ensemble Method
    best_result = max(ensemble_results, key=lambda x: x[2])
    best_name, best_ensemble, best_f1 = best_result[0], best_result[1], best_result[2]

    print("\n" + "=" * 80)
    print(f"BEST PERFORMING ENSEMBLE: {best_name} (F1: {best_f1:.4f})")
    print("=" * 80)
    
    # 7. Final Retraining on Full Dataset
    if best_name == "Optimized Blending":
        # Create custom blend class and train it
        final_ensemble = _BlendedEnsemble(base_models, best_result[3], best_result[4])
    else:
        final_ensemble = best_ensemble
        
    print("Retraining best ensemble on FULL training dataset...")
    final_ensemble.fit(X_np, y_np)
    
    return final_ensemble


def _optimize_blending(models, X_tr, y_tr, X_val, y_val):
    """Internal function to search for optimal blend weights and threshold."""
    rf_model = next(model for name, model in models if name == "rf")
    xgb_model = next(model for name, model in models if name == "xgb")
    
    rf_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    
    rf_proba = rf_model.predict_proba(X_val)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    best_blend_f1 = 0
    best_blend_weights = (0.5, 0.5)
    best_blend_threshold = 0.5
    
    # Search over weights and thresholds
    for w1 in np.arange(0.2, 0.8, 0.05):
        w2 = 1.0 - w1
        blended_proba = w1 * rf_proba + w2 * xgb_proba
        
        for threshold in np.arange(0.4, 0.6, 0.02):
            blended_pred = (blended_proba > threshold).astype(int)
            blend_f1 = f1_score(y_val, blended_pred)
            
            if blend_f1 > best_blend_f1:
                best_blend_f1 = blend_f1
                best_blend_weights = (w1, w2)
                best_blend_threshold = threshold
                
    return best_blend_f1, best_blend_weights, best_blend_threshold


class _BlendedEnsemble:
    """Custom class to handle the final blending model."""
    def __init__(self, models, weights, threshold=0.5):
        self.models = models
        self.weights = weights
        self.threshold = threshold
    
    def fit(self, X, y):
        # Train base models on full data
        for _, model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        rf_proba = self.models[0][1].predict_proba(X)[:, 1]
        xgb_proba = self.models[1][1].predict_proba(X)[:, 1]
        
        blended = self.weights[0] * rf_proba + self.weights[1] * xgb_proba
        return (blended > self.threshold).astype(int)


def make_predictions(ensemble: object, X_test: pd.DataFrame):
    """Return class predictions for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = ensemble.predict(X_test_np)
    return preds, None


def create_submission(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = "data/submission_ensemble.csv",
):
    """Create submission matching sample_submission format."""
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
    
    # Train with soft voting (uses probabilities when available)
    ensemble = train_ensemble(X_train, y_train, voting="soft")
    
    predictions, _ = make_predictions(ensemble, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    # To run this script, ensure XGBoost is installed and your processed files are ready.
    # Execute the main function:
    main()