"""
Support Vector Machine (SVM) Solution for Chronic Obstructive Database
"""

import pandas as pd
import numpy as np
import warnings
import os

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
def load_data():
    """Load preprocessed features for Chronic Obstructive database"""
    print("Loading data...")
    
    # Get the script's directory and resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "chronic-obstructive")
    
    # Load preprocessed features
    X_train = pd.read_csv(os.path.join(data_dir, "processed_train_features.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["has_copd_risk"]
    X_test = pd.read_csv(os.path.join(data_dir, "processed_test_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_ids = test_df["patient_id"]
    
    # Basic cleanup
    if X_train.isnull().values.any():
        print("Detected NaNs in training features -> filling with 0.")
        X_train = X_train.fillna(0)
    if X_test.isnull().values.any():
        print("Detected NaNs in test features -> filling with 0.")
        X_test = X_test.fillna(0)
    
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print(f"Data Loaded: {X_train.shape}")
    print(f"Test Data: {X_test.shape}")
    return X_train, y_train, X_test, test_ids

def train_svm_model(X_train, y_train):
    print("\n" + "="*50)
    print("TRAINING SUPPORT VECTOR MACHINE (SVM)")
    print("="*50)
    
    # 1. Split for Honest Validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 2. Define Pipeline
    # RobustScaler is CRITICAL for SVM to handle outliers without shifting the margin
    pipeline = Pipeline([
        ('scaler', RobustScaler()), 
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])

    # 3. Hyperparameter Tuning (The most important part for SVM)
    # C: Stritcness. High C = "Make no mistakes" (Complex). Low C = "Allow mistakes" (Simple).
    # gamma: Reach. High gamma = "Only look at close points". Low gamma = "Look at far points".
    param_grid = {
        'svm__C': [1, 10, 50, 100, 200],  
        'svm__gamma': ['scale', 0.1, 0.5, 1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running Grid Search (Tuning C and Gamma)...")
    grid = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='f1',  # Binary classification uses 'f1'
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_split, y_train_split)

    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.5f}")

    # 4. Validation
    best_model = grid.best_estimator_
    val_preds = best_model.predict(X_val_split)
    val_score = f1_score(y_val_split, val_preds)
    
    print(f"\nValidation F1: {val_score:.5f}")
    print(classification_report(y_val_split, val_preds))

    # 5. Visualization skipped (too many features for 2D plot)
    # With 239 features, 2D visualization is not meaningful

    # 6. Final Training
    print("\nRetraining on FULL dataset...")
    best_model.fit(X_train, y_train)
    
    return best_model

def main():
    # Load
    X_train, y_train, X_test, test_ids = load_data()
    
    # Train
    model = train_svm_model(X_train, y_train)
    
    # Predict
    print("\nGenerating Predictions...")
    preds = model.predict(X_test)
    
    # Save in chronic-obstructive format
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "chronic-obstructive")
    
    sub = pd.DataFrame({
        'patient_id': test_ids,
        'has_copd_risk': preds.astype(int)
    })
    out_path = os.path.join(output_dir, 'submission_svm_new.csv')
    sub.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! SVM Submission saved to: {out_path}")
    print(sub.head())

if __name__ == "__main__":
    main()