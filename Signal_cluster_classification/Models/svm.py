"""
Support Vector Machine (SVM) Solution
"""

import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
ORIGINAL_TRAIN_FILE = 'train.csv'
ORIGINAL_TEST_FILE  = 'test.csv'
OUTPUT_FOLDER = 'output'

def load_data():
    """Load Raw Data (SVM prefers raw signal inputs)"""
    print("Loading data...")
    paths = ['', 'data/']
    train_df, test_df = None, None
    
    for p in paths:
        try:
            if train_df is None:
                train_df = pd.read_csv(os.path.join(p, ORIGINAL_TRAIN_FILE))
                test_df = pd.read_csv(os.path.join(p, ORIGINAL_TEST_FILE))
        except FileNotFoundError:
            continue
            
    if train_df is None:
        raise FileNotFoundError("Could not find 'train.csv'.")

    # We use only the raw signals. 
    # SVM's RBF kernel automatically handles the non-linearity (squares, curves).
    X_train = train_df[['signal_strength', 'response_level']].fillna(0)
    y_train = train_df['category']
    X_test = test_df[['signal_strength', 'response_level']].fillna(0)
    test_ids = test_df['sample_id']
    
    print(f"Data Loaded: {X_train.shape}")
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
        scoring='f1_macro', 
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_split, y_train_split)

    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.5f}")

    # 4. Validation
    best_model = grid.best_estimator_
    val_preds = best_model.predict(X_val_split)
    val_score = f1_score(y_val_split, val_preds, average='macro')
    
    print(f"\nValidation Macro F1: {val_score:.5f}")
    print(classification_report(y_val_split, val_preds))

    # 5. Visualization (Verify the "Margin")
    try:
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
        
        plt.figure(figsize=(10, 6))
        # Create mesh
        x_min, x_max = X_train['signal_strength'].min() - 1, X_train['signal_strength'].max() + 1
        y_min, y_max = X_train['response_level'].min() - 1, X_train['response_level'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
        
        # Predict
        Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Convert to numbers for plotting
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(y_train)
        Z = le.transform(Z).reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        sns.scatterplot(x=X_train_split['signal_strength'], y=X_train_split['response_level'], hue=y_train_split, palette='viridis', edgecolor='k', s=20)
        plt.title("SVM Decision Boundary (RBF Kernel)")
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'svm_boundary.png'))
        print("Boundary plot saved.")
    except Exception as e:
        print(f"Skipping plot: {e}")

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
    
    # Save
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    sub = pd.DataFrame({'sample_id': test_ids, 'category': preds})
    out_path = os.path.join(OUTPUT_FOLDER, 'submission_svm.csv')
    sub.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! SVM Submission saved to: {out_path}")

if __name__ == "__main__":
    main()