"""
Advanced Logistic Regression Model for SignalCluster Classification
Structure adapted for professional deployment.
"""

import pandas as pd
import numpy as np
import warnings
import os  # <--- Added this to handle folder creation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure your files are in the SAME folder as this script, or update these paths
TRAIN_FEATURES_FILE = 'data/processed_train_features.csv'
TEST_FEATURES_FILE  = 'data/processed_test_features.csv'
ORIGINAL_TRAIN_FILE = 'data/train.csv'
ORIGINAL_TEST_FILE  = 'data/test.csv'
OUTPUT_FOLDER = 'output' # <--- Name of the folder where results will go

def load_data():
    """Load processed features, target, and IDs with error handling"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # 1. Load Processed Features
    try:
        X_train = pd.read_csv(TRAIN_FEATURES_FILE)
        X_test = pd.read_csv(TEST_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find file: {e.filename}")
        print("Suggestion: Ensure all CSV files are in the same folder as this script.")
        return None, None, None, None
    
    # 2. Load Target and IDs (from original files)
    try:
        train_orig = pd.read_csv(ORIGINAL_TRAIN_FILE)
        test_orig = pd.read_csv(ORIGINAL_TEST_FILE)
        
        y_train = train_orig['category']
        test_ids = test_orig['sample_id']
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find original file: {e.filename}")
        print("Suggestion: Ensure 'train.csv' and 'test.csv' are also in this folder.")
        return None, None, None, None

    # 3. Handle NaN/Inf values (Safety Check)
    if X_train.isnull().sum().sum() > 0:
        print("Filling NaN values in training data with 0...")
        X_train = X_train.fillna(0)
    if X_test.isnull().sum().sum() > 0:
        X_test = X_test.fillna(0)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape:   {y_train.shape}")
    print(f"Test features shape:     {X_test.shape}")
    
    return X_train, y_train, X_test, test_ids

def train_model(X_train, y_train):
    """Train Logistic Regression with GridSearchCV and Validation"""
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 80)
    
    # 1. Split data for Hold-out Validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set:   {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val_split.shape[0]} samples")

    # 2. Setup Hyperparameter Tuning
    print("\nSetting up GridSearchCV...")
    model = LogisticRegression(class_weight='balanced', max_iter=5000, solver='lbfgs')
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1
    )
    
    # 3. Train on the Split Training Data
    print("Running Grid Search on training split...")
    grid.fit(X_train_split, y_train_split)
    
    best_model = grid.best_estimator_
    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Score (Macro F1): {grid.best_score_:.4f}")

    # 4. Evaluate on Validation Set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    
    y_val_pred = best_model.predict(X_val_split)
    
    macro_f1 = f1_score(y_val_split, y_val_pred, average='macro')
    print(f"Validation Macro F1-Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val_split, y_val_pred))
    
    # 5. Retrain on FULL Training Set
    print("\n" + "=" * 80)
    print("RETRAINING ON FULL DATASET")
    print("=" * 80)
    print("Retraining best model on all available data (Train + Validation)...")
    
    final_model = grid.best_estimator_
    final_model.fit(X_train, y_train)
    
    return final_model

def make_predictions(model, X_test):
    """Make predictions on test set"""
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    print("Predicting on test set...")
    predictions = model.predict(X_test)
    
    return predictions

def create_submission(test_ids, predictions, filename='submission.csv'):
    """Create submission file inside output folder"""
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    # 1. Create Output Folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Creating folder: '{OUTPUT_FOLDER}'...")
        os.makedirs(OUTPUT_FOLDER)
    
    # 2. Create DataFrame
    submission_df = pd.DataFrame({
        'sample_id': test_ids,
        'category': predictions
    })
    
    # 3. Save to the Output Folder
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Shape: {submission_df.shape}")
    print("\nFirst few predictions:")
    print(submission_df.head())
    
    return submission_df

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("SIGNAL CLUSTER CLASSIFICATION - LOGISTIC REGRESSION")
    print("=" * 80)
    
    # 1. Load
    X_train, y_train, X_test, test_ids = load_data()
    
    # Only proceed if data loaded successfully
    if X_train is not None:
        # 2. Train
        model = train_model(X_train, y_train)
        
        # 3. Predict
        predictions = make_predictions(model, X_test)
        
        # 4. Submit
        create_submission(test_ids, predictions)
        
        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)

if __name__ == "__main__":
    main()