"""
Basic Neural Network Model for COPD Risk Prediction
Single hidden layer neural network
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load processed features and target"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load processed features
    X_train = pd.read_csv('processed_train_features.csv')
    y_train = pd.read_csv('train_target.csv')['has_copd_risk']
    X_test = pd.read_csv('processed_test_features.csv')
    
    # Handle NaN values in training data
    if X_train.isnull().sum().sum() > 0:
        print("Filling NaN values in training data with 0...")
        X_train = X_train.fillna(0)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Load test patient IDs
    test_df = pd.read_csv('test.csv')
    test_patient_ids = test_df['patient_id']
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, test_patient_ids

def train_model(X_train, y_train):
    """Train a basic neural network with 1 hidden layer"""
    print("\n" + "=" * 80)
    print("TRAINING NEURAL NETWORK")
    print("=" * 80)
    
    # Split data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTraining set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val_split.shape[0]} samples")
    
    # Create neural network with 1 hidden layer
    # Using MLPClassifier from sklearn
    # hidden_layer_sizes=(100,) means 1 hidden layer with 100 neurons
    print("\nCreating neural network...")
    print("Architecture: Input -> Hidden Layer (100 neurons) -> Output")
    
    model = MLPClassifier(
        hidden_layer_sizes=(100,),  # 1 hidden layer with 100 neurons
        activation='relu',            # ReLU activation
        solver='adam',                # Adam optimizer
        alpha=0.0001,                 # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,                 # Maximum iterations
        shuffle=True,
        random_state=42,
        tol=0.0001,
        verbose=True,
        warm_start=False,
        early_stopping=True,          # Early stopping to prevent overfitting
        validation_fraction=0.1,     # 10% for validation during training
        n_iter_no_change=10          # Stop if no improvement for 10 iterations
    )
    
    print("\nTraining model...")
    model.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    
    y_val_pred = model.predict(X_val_split)
    y_val_pred_proba = model.predict_proba(X_val_split)[:, 1]
    
    # Calculate metrics
    accuracy = model.score(X_val_split, y_val_split)
    f1 = f1_score(y_val_split, y_val_pred)
    auc = roc_auc_score(y_val_split, y_val_pred_proba)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-Score: {f1:.4f}")
    print(f"Validation AUC-ROC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val_split, y_val_pred, 
                                target_names=['No Risk (0)', 'Has Risk (1)']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val_split, y_val_pred))
    
    # Retrain on full training set
    print("\n" + "=" * 80)
    print("RETRAINING ON FULL TRAINING SET")
    print("=" * 80)
    
    model_final = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=42,
        tol=0.0001,
        verbose=True,
        warm_start=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    print("\nTraining on full dataset...")
    model_final.fit(X_train, y_train)
    
    return model_final

def make_predictions(model, X_test):
    """Make predictions on test set"""
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    # Handle NaN values
    print("Checking for NaN values...")
    nan_count = X_test.isnull().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values. Filling with 0...")
        X_test = X_test.fillna(0)
    
    # Handle infinite values
    inf_count = np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"Found {inf_count} infinite values. Replacing with 0...")
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print("Predicting on test set...")
    predictions = model.predict(X_test)
    prediction_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for val, count in zip(unique, counts):
        pct = count / len(predictions) * 100
        print(f"  Class {val}: {count} ({pct:.2f}%)")
    
    return predictions, prediction_proba

def create_submission(test_patient_ids, predictions, output_file='submission.csv'):
    """Create submission file"""
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission_df = pd.DataFrame({
        'patient_id': test_patient_ids,
        'has_copd_risk': predictions.astype(int)
    })
    
    # Ensure patient_id is in the same order as test.csv
    submission_df = submission_df.sort_values('patient_id')
    
    submission_df.to_csv(output_file, index=False)
    print(f"\n✓ Submission file created: {output_file}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Columns: {submission_df.columns.tolist()}")
    
    print("\nFirst few predictions:")
    print(submission_df.head(10))
    
    return submission_df

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("COPD RISK PREDICTION - BASIC NEURAL NETWORK")
    print("=" * 80)
    
    # Load data
    X_train, y_train, X_test, test_patient_ids = load_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions, prediction_proba = make_predictions(model, X_test)
    
    # Create submission
    submission_df = create_submission(test_patient_ids, predictions)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nSubmission file saved as: submission.csv")
    print(f"Ready for submission!\n")

if __name__ == "__main__":
    main()

